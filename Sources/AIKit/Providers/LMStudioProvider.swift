import Foundation
import Logging
import NetKit

// MARK: - LMStudioProvider

/// `AgentProviding` over LM Studio's OpenAI-compatible API.
/// Enables local model fallback when cloud providers hit rate limits.
///
/// LM Studio exposes POST /v1/chat/completions at http://127.0.0.1:1234 by default.
/// Supports any model loaded in LM Studio — model name is configurable.
///
/// The wire work belongs to `OpenAIProvider`: this type holds LM Studio's
/// configuration, renders a prompt into an `AIRequest`, and translates what
/// comes back into `LMStudioError` so a chain's fallback rule keeps reading the
/// cases it always read. It used to carry its own `URLSession` and
/// `[String: Any]` body, a second implementation of a wire format the package
/// already had (BR-DISC-A).
public struct LMStudioProvider: AgentProviding, Sendable {
    /// Base URL for the LM Studio server (no trailing slash).
    public let baseURL: String
    /// Model identifier as loaded in LM Studio.
    public let model: String
    /// Request timeout in seconds, as a ceiling on the caller's own timeout.
    public let requestTimeout: TimeInterval
    /// Logger for diagnostics.
    private let logger: Logger
    /// Set only when a caller injects a transport, which tests do (BR-AIL-12).
    /// Left nil in production because the timeout is an argument to `run`, not
    /// to `init`, so the transport that carries it is built per call.
    private let injectedNetworkService: (any NetworkProtocol)?

    // MARK: - Errors

    public enum LMStudioError: Error, Equatable, Sendable {
        case connectionRefused(String)
        case rateLimited
        case invalidResponse(String)
        case httpError(Int, String)
        case emptyContent
    }

    // MARK: - Init

    /// Create a new LMStudioProvider.
    /// - Parameters:
    ///   - baseURL: LM Studio server URL. Defaults to env `LMSTUDIO_URL` or `http://127.0.0.1:1234`.
    ///   - model: Model name. Defaults to env `LMSTUDIO_MODEL` or `local-model`.
    ///   - requestTimeout: Ceiling for a single request in seconds. Defaults to 300.
    ///   - logger: Logger for diagnostics.
    ///   - networkService: Transport override. Tests inject a mock; production
    ///     leaves it nil so `run` can build one carrying the caller's timeout.
    public init(
        baseURL: String? = nil,
        model: String? = nil,
        requestTimeout: TimeInterval = 300,
        logger: Logger = Logger(label: "shikki.lmstudio-provider"),
        networkService: (any NetworkProtocol)? = nil
    ) {
        self.baseURL = baseURL
            ?? ProcessInfo.processInfo.environment["LMSTUDIO_URL"]
            ?? "http://127.0.0.1:1234"
        self.model = model
            ?? ProcessInfo.processInfo.environment["LMSTUDIO_MODEL"]
            ?? "local-model"
        self.requestTimeout = requestTimeout
        self.logger = logger
        injectedNetworkService = networkService
    }

    // MARK: - AgentProviding

    public func run(prompt: String, timeout: TimeInterval) async throws -> String {
        logger.info("LM Studio request", metadata: [
            "model": "\(model)",
            "url": "\(baseURL)",
        ])

        let response: AIResponse
        do {
            response = try await provider(timeout: timeout).complete(request: makeRequest(prompt: prompt))
        } catch {
            throw mapped(error)
        }

        let content = response.content.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !content.isEmpty else {
            throw LMStudioError.emptyContent
        }
        return content
    }

    // MARK: - Request Building

    /// The provider that does the wire work, built for one call because the
    /// timeout it must honour is that call's argument.
    func provider(timeout: TimeInterval) -> OpenAIProvider {
        OpenAIProvider(
            id: "lmstudio",
            displayName: "LM Studio",
            capabilities: .textGeneration,
            baseURL: baseURL,
            modelName: model,
            apiKey: nil,
            networkService: injectedNetworkService
                ?? TimeoutNetworkService(timeoutInterval: min(timeout, requestTimeout))
        )
    }

    /// The typed request for a raw prompt.
    ///
    /// `maxTokens` is -1, which is what this provider has always sent: LM Studio
    /// reads it as "answer until the context is full", and a positive ceiling
    /// would start truncating answers that are not truncated today.
    func makeRequest(prompt: String) -> AIRequest {
        AIRequest(
            messages: [AIMessage(role: .user, content: prompt)],
            temperature: 0.7,
            maxTokens: -1
        )
    }

    // MARK: - Error Mapping

    /// Translates what the transport and `OpenAIProvider` throw into this
    /// provider's own cases, which `FallbackProviderChain.isFallbackEligible`
    /// reads: `.connectionRefused` and `.rateLimited` mean try the next
    /// provider, the rest mean the server answered and said no.
    func mapped(_ error: any Error) -> LMStudioError {
        switch error {
        case let error as LMStudioError:
            return error
        case let error as AIKitError:
            return mapped(aiKitError: error)
        case let error as NetworkError:
            return mapped(networkError: error)
        case let error as URLError:
            return mapped(urlError: error)
        default:
            return .invalidResponse("\(error)")
        }
    }

    private func mapped(aiKitError: AIKitError) -> LMStudioError {
        switch aiKitError {
        case .requestFailed(let message) where message == OpenAIProvider.emptyReplyMessage:
            return .emptyContent
        case .requestFailed(let message):
            // A 200 reply carrying an error body: the server's own words
            // (BR-AIL-10). It answered, so this is not fallback-eligible.
            return .invalidResponse(message)
        default:
            return .invalidResponse(aiKitError.localizedDescription)
        }
    }

    private func mapped(networkError: NetworkError) -> LMStudioError {
        switch networkError {
        case .unexpectedStatusCode(429, _):
            return .rateLimited
        case .unexpectedStatusCode(let code, _):
            // NetKit discards a non-2xx body, so the status code and the URL
            // are all there is to report here. The 200-with-error-body path
            // above is the one that carries the server's text.
            return .httpError(code, "HTTP \(code) from \(baseURL)")
        case .jsonParsingFailed(let error):
            return .invalidResponse("Failed to decode response: \(error)")
        case .requestFailed(let description):
            return .invalidResponse(description)
        case .invalidData:
            return .invalidResponse("Empty response body")
        case .wsError(let description):
            return .invalidResponse(description)
        case .unknown(let underlying):
            // NetKit wraps a transport failure here, and `URLError` inside it is
            // exactly what decides fallback — so unwrap instead of reporting
            // "unknown".
            return mapped(underlying)
        }
    }

    private func mapped(urlError: URLError) -> LMStudioError {
        switch urlError.code {
        case .cannotConnectToHost, .networkConnectionLost, .timedOut, .cannotFindHost:
            return .connectionRefused(
                "Cannot connect to LM Studio at \(baseURL): \(urlError.localizedDescription)"
            )
        default:
            return .connectionRefused(
                "Network error: \(urlError.localizedDescription)"
            )
        }
    }
}
