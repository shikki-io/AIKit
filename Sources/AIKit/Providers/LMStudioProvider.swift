import Foundation
import Logging
import NetKit

// MARK: - LMStudioProvider

/// What is LM Studio-specific about an LM Studio server, beyond the wire.
///
/// The wire work is ``HTTPAIProvider`` with the `openai-chat` codec. What is
/// left here, and could not live anywhere else, is two things:
///
/// - an `AgentProviding` face (prompt in, text out) with LM Studio's own error
///   vocabulary, which a `FallbackProviderChain` reads;
/// - the model MANAGEMENT calls, which are on LM Studio's native API and have
///   no equivalent on the OpenAI surface: load, unload, list, download.
///
/// It carried its own `URLSession` and `[String: Any]` body once — a second
/// implementation of a wire format the package already had (BR-DISC-A).
public struct LMStudioProvider: AgentProviding, Sendable {
    /// Base URL for the LM Studio server, as it was configured. A server
    /// ROOT, no path (BR-AIL-09).
    public let baseURL: String
    /// The same server, parsed and validated once in `init` (BR-AIL-18).
    public let address: AIServerAddress
    /// Model identifier as loaded in LM Studio.
    public let model: String
    /// Request timeout in seconds, as a ceiling on the caller's own timeout.
    public let requestTimeout: TimeInterval
    /// Logger for diagnostics.
    private let logger: Logger
    /// Set only when a caller injects a transport, which tests do (BR-AIL-12).
    /// Left nil in production because the timeout is an argument to `run`, not
    /// to `init`, so the transport that carries it is built per call.
    private let injectedTransport: (any AIHTTPTransport)?

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
    ///   - baseURL: LM Studio server root, with no path (BR-AIL-09). Falls
    ///     back to env `LMSTUDIO_URL`, and to NOTHING after that.
    ///   - model: Model name. Defaults to env `LMSTUDIO_MODEL` or `local-model`.
    ///   - requestTimeout: Ceiling for a single request in seconds. Defaults to 300.
    ///   - logger: Logger for diagnostics.
    ///   - transport: Transport override. Tests inject a stub; production
    ///     leaves it nil so `run` can build one carrying the caller's timeout.
    ///
    /// Throwing since W21. The address used to end
    /// `?? "http://127.0.0.1:1234"`, so a host that had configured no LM
    /// Studio at all was given one on its own loopback: the request went out,
    /// failed to connect, and reported a dead server rather than an empty
    /// setting (BR-AIL-18). There is no localhost default here now — an
    /// unconfigured provider refuses to exist.
    ///
    /// The address is also VALIDATED here, not at first use, so a bad one is
    /// named where it was written.
    ///
    /// FIXME(spec 8badc87f-e9e0-4bfd-913f-a52f52ba70fc W22): BR-AIL-20 forbids a library initializer reading an
    /// environment variable, and the two reads below are the ones it names.
    /// W22 of spec ai-layer-aikit-core-2026-10-31 makes `baseURL` a required
    /// argument and deletes `LMSTUDIO_URL` and `LMSTUDIO_MODEL` from Sources,
    /// with the values arriving as an ``AIProviderEntry`` through CoreKit DI.
    ///
    /// - Throws: ``AIKitError/serverAddressNotConfigured(setting:)`` when
    ///   neither the argument nor the environment names a server, and
    ///   ``AIKitError/invalidServerAddress(value:reason:)`` when what they
    ///   name is not a server root.
    public init(
        baseURL: String? = nil,
        model: String? = nil,
        requestTimeout: TimeInterval = AIProviderEntry.defaultRequestTimeout,
        logger: Logger = Logger(label: "shikki.lmstudio-provider"),
        transport: (any AIHTTPTransport)? = nil
    ) throws(AIKitError) {
        guard let configured = baseURL ?? ProcessInfo.processInfo.environment["LMSTUDIO_URL"] else {
            throw .serverAddressNotConfigured(setting: "LMSTUDIO_URL")
        }
        // Validated here so `entry()` cannot be the first thing to notice, and
        // so `address` is the one copy of the parsed value.
        self.address = try AIServerAddress(validating: configured)
        self.baseURL = configured
        self.model =
            model
            ?? ProcessInfo.processInfo.environment["LMSTUDIO_MODEL"]
            ?? "local-model"
        self.requestTimeout = requestTimeout
        self.logger = logger
        injectedTransport = transport
    }

    // MARK: - AgentProviding

    public func run(prompt: String, timeout: TimeInterval) async throws -> String {
        logger.info(
            "LM Studio request",
            metadata: [
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

    /// The settings entry this provider's requests are built from.
    ///
    /// Non-throwing: the address was validated in `init`, which is what
    /// "validated once" means (BR-AIL-18).
    func entry() -> AIProviderEntry {
        AIProviderEntry(
            id: "lmstudio",
            family: .openAIChat,
            address: address,
            model: model,
            requestTimeout: requestTimeout
        )
    }

    /// The provider that does the wire work, built for one call because the
    /// timeout it must honour is that call's argument.
    ///
    /// Non-throwing for the same reason `entry()` is: the address was
    /// validated in `init`.
    func provider(timeout: TimeInterval) -> HTTPAIProvider {
        HTTPAIProvider(
            entry: entry(),
            codec: OpenAIChatCodec(),
            network: injectedTransport
                ?? URLSessionHTTPTransport(timeout: min(timeout, requestTimeout)),
            displayName: "LM Studio"
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

    // MARK: - Model management (LM Studio's native API)

    /// Every model on the host, loaded or not — `GET /api/v1/models`.
    ///
    /// On LM Studio's own API rather than the OpenAI-compatible one, because
    /// `GET /v1/models` lists only what is LOADED. This is the half of this
    /// server that has no OpenAI equivalent, and the reason this type still
    /// exists after the wire moved out.
    public func hostedModels() async throws -> [LMStudioHostedModel] {
        let transport = injectedTransport ?? URLSessionHTTPTransport(timeout: requestTimeout)
        let reply: LMStudioModelsReply = try await transport.sendRequest(
            endpoint: LMStudioEndPoint.models(address: address, credential: nil)
        )
        return reply.data
    }

    /// Asks the host to load a model — `POST /api/v1/models/load`.
    public func load(model: String, contextLength: Int? = nil, ttl: Int? = nil) async throws {
        try await send(
            LMStudioEndPoint.loadModel(
                address: address,
                credential: nil,
                body: LMStudioModelLoadRequest(model: model, contextLength: contextLength, ttl: ttl)
            ),
            address: address
        )
    }

    /// Asks the host to unload a model — `POST /api/v1/models/unload`.
    public func unload(model: String) async throws {
        try await send(
            LMStudioEndPoint.unloadModel(
                address: address, credential: nil, body: LMStudioModelUnloadRequest(model: model)),
            address: address
        )
    }

    /// Sends a management call and reads the status, so a refusal is an error
    /// rather than a silently discarded reply (BR-AIL-19).
    ///
    /// The route in the error comes from the endpoint, never from a literal
    /// repeated here: the enum is the SSoT for it (BR-AIL-16).
    private func send(_ endPoint: any EndPoint, address: AIServerAddress) async throws {
        let transport = injectedTransport ?? URLSessionHTTPTransport(timeout: requestTimeout)
        let reply = try await transport.send(try transport.createRequest(endPoint: endPoint))
        guard 200..<300 ~= reply.status else {
            throw AIKitError.http(
                status: reply.status,
                body: reply.text,
                target: AIRequestTarget(entryID: "lmstudio", address: address, route: endPoint.path)
            )
        }
    }

    // MARK: - Error Mapping

    /// Translates what the transport and ``HTTPAIProvider`` throw into this
    /// provider's own cases, which `FallbackProviderChain.isFallbackEligible`
    /// reads: `.connectionRefused` and `.rateLimited` mean try the next
    /// provider, the rest mean the server answered and said no.
    func mapped(_ error: any Error) -> LMStudioError {
        switch error {
        case let error as LMStudioError:
            return error
        case let error as AIKitError:
            return mapped(aiKitError: error)
        case let error as AIHTTPTransportError:
            return mapped(transportError: error)
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
        case .emptyReply:
            // A CASE, not a sentence. This used to read
            // `message.hasPrefix(AIWireStandard.emptyReplyPrefix(…))`, so the
            // same wire event got two verdicts depending on how its message
            // had been composed (AIKit#18, panel finding on
            // `LMStudioProvider.swift:150`).
            return .emptyContent
        case .http(let status, let body, _):
            // BR-AIL-19: the status decides, and the server's own text travels
            // with it. 429 is its own case because that is the one a chain
            // falls back on.
            if status == 429 {
                return .rateLimited
            }
            return .httpError(status, body.isEmpty ? "HTTP \(status) from \(baseURL)" : body)
        case .requestFailed(let message):
            // A 2xx reply carrying an error body: the server's own words
            // (BR-AIL-10). It answered, so this is not fallback-eligible.
            return .invalidResponse(message)
        case .replyUndecodable(_, let reason):
            return .invalidResponse("Failed to decode response: \(reason)")
        case .streamLineTooLong(let cap, _):
            return .invalidResponse("A streamed line passed the \(cap)-byte cap")
        case .unsupported(let field, let family):
            return .invalidResponse("\(family.displayName) cannot carry '\(field)'")
        case .invalidServerAddress(let value, let reason):
            // Deliberately NOT a connection failure: nothing was sent, and
            // reporting it as one would hide a misconfigured address behind a
            // retry on the next provider.
            return .invalidResponse("Not a server address: '\(value)' — \(reason)")
        default:
            return .invalidResponse(aiKitError.localizedDescription)
        }
    }

    private func mapped(transportError: AIHTTPTransportError) -> LMStudioError {
        switch transportError {
        case .lineExceededCap(let cap):
            return .invalidResponse("A streamed line passed the \(cap)-byte cap")
        case .notAnHTTPResponse:
            return .invalidResponse("The answer was not an HTTP answer")
        }
    }

    private func mapped(networkError: NetworkError) -> LMStudioError {
        switch networkError {
        case .unexpectedStatusCode(429, _):
            return .rateLimited
        case .unexpectedStatusCode(let code, _):
            // Reached only through NetKit's own `sendRequest`, which the
            // management calls above use for a decodable reply. The completion
            // path reads the status itself and keeps the body.
            return .httpError(code, "HTTP \(code) from \(baseURL)")
        case .jsonParsingFailed(let error):
            return .invalidResponse("Failed to decode response: \(error)")
        case .requestFailed(let description):
            return .invalidResponse(description)
        case .invalidData:
            return .invalidResponse("Empty response body")
        case .wsError(let description):
            return .invalidResponse(description)
        case .invalidURL(let description):
            // NetKit 3.0.0: the endpoint formed no URL, so NOTHING was sent.
            // Deliberately NOT fallback-eligible — this is a misconfigured
            // address, and trying the next provider would hide the settings
            // bug behind a working one.
            return .invalidResponse("Endpoint formed no URL: \(description)")
        case .bodyEncodingFailed(let cause):
            // Also never sent, and never a server's fault: a request body JSON
            // cannot represent. Falling back would send the same bad body to
            // the next provider and report its refusal instead of this cause.
            return .invalidResponse("Request body could not be encoded: \(cause)")
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

// MARK: - Hosted models

/// One model on an LM Studio host, as `GET /api/v1/models` reports it.
public struct LMStudioHostedModel: Sendable, Decodable, Equatable {
    public let id: String
    /// `llm`, `embeddings`, … as the host classifies it.
    public let type: String?
    /// Whether the host currently has it in memory.
    public let isLoaded: Bool?

    enum CodingKeys: String, CodingKey {
        case id, type
        case isLoaded = "is_loaded"
    }
}

/// The `GET /api/v1/models` envelope.
struct LMStudioModelsReply: Sendable, Decodable {
    let data: [LMStudioHostedModel]
}
