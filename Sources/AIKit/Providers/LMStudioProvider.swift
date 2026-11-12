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

    /// The settings row this provider serves: who it is, where it is, which
    /// model, and how long a request may take.
    ///
    /// Held rather than copied field by field (W22, BR-AIL-20). A local LM
    /// Studio and the one on the Mac across the desk are the same type with a
    /// different address, and that address came from settings — validated once
    /// when the document loaded, so `run` has nothing left to parse.
    public let entry: AIProviderEntry

    /// What settings call this provider, and what a failed attempt in a
    /// fallback chain names (``AgentProviding/id``).
    public var id: String { entry.id }
    /// The LM Studio server root. A server ROOT, no path (BR-AIL-09).
    ///
    /// Non-optional because the init refuses an entry without one, so there is
    /// no empty-string sentinel here for a caller to mistake for an address.
    public let baseURL: String
    /// Model identifier as loaded in LM Studio.
    public var model: String { entry.model }
    /// Request timeout in seconds, as a ceiling on the caller's own timeout.
    public var requestTimeout: TimeInterval { entry.requestTimeout }
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

    /// Serves one configured LM Studio entry.
    ///
    /// - Parameters:
    ///   - entry: the settings row. Its address was validated when the
    ///     document loaded (BR-AIL-18).
    ///   - logger: Logger for diagnostics.
    ///   - transport: Transport override. Tests inject a stub; production
    ///     leaves it nil so `run` can build one carrying the caller's timeout.
    /// - Throws: ``AIKitError/settings(file:entry:reason:)`` when the row has
    ///   no server address — a `cli-subprocess` or `mlx-in-process` row, which
    ///   this provider cannot serve. Refused HERE rather than carried as an
    ///   empty `baseURL` that reads as an address until the first request.
    public init(
        entry: AIProviderEntry,
        logger: Logger = Logger(label: "shikki.lmstudio-provider"),
        transport: (any AIHTTPTransport)? = nil
    ) throws(AIKitError) {
        baseURL = try entry.serverAddress().description
        self.entry = entry
        self.logger = logger
        injectedTransport = transport
    }

    /// Serves an LM Studio server at a configured address.
    ///
    /// Non-optional and throwing, both deliberately (W22, BR-AIL-18 and
    /// BR-AIL-20). What this replaces read `LMSTUDIO_URL` and `LMSTUDIO_MODEL`
    /// from the process environment and fell back to `http://127.0.0.1:1234`
    /// and `local-model`, so a host that had set neither got a provider
    /// pointing at its own machine and a model name no server has — and a
    /// library initializer decided it. Both reads are gone from Sources; the
    /// address is now validated here, once, instead of at the first request.
    ///
    /// - Throws: ``AIKitError/invalidServerAddress(value:reason:)`` naming the
    ///   value.
    public init(
        id: String = "lmstudio",
        baseURL: String,
        model: String,
        requestTimeout: TimeInterval = AIProviderEntry.defaultRequestTimeout,
        logger: Logger = Logger(label: "shikki.lmstudio-provider"),
        transport: (any AIHTTPTransport)? = nil
    ) throws(AIKitError) {
        try self.init(
            entry: try AIProviderEntry(
                id: id,
                family: .openAIChat,
                address: baseURL,
                model: model,
                requestTimeout: requestTimeout
            ),
            logger: logger,
            transport: transport
        )
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

    /// The provider that does the wire work, built for one call because the
    /// timeout it must honour is that call's argument.
    ///
    /// Nothing is validated here any more: the entry arrived validated.
    func provider(timeout: TimeInterval) -> HTTPAIProvider {
        HTTPAIProvider(
            entry: entry,
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
        let address = try entry.serverAddress()
        let transport = injectedTransport ?? URLSessionHTTPTransport(timeout: requestTimeout)
        let reply: LMStudioModelsReply = try await transport.sendRequest(
            endpoint: LMStudioEndPoint.models(address: address, credential: nil)
        )
        return reply.data
    }

    /// Asks the host to load a model — `POST /api/v1/models/load`.
    public func load(model: String, contextLength: Int? = nil, ttl: Int? = nil) async throws {
        let address = try entry.serverAddress()
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
        let address = try entry.serverAddress()
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
                target: AIRequestTarget(entryID: id, address: address, route: endPoint.path)
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
        case .settings:
            // Same reasoning, one level up: the DOCUMENT is wrong, so no
            // provider in it is "down" and falling back would hide the edit
            // the operator has to make.
            return .invalidResponse(aiKitError.localizedDescription)
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
public struct LMStudioHostedModel: Sendable, Codable, Equatable {
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
///
/// `Encodable` for the reason given on ``ChatCompletionResponseDTO/model``: a
/// test fixture is MINTED from this declaration rather than hand-written
/// beside it (W24).
struct LMStudioModelsReply: Sendable, Codable, Equatable {
    let data: [LMStudioHostedModel]
}
