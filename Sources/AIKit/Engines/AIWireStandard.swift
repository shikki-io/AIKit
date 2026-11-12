import Foundation
import NetKit

// MARK: - AIWireStandard

/// A wire standard an AI server speaks, declared rather than discovered.
///
/// One server speaks several of these at once. Measured on the operator's LM
/// Studio on 2026-10-31, every case below answered HTTP 200 with
/// `mistralai_devstral-small-2-24b-instruct-2512-mlx` loaded — the same model,
/// the same port, four different request and reply shapes. So a server is not
/// identified by its wire standard, and a wire standard is not identified by
/// its server: the pair is what a provider declares.
///
/// BR-AIL-09. `kind` in `providers.toml` names the standard, and the standard
/// owns the whole path, version prefix included. Nothing parses a configured
/// URL looking for `/v1`, and nothing sends a probe to find out what a server
/// speaks before dispatching: discovery proves (`shi providers discover`, W8),
/// runtime trusts what `kind` declared.
public enum AIWireStandard: String, Sendable, Codable, CaseIterable {
    /// OpenAI chat completions: `/v1/chat/completions`.
    case openAIChat = "openai-chat"
    /// OpenAI responses: `/v1/responses`.
    case openAIResponses = "openai-responses"
    /// Anthropic messages: `/v1/messages`.
    case anthropicMessages = "anthropic-messages"
    /// LM Studio's own API: `/api/v1/chat`.
    case lmStudioNative = "lmstudio-native"

    /// The full path this standard posts to, version prefix included.
    ///
    /// This is the only place any of these four paths is written. A caller that
    /// wants a URL asks the standard; it never builds one by appending to a
    /// configured string, which is the mistake BR-AIL-09 exists to forbid.
    public var path: String {
        switch self {
        case .openAIChat: "/v1/chat/completions"
        case .openAIResponses: "/v1/responses"
        case .anthropicMessages: "/v1/messages"
        case .lmStudioNative: "/api/v1/chat"
        }
    }

    /// Human-readable name, for `shi providers discover` output and errors.
    public var displayName: String {
        switch self {
        case .openAIChat: "OpenAI chat completions"
        case .openAIResponses: "OpenAI responses"
        case .anthropicMessages: "Anthropic messages"
        case .lmStudioNative: "LM Studio native"
        }
    }

    /// Reads the standard a `providers.toml` `kind` names.
    ///
    /// `openai-compatible` is the value all 7 existing provider profiles carry,
    /// written when nothing read `kind` at all. It means the chat-completions
    /// standard, which is what those profiles have always talked to, so it maps
    /// there rather than failing a config that was valid yesterday.
    ///
    /// Mistral (La Plateforme, and a local mistral.rs server) speaks the same
    /// chat-completions standard — `POST /v1/chat/completions`, bearer auth,
    /// the same request and reply shapes — so `mistral` and the two
    /// `AIProviderID` spellings map to `.openAIChat` instead of a fifth family
    /// that would duplicate it.
    public init?(kind: String) {
        let normalised = kind.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        switch normalised {
        case "openai-compatible", "openai", "lmstudio", "ollama",
            "mistral", AIProviderID.mistralLaPlateforme.rawValue, AIProviderID.mistralRsLocal.rawValue:
            self = .openAIChat
        case "anthropic":
            self = .anthropicMessages
        default:
            guard let parsed = AIWireStandard(rawValue: normalised) else { return nil }
            self = parsed
        }
    }
}

// MARK: - AIWireServer

/// A server root: scheme, host, port, and nothing else.
///
/// BR-AIL-09 makes `base_url` the server root, so this type's job is to refuse
/// anything more. A root that still carries `/v1` is a configuration mistake
/// that the standard's own path would turn into `/v1/v1/chat/completions` —
/// which LM Studio answers with HTTP 200 and an error body, the failure the
/// silent-fail table calls "looks like a bad reply". Naming it here, at parse
/// time, is cheaper than reading it back out of a reply. Nothing is stripped:
/// the caller's configuration is wrong and gets told so.
public struct AIWireServer: Sendable, Equatable {
    public let scheme: String
    public let host: String
    public let port: Int?

    public init(scheme: String, host: String, port: Int?) {
        self.scheme = scheme
        self.host = host
        self.port = port
    }

    /// Reads a server root such as `http://127.0.0.1:1234`.
    /// - Throws: `AIKitError.requestFailed` naming the leftover path when the
    ///   root carries one, or the string when it is not a URL at all.
    public init(serverRoot: String) throws {
        let trimmed = serverRoot.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let components = URLComponents(string: trimmed), let host = components.host, !host.isEmpty else {
            throw AIKitError.requestFailed(
                "Not a server root: \(serverRoot.isEmpty ? "<empty>" : serverRoot). Expected something like http://127.0.0.1:1234"
            )
        }
        // A trailing slash is a root written with a slash, not a path.
        let leftover = components.path.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        guard leftover.isEmpty else {
            throw AIKitError.requestFailed(
                """
                base_url must be a server root with no path, but \(trimmed) carries '/\(leftover)'. \
                The wire standard owns the path (BR-AIL-09) — drop '/\(leftover)' from the configuration.
                """
            )
        }
        self.init(scheme: components.scheme ?? "http", host: host, port: components.port)
    }

    /// The root as written, for an error that has to name where it pointed.
    public var description: String {
        var text = "\(scheme)://\(host)"
        if let port { text += ":\(port)" }
        return text
    }
}

// MARK: - AIWireTarget

/// Where a request goes and as whom: one provider's configuration.
///
/// This is one `providers.toml` row minus the `kind` — the standard is chosen
/// separately, which is the whole point of BR-AIL-09, so one target can be
/// called on two standards and one standard on two targets. Grouping these four
/// also keeps the call sites below readable: a standard's `complete` takes a
/// request, a target and a transport, not a seven-argument list in which two
/// `String`s could be swapped without the compiler noticing.
public struct AIWireTarget: Sendable {
    /// Scheme, host and port — no path (BR-AIL-09).
    public let server: AIWireServer
    /// Bearer token or `x-api-key` value; nil for a local server.
    public let apiKey: String?
    /// The model name as the server knows it.
    public let model: String
    /// The `AIProvider.id` every answer from this target carries (BR-AIL-06).
    public let providerID: String

    public init(server: AIWireServer, apiKey: String? = nil, model: String, providerID: String) {
        self.server = server
        self.apiKey = apiKey
        self.model = model
        self.providerID = providerID
    }

    /// Builds a target from a configured server root.
    /// - Throws: `AIKitError.requestFailed` when `serverRoot` is not a root.
    public init(serverRoot: String, apiKey: String? = nil, model: String, providerID: String) throws {
        self.init(
            server: try AIWireServer(serverRoot: serverRoot),
            apiKey: apiKey,
            model: model,
            providerID: providerID
        )
    }
}

// MARK: - Dispatch

extension AIWireStandard {
    /// Builds the endpoint this standard declares for one request.
    ///
    /// Returned erased because the standards are three different `EndPoint`
    /// enums; a caller that only needs a `URLRequest` — `shi providers
    /// discover`, a test asserting a path — does not care which.
    ///
    /// - Throws: An encoding error when the request body cannot be serialised.
    ///   The caller must propagate this rather than silently sending a bodyless
    ///   request.
    func endPoint(target: AIWireTarget, request: AIRequest, stream: Bool) throws -> any EndPoint {
        switch self {
        case .openAIChat:
            let endpoint = OpenAIEndPoint.chat(
                server: target.server,
                apiKey: target.apiKey,
                body: ChatCompletionRequest(request: request, model: target.model, stream: stream)
            )
            _ = try endpoint.bodyObject()
            return endpoint
        case .openAIResponses:
            let endpoint = OpenAIEndPoint.responses(
                server: target.server,
                apiKey: target.apiKey,
                model: target.model,
                request: request,
                stream: stream
            )
            _ = try endpoint.bodyObject()
            return endpoint
        case .anthropicMessages:
            let endpoint = AnthropicEndPoint.messages(
                server: target.server,
                apiKey: target.apiKey,
                model: target.model,
                request: request,
                stream: stream
            )
            _ = try endpoint.bodyObject()
            return endpoint
        case .lmStudioNative:
            let endpoint = LMStudioEndPoint.chat(
                server: target.server,
                apiKey: target.apiKey,
                model: target.model,
                request: request,
                stream: stream
            )
            _ = try endpoint.bodyObject()
            return endpoint
        }
    }

    /// Sends one request on this standard and maps the reply.
    ///
    /// The transport is injected, so no unit test reaches a server (BR-AIL-12).
    /// A transport failure travels as the type it was thrown as, because a
    /// caller deciding whether to fall back needs to tell a refused connection
    /// from a server that answered (BR-AIL-07, W5).
    func complete(
        request: AIRequest,
        target: AIWireTarget,
        networkService: any NetworkProtocol
    ) async throws -> AIResponse {
        let start = ContinuousClock.now
        let endpoint = try endPoint(target: target, request: request, stream: false)
        let map: (Int, String) throws -> AIResponse
        switch self {
        case .openAIChat:
            let reply: ChatCompletionReply = try await networkService.sendRequest(endpoint: endpoint)
            map = { try reply.response(requestedModel: target.model, providerID: target.providerID, latencyMs: $0, requestURL: $1) }
        case .openAIResponses:
            let reply: ResponsesReply = try await networkService.sendRequest(endpoint: endpoint)
            map = { try reply.response(requestedModel: target.model, providerID: target.providerID, latencyMs: $0, requestURL: $1) }
        case .anthropicMessages:
            let reply: AnthropicMessagesReply = try await networkService.sendRequest(endpoint: endpoint)
            map = { try reply.response(requestedModel: target.model, providerID: target.providerID, latencyMs: $0, requestURL: $1) }
        case .lmStudioNative:
            let reply: LMStudioNativeReply = try await networkService.sendRequest(endpoint: endpoint)
            map = { try reply.response(requestedModel: target.model, providerID: target.providerID, latencyMs: $0, requestURL: $1) }
        }
        let elapsed = start.duration(to: .now)
        let latencyMs = Int(elapsed.components.seconds * 1000 + elapsed.components.attoseconds / 1_000_000_000_000_000)
        // `try?` on purpose, and only here: the request this labels has
        // already been sent and answered, so rebuilding it cannot fail for a
        // reason the caller could act on, and a label is not worth turning a
        // good reply into a thrown error. The composed description is the
        // fallback, as it was when NetKit 3.0.0 made this builder throwing.
        let requestURL =
            (try? networkService.createRequest(endPoint: endpoint))?.url?.absoluteString
            ?? "\(target.server.description)\(path)"
        return try map(latencyMs, requestURL)
    }
}

// MARK: - Body encoding

/// Encodes a request body to a JSON object, once, so an encoding failure
/// reaches the caller as a thrown error rather than a bodyless request.
func encodeWireJSONObject<Body: Encodable>(_ body: Body, label: String) throws -> [String: Any] {
    let data = try JSONEncoder().encode(body)
    guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
        throw EncodingError.invalidValue(
            body,
            .init(codingPath: [], debugDescription: "Request body for \(label) did not encode to a JSON object")
        )
    }
    return object
}

// MARK: - Shared mapping helpers

extension AIWireStandard {
    /// The message thrown when a standard's reply carried no text at all.
    ///
    /// One constant rather than four sentences that drift apart; the suites
    /// match on this value, so thrower and matcher read the same code path.
    static func emptyReplyMessage(_ standard: AIWireStandard, requestURL: String) -> String {
        "\(emptyReplyPrefix(standard)) — from \(requestURL)"
    }

    /// The part of the empty-reply message that does not depend on the URL.
    ///
    /// A caller that has to recognise the failure (`LMStudioProvider` keeps its
    /// own `.emptyContent` case) matches this, so thrower and matcher still read
    /// one definition even though the message also names the request URL.
    static func emptyReplyPrefix(_ standard: AIWireStandard) -> String {
        "\(standard.displayName) answered with no content"
    }
}

/// How a conversation is rendered for a standard that takes one prompt string
/// rather than a role array.
///
/// `/api/v1/chat` is measured to reject a role array outright: an `input` array
/// must be typed content parts (`Expected 'text' | 'image'`), and a bare string
/// is the shape that works for a conversation. A single user message therefore
/// travels as exactly what the caller wrote — no labels invented around it —
/// and only a genuinely multi-part conversation gets role labels, because
/// without them the model cannot tell who said what.
enum AIWirePromptFlattening {
    static func flatten(_ request: AIRequest) -> String {
        let system = request.systemPrompt?.trimmingCharacters(in: .whitespacesAndNewlines)
        let hasSystem = !(system ?? "").isEmpty

        if !hasSystem, request.messages.count == 1, let only = request.messages.first, only.role == .user {
            return only.content
        }

        var parts: [String] = []
        if let system, hasSystem {
            parts.append("system: \(system)")
        }
        for message in request.messages {
            parts.append("\(message.role.rawValue): \(message.content)")
        }
        return parts.joined(separator: "\n\n")
    }
}
