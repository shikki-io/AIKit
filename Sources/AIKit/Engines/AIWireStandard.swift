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
    public init?(kind: String) {
        let normalised = kind.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        switch normalised {
        case "openai-compatible", "openai", "lmstudio", "ollama":
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

// MARK: - AIWireEndPoint

/// The one declaration shape every wire standard is built from.
///
/// A standard contributes a path, a method, its headers and a request body; the
/// transport concerns — where the server is, how a bearer token is attached,
/// how a body becomes JSON — are written once, here. Adding a standard is a
/// family plus a case, never another `EndPoint` conformance that re-decides
/// what a header is.
struct AIWireEndPoint<Body: Encodable & Sendable>: EndPoint {
    let server: AIWireServer
    let standard: AIWireStandard
    let apiKey: String?
    /// Headers the standard needs beyond the shared ones, for example
    /// Anthropic's `anthropic-version`.
    let standardHeaders: [String: String]
    let requestBody: Body

    var host: String { server.host }
    var port: Int? { server.port }
    var scheme: String { server.scheme }
    var apiPath: String { "" }
    var apiFilePath: String { "" }
    var path: String { standard.path }
    var method: RequestMethod { .POST }

    var header: [String: String]? {
        var headers = ["Content-Type": "application/json"]
        for (name, value) in standardHeaders {
            headers[name] = value
        }
        if let apiKey, !apiKey.isEmpty {
            // Anthropic reads the key from `x-api-key`; everything else reads a
            // bearer token. The standard says which, so this stays a lookup.
            switch standard {
            case .anthropicMessages:
                headers["x-api-key"] = apiKey
            case .openAIChat, .openAIResponses, .lmStudioNative:
                headers["Authorization"] = "Bearer \(apiKey)"
            }
        }
        return headers
    }

    var queryParams: [String: Any]? { nil }

    var body: [String: Any]? {
        guard let data = try? JSONEncoder().encode(requestBody),
            let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return nil }
        return dict
    }
}

// MARK: - AIWireFamily

/// What one wire standard has to supply: a request body built from an
/// `AIRequest`, and an `AIResponse` read back out of its reply.
///
/// Two directions, both declared, which is what lets a test pin a captured
/// reply both ways (T-AIL-17) instead of asserting that a hand-built dictionary
/// matches a hand-built expectation.
protocol AIWireFamily {
    associatedtype RequestBody: Encodable & Sendable
    associatedtype Reply: Decodable & Sendable

    /// The standard this family implements.
    static var standard: AIWireStandard { get }
    /// Headers this standard needs beyond `Content-Type` and the key.
    static var standardHeaders: [String: String] { get }

    /// Renders a typed request onto this standard's request shape.
    static func requestBody(for request: AIRequest, model: String, stream: Bool) -> RequestBody

    /// Reads this standard's reply back into the typed response.
    /// - Parameters:
    ///   - requestedModel: What the caller asked for, used only when the reply
    ///     does not name a model itself.
    ///   - target: The URL the reply came from, for an error that must name it.
    static func response(
        from reply: Reply,
        requestedModel: String,
        providerID: String,
        latencyMs: Int,
        requestURL: String
    ) throws -> AIResponse
}

extension AIWireFamily {
    static var standardHeaders: [String: String] { [:] }
}

// MARK: - Dispatch

extension AIWireStandard {
    /// Builds the endpoint this standard declares for one request.
    ///
    /// Returned erased because the four families carry four different body
    /// types; a caller that only needs a `URLRequest` — `shi providers
    /// discover`, a test asserting a path — does not care which.
    func endPoint(target: AIWireTarget, request: AIRequest, stream: Bool) -> any EndPoint {
        switch self {
        case .openAIChat:
            Self.endPoint(OpenAIChatWireFamily.self, target, request, stream)
        case .openAIResponses:
            Self.endPoint(OpenAIResponsesWireFamily.self, target, request, stream)
        case .anthropicMessages:
            Self.endPoint(AnthropicMessagesWireFamily.self, target, request, stream)
        case .lmStudioNative:
            Self.endPoint(LMStudioNativeWireFamily.self, target, request, stream)
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
        switch self {
        case .openAIChat:
            try await Self.send(OpenAIChatWireFamily.self, request, target, networkService)
        case .openAIResponses:
            try await Self.send(OpenAIResponsesWireFamily.self, request, target, networkService)
        case .anthropicMessages:
            try await Self.send(AnthropicMessagesWireFamily.self, request, target, networkService)
        case .lmStudioNative:
            try await Self.send(LMStudioNativeWireFamily.self, request, target, networkService)
        }
    }

    // MARK: - Private

    /// The one endpoint builder every family goes through.
    private static func endPoint<Family: AIWireFamily>(
        _ family: Family.Type,
        _ target: AIWireTarget,
        _ request: AIRequest,
        _ stream: Bool
    ) -> AIWireEndPoint<Family.RequestBody> {
        AIWireEndPoint(
            server: target.server,
            standard: Family.standard,
            apiKey: target.apiKey,
            standardHeaders: Family.standardHeaders,
            requestBody: Family.requestBody(for: request, model: target.model, stream: stream)
        )
    }

    /// The one send-and-map path every family goes through.
    private static func send<Family: AIWireFamily>(
        _ family: Family.Type,
        _ request: AIRequest,
        _ target: AIWireTarget,
        _ networkService: any NetworkProtocol
    ) async throws -> AIResponse {
        let start = ContinuousClock.now
        let endpoint = endPoint(family, target, request, false)
        let reply: Family.Reply = try await networkService.sendRequest(endpoint: endpoint)
        let elapsed = start.duration(to: .now)
        let latencyMs = Int(elapsed.components.seconds * 1000 + elapsed.components.attoseconds / 1_000_000_000_000_000)
        let requestURL = networkService.createRequest(endPoint: endpoint).url?.absoluteString
            ?? "\(target.server.description)\(Family.standard.path)"
        return try Family.response(
            from: reply,
            requestedModel: target.model,
            providerID: target.providerID,
            latencyMs: latencyMs,
            requestURL: requestURL
        )
    }
}

// MARK: - Shared mapping helpers

extension AIWireStandard {
    /// The message thrown when a standard's reply carried no text at all.
    ///
    /// One constant rather than four sentences that drift apart; the suites
    /// match on this value, so thrower and matcher read the same code path.
    static func emptyReplyMessage(_ standard: AIWireStandard, requestURL: String) -> String {
        "\(standard.displayName) answered with no content — from \(requestURL)"
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
