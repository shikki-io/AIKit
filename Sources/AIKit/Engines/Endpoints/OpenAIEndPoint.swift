import Foundation
import NetKit

/// The OpenAI-shaped HTTP API: one case per documented route.
///
/// BR-AIL-16 — this enum is the SSoT for this API (AIKit#18 r4493304830,
/// r4493409885). Every route the surface documents is a case, including the
/// ones AIKit does not call yet, and the route strings exist NOWHERE else in
/// the package.
///
/// ROUTE TABLE, captured 2026-11-12 with
/// `lightpanda fetch --dump markdown --strip-mode full`
/// from <https://lmstudio.ai/docs/developer/openai-compat> — the
/// OpenAI-compatible surface AIKit actually addresses, documented there with
/// its methods:
///
/// | Route                  | Method |
/// |------------------------|--------|
/// | `/v1/models`           | GET    |
/// | `/v1/responses`        | POST   |
/// | `/v1/chat/completions` | POST   |
/// | `/v1/completions`      | POST   |
/// | `/v1/embeddings`       | POST   |
///
/// `/v1/decisions` is in that table too and is deliberately NOT here: it is an
/// LM Studio extension of the OpenAI shape rather than part of it, and it
/// belongs to ``LMStudioEndPoint``'s surface if AIKit ever calls it.
///
/// Anthropic and LM Studio's native API are different APIs with their own
/// enums (``AnthropicEndPoint``, ``LMStudioEndPoint``). The server, the bearer
/// token and the way a body becomes JSON are written once, on this enum; a case
/// contributes only its path, its method and the body it carries.
enum OpenAIEndPoint: WireEndPoint {

    // MARK: - End points declaration

    /// `GET /v1/models` — what this server serves.
    case models(address: AIServerAddress, credential: String?)
    /// `POST /v1/chat/completions`.
    case chatCompletions(address: AIServerAddress, credential: String?, body: ChatCompletionRequest)
    /// `POST /v1/responses` — the prompt is `input`, the system prompt
    /// `instructions`.
    case responses(address: AIServerAddress, credential: String?, body: ResponsesRequest)
    /// `POST /v1/completions` — the legacy text surface. No caller in AIKit
    /// today; declared because the API documents it (BR-AIL-16).
    case completions(address: AIServerAddress, credential: String?, body: CompletionsRequest)
    /// `POST /v1/embeddings`. No caller in AIKit today; declared for the same
    /// reason.
    case embeddings(address: AIServerAddress, credential: String?, body: EmbeddingsRequest)

    // MARK: - Routes

    /// Every route this enum declares, with its method — the table above as
    /// data, so a test can compare the enum against the captured document
    /// (T-AIL-39).
    static let routes: [(path: String, method: RequestMethod)] = [
        ("/v1/models", .GET),
        ("/v1/chat/completions", .POST),
        ("/v1/responses", .POST),
        ("/v1/completions", .POST),
        ("/v1/embeddings", .POST),
    ]

    // MARK: - Properties

    private var address: AIServerAddress {
        switch self {
        case .models(let address, _),
            .chatCompletions(let address, _, _),
            .responses(let address, _, _),
            .completions(let address, _, _),
            .embeddings(let address, _, _):
            address
        }
    }

    private var credential: String? {
        switch self {
        case .models(_, let credential),
            .chatCompletions(_, let credential, _),
            .responses(_, let credential, _),
            .completions(_, let credential, _),
            .embeddings(_, let credential, _):
            credential
        }
    }

    var scheme: String { address.scheme }
    var host: String { address.host }
    var port: Int? { address.port }
    var apiPath: String { "" }
    var apiFilePath: String { "" }

    var path: String {
        switch self {
        case .models: "/v1/models"
        case .chatCompletions: "/v1/chat/completions"
        case .responses: "/v1/responses"
        case .completions: "/v1/completions"
        case .embeddings: "/v1/embeddings"
        }
    }

    var method: RequestMethod {
        switch self {
        case .models: .GET
        case .chatCompletions, .responses, .completions, .embeddings: .POST
        }
    }

    var header: [String: String]? {
        var headers = ["Content-Type": "application/json"]
        if let credential, !credential.isEmpty {
            headers["Authorization"] = "Bearer \(credential)"
        }
        return headers
    }

    var queryParams: [String: Any]? { nil }

    var body: [String: Any]? {
        // Safety: whoever built the endpoint ran `bodyObject()`, which throws
        // before a request that cannot be encoded is ever sent.
        try? bodyObject()
    }

    /// The request body as a JSON object; nil for a call that has none.
    /// - Throws: An encoding error when the body cannot be serialised. The
    ///   caller propagates it rather than sending a bodyless request.
    func bodyObject() throws -> [String: Any]? {
        switch self {
        case .models:
            return nil
        case .chatCompletions(_, _, let body):
            return try encodeWireJSONObject(body, label: "ChatCompletionRequest")
        case .responses(_, _, let body):
            return try encodeWireJSONObject(body, label: "ResponsesRequest")
        case .completions(_, _, let body):
            return try encodeWireJSONObject(body, label: "CompletionsRequest")
        case .embeddings(_, _, let body):
            return try encodeWireJSONObject(body, label: "EmbeddingsRequest")
        }
    }
}
