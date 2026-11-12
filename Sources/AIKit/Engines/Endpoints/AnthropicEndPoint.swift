import Foundation
import NetKit

/// The Anthropic Messages API: one case per documented route.
///
/// BR-AIL-16, answering AIKit#18 r4493313756 ("It's normal the Anthrophic
/// protocol has only one endpoint?" — it was not; it had one case because one
/// call existed).
///
/// ROUTE TABLE, captured 2026-11-12 with
/// `lightpanda fetch --dump markdown --strip-mode full` from
/// <https://docs.anthropic.com/en/api/messages> (which serves
/// platform.claude.com's reference, API version 2026-10-21) and
/// <https://platform.claude.com/docs/en/api/http/models/list>:
///
/// | Route                       | Method | Documented as          |
/// |-----------------------------|--------|------------------------|
/// | `/v1/messages`              | POST   | Create a Message       |
/// | `/v1/messages/count_tokens` | POST   | Count tokens in a Message |
/// | `/v1/models`                | GET    | List Models            |
/// | `/v1/models/{model_id}`     | GET    | Get a Model            |
///
/// That is the Messages-and-Models surface, which is what AIKit addresses. The
/// same reference documents Batches, Files, Skills, Managed Agents, Tunnels and
/// a Compliance API — separate surfaces with their own request families, each
/// of which would be its own enum when AIKit has a reason to call it. Declaring
/// them here with no DTOs would be a route list pretending to be a client.
///
/// Served by a local LM Studio as well as by Anthropic: LM Studio's
/// Anthropic-compatible surface (<https://lmstudio.ai/docs/developer/anthropic-compat>,
/// captured the same day) exposes `/v1/messages`, measured answering HTTP 200
/// on the operator's machine with a Mistral model loaded. That is the argument
/// for declaring a family separately from a server: `anthropic-messages` says
/// nothing about who is on the other end, only about the shape of the
/// conversation.
enum AnthropicEndPoint: WireEndPoint {

    // MARK: - End points declaration

    /// `POST /v1/messages`.
    case messages(address: AIServerAddress, credential: String?, body: AnthropicMessagesRequest)
    /// `POST /v1/messages/count_tokens`. No caller in AIKit today; declared
    /// because the API documents it (BR-AIL-16).
    case countTokens(address: AIServerAddress, credential: String?, body: AnthropicCountTokensRequest)
    /// `GET /v1/models`.
    case models(address: AIServerAddress, credential: String?)
    /// `GET /v1/models/{model_id}`.
    case model(address: AIServerAddress, credential: String?, id: String)

    // MARK: - Routes

    /// Every route this enum declares, with its method (T-AIL-39). The
    /// templated segment is written as the reference writes it.
    static let routes: [(path: String, method: RequestMethod)] = [
        ("/v1/messages", .POST),
        ("/v1/messages/count_tokens", .POST),
        ("/v1/models", .GET),
        ("/v1/models/{model_id}", .GET),
    ]

    // MARK: - Properties

    private var address: AIServerAddress {
        switch self {
        case .messages(let address, _, _),
            .countTokens(let address, _, _),
            .models(let address, _),
            .model(let address, _, _):
            address
        }
    }

    private var credential: String? {
        switch self {
        case .messages(_, let credential, _),
            .countTokens(_, let credential, _),
            .models(_, let credential),
            .model(_, let credential, _):
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
        case .messages: "/v1/messages"
        case .countTokens: "/v1/messages/count_tokens"
        case .models: "/v1/models"
        case .model(_, _, let id): "/v1/models/\(id)"
        }
    }

    var method: RequestMethod {
        switch self {
        case .messages, .countTokens: .POST
        case .models, .model: .GET
        }
    }

    var header: [String: String]? {
        // Anthropic requires an API version header, and sends a 400 naming it
        // when it is missing. LM Studio does not require it; sending it always
        // keeps one request shape for both ends.
        var headers = ["Content-Type": "application/json", "anthropic-version": Self.apiVersion]
        if let credential, !credential.isEmpty {
            // LM Studio's compatible surface accepts either header; Anthropic
            // reads `x-api-key`. Sending that one satisfies both.
            headers["x-api-key"] = credential
        }
        return headers
    }

    var queryParams: [String: Any]? { nil }

    var body: [String: Any]? {
        // Safety: whoever built the endpoint ran `bodyObject()`, which throws
        // before a request that cannot be encoded is ever sent.
        try? bodyObject()
    }

    /// - Throws: An encoding error when the body cannot be serialised.
    func bodyObject() throws -> [String: Any]? {
        switch self {
        case .messages(_, _, let body):
            return try encodeWireJSONObject(body, label: "AnthropicMessagesRequest")
        case .countTokens(_, _, let body):
            return try encodeWireJSONObject(body, label: "AnthropicCountTokensRequest")
        case .models, .model:
            return nil
        }
    }

    /// The `anthropic-version` this client declares.
    ///
    /// Pinned rather than tracking the reference's newest: a version header is
    /// a contract about the reply shape the DTOs below decode, so it moves when
    /// those are re-read against a capture, not on its own.
    static let apiVersion = "2023-06-01"
}
