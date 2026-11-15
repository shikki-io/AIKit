import Foundation
import NetKit

/// LM Studio's own REST API: one case per documented route.
///
/// BR-AIL-16, answering AIKit#18 r4493310420 ("It's normal the lm studio
/// protocol has only one endpoint?" — it was not; it had one case because one
/// call existed. This API is mostly model MANAGEMENT, which is the half that
/// was missing).
///
/// ROUTE TABLE, captured 2026-11-12 with
/// `lightpanda fetch --dump markdown --strip-mode full` from
/// <https://lmstudio.ai/docs/developer/rest> — its "Supported endpoints" table,
/// verbatim, including the methods:
///
/// | Route                            | Method |
/// |----------------------------------|--------|
/// | `/api/v1/chat`                   | POST   |
/// | `/api/v1/models`                 | GET    |
/// | `/api/v1/models/load`            | POST   |
/// | `/api/v1/models/unload`          | POST   |
/// | `/api/v1/models/download`        | POST   |
/// | `/api/v1/models/download/status` | GET    |
///
/// The v0 API (`/api/v0/…`) is documented as superseded by this one in LM
/// Studio 0.4.0 and is not declared. LM Studio's OpenAI- and
/// Anthropic-compatible surfaces are different APIs on the same port and live
/// in ``OpenAIEndPoint`` and ``AnthropicEndPoint``.
///
/// WHAT THIS API WILL NOT DO, from the same capture's "Inference endpoint
/// comparison" table and measured against the live server on 2026-11-12:
///
/// - Custom tools: documented ❌.
/// - Assistant messages in the request: documented ❌. So there is no
///   conversation shape here, and ``LMStudioNativeCodec`` refuses a role it
///   cannot keep rather than flattening one.
/// - `input` is a string or an array of typed parts. A role array is refused
///   outright (`Invalid discriminator value. Expected 'text' | 'image'`), and a
///   part is `{"type": "text", "content": …}` — measured: `{"type": "text",
///   "text": …}` answers `'input.0.content' is required, Unrecognized key(s) in
///   object: 'text'`.
/// - The system prompt has its own documented top-level field,
///   `system_prompt`. Measured honoured: it raised `input_tokens` to 19 and
///   changed the answer.
/// - `max_tokens` is refused (`Unrecognized key(s) in object: 'max_tokens'`);
///   the budget field is `max_output_tokens`.
/// - An unknown model is a 404 — this is the only one of the four families that
///   validates the model name instead of answering with whatever is loaded.
enum LMStudioEndPoint: WireEndPoint {

    // MARK: - End points declaration

    /// `POST /api/v1/chat`.
    case chat(address: AIServerAddress, credential: String?, body: LMStudioNativeRequest)
    /// `GET /api/v1/models` — every model on the host, loaded or not.
    case models(address: AIServerAddress, credential: String?)
    /// `POST /api/v1/models/load`.
    case loadModel(address: AIServerAddress, credential: String?, body: LMStudioModelLoadRequest)
    /// `POST /api/v1/models/unload`.
    case unloadModel(address: AIServerAddress, credential: String?, body: LMStudioModelUnloadRequest)
    /// `POST /api/v1/models/download`.
    case downloadModel(address: AIServerAddress, credential: String?, body: LMStudioModelDownloadRequest)
    /// `GET /api/v1/models/download/status`.
    case downloadStatus(address: AIServerAddress, credential: String?)

    // MARK: - Routes

    /// Every route this enum declares, with its method (T-AIL-39).
    static let routes: [(path: String, method: RequestMethod)] = [
        ("/api/v1/chat", .POST),
        ("/api/v1/models", .GET),
        ("/api/v1/models/load", .POST),
        ("/api/v1/models/unload", .POST),
        ("/api/v1/models/download", .POST),
        ("/api/v1/models/download/status", .GET),
    ]

    // MARK: - Properties

    private var address: AIServerAddress {
        switch self {
        case .chat(let address, _, _),
            .models(let address, _),
            .loadModel(let address, _, _),
            .unloadModel(let address, _, _),
            .downloadModel(let address, _, _),
            .downloadStatus(let address, _):
            address
        }
    }

    private var credential: String? {
        switch self {
        case .chat(_, let credential, _),
            .models(_, let credential),
            .loadModel(_, let credential, _),
            .unloadModel(_, let credential, _),
            .downloadModel(_, let credential, _),
            .downloadStatus(_, let credential):
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
        case .chat: "/api/v1/chat"
        case .models: "/api/v1/models"
        case .loadModel: "/api/v1/models/load"
        case .unloadModel: "/api/v1/models/unload"
        case .downloadModel: "/api/v1/models/download"
        case .downloadStatus: "/api/v1/models/download/status"
        }
    }

    var method: RequestMethod {
        switch self {
        case .chat, .loadModel, .unloadModel, .downloadModel: .POST
        case .models, .downloadStatus: .GET
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

    /// - Throws: An encoding error when the body cannot be serialised.
    func bodyObject() throws -> [String: Any]? {
        switch self {
        case .chat(_, _, let body):
            return try encodeWireJSONObject(body, label: "LMStudioNativeRequest")
        case .loadModel(_, _, let body):
            return try encodeWireJSONObject(body, label: "LMStudioModelLoadRequest")
        case .unloadModel(_, _, let body):
            return try encodeWireJSONObject(body, label: "LMStudioModelUnloadRequest")
        case .downloadModel(_, _, let body):
            return try encodeWireJSONObject(body, label: "LMStudioModelDownloadRequest")
        case .models, .downloadStatus:
            return nil
        }
    }
}
