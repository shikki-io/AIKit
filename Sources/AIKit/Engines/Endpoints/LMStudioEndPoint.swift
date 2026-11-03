import Foundation
import NetKit

/// LM Studio's own API: `/api/v1/chat`.
///
/// The narrowest of the wire standards. Measured on 2026-10-31, it differs from
/// the OpenAI surfaces in three ways that each produce an HTTP 400 rather than
/// something ignored:
///
/// - `input` is a prompt string. A role array is refused with
///   `Invalid discriminator value. Expected 'text' | 'image'`, so this standard
///   has no conversation shape at this version.
/// - `max_tokens` is refused with `Unrecognized key(s) in object:
///   'max_tokens'`. The budget field is `max_output_tokens`.
/// - An unknown model is a 404 — this is the only one of the four that
///   validates the model name instead of answering with whatever is loaded.
enum LMStudioEndPoint: EndPoint {

    // MARK: - End points declaration

    case chat(server: AIWireServer, apiKey: String?, model: String, request: AIRequest, stream: Bool)

    // MARK: - Properties

    private var server: AIWireServer {
        switch self {
        case .chat(let server, _, _, _, _): server
        }
    }

    private var apiKey: String? {
        switch self {
        case .chat(_, let apiKey, _, _, _): apiKey
        }
    }

    var scheme: String { server.scheme }
    var host: String { server.host }
    var port: Int? { server.port }
    var apiPath: String { "" }
    var apiFilePath: String { "" }

    var path: String {
        switch self {
        case .chat: AIWireStandard.lmStudioNative.path
        }
    }

    var method: RequestMethod { .POST }

    var header: [String: String]? {
        var headers = ["Content-Type": "application/json"]
        if let apiKey, !apiKey.isEmpty {
            headers["Authorization"] = "Bearer \(apiKey)"
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
    func bodyObject() throws -> [String: Any] {
        switch self {
        case .chat(_, _, let model, let request, let stream):
            let body = LMStudioNativeRequest(
                model: model,
                input: AIWirePromptFlattening.flatten(request),
                temperature: request.temperature,
                maxOutputTokens: request.maxTokens,
                stream: stream
            )
            return try encodeWireJSONObject(body, label: AIWireStandard.lmStudioNative.rawValue)
        }
    }
}

// MARK: - Reply

extension LMStudioNativeReply {
    /// Reads this reply into the typed response.
    func response(
        requestedModel: String,
        providerID: String,
        latencyMs: Int,
        requestURL: String
    ) throws -> AIResponse {
        let response: LMStudioNativeResponseDTO
        switch self {
        case .response(let value):
            response = value
        case .serverError(let message):
            // BR-AIL-10.
            throw AIKitError.requestFailed("\(message) — from \(requestURL)")
        }

        let text = response.text
        guard !text.isEmpty else {
            throw AIKitError.requestFailed(AIWireStandard.emptyReplyMessage(.lmStudioNative, requestURL: requestURL))
        }

        return AIResponse(
            content: text,
            // This standard names the loaded instance rather than a model id.
            model: response.modelInstanceId ?? requestedModel,
            tokensUsed: TokenUsage(
                prompt: response.stats?.inputTokens,
                completion: response.stats?.totalOutputTokens
            ),
            latencyMs: latencyMs,
            providerID: providerID
        )
    }
}
