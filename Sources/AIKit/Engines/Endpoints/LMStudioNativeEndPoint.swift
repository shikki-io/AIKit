import Foundation
import NetKit

/// `.lmStudioNative` as a declared standard: `/api/v1/chat`.
///
/// LM Studio's own API, and the narrowest of the four. Measured on 2026-10-31,
/// it differs from the OpenAI surfaces in three ways that each produce an HTTP
/// 400 rather than something ignored:
///
/// - `input` is a prompt string. A role array is refused with
///   `Invalid discriminator value. Expected 'text' | 'image'`, so this standard
///   has no conversation shape at this version.
/// - `max_tokens` is refused with `Unrecognized key(s) in object:
///   'max_tokens'`. The budget field is `max_output_tokens`.
/// - An unknown model is a 404 — this is the only one of the four that
///   validates the model name instead of answering with whatever is loaded.
///
/// The spec's Phase 1 audit lists this path as answering a `messages` array;
/// it does not, and that is why the request body here is not the chat one.
enum LMStudioNativeWireFamily: AIWireFamily {
    static var standard: AIWireStandard { .lmStudioNative }

    static func requestBody(for request: AIRequest, model: String, stream: Bool) -> LMStudioNativeRequest {
        LMStudioNativeRequest(
            model: model,
            input: AIWirePromptFlattening.flatten(request),
            temperature: request.temperature,
            maxOutputTokens: request.maxTokens,
            stream: stream
        )
    }

    static func response(
        from reply: LMStudioNativeReply,
        requestedModel: String,
        providerID: String,
        latencyMs: Int,
        requestURL: String
    ) throws -> AIResponse {
        let response: LMStudioNativeResponseDTO
        switch reply {
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
                prompt: response.stats?.inputTokens ?? 0,
                completion: response.stats?.totalOutputTokens ?? 0
            ),
            latencyMs: latencyMs,
            providerID: providerID
        )
    }
}
