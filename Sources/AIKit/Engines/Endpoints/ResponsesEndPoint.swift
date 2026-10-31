import Foundation
import NetKit

/// `.openAIResponses` as a declared standard: `/v1/responses`.
///
/// The second OpenAI wire standard, and a different shape from chat completions
/// rather than a version of it — the prompt is `input`, the system prompt is
/// `instructions`, the budget is `max_output_tokens`, and the answer is nested
/// two levels down in `output[].content[].text`. One LM Studio serves this and
/// `/v1/chat/completions` at the same time, which is why the standard is a
/// declaration and not a property of the server.
enum OpenAIResponsesWireFamily: AIWireFamily {
    static var standard: AIWireStandard { .openAIResponses }

    static func requestBody(for request: AIRequest, model: String, stream: Bool) -> ResponsesRequest {
        ResponsesRequest(
            model: model,
            input: request.messages.map { .init(role: $0.role.rawValue, content: $0.content) },
            // This standard's own field for a system prompt. It is not sent as
            // a pseudo-message: the reply echoes `instructions` back verbatim,
            // so the server reads it there.
            instructions: request.systemPrompt.flatMap { $0.isEmpty ? nil : $0 },
            maxOutputTokens: request.maxTokens,
            temperature: request.temperature,
            stream: stream
        )
    }

    static func response(
        from reply: ResponsesReply,
        requestedModel: String,
        providerID: String,
        latencyMs: Int,
        requestURL: String
    ) throws -> AIResponse {
        let response: ResponsesResponseDTO
        switch reply {
        case .response(let value):
            response = value
        case .serverError(let message):
            // BR-AIL-10. Reached only when the error object actually decoded:
            // this envelope sends `"error": null` on success, and a null is not
            // an error body.
            throw AIKitError.requestFailed("\(message) — from \(requestURL)")
        }

        let text = response.text
        guard !text.isEmpty else {
            throw AIKitError.requestFailed(AIWireStandard.emptyReplyMessage(.openAIResponses, requestURL: requestURL))
        }

        return AIResponse(
            content: text,
            model: response.model ?? requestedModel,
            tokensUsed: TokenUsage(
                prompt: response.usage?.inputTokens ?? 0,
                completion: response.usage?.outputTokens ?? 0
            ),
            latencyMs: latencyMs,
            providerID: providerID
        )
    }
}
