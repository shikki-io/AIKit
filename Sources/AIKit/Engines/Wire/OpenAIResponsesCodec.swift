import Foundation
import NetKit

/// `openai-responses` — `POST /v1/responses`.
///
/// Keeps a conversation as roles in `input` and the system prompt in this
/// surface's own `instructions` field. It is the newer OpenAI surface and the
/// one LM Studio pairs with stateful chats and MCP integrations.
public struct OpenAIResponsesCodec: WireCodec {

    public init() {}

    public var family: AIProviderFamily { .openAIResponses }

    public var capabilities: WireCodecCapabilities {
        WireCodecCapabilities(
            // No `.tool`: a tool RESULT travels as a `function_call_output`
            // item on this surface rather than as a role, and AIKit does not
            // build one. Refusing is the point — the alternative was sending a
            // `"role": "tool"` the server does not define.
            roles: [.system, .user, .assistant],
            systemPrompt: true,
            // The surface DOES take custom tools; this codec does not map them
            // yet, so it says so instead of dropping them (BR-AIL-17).
            tools: false,
            responseFormats: [],
            inputKinds: [],
            streaming: true
        )
    }

    // MARK: - Request

    public func body(
        for request: AIRequest,
        model: String,
        stream: Bool
    ) throws(AIKitError) -> any Encodable & Sendable {
        try refuseWhatItCannotCarry(request, streaming: stream)
        return responsesRequest(for: request, model: model, stream: stream)
    }

    public func endpoint(
        for request: AIRequest,
        entry: AIProviderEntry,
        credential: String?,
        stream: Bool
    ) throws(AIKitError) -> any EndPoint {
        try refuseWhatItCannotCarry(request, streaming: stream)
        return try proved(
            OpenAIEndPoint.responses(
                address: entry.address,
                credential: credential,
                body: responsesRequest(for: request, model: entry.model, stream: stream)
            )
        )
    }

    private func responsesRequest(for request: AIRequest, model: String, stream: Bool) -> ResponsesRequest {
        ResponsesRequest(
            model: model,
            input: request.messages.map { .init(role: $0.role.rawValue, content: $0.content) },
            // This surface's own field for a system prompt. It is not sent as a
            // pseudo-message: the reply echoes `instructions` back verbatim, so
            // the server reads it there.
            instructions: request.systemPrompt.flatMap { $0.isEmpty ? nil : $0 },
            maxOutputTokens: request.maxTokens,
            temperature: request.temperature,
            stream: stream
        )
    }

    // MARK: - Reply

    public func response(from data: Data, context: WireReplyContext) throws(AIKitError) -> AIResponse {
        let reply = try decode(ResponsesReply.self, from: data, context: context)
        let value: ResponsesResponseDTO
        switch reply {
        case .response(let decoded):
            value = decoded
        case .serverError(let message):
            // BR-AIL-10. Reached only when the error object actually decoded:
            // this envelope sends `"error": null` on success, and a null is not
            // an error body.
            throw .requestFailed("\(message) — from \(context.target)")
        }

        let text = value.text
        guard !text.isEmpty else {
            throw .emptyReply(target: context.target)
        }

        return AIResponse(
            content: text,
            model: value.model ?? context.requestedModel,
            tokensUsed: TokenUsage(prompt: value.usage?.inputTokens, completion: value.usage?.outputTokens),
            latencyMs: context.latencyMs,
            providerID: context.providerID
        )
    }

    public func chunk(fromLine line: String, context: WireReplyContext) throws(AIKitError) -> AIChunk? {
        guard let payload = ssePayload(of: line) else { return nil }
        guard let data = payload.data(using: .utf8) else {
            throw .replyUndecodable(target: context.target, reason: "an SSE chunk is not valid UTF-8")
        }
        let event = try decode(ResponsesStreamEvent.self, from: data, context: context)
        switch event.type {
        case ResponsesStreamEvent.textDelta:
            let delta = event.delta ?? ""
            return delta.isEmpty ? nil : AIChunk(delta: delta, isComplete: false)
        case ResponsesStreamEvent.completed:
            return AIChunk(delta: "", isComplete: true)
        default:
            // Lifecycle: `response.created`, `response.in_progress`,
            // `response.output_item.added`, `response.content_part.added`,
            // `response.output_text.done`, … None carries answer text.
            return nil
        }
    }
}
