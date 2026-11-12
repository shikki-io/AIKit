import Foundation
import NetKit

/// `anthropic-messages` — `POST /v1/messages`.
///
/// Anthropic itself, and LM Studio's Anthropic-compatible surface. Before this
/// wave AIKit could not talk this wire at all: the family was declared and had
/// no production caller (AIKit#18, panel finding on
/// `AIWireStandard.swift:185,234`).
public struct AnthropicMessagesCodec: WireCodec {

    public init() {}

    public var family: AIProviderFamily { .anthropicMessages }

    public var capabilities: WireCodecCapabilities {
        WireCodecCapabilities(
            // `.system` is here because a system message in the array is
            // LIFTED to the top-level `system` field, which this surface
            // requires — Anthropic rejects a `system` role inside `messages`.
            // `.tool` is not: a tool result is a `tool_result` content block
            // here, and this codec does not build content blocks.
            roles: [.system, .user, .assistant],
            systemPrompt: true,
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
        return messagesRequest(for: request, model: model, stream: stream)
    }

    public func endpoint(
        for request: AIRequest,
        entry: AIProviderEntry,
        credential: String?,
        stream: Bool
    ) throws(AIKitError) -> any EndPoint {
        try refuseWhatItCannotCarry(request, streaming: stream)
        return try proved(
            AnthropicEndPoint.messages(
                address: try entry.serverAddress(),
                credential: credential,
                body: messagesRequest(for: request, model: entry.model, stream: stream)
            )
        )
    }

    private func messagesRequest(for request: AIRequest, model: String, stream: Bool) -> AnthropicMessagesRequest {
        AnthropicMessagesRequest(
            model: model,
            messages: request.messages
                .filter { $0.role != .system }
                .map { .init(role: $0.role.rawValue, content: $0.content) },
            system: Self.systemPrompt(for: request),
            maxTokens: request.maxTokens,
            temperature: request.temperature,
            stream: stream
        )
    }

    /// The system prompt this surface sends top-level: the request's own, plus
    /// any system message that arrived in the array, joined in order.
    ///
    /// This is a MOVE, not a flattening: the text keeps its own field and no
    /// role label is invented around it. Measured against LM Studio — sending
    /// it here changed the answer and raised `input_tokens` from 4 to 18, so
    /// the field is read.
    static func systemPrompt(for request: AIRequest) -> String? {
        var parts: [String] = []
        if let prompt = request.systemPrompt, !prompt.isEmpty {
            parts.append(prompt)
        }
        parts.append(contentsOf: request.messages.filter { $0.role == .system }.map(\.content))
        let joined = parts.joined(separator: "\n\n")
        return joined.isEmpty ? nil : joined
    }

    // MARK: - Reply

    public func response(from data: Data, context: WireReplyContext) throws(AIKitError) -> AIResponse {
        let reply = try decode(AnthropicMessagesReply.self, from: data, context: context)
        let message: AnthropicMessagesResponseDTO
        switch reply {
        case .message(let value):
            message = value
        case .serverError(let text):
            // BR-AIL-10.
            throw .requestFailed("\(text) — from \(context.target)")
        }

        let text = message.text
        guard !text.isEmpty else {
            throw .emptyReply(target: context.target)
        }

        return AIResponse(
            content: text,
            model: message.model ?? context.requestedModel,
            tokensUsed: TokenUsage(prompt: message.usage?.inputTokens, completion: message.usage?.outputTokens),
            latencyMs: context.latencyMs,
            providerID: context.providerID
        )
    }

    public func chunk(fromLine line: String, context: WireReplyContext) throws(AIKitError) -> AIChunk? {
        guard let payload = ssePayload(of: line) else { return nil }
        guard let data = payload.data(using: .utf8) else {
            throw .replyUndecodable(target: context.target, reason: "an SSE chunk is not valid UTF-8")
        }
        let event = try decode(AnthropicStreamEvent.self, from: data, context: context)
        switch event.type {
        case AnthropicStreamEvent.contentBlockDelta:
            // Only a `text_delta` is the answer. A `thinking_delta` or an
            // `input_json_delta` arrives on the same event type and is not.
            guard event.delta?.type == "text_delta", let delta = event.delta?.text, !delta.isEmpty else {
                return nil
            }
            return AIChunk(delta: delta, isComplete: false)
        case AnthropicStreamEvent.messageStop:
            return AIChunk(delta: "", isComplete: true)
        default:
            // `message_start`, `content_block_start`, `content_block_stop`,
            // `message_delta`, `ping`. None carries answer text.
            return nil
        }
    }
}
