import Foundation
import NetKit

/// `.anthropicMessages` as a declared standard: `/v1/messages`.
///
/// Anthropic's wire standard, served here by a local LM Studio rather than by
/// Anthropic — measured answering HTTP 200 on the operator's machine with a
/// Mistral model loaded. That is the whole argument for declaring a standard
/// separately from a server: `anthropic-messages` says nothing about who is on
/// the other end, only about the shape of the conversation.
enum AnthropicMessagesWireFamily: AIWireFamily {
    static var standard: AIWireStandard { .anthropicMessages }

    /// Anthropic requires an API version header, and sends a 400 naming it when
    /// it is missing. LM Studio does not require it; sending it always keeps one
    /// request shape for both ends.
    static var standardHeaders: [String: String] { ["anthropic-version": "2023-06-01"] }

    static func requestBody(for request: AIRequest, model: String, stream: Bool) -> AnthropicMessagesRequest {
        AnthropicMessagesRequest(
            model: model,
            // Anthropic rejects a `system` role inside the array — the system
            // prompt is a top-level field on this standard — so a system
            // message that arrived in `messages` is lifted out rather than sent
            // where it would be refused.
            messages: request.messages
                .filter { $0.role != .system }
                .map { .init(role: $0.role.rawValue, content: $0.content) },
            system: systemPrompt(for: request),
            maxTokens: request.maxTokens,
            temperature: request.temperature,
            stream: stream
        )
    }

    static func response(
        from reply: AnthropicMessagesReply,
        requestedModel: String,
        providerID: String,
        latencyMs: Int,
        requestURL: String
    ) throws -> AIResponse {
        let message: AnthropicMessagesResponseDTO
        switch reply {
        case .message(let value):
            message = value
        case .serverError(let text):
            // BR-AIL-10.
            throw AIKitError.requestFailed("\(text) — from \(requestURL)")
        }

        let text = message.text
        guard !text.isEmpty else {
            throw AIKitError.requestFailed(AIWireStandard.emptyReplyMessage(.anthropicMessages, requestURL: requestURL))
        }

        return AIResponse(
            content: text,
            model: message.model ?? requestedModel,
            tokensUsed: TokenUsage(
                prompt: message.usage?.inputTokens ?? 0,
                completion: message.usage?.outputTokens ?? 0
            ),
            latencyMs: latencyMs,
            providerID: providerID
        )
    }

    // MARK: - Private

    /// The system prompt this standard sends top-level: the request's own, plus
    /// any system message that arrived in the array, joined in order.
    private static func systemPrompt(for request: AIRequest) -> String? {
        var parts: [String] = []
        if let prompt = request.systemPrompt, !prompt.isEmpty {
            parts.append(prompt)
        }
        parts.append(contentsOf: request.messages.filter { $0.role == .system }.map(\.content))
        let joined = parts.joined(separator: "\n\n")
        return joined.isEmpty ? nil : joined
    }
}
