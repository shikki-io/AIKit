import Foundation
import NetKit

/// The Anthropic messages API: `/v1/messages`.
///
/// Served here by a local LM Studio as well as by Anthropic — measured
/// answering HTTP 200 on the operator's machine with a Mistral model loaded.
/// That is the argument for declaring a standard separately from a server:
/// `anthropic-messages` says nothing about who is on the other end, only about
/// the shape of the conversation.
enum AnthropicEndPoint: EndPoint {

    // MARK: - End points declaration

    case messages(server: AIWireServer, apiKey: String?, model: String, request: AIRequest, stream: Bool)

    // MARK: - Properties

    private var server: AIWireServer {
        switch self {
        case .messages(let server, _, _, _, _): server
        }
    }

    private var apiKey: String? {
        switch self {
        case .messages(_, let apiKey, _, _, _): apiKey
        }
    }

    var scheme: String { server.scheme }
    var host: String { server.host }
    var port: Int? { server.port }
    var apiPath: String { "" }
    var apiFilePath: String { "" }

    var path: String {
        switch self {
        case .messages: AIWireStandard.anthropicMessages.path
        }
    }

    var method: RequestMethod { .POST }

    var header: [String: String]? {
        // Anthropic requires an API version header, and sends a 400 naming it
        // when it is missing. LM Studio does not require it; sending it always
        // keeps one request shape for both ends.
        var headers = ["Content-Type": "application/json", "anthropic-version": "2023-06-01"]
        if let apiKey, !apiKey.isEmpty {
            headers["x-api-key"] = apiKey
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
        case .messages(_, _, let model, let request, let stream):
            let body = AnthropicMessagesRequest(
                model: model,
                // Anthropic rejects a `system` role inside the array — the system
                // prompt is a top-level field on this standard — so a system
                // message that arrived in `messages` is lifted out rather than sent
                // where it would be refused.
                messages: request.messages
                    .filter { $0.role != .system }
                    .map { .init(role: $0.role.rawValue, content: $0.content) },
                system: Self.systemPrompt(for: request),
                maxTokens: request.maxTokens,
                temperature: request.temperature,
                stream: stream
            )
            return try encodeWireJSONObject(body, label: AIWireStandard.anthropicMessages.rawValue)
        }
    }

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

// MARK: - Reply

extension AnthropicMessagesReply {
    /// Reads this reply into the typed response.
    func response(
        requestedModel: String,
        providerID: String,
        latencyMs: Int,
        requestURL: String
    ) throws -> AIResponse {
        let message: AnthropicMessagesResponseDTO
        switch self {
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
                prompt: message.usage?.inputTokens,
                completion: message.usage?.outputTokens
            ),
            latencyMs: latencyMs,
            providerID: providerID
        )
    }
}
