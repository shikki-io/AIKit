import Foundation
import NetKit

/// `openai-chat` — `POST /v1/chat/completions`.
///
/// The widest of the four families: it keeps every role, carries tool
/// definitions and tool results, and takes a response format. OpenAI itself,
/// LM Studio, Ollama and any proxy in front of them speak it, which is why it
/// is the family the legacy `openai-compatible` `kind` maps to.
public struct OpenAIChatCodec: WireCodec {

    public init() {}

    public var family: AIProviderFamily { .openAIChat }

    public var capabilities: WireCodecCapabilities {
        WireCodecCapabilities(
            // All four, including `.tool`: this is the only family of the four
            // that carries a tool RESULT back, keyed by the id the server
            // issued.
            roles: [.system, .user, .assistant, .tool],
            systemPrompt: true,
            tools: true,
            responseFormats: [.text, .json],
            // No media. This family does carry images, as typed content parts
            // with a data URL, and AIKit does not build them — so an image
            // input is REFUSED rather than dropped, which is what used to
            // happen while `AICapabilities.vision` advertised it
            // (AIKit#18, `OpenAIEndPoint.swift:129`). The part builder belongs
            // to whichever wave gives AIKit a vision caller.
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
        return try chatRequest(for: request, model: model, stream: stream)
    }

    public func endpoint(
        for request: AIRequest,
        entry: AIProviderEntry,
        credential: String?,
        stream: Bool
    ) throws(AIKitError) -> any EndPoint {
        try refuseWhatItCannotCarry(request, streaming: stream)
        let body = try chatRequest(for: request, model: entry.model, stream: stream)
        return try proved(
            OpenAIEndPoint.chatCompletions(address: try entry.serverAddress(), credential: credential, body: body)
        )
    }

    /// Renders a typed request onto the chat-completions shape.
    private func chatRequest(
        for request: AIRequest,
        model: String,
        stream: Bool
    ) throws(AIKitError) -> ChatCompletionRequest {
        var messages: [ChatCompletionRequest.ChatMessage] = []
        if let systemPrompt = request.systemPrompt, !systemPrompt.isEmpty {
            messages.append(.init(role: "system", content: systemPrompt))
        }
        for message in request.messages {
            var wireToolCalls: [ChatCompletionRequest.ToolCallRequest]?
            if let calls = message.toolCalls {
                var rendered: [ChatCompletionRequest.ToolCallRequest] = []
                for call in calls {
                    // The id is the SERVER's. `refuseWhatItCannotCarry` has
                    // already rejected a missing one; this re-check is what
                    // keeps the alternative from being a minted `UUID()`, which
                    // is what this call site used to do.
                    guard let id = call.id, !id.isEmpty else {
                        throw .unsupported(field: "messages[].toolCalls[].id is missing", family: family)
                    }
                    rendered.append(
                        .init(id: id, type: "function", function: .init(name: call.name, arguments: call.arguments))
                    )
                }
                wireToolCalls = rendered
            }
            messages.append(
                .init(
                    role: message.role.rawValue,
                    // Nil, not "", for an assistant turn that is only tool
                    // calls: the wire wants the key absent there.
                    content: message.content.isEmpty ? nil : message.content,
                    toolCalls: wireToolCalls,
                    toolCallId: message.toolCallId
                )
            )
        }

        let tools: [ChatCompletionRequest.ToolDefinition]? = request.tools?.map { tool in
            .init(
                type: "function",
                function: .init(
                    name: tool.name,
                    description: tool.description,
                    parameters: JSONFragment(rawJSON: tool.parametersSchema)
                )
            )
        }

        return ChatCompletionRequest(
            model: model,
            messages: messages,
            temperature: request.temperature,
            maxTokens: request.maxTokens,
            stream: stream,
            tools: tools,
            responseFormat: request.responseFormat.map { .init(type: $0 == .json ? "json_object" : "text") }
        )
    }

    // MARK: - Reply

    public func response(from data: Data, context: WireReplyContext) throws(AIKitError) -> AIResponse {
        let reply = try decode(ChatCompletionReply.self, from: data, context: context)
        let completion: ChatCompletionResponseDTO
        switch reply {
        case .completion(let value):
            completion = value
        case .serverError(let message):
            // BR-AIL-10: a 2xx carrying an error body. The server's own words,
            // and where they came from.
            throw .requestFailed("\(message) — from \(context.target)")
        }

        guard let choice = completion.choices.first else {
            throw .emptyReply(target: context.target)
        }

        return AIResponse(
            content: choice.message.content ?? "",
            // The reply names the model that answered; `requestedModel` is only
            // the fallback, because this surface is measured to ignore an
            // unknown model and answer with whatever is loaded (BR-AIL-06).
            model: completion.model ?? context.requestedModel,
            tokensUsed: TokenUsage(
                prompt: completion.usage?.promptTokens,
                completion: completion.usage?.completionTokens
            ),
            latencyMs: context.latencyMs,
            toolCalls: choice.message.toolCalls?.map {
                AIToolCall(id: $0.id, name: $0.function.name, arguments: $0.function.arguments)
            },
            providerID: context.providerID
        )
    }

    public func chunk(fromLine line: String, context: WireReplyContext) throws(AIKitError) -> AIChunk? {
        guard let payload = ssePayload(of: line) else { return nil }
        if payload == "[DONE]" {
            return AIChunk(delta: "", isComplete: true)
        }
        guard let data = payload.data(using: .utf8) else {
            throw .replyUndecodable(target: context.target, reason: "an SSE chunk is not valid UTF-8")
        }
        let chunk = try decode(ChatCompletionChunkDTO.self, from: data, context: context)
        guard let choice = chunk.choices.first else { return nil }

        let delta = choice.delta.content ?? ""
        let isComplete = choice.finishReason != nil
        // A chunk that carries neither text nor an ending is not news: the
        // captured stream sends `{"delta":{"role":"assistant"}}` first and
        // `{"delta":{"content":""}}` mid-run.
        guard !delta.isEmpty || isComplete else { return nil }
        return AIChunk(delta: delta, isComplete: isComplete)
    }
}
