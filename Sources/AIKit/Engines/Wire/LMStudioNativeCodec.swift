import Foundation
import NetKit

/// `lmstudio-native` — `POST /api/v1/chat`.
///
/// The narrowest of the four, and the one whose narrowness used to be hidden.
/// LM Studio's own reference says this surface takes no custom tools and no
/// assistant messages in the request (captured table on ``LMStudioEndPoint``),
/// and the live server refuses a role array outright. The previous codec
/// answered that by FLATTENING a conversation into `"system: …\n\nuser: …"`
/// — inventing a transcript the model then read as content, and reporting
/// nothing (AIKit#18, `AIWireStandard.swift:310-322`).
///
/// This codec keeps what the surface keeps structurally and refuses the rest:
///
/// - a system prompt goes in the documented top-level `system_prompt` field,
/// - user messages become typed `{"type": "text", "content": …}` parts, one
///   per message, so a multi-turn user conversation survives in order,
/// - an assistant or tool message is refused by role, because this surface has
///   nowhere to put one.
public struct LMStudioNativeCodec: WireCodec {

    public init() {}

    public var family: AIProviderFamily { .lmStudioNative }

    public var capabilities: WireCodecCapabilities {
        WireCodecCapabilities(
            // `.system` is carried by `system_prompt`, `.user` by an `input`
            // part. `.assistant` and `.tool` are documented ❌ and have no
            // field at all, so they are refused rather than relabelled.
            roles: [.system, .user],
            systemPrompt: true,
            // Documented ❌ for custom tools. (The surface reaches tools only
            // through LM Studio's own `integrations`, which is a different
            // thing from an `AITool` the caller defines.)
            tools: false,
            responseFormats: [],
            // The surface takes an image as a `{"type": "image", "data_url":
            // …}` part; AIKit builds no data URLs, so an image input is
            // refused rather than dropped.
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
        return try nativeRequest(for: request, model: model, stream: stream)
    }

    public func endpoint(
        for request: AIRequest,
        entry: AIProviderEntry,
        credential: String?,
        stream: Bool
    ) throws(AIKitError) -> any EndPoint {
        try refuseWhatItCannotCarry(request, streaming: stream)
        return try proved(
            LMStudioEndPoint.chat(
                address: entry.address,
                credential: credential,
                body: try nativeRequest(for: request, model: entry.model, stream: stream)
            )
        )
    }

    private func nativeRequest(
        for request: AIRequest,
        model: String,
        stream: Bool
    ) throws(AIKitError) -> LMStudioNativeRequest {
        let parts = request.messages
            .filter { $0.role == .user }
            .map { LMStudioInputPart.text($0.content) }
        guard !parts.isEmpty else {
            // Nothing to ask. `input` is required, and an empty array is a 400
            // — so this is named here rather than discovered on the wire.
            throw .unsupported(field: "messages: no user message to send as 'input'", family: family)
        }

        return LMStudioNativeRequest(
            model: model,
            input: parts,
            systemPrompt: Self.systemPrompt(for: request),
            temperature: request.temperature,
            // `max_tokens` is a 400 on this surface; `max_output_tokens` is the
            // budget field. A non-positive budget is omitted, because the
            // caller's "-1 means until the context is full" convention is an
            // OpenAI one and this server validates its keys.
            maxOutputTokens: request.maxTokens > 0 ? request.maxTokens : nil,
            stream: stream
        )
    }

    /// The `system_prompt` this surface documents: the request's own, plus any
    /// system message that arrived in the array, joined in order.
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
        let reply = try decode(LMStudioNativeReply.self, from: data, context: context)
        let value: LMStudioNativeResponseDTO
        switch reply {
        case .response(let decoded):
            value = decoded
        case .serverError(let message):
            // BR-AIL-10.
            throw .requestFailed("\(message) — from \(context.target)")
        }

        let text = value.text
        guard !text.isEmpty else {
            throw .emptyReply(target: context.target)
        }

        return AIResponse(
            content: text,
            // This surface names the loaded instance rather than a model id.
            model: value.modelInstanceId ?? context.requestedModel,
            tokensUsed: TokenUsage(prompt: value.stats?.inputTokens, completion: value.stats?.totalOutputTokens),
            latencyMs: context.latencyMs,
            providerID: context.providerID
        )
    }

    public func chunk(fromLine line: String, context: WireReplyContext) throws(AIKitError) -> AIChunk? {
        guard let payload = ssePayload(of: line) else { return nil }
        guard let data = payload.data(using: .utf8) else {
            throw .replyUndecodable(target: context.target, reason: "an SSE chunk is not valid UTF-8")
        }
        let event = try decode(LMStudioStreamEvent.self, from: data, context: context)
        switch event.type {
        case LMStudioStreamEvent.messageDelta:
            let delta = event.content ?? ""
            return delta.isEmpty ? nil : AIChunk(delta: delta, isComplete: false)
        case LMStudioStreamEvent.chatEnd:
            return AIChunk(delta: "", isComplete: true)
        default:
            // `chat.start`, `prompt_processing.start`,
            // `prompt_processing.progress`, `message.start`, `message.end`.
            // Lifecycle, not answer text.
            return nil
        }
    }
}
