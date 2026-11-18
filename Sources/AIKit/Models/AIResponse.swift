import Foundation

/// Response from an AI provider.
public struct AIResponse: Sendable, Codable, Equatable {
    public var content: String
    public var model: String
    public var tokensUsed: TokenUsage
    public var latencyMs: Int
    public var toolCalls: [AIToolCall]?
    /// The `AIProvider.id` that actually answered.
    ///
    /// After a fallback this is the provider that succeeded, not the one that
    /// was asked first — which is the whole point of recording it: a response
    /// that cannot name its author hides a misrouted request.
    ///
    /// Empty only when read from a payload written before provenance was
    /// recorded; a live provider always fills it, because the memberwise
    /// initialiser requires it.
    public var providerID: String
    /// Media the provider produced, by reference or by value.
    /// Empty for a text-only answer, which arrives in `content`.
    public var outputs: [AIOutput]

    /// Builds a response, naming the provider that produced it.
    ///
    /// `providerID` has no default on purpose. It carried `""` when provenance
    /// was introduced, which meant a provider that forgot to name itself
    /// compiled and shipped an anonymous answer — and after a fallback an
    /// anonymous answer hides which provider actually served the request
    /// (BR-AIL-06). Requiring it moves that from a runtime blank to a compile
    /// error. Decoding keeps its tolerance: an *old stored payload* has no
    /// provenance to give and must still be readable, which is
    /// `init(from:)`'s job, not this one's.
    public init(
        content: String,
        model: String,
        tokensUsed: TokenUsage,
        latencyMs: Int,
        toolCalls: [AIToolCall]? = nil,
        providerID: String,
        outputs: [AIOutput] = []
    ) {
        self.content = content
        self.model = model
        self.tokensUsed = tokensUsed
        self.latencyMs = latencyMs
        self.toolCalls = toolCalls
        self.providerID = providerID
        self.outputs = outputs
    }

    enum CodingKeys: String, CodingKey {
        case content
        case model
        case tokensUsed
        case latencyMs
        case toolCalls
        case providerID
        case outputs
    }

    /// Decodes a response, including one stored before `providerID` and
    /// `outputs` existed. Responses are cached by callers, so a payload from an
    /// earlier version must not throw on read: it simply has no provenance.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        content = try container.decode(String.self, forKey: .content)
        model = try container.decode(String.self, forKey: .model)
        tokensUsed = try container.decode(TokenUsage.self, forKey: .tokensUsed)
        latencyMs = try container.decode(Int.self, forKey: .latencyMs)
        toolCalls = try container.decodeIfPresent([AIToolCall].self, forKey: .toolCalls)
        providerID = try container.decodeIfPresent(String.self, forKey: .providerID) ?? ""
        outputs = try container.decodeIfPresent([AIOutput].self, forKey: .outputs) ?? []
    }
}

/// A media output produced by a provider.
///
/// The mirror of `AIInput`: a generated image, a rendered video, a transcript
/// written to a file. Text that fits in a string belongs in
/// `AIResponse.content` instead.
public struct AIOutput: Sendable, Codable, Equatable {
    public var kind: AIMediaKind
    public var source: AIMediaSource
    /// IANA media type when the provider reports one, for example `video/mp4`.
    public var mimeType: String?

    public init(kind: AIMediaKind, source: AIMediaSource, mimeType: String? = nil) {
        self.kind = kind
        self.source = source
        self.mimeType = mimeType
    }

    public init(kind: AIMediaKind, url: URL, mimeType: String? = nil) {
        self.init(kind: kind, source: .url(url), mimeType: mimeType)
    }

    public init(kind: AIMediaKind, data: Data, mimeType: String? = nil) {
        self.init(kind: kind, source: .data(data), mimeType: mimeType)
    }

    /// The output's location, when it is carried by reference.
    public var url: URL? {
        if case .url(let url) = source { return url }
        return nil
    }

    /// The output's bytes, when it is carried by value.
    public var data: Data? {
        if case .data(let data) = source { return data }
        return nil
    }
}

/// A tool call returned by the model.
public struct AIToolCall: Sendable, Codable, Equatable {
    /// The server-assigned ID for this call.
    ///
    /// Required when the caller needs to send a tool result back: the tool
    /// message's `toolCallId` must echo this value. `nil` only when read from
    /// a payload written before the field existed.
    public var id: String?
    public var name: String
    public var arguments: String

    public init(id: String? = nil, name: String, arguments: String) {
        self.id = id
        self.name = name
        self.arguments = arguments
    }

    enum CodingKeys: String, CodingKey {
        case id
        case name
        case arguments
    }

    /// Decodes a tool call, tolerating an absent `id` for backward-compat.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decodeIfPresent(String.self, forKey: .id)
        name = try container.decode(String.self, forKey: .name)
        arguments = try container.decode(String.self, forKey: .arguments)
    }
}

/// Token usage statistics.
///
/// A count is `nil` when the provider did not report it. It is never 0 by
/// default: a missing count read as zero silently under-charges a budget and
/// passes a limit check, so a consumer that needs a number decides explicitly
/// how to treat `nil`.
public struct TokenUsage: Sendable, Codable, Equatable {
    public var prompt: Int?
    public var completion: Int?
    /// The sum, or `nil` when either side is unknown: half a sum is not a total.
    public var total: Int? {
        guard let prompt, let completion else { return nil }
        return prompt + completion
    }

    public init(prompt: Int?, completion: Int?) {
        self.prompt = prompt
        self.completion = completion
    }

    enum CodingKeys: String, CodingKey {
        case prompt
        case completion
        case total
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        prompt = try container.decodeIfPresent(Int.self, forKey: .prompt)
        completion = try container.decodeIfPresent(Int.self, forKey: .completion)
        // total is computed, ignore decoded value
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encodeIfPresent(prompt, forKey: .prompt)
        try container.encodeIfPresent(completion, forKey: .completion)
        try container.encodeIfPresent(total, forKey: .total)
    }
}

/// A single chunk from a streaming response.
///
/// Door 2 (TALK) of BR-AIL-21: a consumer streams through `AIProvider.stream`
/// and never sees a provider's chunk shape. Until now that promise held only
/// for TEXT — `delta` plus `isComplete` — while `OpenAIChatCodec.chunk` decoded
/// the role, the tool-call deltas and the finish reason from
/// `ChatCompletionChunkDTO` and threw all three away. A chunk carrying only a
/// tool-call delta (no text, no finish reason) was dropped entirely, so
/// streaming a tool call through AIKit produced nothing at all and a consumer
/// that needed one had to re-implement the wire shape. That is why shikki keeps
/// its own `ChatCompletionReply` / `ToolCall` in `LocalHTTPAgentRunner`.
///
/// The fields below are what the DTO already parsed. Nothing new is read off
/// the wire; what changes is that the public model stops narrowing it.
public struct AIChunk: Sendable, Codable, Equatable {
    /// The text this chunk adds. Empty for a chunk that carries only a
    /// tool-call delta or only an ending.
    public var delta: String
    public var isComplete: Bool

    /// The role the provider declared for the message being streamed, when it
    /// declared one. OpenAI-compatible servers send it on the first chunk only.
    public var role: AIMessage.Role?

    /// The tool calls this chunk advances, in wire order.
    ///
    /// Streaming tool calls arrive in pieces: the first chunk carries the id
    /// and the function name, later chunks carry argument text only. The
    /// consumer accumulates by ``AIToolCallDelta/index``.
    public var toolCallDeltas: [AIToolCallDelta]?

    /// Why the provider stopped, once it has. `nil` on every chunk before the
    /// last. `isComplete` stays the plain answer to "is this the end"; this is
    /// the reason, which a caller needs to tell a completed answer from a
    /// truncated one or from one that ended to run a tool.
    public var finishReason: AIFinishReason?

    /// Which choice this chunk belongs to, when the provider numbers them.
    public var choiceIndex: Int?

    /// Every parameter after `isComplete` defaults, so the two-argument form
    /// every existing caller and `AIKitPublicAPI.txt:453` already use keeps
    /// compiling unchanged.
    public init(
        delta: String,
        isComplete: Bool = false,
        role: AIMessage.Role? = nil,
        toolCallDeltas: [AIToolCallDelta]? = nil,
        finishReason: AIFinishReason? = nil,
        choiceIndex: Int? = nil
    ) {
        self.delta = delta
        self.isComplete = isComplete
        self.role = role
        self.toolCallDeltas = toolCallDeltas
        self.finishReason = finishReason
        self.choiceIndex = choiceIndex
    }
}

/// One step of a tool call being streamed.
///
/// `argumentsFragment` is deliberately a `String` and NOT a typed JSON value,
/// even once `CoreKit.JSONValue` becomes `Codable`: a provider streams the
/// arguments as partial JSON text, and a fragment such as `{"pa` is not valid
/// JSON until the whole call has arrived. Typing it would promise a parse that
/// cannot succeed mid-stream. The typed-JSON question belongs to the COMPLETE
/// call — `AITool.parametersSchema` and `AIToolCall.arguments`, both `String`
/// today — and is left alone here on purpose (see this PR's body).
public struct AIToolCallDelta: Sendable, Codable, Equatable {
    /// The call this fragment belongs to. Fragments for one call share it, and
    /// it is how a consumer concatenates them. Providers that omit it are
    /// streaming a single call, which is index 0.
    public var index: Int
    /// The server-assigned call id, sent once, on the first fragment.
    public var id: String?
    /// The function name, sent once, on the first fragment.
    public var name: String?
    /// This chunk's slice of the argument JSON text.
    public var argumentsFragment: String?

    public init(index: Int, id: String? = nil, name: String? = nil, argumentsFragment: String? = nil) {
        self.index = index
        self.id = id
        self.name = name
        self.argumentsFragment = argumentsFragment
    }
}

/// Why a provider stopped generating.
///
/// A `RawRepresentable` struct rather than an enum, on purpose: the wire
/// vocabulary is open. A provider we have never seen may answer
/// `"max_output_tokens"`, and an enum would force either a lossy `.other`
/// mapping or a decode failure on a response that is otherwise perfectly good.
/// This round-trips any value and still lets call sites switch on the four
/// everyone sends.
public struct AIFinishReason: RawRepresentable, Sendable, Codable, Equatable, Hashable {
    public let rawValue: String

    public init(rawValue: String) {
        self.rawValue = rawValue
    }

    /// The model finished its answer.
    public static let stop = AIFinishReason(rawValue: "stop")
    /// The model hit the token ceiling; the answer is truncated.
    public static let length = AIFinishReason(rawValue: "length")
    /// The model stopped to call a tool — the caller is expected to run it and
    /// send the result back.
    public static let toolCalls = AIFinishReason(rawValue: "tool_calls")
    /// The provider's content filter stopped it.
    public static let contentFilter = AIFinishReason(rawValue: "content_filter")
}
