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
public struct TokenUsage: Sendable, Codable, Equatable {
    public var prompt: Int
    public var completion: Int
    public var total: Int { prompt + completion }

    public init(prompt: Int, completion: Int) {
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
        prompt = try container.decode(Int.self, forKey: .prompt)
        completion = try container.decode(Int.self, forKey: .completion)
        // total is computed, ignore decoded value
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(prompt, forKey: .prompt)
        try container.encode(completion, forKey: .completion)
        try container.encode(total, forKey: .total)
    }
}

/// A single chunk from a streaming response.
public struct AIChunk: Sendable {
    public var delta: String
    public var isComplete: Bool

    public init(delta: String, isComplete: Bool = false) {
        self.delta = delta
        self.isComplete = isComplete
    }
}
