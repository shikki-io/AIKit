import Foundation

/// A request to an AI provider.
public struct AIRequest: Sendable, Codable {
    public var messages: [AIMessage]
    public var systemPrompt: String?
    public var temperature: Double
    public var maxTokens: Int
    public var model: ModelIdentifier?
    public var tools: [AITool]?
    public var responseFormat: ResponseFormat?
    /// Media the request carries, by reference or by value.
    /// Empty for a text-only request, which is why it is not optional:
    /// "no inputs" and "an empty list of inputs" are the same thing.
    public var inputs: [AIInput]

    public init(
        messages: [AIMessage],
        systemPrompt: String? = nil,
        temperature: Double = 0.7,
        maxTokens: Int = 1024,
        model: ModelIdentifier? = nil,
        tools: [AITool]? = nil,
        responseFormat: ResponseFormat? = nil,
        inputs: [AIInput] = []
    ) {
        self.messages = messages
        self.systemPrompt = systemPrompt
        self.temperature = temperature
        self.maxTokens = maxTokens
        self.model = model
        self.tools = tools
        self.responseFormat = responseFormat
        self.inputs = inputs
    }

    enum CodingKeys: String, CodingKey {
        case messages
        case systemPrompt
        case temperature
        case maxTokens
        case model
        case tools
        case responseFormat
        case inputs
    }

    /// Decodes a request, including one stored before `inputs` existed.
    /// Callers persist `AIRequest`, so a payload written by an earlier version
    /// must stay readable: a missing `inputs` key means no media, not a failure.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        messages = try container.decode([AIMessage].self, forKey: .messages)
        systemPrompt = try container.decodeIfPresent(String.self, forKey: .systemPrompt)
        temperature = try container.decode(Double.self, forKey: .temperature)
        maxTokens = try container.decode(Int.self, forKey: .maxTokens)
        model = try container.decodeIfPresent(ModelIdentifier.self, forKey: .model)
        tools = try container.decodeIfPresent([AITool].self, forKey: .tools)
        responseFormat = try container.decodeIfPresent(ResponseFormat.self, forKey: .responseFormat)
        inputs = try container.decodeIfPresent([AIInput].self, forKey: .inputs) ?? []
    }
}

/// What a media payload is, in either direction.
public enum AIMediaKind: String, Sendable, Codable {
    case audio
    case image
    case video
    case document
    /// A text artifact a provider produced as a file rather than as `content`.
    case text
}

/// Where a media payload lives: by reference, or by value.
///
/// A URL keeps a large file out of memory and out of every log that prints a
/// request; data is for a payload that has no file behind it. A source is one
/// or the other, never both and never neither.
public enum AIMediaSource: Sendable, Equatable {
    case url(URL)
    case data(Data)
}

extension AIMediaSource: Codable {
    enum CodingKeys: String, CodingKey {
        case url
        case data
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        if let url = try container.decodeIfPresent(URL.self, forKey: .url) {
            self = .url(url)
        } else if let data = try container.decodeIfPresent(Data.self, forKey: .data) {
            self = .data(data)
        } else {
            throw DecodingError.dataCorrupted(
                DecodingError.Context(
                    codingPath: container.codingPath,
                    debugDescription: "A media source carries either a url or data; this payload carries neither."
                )
            )
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .url(let url):
            try container.encode(url, forKey: .url)
        case .data(let data):
            try container.encode(data, forKey: .data)
        }
    }
}

/// A media input carried by an `AIRequest`.
///
/// An audio file for transcription, an image for a vision model: the kind says
/// what it is, the source says where it is, and the provider's
/// `AICapabilities` decide whether it can be served at all.
public struct AIInput: Sendable, Codable, Equatable {
    public var kind: AIMediaKind
    public var source: AIMediaSource
    /// IANA media type when the caller knows it, for example `audio/wav`.
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

    /// The input's location, when it is carried by reference.
    public var url: URL? {
        if case .url(let url) = source { return url }
        return nil
    }

    /// The input's bytes, when it is carried by value.
    public var data: Data? {
        if case .data(let data) = source { return data }
        return nil
    }
}

/// A single message in a conversation.
public struct AIMessage: Sendable, Codable, Equatable {
    public enum Role: String, Sendable, Codable {
        case system
        case user
        case assistant
        case tool
    }

    public var role: Role
    /// The text content of the message.
    ///
    /// Empty for an assistant message that carries only `toolCalls` and has no
    /// parallel text. On the wire those messages need `"content": null`, so the
    /// engine checks `content.isEmpty && toolCalls != nil` before deciding
    /// whether to omit the field. Keeping the type `String` (not optional)
    /// preserves backward-compat for every caller that builds a plain text
    /// message and never thinks about tool loops.
    public var content: String
    public var images: [Data]?
    /// Tool calls the assistant is requesting, when `role == .assistant`.
    ///
    /// Each call carries the `id` the server assigned: the tool result message
    /// that follows must echo that same `id` in `toolCallId`.
    public var toolCalls: [AIToolCall]?
    /// Server-assigned ID of the tool call this message is responding to,
    /// when `role == .tool`.
    public var toolCallId: String?

    public init(
        role: Role,
        content: String,
        images: [Data]? = nil,
        toolCalls: [AIToolCall]? = nil,
        toolCallId: String? = nil
    ) {
        self.role = role
        self.content = content
        self.images = images
        self.toolCalls = toolCalls
        self.toolCallId = toolCallId
    }

    enum CodingKeys: String, CodingKey {
        case role
        case content
        case images
        case toolCalls
        case toolCallId
    }

    /// Decodes a message, including one written before `toolCalls` and
    /// `toolCallId` existed. Stored conversations must still decode without
    /// those keys.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        role = try container.decode(Role.self, forKey: .role)
        content = try container.decodeIfPresent(String.self, forKey: .content) ?? ""
        images = try container.decodeIfPresent([Data].self, forKey: .images)
        toolCalls = try container.decodeIfPresent([AIToolCall].self, forKey: .toolCalls)
        toolCallId = try container.decodeIfPresent(String.self, forKey: .toolCallId)
    }
}

/// A tool available for function calling.
public struct AITool: Sendable, Codable, Equatable {
    public var name: String
    public var description: String
    public var parametersSchema: String

    public init(name: String, description: String, parametersSchema: String) {
        self.name = name
        self.description = description
        self.parametersSchema = parametersSchema
    }
}

/// Response format hint.
public enum ResponseFormat: String, Sendable, Codable {
    case text
    case json
}
