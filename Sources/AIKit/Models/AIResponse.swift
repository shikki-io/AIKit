import Foundation

/// Response from an AI provider.
public struct AIResponse: Sendable, Codable, Equatable {
    public var content: String
    public var model: String
    public var tokensUsed: TokenUsage
    public var latencyMs: Int
    public var toolCalls: [AIToolCall]?

    public init(
        content: String,
        model: String,
        tokensUsed: TokenUsage,
        latencyMs: Int,
        toolCalls: [AIToolCall]? = nil
    ) {
        self.content = content
        self.model = model
        self.tokensUsed = tokensUsed
        self.latencyMs = latencyMs
        self.toolCalls = toolCalls
    }
}

/// A tool call returned by the model.
public struct AIToolCall: Sendable, Codable, Equatable {
    public var name: String
    public var arguments: String

    public init(name: String, arguments: String) {
        self.name = name
        self.arguments = arguments
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
