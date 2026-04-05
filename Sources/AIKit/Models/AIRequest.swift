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

    public init(
        messages: [AIMessage],
        systemPrompt: String? = nil,
        temperature: Double = 0.7,
        maxTokens: Int = 1024,
        model: ModelIdentifier? = nil,
        tools: [AITool]? = nil,
        responseFormat: ResponseFormat? = nil
    ) {
        self.messages = messages
        self.systemPrompt = systemPrompt
        self.temperature = temperature
        self.maxTokens = maxTokens
        self.model = model
        self.tools = tools
        self.responseFormat = responseFormat
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
    public var content: String
    public var images: [Data]?

    public init(role: Role, content: String, images: [Data]? = nil) {
        self.role = role
        self.content = content
        self.images = images
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
