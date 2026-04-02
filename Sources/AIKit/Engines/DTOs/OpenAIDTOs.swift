import Foundation

// MARK: - Chat Completion Request

/// OpenAI-compatible chat completion request body.
struct ChatCompletionRequest: Encodable, Sendable {
    let model: String
    let messages: [ChatMessage]
    let temperature: Double
    let maxTokens: Int
    let stream: Bool
    let tools: [ToolDefinition]?
    let responseFormat: ResponseFormatDTO?

    enum CodingKeys: String, CodingKey {
        case model, messages, temperature, stream, tools
        case maxTokens = "max_tokens"
        case responseFormat = "response_format"
    }

    struct ChatMessage: Encodable, Sendable {
        let role: String
        let content: String
    }

    struct ToolDefinition: Encodable, Sendable {
        let type: String
        let function: FunctionDef

        struct FunctionDef: Encodable, Sendable {
            let name: String
            let description: String
            let parameters: JSONFragment
        }
    }

    struct ResponseFormatDTO: Encodable, Sendable {
        let type: String
    }
}

// MARK: - Chat Completion Response

/// OpenAI-compatible chat completion response.
struct ChatCompletionResponseDTO: Decodable, Sendable {
    let id: String
    let choices: [Choice]
    let usage: Usage?

    struct Choice: Decodable, Sendable {
        let index: Int
        let message: Message
        let finishReason: String?

        enum CodingKeys: String, CodingKey {
            case index, message
            case finishReason = "finish_reason"
        }
    }

    struct Message: Decodable, Sendable {
        let role: String
        let content: String?
        let toolCalls: [ToolCall]?

        enum CodingKeys: String, CodingKey {
            case role, content
            case toolCalls = "tool_calls"
        }
    }

    struct ToolCall: Decodable, Sendable {
        let id: String?
        let type: String?
        let function: FunctionCall

        struct FunctionCall: Decodable, Sendable {
            let name: String
            let arguments: String
        }
    }

    struct Usage: Decodable, Sendable {
        let promptTokens: Int
        let completionTokens: Int
        let totalTokens: Int

        enum CodingKeys: String, CodingKey {
            case promptTokens = "prompt_tokens"
            case completionTokens = "completion_tokens"
            case totalTokens = "total_tokens"
        }
    }
}

// MARK: - Streaming Chunk

/// A single SSE chunk from a streaming chat completion.
struct ChatCompletionChunkDTO: Decodable, Sendable {
    let id: String
    let choices: [Choice]

    struct Choice: Decodable, Sendable {
        let index: Int
        let delta: Delta
        let finishReason: String?

        enum CodingKeys: String, CodingKey {
            case index, delta
            case finishReason = "finish_reason"
        }
    }

    struct Delta: Decodable, Sendable {
        let role: String?
        let content: String?
        let toolCalls: [ToolCallDelta]?

        enum CodingKeys: String, CodingKey {
            case role, content
            case toolCalls = "tool_calls"
        }
    }

    struct ToolCallDelta: Decodable, Sendable {
        let index: Int?
        let id: String?
        let type: String?
        let function: FunctionDelta?

        struct FunctionDelta: Decodable, Sendable {
            let name: String?
            let arguments: String?
        }
    }
}

// MARK: - Models List

/// Response from /v1/models endpoint.
struct ModelsListResponseDTO: Decodable, Sendable {
    let data: [ModelInfo]

    struct ModelInfo: Decodable, Sendable {
        let id: String
        let ownedBy: String?
        let created: Int?

        enum CodingKeys: String, CodingKey {
            case id
            case ownedBy = "owned_by"
            case created
        }
    }
}

// MARK: - JSON Fragment

/// Wraps a raw JSON string so it encodes as a JSON object (not an escaped string).
struct JSONFragment: Encodable, Sendable {
    let rawJSON: String

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        if let data = rawJSON.data(using: .utf8),
           let object = try? JSONSerialization.jsonObject(with: data) {
            // Re-serialize through the encoder to produce valid nested JSON.
            let reEncoded = try JSONSerialization.data(withJSONObject: object)
            let rawValue = try JSONDecoder().decode(AnyCodable.self, from: reEncoded)
            try container.encode(rawValue)
        } else {
            // Fallback: encode as a plain string.
            try container.encode(rawJSON)
        }
    }
}

/// Minimal type-erased Codable wrapper for arbitrary JSON.
private enum AnyCodable: Codable, Sendable {
    case string(String)
    case int(Int)
    case double(Double)
    case bool(Bool)
    case array([AnyCodable])
    case dictionary([String: AnyCodable])
    case null

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let value = try? container.decode(Bool.self) {
            self = .bool(value)
        } else if let value = try? container.decode(Int.self) {
            self = .int(value)
        } else if let value = try? container.decode(Double.self) {
            self = .double(value)
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode([AnyCodable].self) {
            self = .array(value)
        } else if let value = try? container.decode([String: AnyCodable].self) {
            self = .dictionary(value)
        } else if container.decodeNil() {
            self = .null
        } else {
            throw DecodingError.dataCorruptedError(in: container, debugDescription: "Unsupported JSON value")
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let value): try container.encode(value)
        case .int(let value): try container.encode(value)
        case .double(let value): try container.encode(value)
        case .bool(let value): try container.encode(value)
        case .array(let value): try container.encode(value)
        case .dictionary(let value): try container.encode(value)
        case .null: try container.encodeNil()
        }
    }
}
