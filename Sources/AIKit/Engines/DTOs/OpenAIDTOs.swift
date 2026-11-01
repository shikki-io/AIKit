import Foundation

// MARK: - Chat Completion Request

// CodingKey for ChatCompletionRequest.ChatMessage fields.
// Defined at file scope to avoid a 2-level nesting violation (SwiftLint `nesting`).
private enum ChatMessageKey: String, CodingKey {
    case role, content
    case toolCalls = "tool_calls"
    case toolCallId = "tool_call_id"
}

/// Function call details inside a ToolCallRequest.
/// Defined at file scope to avoid a 2-level nesting violation (SwiftLint `nesting`).
struct ToolCallFunctionRequest: Encodable, Sendable {
    let name: String
    let arguments: String
}

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
        /// Nil for an assistant message that carries only `toolCalls`.
        let content: String?
        /// Tool calls the assistant is requesting (`role == "assistant"`).
        let toolCalls: [ToolCallRequest]?
        /// ID of the tool call being responded to (`role == "tool"`).
        let toolCallId: String?

        init(role: String, content: String?, toolCalls: [ToolCallRequest]? = nil, toolCallId: String? = nil) {
            self.role = role
            self.content = content
            self.toolCalls = toolCalls
            self.toolCallId = toolCallId
        }

        func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: ChatMessageKey.self)
            try container.encode(role, forKey: .role)
            // Only encode content when present; omitting it produces the
            // correct wire format for a tool-calls-only assistant message.
            if let content {
                try container.encode(content, forKey: .content)
            }
            try container.encodeIfPresent(toolCalls, forKey: .toolCalls)
            try container.encodeIfPresent(toolCallId, forKey: .toolCallId)
        }
    }

    /// A tool call sent IN a request (assistant-turn replay).
    struct ToolCallRequest: Encodable, Sendable {
        let id: String
        let type: String
        let function: ToolCallFunctionRequest
    }

    struct ToolDefinition: Encodable, Sendable {
        let type: String
        let function: FunctionDef

        // swiftlint:disable:next nesting
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
    /// The model that actually answered.
    ///
    /// Optional because a reply is not required to echo it, and because this
    /// type is decode-only — no call site builds it memberwise. Measured on LM
    /// Studio 2026-10-31: this surface *ignores* an unknown `model` in the
    /// request and answers with whatever is loaded, so the request's model name
    /// is not evidence of who answered and the reply's value is (BR-AIL-06).
    let model: String?
    let choices: [Choice]
    let usage: Usage?

    struct Choice: Decodable, Sendable {
        let index: Int
        let message: Message
        let finishReason: String?

        // swiftlint:disable:next nesting
        enum CodingKeys: String, CodingKey {
            case index, message
            case finishReason = "finish_reason"
        }
    }

    struct Message: Decodable, Sendable {
        let role: String
        let content: String?
        let toolCalls: [ToolCall]?

        // swiftlint:disable:next nesting
        enum CodingKeys: String, CodingKey {
            case role, content
            case toolCalls = "tool_calls"
        }
    }

    struct ToolCall: Decodable, Sendable {
        let id: String?
        let type: String?
        let function: FunctionCall

        // swiftlint:disable:next nesting
        struct FunctionCall: Decodable, Sendable {
            let name: String
            let arguments: String
        }
    }

    struct Usage: Decodable, Sendable {
        let promptTokens: Int
        let completionTokens: Int
        let totalTokens: Int

        // swiftlint:disable:next nesting
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

        // swiftlint:disable:next nesting
        enum CodingKeys: String, CodingKey {
            case index, delta
            case finishReason = "finish_reason"
        }
    }

    struct Delta: Decodable, Sendable {
        let role: String?
        let content: String?
        let toolCalls: [ToolCallDelta]?

        // swiftlint:disable:next nesting
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

        // swiftlint:disable:next nesting
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

        // swiftlint:disable:next nesting
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
