import Foundation
import Testing
@testable import AIKit

@Suite("OpenAI DTOs")
struct OpenAIDTOTests {

    // MARK: - ChatCompletionRequest Encoding

    @Test("ChatCompletionRequest encodes correctly")
    func requestEncodesCorrectly() throws {
        let request = ChatCompletionRequest(
            model: "gpt-4",
            messages: [
                .init(role: "system", content: "You are helpful."),
                .init(role: "user", content: "Hello"),
            ],
            temperature: 0.7,
            maxTokens: 1024,
            stream: false,
            tools: nil,
            responseFormat: nil
        )

        let data = try JSONEncoder().encode(request)
        let dict = try JSONSerialization.jsonObject(with: data) as! [String: Any]

        #expect(dict["model"] as? String == "gpt-4")
        #expect(dict["temperature"] as? Double == 0.7)
        #expect(dict["max_tokens"] as? Int == 1024)
        #expect(dict["stream"] as? Bool == false)

        let messages = dict["messages"] as! [[String: Any]]
        #expect(messages.count == 2)
        #expect(messages[0]["role"] as? String == "system")
        #expect(messages[1]["content"] as? String == "Hello")
    }

    @Test("ChatCompletionRequest encodes tools")
    func requestEncodesTools() throws {
        let request = ChatCompletionRequest(
            model: "gpt-4",
            messages: [.init(role: "user", content: "Weather?")],
            temperature: 0.7,
            maxTokens: 256,
            stream: false,
            tools: [
                .init(
                    type: "function",
                    function: .init(
                        name: "get_weather",
                        description: "Get weather for a city",
                        parameters: JSONFragment(rawJSON: "{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}}}")
                    )
                ),
            ],
            responseFormat: nil
        )

        let data = try JSONEncoder().encode(request)
        let dict = try JSONSerialization.jsonObject(with: data) as! [String: Any]

        let tools = dict["tools"] as! [[String: Any]]
        #expect(tools.count == 1)
        #expect(tools[0]["type"] as? String == "function")

        let function = tools[0]["function"] as! [String: Any]
        #expect(function["name"] as? String == "get_weather")
    }

    @Test("ChatCompletionRequest encodes response_format")
    func requestEncodesResponseFormat() throws {
        let request = ChatCompletionRequest(
            model: "gpt-4",
            messages: [.init(role: "user", content: "Return JSON")],
            temperature: 0.5,
            maxTokens: 512,
            stream: false,
            tools: nil,
            responseFormat: .init(type: "json_object")
        )

        let data = try JSONEncoder().encode(request)
        let dict = try JSONSerialization.jsonObject(with: data) as! [String: Any]

        let format = dict["response_format"] as! [String: Any]
        #expect(format["type"] as? String == "json_object")
    }

    // MARK: - ChatCompletionResponse Decoding

    @Test("ChatCompletionResponse decodes correctly")
    func responseDecodesCorrectly() throws {
        let json = """
        {
            "id": "chatcmpl-abc123",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hi there!"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }
        """

        let response = try JSONDecoder().decode(ChatCompletionResponseDTO.self, from: Data(json.utf8))

        #expect(response.id == "chatcmpl-abc123")
        #expect(response.choices.count == 1)
        #expect(response.choices[0].message.content == "Hi there!")
        #expect(response.choices[0].message.role == "assistant")
        #expect(response.choices[0].finishReason == "stop")
        #expect(response.usage?.promptTokens == 10)
        #expect(response.usage?.completionTokens == 5)
        #expect(response.usage?.totalTokens == 15)
    }

    @Test("ChatCompletionResponse decodes tool calls")
    func responseDecodesToolCalls() throws {
        let json = """
        {
            "id": "chatcmpl-tool",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "search", "arguments": "{\\"query\\":\\"swift\\"}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28}
        }
        """

        let response = try JSONDecoder().decode(ChatCompletionResponseDTO.self, from: Data(json.utf8))

        #expect(response.choices[0].message.content == nil)
        #expect(response.choices[0].message.toolCalls?.count == 1)

        let toolCall = response.choices[0].message.toolCalls![0]
        #expect(toolCall.id == "call_abc")
        #expect(toolCall.function.name == "search")
        #expect(toolCall.function.arguments == "{\"query\":\"swift\"}")
    }

    // MARK: - ChatCompletionChunk Decoding

    @Test("ChatCompletionChunk decodes streaming format")
    func chunkDecodesStreaming() throws {
        let json = """
        {
            "id": "chatcmpl-stream1",
            "choices": [{
                "index": 0,
                "delta": {"content": "Hello"},
                "finish_reason": null
            }]
        }
        """

        let chunk = try JSONDecoder().decode(ChatCompletionChunkDTO.self, from: Data(json.utf8))

        #expect(chunk.id == "chatcmpl-stream1")
        #expect(chunk.choices.count == 1)
        #expect(chunk.choices[0].delta.content == "Hello")
        #expect(chunk.choices[0].finishReason == nil)
    }

    @Test("ChatCompletionChunk decodes final chunk with finish_reason")
    func chunkDecodesFinal() throws {
        let json = """
        {
            "id": "chatcmpl-stream2",
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        }
        """

        let chunk = try JSONDecoder().decode(ChatCompletionChunkDTO.self, from: Data(json.utf8))

        #expect(chunk.choices[0].delta.content == nil)
        #expect(chunk.choices[0].finishReason == "stop")
    }

    @Test("ChatCompletionChunk decodes role-only delta")
    func chunkDecodesRoleDelta() throws {
        let json = """
        {
            "id": "chatcmpl-stream0",
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant"},
                "finish_reason": null
            }]
        }
        """

        let chunk = try JSONDecoder().decode(ChatCompletionChunkDTO.self, from: Data(json.utf8))

        #expect(chunk.choices[0].delta.role == "assistant")
        #expect(chunk.choices[0].delta.content == nil)
    }

    // MARK: - ModelsListResponse Decoding

    @Test("ModelsListResponse decodes model list")
    func modelsListDecodes() throws {
        let json = """
        {
            "data": [
                {"id": "gpt-4", "owned_by": "openai", "created": 1700000000},
                {"id": "gpt-3.5-turbo", "owned_by": "openai", "created": 1699000000},
                {"id": "whisper-1", "owned_by": "openai-internal"}
            ]
        }
        """

        let response = try JSONDecoder().decode(ModelsListResponseDTO.self, from: Data(json.utf8))

        #expect(response.data.count == 3)
        #expect(response.data[0].id == "gpt-4")
        #expect(response.data[0].ownedBy == "openai")
        #expect(response.data[0].created == 1700000000)
        #expect(response.data[2].id == "whisper-1")
        #expect(response.data[2].ownedBy == "openai-internal")
    }

    @Test("ModelsListResponse decodes with missing optional fields")
    func modelsListDecodesMinimal() throws {
        let json = """
        {"data": [{"id": "local-model"}]}
        """

        let response = try JSONDecoder().decode(ModelsListResponseDTO.self, from: Data(json.utf8))

        #expect(response.data.count == 1)
        #expect(response.data[0].id == "local-model")
        #expect(response.data[0].ownedBy == nil)
        #expect(response.data[0].created == nil)
    }

    // MARK: - ChatCompletionChunk Tool Call Delta

    @Test("ChatCompletionChunk decodes tool call delta")
    func chunkDecodesToolCallDelta() throws {
        let json = """
        {
            "id": "chatcmpl-tooldelta",
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_xyz",
                        "type": "function",
                        "function": {"name": "calc", "arguments": "{\\"x\\":1}"}
                    }]
                },
                "finish_reason": null
            }]
        }
        """

        let chunk = try JSONDecoder().decode(ChatCompletionChunkDTO.self, from: Data(json.utf8))

        let toolDelta = chunk.choices[0].delta.toolCalls?.first
        #expect(toolDelta?.id == "call_xyz")
        #expect(toolDelta?.function?.name == "calc")
        #expect(toolDelta?.function?.arguments == "{\"x\":1}")
    }
}
