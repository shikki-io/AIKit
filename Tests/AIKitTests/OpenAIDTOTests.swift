import Foundation
import Testing

@testable import AIKit

// MARK: - OpenAIDTOTests
//
// W24 of spec ai-layer-aikit-core-2026-10-31, threads AIKit#18 r4500592040,
// r4500589635 and siblings: *"We don't have typed model to have fake data?"*
//
// Every payload these tests read is now MINTED from the DTO that decodes it.
// What that changes, and what it must NOT lose:
//
//   GAINED. One statement of each wire shape. A field renamed on the DTO is a
//   compile error in the fixture, where before it left the hand-written
//   payload carrying the old key, `JSONDecoder` ignoring it, the optional
//   swallowing it, and the test passing.
//
//   AT RISK. A symmetric round trip cannot catch a WRONG wire key: rename both
//   ends of `finishReason = "finish_reason"` to `"finishReason"` and encode
//   then decode still agrees. So every test below that used to pin a key by
//   writing it out now asserts the key EXPLICITLY on the encoded object
//   (`#expect(object["finish_reason"] != nil)`). The key names stay proven;
//   only the payload around them stops being hand-written.
//
//   The other half of that evidence is `FixturePinTests`, which pins the four
//   CAPTURED server replies — real bytes, asymmetric — to these same DTOs.

/// The JSON object a DTO encodes to, for asserting wire key names.
private func wireObject(_ value: some Encodable) throws -> [String: Any] {
    let data = try JSONEncoder().encode(value)
    return try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
}

/// Encode, then decode, and hand back both the model and its wire object.
private func roundTrip<T: Codable>(_ value: T) throws -> (model: T, wire: [String: Any]) {
    let data = try JSONEncoder().encode(value)
    return (try JSONDecoder().decode(T.self, from: data), try wireObject(value))
}

@Suite("OpenAI DTOs")
struct OpenAIDTOTests {

    // MARK: - ChatCompletionRequest encoding

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

        let dict = try wireObject(request)

        #expect(dict["model"] as? String == "gpt-4")
        #expect(dict["temperature"] as? Double == 0.7)
        #expect(dict["max_tokens"] as? Int == 1024)
        #expect(dict["stream"] as? Bool == false)

        let messages = try #require(dict["messages"] as? [[String: Any]])
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
                        parameters: JSONFragment(rawJSON: fixtureToolSchemaWithProperties)
                    )
                ),
            ],
            responseFormat: nil
        )

        let dict = try wireObject(request)
        let tools = try #require(dict["tools"] as? [[String: Any]])
        #expect(tools.count == 1)
        #expect(tools[0]["type"] as? String == "function")

        let function = try #require(tools[0]["function"] as? [String: Any])
        #expect(function["name"] as? String == "get_weather")
        // The schema rides as a nested OBJECT, not as an escaped string.
        let parameters = try #require(function["parameters"] as? [String: Any])
        #expect(parameters["type"] as? String == "object")
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

        let format = try #require(try wireObject(request)["response_format"] as? [String: Any])
        #expect(format["type"] as? String == "json_object")
    }

    // MARK: - ChatCompletionResponse

    @Test("ChatCompletionResponse round trips, and keeps its snake_case wire keys")
    func responseRoundTrips() throws {
        let reply = ChatCompletionResponseDTO(
            id: "chatcmpl-abc123",
            model: fakeReplyModel,
            choices: [
                .init(
                    index: 0,
                    message: .init(role: "assistant", content: "Hi there!", toolCalls: nil),
                    finishReason: "stop"
                ),
            ],
            usage: .init(promptTokens: 10, completionTokens: 5, totalTokens: 15)
        )

        let (decoded, wire) = try roundTrip(reply)

        #expect(decoded == reply)
        #expect(decoded.id == "chatcmpl-abc123")
        #expect(decoded.choices.count == 1)
        #expect(decoded.choices[0].message.content == "Hi there!")
        #expect(decoded.choices[0].message.role == "assistant")
        #expect(decoded.choices[0].finishReason == "stop")
        #expect(decoded.usage?.promptTokens == 10)
        #expect(decoded.usage?.completionTokens == 5)
        #expect(decoded.usage?.totalTokens == 15)

        // The keys, asserted rather than implied by a symmetric round trip.
        let choice = try #require((wire["choices"] as? [[String: Any]])?.first)
        #expect(choice["finish_reason"] as? String == "stop")
        let usage = try #require(wire["usage"] as? [String: Any])
        #expect(usage["prompt_tokens"] as? Int == 10)
        #expect(usage["completion_tokens"] as? Int == 5)
        #expect(usage["total_tokens"] as? Int == 15)
    }

    @Test("ChatCompletionResponse carries tool calls, under tool_calls")
    func responseRoundTripsToolCalls() throws {
        let reply = ChatCompletionResponseDTO(
            id: "chatcmpl-tool",
            model: fakeReplyModel,
            choices: [
                .init(
                    index: 0,
                    message: .init(
                        role: "assistant",
                        content: nil,
                        toolCalls: [
                            .init(
                                id: "call_abc",
                                type: "function",
                                function: .init(name: "search", arguments: fixtureToolArguments)
                            ),
                        ]
                    ),
                    finishReason: "tool_calls"
                ),
            ],
            usage: .init(promptTokens: 20, completionTokens: 8, totalTokens: 28)
        )

        let (decoded, wire) = try roundTrip(reply)

        #expect(decoded == reply)
        #expect(decoded.choices[0].message.content == nil)
        #expect(decoded.choices[0].message.toolCalls?.count == 1)

        let toolCall = try #require(decoded.choices[0].message.toolCalls?.first)
        #expect(toolCall.id == "call_abc")
        #expect(toolCall.function.name == "search")
        // The arguments stay an opaque STRING — the tool's own object, which
        // AIKit carries through rather than models.
        #expect(toolCall.function.arguments == fixtureToolArguments)

        let message = try #require(
            ((wire["choices"] as? [[String: Any]])?.first)?["message"] as? [String: Any])
        #expect(message["tool_calls"] != nil)
        let arguments = try #require((message["tool_calls"] as? [[String: Any]])?.first)
        #expect(arguments["function"] != nil)
    }

    // MARK: - ChatCompletionChunk

    @Test("ChatCompletionChunk round trips a content delta")
    func chunkRoundTripsContent() throws {
        let chunk = ChatCompletionChunkDTO(
            id: "chatcmpl-stream1",
            choices: [.init(index: 0, delta: .init(role: nil, content: "Hello", toolCalls: nil), finishReason: nil)]
        )

        let (decoded, _) = try roundTrip(chunk)

        #expect(decoded == chunk)
        #expect(decoded.id == "chatcmpl-stream1")
        #expect(decoded.choices.count == 1)
        #expect(decoded.choices[0].delta.content == "Hello")
        #expect(decoded.choices[0].finishReason == nil)
    }

    @Test("ChatCompletionChunk round trips a final chunk, under finish_reason")
    func chunkRoundTripsFinal() throws {
        let chunk = ChatCompletionChunkDTO(
            id: "chatcmpl-stream2",
            choices: [.init(index: 0, delta: .init(role: nil, content: nil, toolCalls: nil), finishReason: "stop")]
        )

        let (decoded, wire) = try roundTrip(chunk)

        #expect(decoded == chunk)
        #expect(decoded.choices[0].delta.content == nil)
        #expect(decoded.choices[0].finishReason == "stop")
        let choice = try #require((wire["choices"] as? [[String: Any]])?.first)
        #expect(choice["finish_reason"] as? String == "stop")
    }

    @Test("ChatCompletionChunk round trips a role-only delta")
    func chunkRoundTripsRoleDelta() throws {
        let chunk = ChatCompletionChunkDTO(
            id: "chatcmpl-stream0",
            choices: [
                .init(index: 0, delta: .init(role: "assistant", content: nil, toolCalls: nil), finishReason: nil)
            ]
        )

        let (decoded, _) = try roundTrip(chunk)

        #expect(decoded == chunk)
        #expect(decoded.choices[0].delta.role == "assistant")
        #expect(decoded.choices[0].delta.content == nil)
    }

    @Test("ChatCompletionChunk round trips a tool-call delta")
    func chunkRoundTripsToolCallDelta() throws {
        let chunk = ChatCompletionChunkDTO(
            id: "chatcmpl-tooldelta",
            choices: [
                .init(
                    index: 0,
                    delta: .init(
                        role: nil,
                        content: nil,
                        toolCalls: [
                            .init(
                                index: 0,
                                id: "call_xyz",
                                type: "function",
                                function: .init(name: "calc", arguments: fixtureToolArgumentsCity)
                            ),
                        ]
                    ),
                    finishReason: nil
                ),
            ]
        )

        let (decoded, wire) = try roundTrip(chunk)

        #expect(decoded == chunk)
        let toolDelta = try #require(decoded.choices[0].delta.toolCalls?.first)
        #expect(toolDelta.id == "call_xyz")
        #expect(toolDelta.function?.name == "calc")
        #expect(toolDelta.function?.arguments == fixtureToolArgumentsCity)

        let delta = try #require(((wire["choices"] as? [[String: Any]])?.first)?["delta"] as? [String: Any])
        #expect(delta["tool_calls"] != nil)
    }

    // MARK: - ModelsListResponse

    @Test("ModelsListResponse round trips a model list, under owned_by")
    func modelsListRoundTrips() throws {
        let list = ModelsListResponseDTO(data: [
            .init(id: "gpt-4", ownedBy: "openai", created: 1_700_000_000),
            .init(id: "gpt-3.5-turbo", ownedBy: "openai", created: 1_699_000_000),
            .init(id: "whisper-1", ownedBy: "openai-internal", created: nil),
        ])

        let (decoded, wire) = try roundTrip(list)

        #expect(decoded == list)
        #expect(decoded.data.count == 3)
        #expect(decoded.data[0].id == "gpt-4")
        #expect(decoded.data[0].ownedBy == "openai")
        #expect(decoded.data[0].created == 1_700_000_000)
        #expect(decoded.data[2].id == "whisper-1")
        #expect(decoded.data[2].ownedBy == "openai-internal")
        // `created` was minted absent on the third row and must stay absent.
        #expect(decoded.data[2].created == nil)

        let rows = try #require(wire["data"] as? [[String: Any]])
        #expect(rows[0]["owned_by"] as? String == "openai")
        #expect(rows[2]["created"] == nil)
    }

    @Test("ModelsListResponse round trips with the optional fields missing")
    func modelsListRoundTripsMinimal() throws {
        let list = ModelsListResponseDTO(data: [.init(id: "local-model", ownedBy: nil, created: nil)])

        let (decoded, wire) = try roundTrip(list)

        #expect(decoded == list)
        #expect(decoded.data.count == 1)
        #expect(decoded.data[0].id == "local-model")
        #expect(decoded.data[0].ownedBy == nil)
        #expect(decoded.data[0].created == nil)
        // A nil optional is OMITTED, not encoded as null — the shape a minimal
        // server really sends, and what makes the decode above meaningful.
        let row = try #require((wire["data"] as? [[String: Any]])?.first)
        #expect(row.keys.sorted() == ["id"])
    }

    // MARK: - ChatMessage tool fields

    @Test("ChatMessage encodes assistant message with tool_calls")
    func assistantMessageEncodesToolCalls() throws {
        let request = ChatCompletionRequest(
            model: "m",
            messages: [
                .init(
                    role: "assistant",
                    content: nil,
                    toolCalls: [
                        .init(
                            id: "call_001",
                            type: "function",
                            function: .init(name: "search", arguments: fixtureToolArguments)
                        ),
                    ],
                    toolCallId: nil
                ),
            ],
            temperature: 0.7,
            maxTokens: 256,
            stream: false,
            tools: nil,
            responseFormat: nil
        )

        let messages = try #require(try wireObject(request)["messages"] as? [[String: Any]])
        let msg = messages[0]

        // content omitted when nil
        #expect(msg["content"] == nil)
        #expect(msg["role"] as? String == "assistant")

        let toolCalls = try #require(msg["tool_calls"] as? [[String: Any]])
        #expect(toolCalls.count == 1)
        #expect(toolCalls[0]["id"] as? String == "call_001")
        #expect(toolCalls[0]["type"] as? String == "function")

        let fn = try #require(toolCalls[0]["function"] as? [String: Any])
        #expect(fn["name"] as? String == "search")
        #expect(fn["arguments"] as? String == fixtureToolArguments)
    }

    @Test("ChatMessage encodes tool result message with tool_call_id")
    func toolMessageEncodesToolCallId() throws {
        let request = ChatCompletionRequest(
            model: "m",
            messages: [.init(role: "tool", content: "42°C", toolCalls: nil, toolCallId: "call_001")],
            temperature: 0.7,
            maxTokens: 256,
            stream: false,
            tools: nil,
            responseFormat: nil
        )

        let messages = try #require(try wireObject(request)["messages"] as? [[String: Any]])
        let msg = messages[0]

        #expect(msg["role"] as? String == "tool")
        #expect(msg["content"] as? String == "42°C")
        #expect(msg["tool_call_id"] as? String == "call_001")
        #expect(msg["tool_calls"] == nil)
    }

    @Test("ChatMessage omits content when empty and toolCalls present")
    func assistantMessageOmitsEmptyContent() throws {
        // When AIMessage.content is empty and the message has tool calls,
        // the wire format must omit content (nil), not encode "".
        let request = ChatCompletionRequest(
            model: "m",
            messages: [
                .init(
                    role: "assistant",
                    content: nil,
                    toolCalls: [
                        .init(
                            id: "x", type: "function",
                            function: .init(name: "f", arguments: fixtureEmptyJSONObject)),
                    ],
                    toolCallId: nil
                ),
            ],
            temperature: 0.7,
            maxTokens: 256,
            stream: false,
            tools: nil,
            responseFormat: nil
        )
        let messages = try #require(try wireObject(request)["messages"] as? [[String: Any]])
        // Nil content → key absent from dict (encodeIfPresent / encode(String?) omits nil)
        #expect(messages[0]["content"] == nil)
    }

    // MARK: - JSONFragment error handling

    @Test("JSONFragment throws on invalid JSON instead of encoding as plain string")
    func jsonFragmentThrowsOnInvalidJSON() {
        let fragment = JSONFragment(rawJSON: "not-valid-json{")
        #expect(throws: (any Error).self) {
            // The pre-review code silently encoded this as a plain string.
            // After fix it must throw so the caller sees the malformed fragment.
            try JSONEncoder().encode(fragment)
        }
    }

    @Test("JSONFragment encodes valid JSON as a nested object")
    func jsonFragmentEncodesValidJSON() throws {
        let fragment = JSONFragment(rawJSON: fixtureToolSchema)
        let data = try JSONEncoder().encode(fragment)
        let decoded = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        #expect(decoded?["type"] as? String == "object")
    }
}
