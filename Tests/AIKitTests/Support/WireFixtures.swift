import Foundation
import Testing

@testable import AIKit

// MARK: - Captured replies
//
// PROVENANCE. Every reply and every SSE line below is a real one, captured
// from the operator's LM Studio at 127.0.0.1:1234 with
// `mistralai_devstral-small-2-24b-instruct-2512-mlx` loaded — the complete
// replies on 2026-10-31 from a prompt of "Reply with exactly: pong", the
// streams on 2026-11-12 from the same prompt. They are pinned verbatim,
// including the fields AIKit does not read (`logprobs`, `stats`,
// `system_fingerprint`, `reasoning_content`, `sequence_number`), because a
// fixture trimmed to what the decoder wants stops being evidence that the
// decoder survives the real thing.
//
// They are Swift literals rather than files under `Tests/AIKitTests/Fixtures/`
// because this repo has no on-disk fixture convention and loading them from
// disk would need a `resources:` declaration in `Package.swift`. W24 of spec
// ai-layer-aikit-core-2026-10-31 pins each one to its typed model both ways;
// a captured server reply is the one raw literal that wave still allows.

/// `/v1/chat/completions` — HTTP 200.
let capturedChatCompletionsReply = """
    {
      "id": "chatcmpl-khd9f53ms98pkmkg2wo91",
      "object": "chat.completion",
      "created": 1793457204,
      "model": "mistralai_devstral-small-2-24b-instruct-2512-mlx",
      "choices": [
        {
          "index": 0,
          "message": {
            "role": "assistant",
            "content": "pong",
            "reasoning_content": "",
            "tool_calls": []
          },
          "logprobs": null,
          "finish_reason": "stop"
        }
      ],
      "usage": {
        "prompt_tokens": 9,
        "completion_tokens": 3,
        "total_tokens": 12,
        "completion_tokens_details": { "reasoning_tokens": 0 }
      },
      "stats": {},
      "system_fingerprint": "mistralai_devstral-small-2-24b-instruct-2512-mlx"
    }
    """

/// `/v1/responses` — HTTP 200. Note `"error": null` on a SUCCESSFUL reply.
let capturedResponsesReply = """
    {
      "id": "resp_b13bdc5d60556b007624d79ebde5ff0358439b1d87179228",
      "object": "response",
      "created_at": 1793457212,
      "completed_at": 1793457212,
      "status": "completed",
      "incomplete_details": null,
      "model": "mistralai_devstral-small-2-24b-instruct-2512-mlx",
      "previous_response_id": null,
      "instructions": null,
      "output": [
        {
          "id": "msg_afef27hbd5u5kzzic98z5o",
          "type": "message",
          "role": "assistant",
          "status": "completed",
          "content": [
            { "type": "output_text", "text": "pong", "annotations": [], "logprobs": [] }
          ]
        }
      ],
      "error": null,
      "tools": [],
      "tool_choice": "auto",
      "usage": {
        "input_tokens": 9,
        "output_tokens": 2,
        "total_tokens": 11,
        "input_tokens_details": { "cached_tokens": 8 },
        "output_tokens_details": { "reasoning_tokens": 0 }
      },
      "max_output_tokens": 16,
      "store": true,
      "service_tier": "default",
      "metadata": {}
    }
    """

/// `/v1/messages` — HTTP 200.
let capturedAnthropicReply = """
    {
      "id": "msg_0302tng5wkeadg5qtbnjrzn",
      "type": "message",
      "role": "assistant",
      "content": [
        { "type": "text", "text": "pong" }
      ],
      "model": "mistralai_devstral-small-2-24b-instruct-2512-mlx",
      "stop_reason": "end_turn",
      "stop_sequence": null,
      "usage": {
        "input_tokens": 9,
        "output_tokens": 3,
        "cache_read_input_tokens": 8
      }
    }
    """

/// `/api/v1/chat` — HTTP 200.
let capturedLMStudioNativeReply = """
    {
      "model_instance_id": "mistralai_devstral-small-2-24b-instruct-2512-mlx",
      "output": [
        { "type": "message", "content": "pong" }
      ],
      "stats": {
        "input_tokens": 9,
        "total_output_tokens": 3,
        "reasoning_output_tokens": 0,
        "tokens_per_second": 35.10915764798229,
        "time_to_first_token_seconds": 0.296
      },
      "response_id": "resp_ce67a81f5f12825ad85e85d8fbad7793f8ecf864afa4a16d"
    }
    """

/// The model every capture above was answered by.
let capturedModel = "mistralai_devstral-small-2-24b-instruct-2512-mlx"

// MARK: - Captured streams

/// `/v1/chat/completions` with `stream: true`, captured 2026-11-12.
let capturedChatSSELines = [
    "data: {\"id\":\"chatcmpl-1pbtb60kxuhjxt37h7fx4c\",\"object\":\"chat.completion.chunk\",\"created\":1794465776,\"model\":\"\(capturedModel)\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"p\"},\"logprobs\":null,\"finish_reason\":null}]}",
    "",
    "data: {\"id\":\"chatcmpl-1pbtb60kxuhjxt37h7fx4c\",\"object\":\"chat.completion.chunk\",\"created\":1794465776,\"model\":\"\(capturedModel)\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ong\"},\"logprobs\":null,\"finish_reason\":null}]}",
    "",
    "data: {\"id\":\"chatcmpl-1pbtb60kxuhjxt37h7fx4c\",\"object\":\"chat.completion.chunk\",\"created\":1794465776,\"model\":\"\(capturedModel)\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"\"},\"logprobs\":null,\"finish_reason\":null}]}",
    "",
    "data: {\"id\":\"chatcmpl-1pbtb60kxuhjxt37h7fx4c\",\"object\":\"chat.completion.chunk\",\"created\":1794465776,\"model\":\"\(capturedModel)\",\"choices\":[{\"index\":0,\"delta\":{},\"logprobs\":null,\"finish_reason\":\"stop\"}]}",
    "",
    "data: [DONE]",
]

/// `/v1/responses` with `stream: true`, captured 2026-11-12. Both the
/// `event:` labels and the lifecycle events are kept, because a reader that
/// only works on a trimmed stream is not a reader for this surface.
let capturedResponsesSSELines = [
    "event: response.created",
    "data: {\"type\":\"response.created\",\"sequence_number\":0}",
    "",
    "event: response.content_part.added",
    "data: {\"type\":\"response.content_part.added\",\"item_id\":\"msg_r9kx9bw0pnaj5u4u354s88\",\"output_index\":0,\"content_index\":0,\"part\":{\"type\":\"output_text\",\"text\":\"\",\"annotations\":[],\"logprobs\":[]},\"sequence_number\":3}",
    "",
    "event: response.output_text.delta",
    "data: {\"type\":\"response.output_text.delta\",\"item_id\":\"msg_r9kx9bw0pnaj5u4u354s88\",\"output_index\":0,\"content_index\":0,\"delta\":\"p\",\"logprobs\":[],\"sequence_number\":4}",
    "",
    "event: response.output_text.delta",
    "data: {\"type\":\"response.output_text.delta\",\"item_id\":\"msg_r9kx9bw0pnaj5u4u354s88\",\"output_index\":0,\"content_index\":0,\"delta\":\"ong\",\"logprobs\":[],\"sequence_number\":5}",
    "",
    "event: response.output_text.done",
    "data: {\"type\":\"response.output_text.done\",\"item_id\":\"msg_r9kx9bw0pnaj5u4u354s88\",\"output_index\":0,\"content_index\":0,\"text\":\"pong\",\"logprobs\":[],\"sequence_number\":6}",
    "",
    "event: response.completed",
    "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_f62c927bf606f5444f8e72f589f2c1c89dc3706cc05ff639\",\"status\":\"completed\"},\"sequence_number\":9}",
]

/// `/v1/messages` with `stream: true`, captured 2026-11-12.
let capturedAnthropicSSELines = [
    "event: message_start",
    "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1ap9mhfh02n0kvwmpfa6lb\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"\(capturedModel)\",\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":9,\"output_tokens\":0,\"cache_read_input_tokens\":8}}}",
    "",
    "event: content_block_start",
    "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}",
    "",
    "event: content_block_delta",
    "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"p\"}}",
    "",
    "event: content_block_delta",
    "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"ong\"}}",
    "",
    "event: content_block_delta",
    "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"\"}}",
    "",
    "event: content_block_stop",
    "data: {\"type\":\"content_block_stop\",\"index\":0}",
    "",
    "event: message_delta",
    "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"input_tokens\":9,\"output_tokens\":3,\"cache_read_input_tokens\":8}}",
    "",
    "event: message_stop",
    "data: {\"type\":\"message_stop\"}",
]

/// `/api/v1/chat` with `stream: true`, captured 2026-11-12. This surface
/// streams its own lifecycle as well as the answer.
let capturedLMStudioSSELines = [
    "event: chat.start",
    "data: {\"type\":\"chat.start\",\"model_instance_id\":\"\(capturedModel)\"}",
    "",
    "event: prompt_processing.start",
    "data: {\"type\":\"prompt_processing.start\"}",
    "",
    "event: prompt_processing.progress",
    "data: {\"type\":\"prompt_processing.progress\",\"progress\":0}",
    "",
    "event: message.start",
    "data: {\"type\":\"message.start\"}",
    "",
    "event: message.delta",
    "data: {\"type\":\"message.delta\",\"content\":\"p\"}",
    "",
    "event: message.delta",
    "data: {\"type\":\"message.delta\",\"content\":\"ong\"}",
    "",
    "event: message.end",
    "data: {\"type\":\"message.end\"}",
    "",
    "event: chat.end",
    "data: {\"type\":\"chat.end\",\"result\":{\"model_instance_id\":\"\(capturedModel)\",\"output\":[{\"type\":\"message\",\"content\":\"pong\"}],\"response_id\":\"resp_881d7dab\"}}",
]

// MARK: - The family table

/// One family, with the captures and route that belong to it.
///
/// Not `private`: a `@Test(arguments:)` case takes it as a parameter, and a
/// parameter type has to be at least as visible as the method.
struct WireFamilyFixture: Sendable, CustomTestStringConvertible {
    let family: AIProviderFamily
    let codec: any WireCodec
    let route: String
    /// A complete 2xx reply, as captured.
    let reply: String
    /// The same answer streamed, as captured.
    let sseLines: [String]
    let promptTokens: Int
    let completionTokens: Int

    /// Names the case in test output by its family, so a failure reads
    /// `anthropic-messages` rather than a struct dump.
    var testDescription: String { family.rawValue }
}

let wireFamilyFixtures: [WireFamilyFixture] = [
    WireFamilyFixture(
        family: .openAIChat,
        codec: OpenAIChatCodec(),
        route: "/v1/chat/completions",
        reply: capturedChatCompletionsReply,
        sseLines: capturedChatSSELines,
        promptTokens: 9,
        completionTokens: 3
    ),
    WireFamilyFixture(
        family: .openAIResponses,
        codec: OpenAIResponsesCodec(),
        route: "/v1/responses",
        reply: capturedResponsesReply,
        sseLines: capturedResponsesSSELines,
        promptTokens: 9,
        completionTokens: 2
    ),
    WireFamilyFixture(
        family: .anthropicMessages,
        codec: AnthropicMessagesCodec(),
        route: "/v1/messages",
        reply: capturedAnthropicReply,
        sseLines: capturedAnthropicSSELines,
        promptTokens: 9,
        completionTokens: 3
    ),
    WireFamilyFixture(
        family: .lmStudioNative,
        codec: LMStudioNativeCodec(),
        route: "/api/v1/chat",
        reply: capturedLMStudioNativeReply,
        sseLines: capturedLMStudioSSELines,
        promptTokens: 9,
        completionTokens: 3
    ),
]

// MARK: - Helpers

/// The one server address every suite builds from, through the validating
/// init. No test writes a URL string of its own (W21's rule, adopted here as
/// the type arrives).
let fixtureServerRoot = "http://127.0.0.1:1234"

func makeFixtureAddress(_ root: String = fixtureServerRoot) throws -> AIServerAddress {
    try AIServerAddress(validating: root)
}

/// A settings entry for one family at the fixture address.
func makeFixtureEntry(
    family: AIProviderFamily,
    id: String = "lmstudio",
    model: String = "requested-model",
    root: String = fixtureServerRoot,
    streamLineByteCap: Int = AIProviderEntry.defaultStreamLineByteCap
) throws -> AIProviderEntry {
    try AIProviderEntry(
        id: id,
        family: family,
        address: root,
        model: model,
        streamLineByteCap: streamLineByteCap
    )
}

/// A plain single-turn request — what every family can carry.
func makeFixtureRequest(
    _ prompt: String = "Reply with exactly: pong",
    systemPrompt: String? = nil,
    maxTokens: Int = 16,
    temperature: Double = 0
) -> AIRequest {
    AIRequest(
        messages: [AIMessage(role: .user, content: prompt)],
        systemPrompt: systemPrompt,
        temperature: temperature,
        maxTokens: maxTokens
    )
}

/// A reply context pointing at the fixture address.
func makeFixtureContext(
    family: AIProviderFamily,
    route: String,
    requestedModel: String = "requested-model",
    providerID: String = "lmstudio"
) throws -> WireReplyContext {
    WireReplyContext(
        target: AIRequestTarget(entryID: providerID, address: try makeFixtureAddress(), route: route),
        requestedModel: requestedModel,
        providerID: providerID,
        latencyMs: 7
    )
}

/// The JSON body of a request, as a dictionary.
func fixtureBodyObject(_ request: URLRequest) throws -> [String: Any] {
    let data = try #require(request.httpBody)
    return try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
}

/// Every chunk a codec reads out of a captured stream.
func fixtureChunks(_ fixture: WireFamilyFixture) throws -> [AIChunk] {
    let context = try makeFixtureContext(family: fixture.family, route: fixture.route)
    return try fixture.sseLines.compactMap { try fixture.codec.chunk(fromLine: $0, context: context) }
}
