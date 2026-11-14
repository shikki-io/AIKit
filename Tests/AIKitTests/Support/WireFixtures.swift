import Foundation
import Testing
@testable import AIKit

// `URLRequest` lives in FoundationNetworking on a swift-corelibs platform
// (W30 of spec ai-layer-aikit-core-2026-10-31).
#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

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

/// Where a test pretends `mlx_lm.server` listens.
///
/// A fixture constant and not an environment read: `MLXEngine` takes its
/// address as an argument since W22 (BR-AIL-20), so a unit test states one
/// instead of depending on what is set on this machine.
let fixtureMLXServerRoot = "http://127.0.0.1:8080"

func makeFixtureMLXAddress() throws -> AIServerAddress {
    try AIServerAddress(validating: fixtureMLXServerRoot)
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

// MARK: - Opaque JSON values
//
// A JSON *value* a typed model carries as a `String` BY DESIGN. AIKit does not
// model a tool's JSON Schema or a tool call's arguments — they belong to the
// tool, not to the wire standard — so there is no DTO to mint them from and no
// encoder that could produce them. They are spelled once here instead of in six
// suites, so "what does a tool schema look like" has one answer.
//
// W24 of spec ai-layer-aikit-core-2026-10-31, declared as `.opaqueValue` in
// `DeclaredEvidence`.

/// A minimal JSON Schema for a tool that takes an object.
let fixtureToolSchema = #"{"type":"object"}"#

/// A schema with a declared property, for a test that needs the nesting.
let fixtureToolSchemaWithProperties = #"{"type":"object","properties":{"city":{"type":"string"}}}"#

/// The empty JSON object, `{}`.
///
/// One constant for the three places it means something: a tool that declares
/// no parameters, a tool call with no arguments, and the body a route that
/// merely ACCEPTS a call answers with (LM Studio's `models/load` and
/// `models/unload`). They are the same two bytes and the same absence of a
/// payload, so three identically-valued constants would be three names for one
/// fact — and `{}` IS a JSON object, which is why it is declared here at all
/// rather than written inline.
let fixtureEmptyJSONObject = "{}"

/// A tool call's `arguments`, the one-field form.
let fixtureToolArguments = #"{"q":"swift"}"#

/// A tool call's `arguments`, a different field, so a test asserting the string
/// arrived unchanged is not asserting against the same bytes as its neighbour.
let fixtureToolArgumentsCity = #"{"city":"Paris"}"#

// MARK: - Fixture addresses, beyond the one server root
//
// AIKit#18 r4500587739: *"I see too many `http://127.0.0.1:1234` in all the
// tests we did not have a SSoT for that? Instead of hardcoded the url of my
// lms?"* — so the operator's LM Studio is spelled ONCE, in
// ``fixtureServerRoot``, and the ratchet in
// `NoLocalhostFallbackRatchetTests` fails any other file in `Tests/` that
// writes it. These are the OTHER servers a test needs, each a different
// service rather than another spelling of the same one.

/// `127.0.0.1:1234` — the host and port of ``fixtureServerRoot``, for an
/// assertion against an error message that quotes the server but not a scheme.
let fixtureServerHostPort = "127.0.0.1:1234"

/// Where a test pretends an MCP server listens. A different service from the
/// AI provider, hence its own root.
let fixtureMCPServerRoot = "http://127.0.0.1:9999"

/// A port nothing listens on, for a health check that must report a server as
/// down. Deliberately high and unregistered.
let fixtureUnreachableServerRoot = "http://127.0.0.1:59999"

// MARK: - Minted replies
//
// WHY MINTING. The operator asked the same question on nine payloads in one
// review (AIKit#18 r4500530906, r4500580810, r4500583527, r4500589635,
// r4500592040, r4500592399, r4500592834, r4500593486, r4500599462): *"it's
// normal it doesn't use typed model with mock data for this?"*
//
// A hand-written payload and the DTO that decodes it are two independent
// statements of one wire shape. When the shape moves, only the DTO gets
// edited: the payload keeps its old key, `JSONDecoder` ignores it, the field
// decodes to nil, the optional swallows it, and the test goes on passing. A
// MINTED payload has ONE statement — the DTO — so the same move is a compile
// error in the fixture instead of a green test.
//
// The fake values below are deliberately NOT the captured ones. When an
// assertion fails, which kind of fixture it was reading is the first thing
// worth knowing, and "minted pong" vs "pong" says it without a stack trace.

/// The answer every minted reply carries.
let fakeReplyText = "minted pong"

/// The model every minted reply says answered. Not ``capturedModel``.
let fakeReplyModel = "fixture-model-7b-minted"

let fakeReplyPromptTokens = 11
let fakeReplyCompletionTokens = 5

/// An id a minted reply carries, so two minted replies are told apart.
let fakeReplyID = "minted-reply-1"

/// Encodes `value` to the JSON text a fixture hands a decoder.
///
/// `sortedKeys` so a minted payload is byte-stable: a test that compares two
/// minted bodies, or that prints one on failure, must not depend on
/// `JSONEncoder`'s dictionary order.
func mintJSONText(_ value: some Encodable) throws -> String {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    return try #require(String(data: try encoder.encode(value), encoding: .utf8))
}

/// A complete 2xx reply for `family`, built from that family's own reply DTO.
///
/// `model: nil` omits the field, which is how a reply that names no model is
/// stated — the shape three of the four surfaces answer with when they are
/// minimal. `promptTokens: nil` likewise omits `usage` entirely, because a
/// reply that claims no usage must not read as zero tokens (a zero silently
/// under-charges a budget and passes a limit check).
func mintReply(
    family: AIProviderFamily,
    text: String = fakeReplyText,
    model: String? = fakeReplyModel,
    promptTokens: Int? = fakeReplyPromptTokens,
    completionTokens: Int? = fakeReplyCompletionTokens
) throws -> String {
    switch family {
    case .openAIChat:
        return try mintJSONText(
            ChatCompletionResponseDTO(
                id: fakeReplyID,
                model: model,
                choices: [
                    ChatCompletionResponseDTO.Choice(
                        index: 0,
                        message: ChatCompletionResponseDTO.Message(role: "assistant", content: text, toolCalls: nil),
                        finishReason: "stop"
                    ),
                ],
                usage: mintChatUsage(promptTokens, completionTokens)
            )
        )

    case .openAIResponses:
        return try mintJSONText(
            ResponsesResponseDTO(
                id: fakeReplyID,
                status: "completed",
                model: model,
                output: [
                    ResponsesOutputItem(
                        type: "message",
                        role: "assistant",
                        content: [ResponsesContentPart(type: "output_text", text: text)]
                    ),
                ],
                usage: promptTokens == nil && completionTokens == nil
                    ? nil
                    : ResponsesUsage(inputTokens: promptTokens, outputTokens: completionTokens)
            )
        )

    case .anthropicMessages:
        return try mintJSONText(
            AnthropicMessagesResponseDTO(
                id: fakeReplyID,
                role: "assistant",
                model: model,
                content: [AnthropicContentBlock(type: "text", text: text)],
                stopReason: "end_turn",
                usage: promptTokens == nil && completionTokens == nil
                    ? nil
                    : AnthropicUsage(inputTokens: promptTokens, outputTokens: completionTokens)
            )
        )

    case .lmStudioNative:
        return try mintJSONText(
            LMStudioNativeResponseDTO(
                responseId: fakeReplyID,
                modelInstanceId: model,
                output: [LMStudioNativeOutputItem(type: "message", content: text)],
                stats: promptTokens == nil && completionTokens == nil
                    ? nil
                    : LMStudioNativeStats(inputTokens: promptTokens, totalOutputTokens: completionTokens)
            )
        )

    case .cliSubprocess, .mlxInProcess, .coreML:
        throw AIKitError.unsupported(field: "a wire reply", family: family)
    }
}

/// `usage` for a chat completion, or nil when neither side is known.
///
/// `total` is only claimed when both sides are: the same rule `TokenUsage`
/// applies, stated here so a minted reply cannot carry an arithmetic that the
/// real server would not have sent.
private func mintChatUsage(_ prompt: Int?, _ completion: Int?) -> ChatCompletionResponseDTO.Usage? {
    guard prompt != nil || completion != nil else { return nil }
    let total = prompt.flatMap { p in completion.map { p + $0 } }
    return ChatCompletionResponseDTO.Usage(promptTokens: prompt, completionTokens: completion, totalTokens: total)
}

/// A 2xx reply that carries no answer — the empty-reply case every codec must
/// report as ``AIKitError/emptyReply(target:)`` rather than as a decode
/// failure. The envelope is well-formed; only the content array is empty.
func mintEmptyReply(family: AIProviderFamily) throws -> String {
    switch family {
    case .openAIChat:
        return try mintJSONText(
            ChatCompletionResponseDTO(id: fakeReplyID, model: fakeReplyModel, choices: [], usage: nil)
        )
    case .openAIResponses:
        return try mintJSONText(
            ResponsesResponseDTO(
                id: fakeReplyID, status: "completed", model: fakeReplyModel, output: [], usage: nil)
        )
    case .anthropicMessages:
        return try mintJSONText(
            AnthropicMessagesResponseDTO(
                id: fakeReplyID, role: "assistant", model: fakeReplyModel,
                content: [], stopReason: "end_turn", usage: nil)
        )
    case .lmStudioNative:
        return try mintJSONText(
            LMStudioNativeResponseDTO(
                responseId: fakeReplyID, modelInstanceId: fakeReplyModel, output: [], stats: nil)
        )
    case .cliSubprocess, .mlxInProcess, .coreML:
        throw AIKitError.unsupported(field: "a wire reply", family: family)
    }
}

/// A models-list reply, as `/v1/models` answers it.
func mintModelsList(_ models: [(id: String, ownedBy: String?)]) throws -> String {
    try mintJSONText(
        ModelsListResponseDTO(
            data: models.map {
                ModelsListResponseDTO.ModelInfo(id: $0.id, ownedBy: $0.ownedBy, created: nil)
            }
        )
    )
}

// MARK: - Minted error envelopes

/// The error body a wire surface answers with, in the nested form
/// `{"error": {"message": "…"}}`.
///
/// This is the ONE minting helper with no production counterpart: AIKit reads
/// an error body and never writes one, so `WireErrorBody` is `Decodable` only.
/// `FixturePinTests.aMintedErrorEnvelopeIsReadByTheRealDecoder` is what keeps
/// the two honest — the production decoder is asserted to read what this
/// writes, in both forms the servers send (BR-EC-02).
struct FixtureWireError: Encodable, Sendable, Equatable {
    struct Body: Encodable, Sendable, Equatable {
        let message: String
        let type: String?
    }

    let error: Body
}

/// The bare form, `{"error": "…"}`, which LM Studio also answers with.
struct FixtureBareWireError: Encodable, Sendable, Equatable {
    let error: String
}

/// `{"error": {"message": message, "type": type}}`.
func mintErrorReply(_ message: String, type: String? = "server_error") throws -> String {
    try mintJSONText(FixtureWireError(error: FixtureWireError.Body(message: message, type: type)))
}

/// `{"error": message}` — the bare-string form.
func mintBareErrorReply(_ message: String) throws -> String {
    try mintJSONText(FixtureBareWireError(error: message))
}

// MARK: - Minted Hugging Face replies
//
// `HFModelInfo` is already `Codable`, so the search and model-detail payloads
// the registry suites read are minted from it directly. The wire keys that are
// NOT the property names — `pipeline_tag`, `library_name` — are the reason
// these fixtures were worth converting: a hand-written payload spelled them
// once in the test and once in the DTO's `CodingKeys`, and only the second got
// maintained.

/// One Hugging Face model row, with the fields the suites assert on.
func makeFixtureHFModel(
    modelId: String = "nvidia/nemotron-3-nano-4b",
    author: String? = "nvidia",
    tags: [String] = ["gguf", "text-generation"],
    downloads: Int = 50_000,
    likes: Int = 1_200,
    lastModified: String? = "2025-11-20T10:30:00.000Z",
    pipelineTag: String? = "text-generation",
    library: String? = "transformers"
) -> HFModelInfo {
    HFModelInfo(
        modelId: modelId,
        author: author,
        tags: tags,
        downloads: downloads,
        likes: likes,
        lastModified: lastModified,
        pipelineTag: pipelineTag,
        library: library
    )
}

/// A `GET /api/models` search reply: a JSON array of rows.
func mintHFSearchReply(_ models: [HFModelInfo]) throws -> Data {
    Data(try mintJSONText(models).utf8)
}

/// A `GET /api/models/<id>` reply: one row.
func mintHFModelReply(_ model: HFModelInfo) throws -> Data {
    Data(try mintJSONText(model).utf8)
}

/// LM Studio's native `GET /api/v1/models` reply.
func mintLMStudioHostedModels(_ models: [LMStudioHostedModel]) throws -> String {
    try mintJSONText(LMStudioModelsReply(data: models))
}
