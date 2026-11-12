import Foundation

// Request and reply shapes for the HTTP families other than chat completions,
// whose DTOs already live in `OpenAIDTOs.swift` and are reused as they are.
//
// Every reply shape here was read off a real reply, captured from the
// operator's LM Studio on 2026-10-31 and re-measured on 2026-11-12 with
// `mistralai_devstral-small-2-24b-instruct-2512-mlx` loaded. Every REQUEST
// shape is from the provider's own captured reference (cited on each family's
// `EndPoint` enum) and, where a field's behaviour mattered, confirmed against
// the live server.
//
// Fields the server sends but AIKit has no use for (`logprobs`, `annotations`,
// `system_fingerprint`, `runtime`, `model_info`) are left undeclared on
// purpose: `JSONDecoder` ignores unknown keys, and declaring them would be a
// second place to maintain a shape nothing reads.
//
// Optionality is not defensive here, it is measured. `usage` is absent from a
// streamed reply, `stop_reason` is null mid-turn, and `/v1/responses` sends
// `"error": null` on success — which is exactly why the reply envelopes below
// test for a *decodable error object* rather than for the key's presence.
//
// The types are flat rather than nested inside their DTO: a `CodingKeys` enum
// inside a nested struct is two levels deep, which this fleet's lint refuses,
// and the snake_case wire keys mean every one of these needs its own
// `CodingKeys`. Prefixed names carry the grouping instead.

// MARK: - Body encoding

/// Encodes a request body to a JSON object, once, so an encoding failure
/// reaches the caller as a thrown error rather than a bodyless request.
func encodeWireJSONObject<Body: Encodable>(_ body: Body, label: String) throws -> [String: Any] {
    let data = try JSONEncoder().encode(body)
    guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
        throw EncodingError.invalidValue(
            body,
            .init(codingPath: [], debugDescription: "Request body for \(label) did not encode to a JSON object")
        )
    }
    return object
}

// MARK: - Shared

/// The error body every one of these surfaces answers with, in both forms
/// seen in the wild: `{"error": {"message": "…"}}` and the bare
/// `{"error": "…"}`.
///
/// One decoder rather than four copies: the envelopes below differ in what
/// their SUCCESS shape is, not in how they say something went wrong.
struct WireErrorBody: Decodable, Sendable {
    let message: String

    private enum Key: String, CodingKey {
        case message
    }

    init(from decoder: Decoder) throws {
        // Shape discovery, nothing more: a bare string decodes as a string, an
        // object does not. Only that `typeMismatch` means "this is the keyed
        // form"; any other error (corrupt data, a missing value) is a real
        // failure and propagates with its key path instead of being swallowed.
        // If the keyed arm then fails too, its own `DecodingError` is what the
        // caller gets.
        do {
            let single = try decoder.singleValueContainer()
            message = try single.decode(String.self)
            return
        } catch DecodingError.typeMismatch {
            // Not a string: fall through to the keyed form.
        }
        let container = try decoder.container(keyedBy: Key.self)
        message = try container.decode(String.self, forKey: .message)
    }
}

/// The key a reply envelope looks for an error object under.
enum WireErrorEnvelopeKey: String, CodingKey {
    case error
}

/// Reads an error body out of a reply envelope, or nil when there is none.
///
/// Nil covers exactly two measured cases: no `error` key, and `"error": null`
/// — the trap `/v1/responses` sets on every SUCCESSFUL reply. Both are tested
/// explicitly. An `error` value that is present, non-null and undecodable is
/// NOT nil: it throws the `DecodingError` with its key path, so a malformed
/// error body reads as what it is instead of silently becoming a success.
func wireErrorMessage(in decoder: Decoder) throws -> String? {
    let container = try decoder.container(keyedBy: WireErrorEnvelopeKey.self)
    guard container.contains(.error), try !container.decodeNil(forKey: .error) else {
        return nil
    }
    return try container.decode(WireErrorBody.self, forKey: .error).message
}

// MARK: - OpenAI completions — /v1/completions

/// Request body for the legacy `POST /v1/completions`.
///
/// Declared because the API documents the route (BR-AIL-16). No AIKit caller
/// builds one today; `AIRequest` is a conversation and this surface takes a
/// bare prompt, so a codec for it would be a lossy mapping nothing has asked
/// for.
struct CompletionsRequest: Encodable, Sendable, Equatable {
    let model: String
    let prompt: String
    let maxTokens: Int?
    let temperature: Double?
    let stream: Bool?

    enum CodingKeys: String, CodingKey {
        case model, prompt, temperature, stream
        case maxTokens = "max_tokens"
    }
}

// MARK: - OpenAI embeddings — /v1/embeddings

/// Request body for `POST /v1/embeddings`. Declared for the reason above.
struct EmbeddingsRequest: Encodable, Sendable, Equatable {
    let model: String
    let input: [String]
}

// MARK: - OpenAI responses — /v1/responses

/// Request body for `/v1/responses`.
///
/// `input` takes the role array (measured: HTTP 200, and `input_tokens` rises
/// with the system text, so the server reads it). The system prompt goes in
/// `instructions`, which is this surface's own field for it rather than a
/// pseudo-message — the reply echoes it back verbatim.
struct ResponsesRequest: Encodable, Sendable {
    let model: String
    let input: [ResponsesInputMessage]
    let instructions: String?
    let maxOutputTokens: Int
    let temperature: Double
    let stream: Bool

    enum CodingKeys: String, CodingKey {
        case model, input, instructions, temperature, stream
        case maxOutputTokens = "max_output_tokens"
    }
}

/// One turn in a `/v1/responses` request.
struct ResponsesInputMessage: Encodable, Sendable, Equatable {
    let role: String
    let content: String
}

/// One reply from `/v1/responses`, in either shape the surface answers with.
enum ResponsesReply: Decodable, Sendable {
    case response(ResponsesResponseDTO)
    case serverError(message: String)

    init(from decoder: Decoder) throws {
        if let message = try wireErrorMessage(in: decoder) {
            self = .serverError(message: message)
            return
        }
        self = .response(try ResponsesResponseDTO(from: decoder))
    }
}

/// A successful `/v1/responses` reply.
struct ResponsesResponseDTO: Decodable, Sendable {
    let id: String
    /// `completed`, or `incomplete` when the token budget cut the answer off.
    let status: String?
    /// The model that actually answered. Measured: this surface ignores an
    /// unknown `model` in the request and answers with whatever is loaded, so
    /// the reply's own value is the only honest provenance (BR-AIL-06).
    let model: String?
    let output: [ResponsesOutputItem]
    let usage: ResponsesUsage?

    /// The assistant's text, concatenated across every `output_text` part.
    ///
    /// Only `output_text` parts are read: a reasoning summary or a tool call
    /// arrives as a different `type` and is not the answer. Tool loops for
    /// local models are out of scope for this spec (#1859), so a tool-call part
    /// is skipped rather than half-mapped.
    var text: String {
        output
            .filter { $0.type == "message" }
            .flatMap { $0.content ?? [] }
            .filter { $0.type == "output_text" }
            .compactMap(\.text)
            .joined()
    }
}

/// One item in a `/v1/responses` reply's `output` array.
struct ResponsesOutputItem: Decodable, Sendable {
    let type: String
    let role: String?
    let content: [ResponsesContentPart]?
}

/// One content part inside a `/v1/responses` output item.
struct ResponsesContentPart: Decodable, Sendable {
    let type: String
    let text: String?
}

/// Token usage on `/v1/responses`.
struct ResponsesUsage: Decodable, Sendable {
    let inputTokens: Int?
    let outputTokens: Int?

    enum CodingKeys: String, CodingKey {
        case inputTokens = "input_tokens"
        case outputTokens = "output_tokens"
    }
}

/// One SSE event from a streamed `/v1/responses`.
///
/// Captured 2026-11-12. The stream is a long sequence of typed events
/// (`response.created`, `response.in_progress`,
/// `response.output_item.added`, `response.content_part.added`, …); the two
/// that carry the answer are the ones read here:
///
/// ```
/// event: response.output_text.delta
/// data: {"type":"response.output_text.delta","delta":"p",…}
/// event: response.completed
/// data: {"type":"response.completed","response":{…}}
/// ```
///
/// The `event:` label repeats the payload's own `type`, so the payload alone is
/// enough and nothing has to track which event it is inside.
struct ResponsesStreamEvent: Decodable, Sendable {
    let type: String
    /// Present on `response.output_text.delta`, and only there.
    let delta: String?

    static let textDelta = "response.output_text.delta"
    static let completed = "response.completed"
}

// MARK: - Anthropic messages — /v1/messages

/// Request body for `/v1/messages`.
///
/// Anthropic puts the system prompt in a top-level `system` field rather than
/// in the message array. Measured against LM Studio: sending it there changed
/// the answer and raised `input_tokens` from 4 to 18, so the field is read and
/// not ignored.
struct AnthropicMessagesRequest: Encodable, Sendable {
    let model: String
    let messages: [AnthropicMessage]
    let system: String?
    let maxTokens: Int
    let temperature: Double
    let stream: Bool

    enum CodingKeys: String, CodingKey {
        case model, messages, system, temperature, stream
        case maxTokens = "max_tokens"
    }
}

/// Request body for `/v1/messages/count_tokens`.
///
/// The same conversation as a message request, minus everything about
/// generating: the route counts what would be sent. Declared because the API
/// documents it (BR-AIL-16); no AIKit caller builds one today.
struct AnthropicCountTokensRequest: Encodable, Sendable {
    let model: String
    let messages: [AnthropicMessage]
    let system: String?
}

/// One turn in a `/v1/messages` request.
struct AnthropicMessage: Encodable, Sendable, Equatable {
    let role: String
    let content: String
}

/// One reply from `/v1/messages`, in either shape the surface answers with.
///
/// Anthropic's error envelope is `{"type": "error", "error": {…}}`, so the
/// discriminator is the nested error object just as it is for the others.
enum AnthropicMessagesReply: Decodable, Sendable {
    case message(AnthropicMessagesResponseDTO)
    case serverError(message: String)

    init(from decoder: Decoder) throws {
        if let message = try wireErrorMessage(in: decoder) {
            self = .serverError(message: message)
            return
        }
        self = .message(try AnthropicMessagesResponseDTO(from: decoder))
    }
}

/// A successful `/v1/messages` reply.
struct AnthropicMessagesResponseDTO: Decodable, Sendable {
    let id: String
    let role: String?
    /// The model that answered; see `ResponsesResponseDTO.model`.
    let model: String?
    let content: [AnthropicContentBlock]
    /// `end_turn`, or `max_tokens` when the budget cut the answer off.
    let stopReason: String?
    let usage: AnthropicUsage?

    enum CodingKeys: String, CodingKey {
        case id, role, model, content, usage
        case stopReason = "stop_reason"
    }

    /// The assistant's text, concatenated across every `text` block.
    /// A `tool_use` block is skipped for the reason given on
    /// `ResponsesResponseDTO.text`.
    var text: String {
        content.filter { $0.type == "text" }.compactMap(\.text).joined()
    }
}

/// One block in a `/v1/messages` reply's `content` array.
struct AnthropicContentBlock: Decodable, Sendable {
    let type: String
    let text: String?
}

/// Token usage on `/v1/messages`.
struct AnthropicUsage: Decodable, Sendable {
    let inputTokens: Int?
    let outputTokens: Int?

    enum CodingKeys: String, CodingKey {
        case inputTokens = "input_tokens"
        case outputTokens = "output_tokens"
    }
}

/// One SSE event from a streamed `/v1/messages`.
///
/// Captured 2026-11-12:
///
/// ```
/// event: content_block_delta
/// data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"p"}}
/// event: message_stop
/// data: {"type":"message_stop"}
/// ```
///
/// `message_start`, `content_block_start`, `content_block_stop` and
/// `message_delta` also arrive and carry no answer text.
struct AnthropicStreamEvent: Decodable, Sendable {
    let type: String
    let delta: AnthropicStreamDelta?

    static let contentBlockDelta = "content_block_delta"
    static let messageStop = "message_stop"
}

/// The `delta` of a `content_block_delta`. `text` is nil for a delta that is
/// not a `text_delta` — a `thinking_delta` or an `input_json_delta`.
struct AnthropicStreamDelta: Decodable, Sendable {
    let type: String?
    let text: String?
}

// MARK: - LM Studio native — /api/v1/chat

/// Request body for `/api/v1/chat`.
///
/// Documented and measured; see ``LMStudioEndPoint`` for the capture and the
/// three request shapes the server refuses. In short: `input` is a typed-part
/// array with no roles, the system prompt is the top-level `system_prompt`, and
/// the budget field is `max_output_tokens` (`max_tokens` is a 400).
///
/// Every field but `input` and `model` is optional and omitted when nil,
/// because an unrecognised key is a 400 on this surface rather than something
/// the server ignores.
struct LMStudioNativeRequest: Encodable, Sendable {
    let model: String
    let input: [LMStudioInputPart]
    let systemPrompt: String?
    let temperature: Double?
    let maxOutputTokens: Int?
    let stream: Bool?

    enum CodingKeys: String, CodingKey {
        case model, input, temperature, stream
        case systemPrompt = "system_prompt"
        case maxOutputTokens = "max_output_tokens"
    }
}

/// One part of an `/api/v1/chat` `input` array.
///
/// The discriminator is `type`, and its value decides which other key is
/// required: `text` wants `content`, `image` wants `data_url`. Measured —
/// sending `{"type":"text","text":…}` answers `'input.0.content' is required,
/// Unrecognized key(s) in object: 'text'`.
struct LMStudioInputPart: Encodable, Sendable, Equatable {
    let type: String
    let content: String?
    let dataURL: String?

    enum CodingKeys: String, CodingKey {
        case type, content
        case dataURL = "data_url"
    }

    /// A text part, the only kind AIKit sends today.
    static func text(_ content: String) -> LMStudioInputPart {
        LMStudioInputPart(type: "text", content: content, dataURL: nil)
    }
}

/// Request body for `POST /api/v1/models/load`.
///
/// Declared because the API documents the route (BR-AIL-16).
struct LMStudioModelLoadRequest: Encodable, Sendable, Equatable {
    let model: String
    /// Context length to load with, when the caller wants one. Omitted
    /// otherwise, so the host's own default applies.
    let contextLength: Int?
    /// Seconds of idleness after which the host may evict the model.
    let ttl: Int?

    enum CodingKeys: String, CodingKey {
        case model, ttl
        case contextLength = "context_length"
    }
}

/// Request body for `POST /api/v1/models/unload`.
struct LMStudioModelUnloadRequest: Encodable, Sendable, Equatable {
    let model: String
}

/// Request body for `POST /api/v1/models/download`.
struct LMStudioModelDownloadRequest: Encodable, Sendable, Equatable {
    let model: String
}

/// One reply from `/api/v1/chat`, in either shape the surface answers with.
///
/// This is the one family measured to *validate* the request: an unknown model
/// is HTTP 404 with an error body, where the three others silently answer with
/// whatever is loaded.
enum LMStudioNativeReply: Decodable, Sendable {
    case response(LMStudioNativeResponseDTO)
    case serverError(message: String)

    init(from decoder: Decoder) throws {
        if let message = try wireErrorMessage(in: decoder) {
            self = .serverError(message: message)
            return
        }
        self = .response(try LMStudioNativeResponseDTO(from: decoder))
    }
}

/// A successful `/api/v1/chat` reply.
///
/// Close to `/v1/responses` and deliberately not the same type: `output[]`
/// carries `content` as a plain string here, where the responses surface nests
/// `content[].text`. Folding the two would mean one optional-riddled shape that
/// matches neither server.
struct LMStudioNativeResponseDTO: Decodable, Sendable {
    let responseId: String?
    /// The loaded model instance that answered — this surface's `model` field.
    let modelInstanceId: String?
    let output: [LMStudioNativeOutputItem]
    let stats: LMStudioNativeStats?

    enum CodingKeys: String, CodingKey {
        case output, stats
        case responseId = "response_id"
        case modelInstanceId = "model_instance_id"
    }

    /// The assistant's text, concatenated across every `message` item.
    var text: String {
        output.filter { $0.type == "message" }.compactMap(\.content).joined()
    }
}

/// One item in an `/api/v1/chat` reply's `output` array.
struct LMStudioNativeOutputItem: Decodable, Sendable {
    let type: String
    let content: String?
}

/// Token counts on `/api/v1/chat`, which calls them stats.
struct LMStudioNativeStats: Decodable, Sendable {
    let inputTokens: Int?
    let totalOutputTokens: Int?

    enum CodingKeys: String, CodingKey {
        case inputTokens = "input_tokens"
        case totalOutputTokens = "total_output_tokens"
    }
}

/// One SSE event from a streamed `/api/v1/chat`.
///
/// Captured 2026-11-12. This family streams its own lifecycle as well as the
/// answer — `chat.start`, `prompt_processing.start`,
/// `prompt_processing.progress`, `message.start`, `message.delta`,
/// `message.end`, `chat.end` — and only the two read here carry an answer:
///
/// ```
/// event: message.delta
/// data: {"type":"message.delta","content":"p"}
/// event: chat.end
/// data: {"type":"chat.end","result":{…}}
/// ```
struct LMStudioStreamEvent: Decodable, Sendable {
    let type: String
    /// Present on `message.delta`, and only there.
    let content: String?

    static let messageDelta = "message.delta"
    static let chatEnd = "chat.end"
}
