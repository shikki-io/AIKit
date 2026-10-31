import Foundation

// Request and reply shapes for the wire standards other than chat completions,
// whose DTOs already live in `OpenAIDTOs.swift` and are reused as they are.
//
// Every shape here was read off a real reply, captured from the operator's LM
// Studio on 2026-10-31 with
// `mistralai_devstral-small-2-24b-instruct-2512-mlx` loaded. Fields the server
// sends but AIKit has no use for (`logprobs`, `annotations`,
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

// MARK: - Shared

/// The error body every one of these three surfaces answers with, in both forms
/// seen in the wild: `{"error": {"message": "…"}}` and the bare
/// `{"error": "…"}`.
///
/// One decoder rather than three copies: the three envelopes below differ in
/// what their SUCCESS shape is, not in how they say something went wrong.
struct WireErrorBody: Decodable, Sendable {
    let message: String

    private enum Key: String, CodingKey {
        case message
    }

    init(from decoder: Decoder) throws {
        if let single = try? decoder.singleValueContainer(), let text = try? single.decode(String.self) {
            message = text
            return
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
/// Nil covers both "no `error` key" and "`error` is null" — the second being
/// the trap `/v1/responses` sets on every SUCCESSFUL reply. Testing for the
/// key's presence would turn every good answer into a failure; testing whether
/// an error *object* decodes is what distinguishes them.
func wireErrorMessage(in decoder: Decoder) -> String? {
    guard let container = try? decoder.container(keyedBy: WireErrorEnvelopeKey.self),
        let body = try? container.decode(WireErrorBody.self, forKey: .error)
    else { return nil }
    return body.message
}

// MARK: - OpenAI responses — /v1/responses

/// Request body for `/v1/responses`.
///
/// `input` takes the role array (measured: HTTP 200, and `input_tokens` rises
/// with the system text, so the server reads it). The system prompt goes in
/// `instructions`, which is this standard's own field for it rather than a
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
struct ResponsesInputMessage: Encodable, Sendable {
    let role: String
    let content: String
}

/// One reply from `/v1/responses`, in either shape the surface answers with.
enum ResponsesReply: Decodable, Sendable {
    case response(ResponsesResponseDTO)
    case serverError(message: String)

    init(from decoder: Decoder) throws {
        if let message = wireErrorMessage(in: decoder) {
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

/// One turn in a `/v1/messages` request.
struct AnthropicMessage: Encodable, Sendable {
    let role: String
    let content: String
}

/// One reply from `/v1/messages`, in either shape the surface answers with.
///
/// Anthropic's error envelope is `{"type": "error", "error": {…}}`, so the
/// discriminator is the nested error object just as it is for the other two.
enum AnthropicMessagesReply: Decodable, Sendable {
    case message(AnthropicMessagesResponseDTO)
    case serverError(message: String)

    init(from decoder: Decoder) throws {
        if let message = wireErrorMessage(in: decoder) {
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

// MARK: - LM Studio native — /api/v1/chat

/// Request body for `/api/v1/chat`.
///
/// Measured, and narrower than it looks:
/// - `input` is a prompt string. A role array is rejected outright
///   (`Invalid discriminator value. Expected 'text' | 'image'`), so this
///   standard has no conversation shape at this version and the prompt is
///   flattened by `AIWirePromptFlattening`.
/// - `max_tokens` is **rejected** (`Unrecognized key(s) in object:
///   'max_tokens'`); the budget field is `max_output_tokens`.
/// - `temperature` and `stream` are accepted.
///
/// Every field but `input` and `model` is therefore optional and omitted when
/// nil, because an unrecognised key is a 400 on this surface rather than
/// something the server ignores.
struct LMStudioNativeRequest: Encodable, Sendable {
    let model: String
    let input: String
    let temperature: Double?
    let maxOutputTokens: Int?
    let stream: Bool?

    enum CodingKeys: String, CodingKey {
        case model, input, temperature, stream
        case maxOutputTokens = "max_output_tokens"
    }
}

/// One reply from `/api/v1/chat`, in either shape the surface answers with.
///
/// This is the one surface measured to *validate* the request: an unknown model
/// is HTTP 404 with an error body, where the three others silently answer with
/// whatever is loaded. NetKit turns a non-2xx into `NetworkError` before a body
/// is decoded, so the error case here covers the 200-with-an-error-body shape
/// the others share.
enum LMStudioNativeReply: Decodable, Sendable {
    case response(LMStudioNativeResponseDTO)
    case serverError(message: String)

    init(from decoder: Decoder) throws {
        if let message = wireErrorMessage(in: decoder) {
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
    /// The loaded model instance that answered — this standard's `model` field.
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
