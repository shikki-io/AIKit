import Foundation
import NetKit

// MARK: - WireCodecCapabilities

/// What a family's wire can actually express.
///
/// Declared as DATA, per family, so one shared check does the refusing
/// (``WireCodec/refuseWhatItCannotCarry(_:streaming:)``) and a family cannot
/// forget to perform it. This is the half that was missing: every codec knew
/// how to build its body and none of them said what it could not hold, so
/// fields were dropped in silence.
struct WireCodecCapabilities: Sendable, Equatable {

    /// Roles this wire can keep as roles. A role outside the set is refused —
    /// never flattened into `"role: text"`, which forged a transcript the
    /// model then read as content (AIKit#18, `AIWireStandard.swift:310-322`).
    let roles: Set<AIMessage.Role>
    /// Whether a system prompt has somewhere to go.
    let systemPrompt: Bool
    /// Whether tool definitions are carried onto this wire by this codec.
    let tools: Bool
    /// Response formats this codec sends. Empty means it sends none.
    let responseFormats: Set<ResponseFormat>
    /// Media input kinds this codec carries. Empty means it carries none.
    let inputKinds: Set<AIMediaKind>
    /// Whether this codec can read a streamed reply.
    let streaming: Bool
}

// MARK: - WireReplyContext

/// What a codec needs to turn bytes into an ``AIResponse``.
struct WireReplyContext: Sendable, Equatable {
    /// Where the request went, for an error that must name it.
    let target: AIRequestTarget
    /// The model that was ASKED for — used only when the reply names none.
    let requestedModel: String
    /// The id every answer carries (BR-AIL-06).
    let providerID: String
    /// Measured round trip.
    let latencyMs: Int
}

// MARK: - WireEndPoint

/// An endpoint whose body is built from a typed value, so that encoding it can
/// FAIL and be reported.
///
/// `NetKit.EndPoint.body` is a non-throwing `[String: Any]?`, so an endpoint
/// that cannot encode its body has only nil to return — and a nil body is
/// indistinguishable from a GET. A request that silently lost its body reads as
/// a server fault, which is why every body-carrying enum here exposes a
/// throwing builder and a codec runs it BEFORE the request is handed to a
/// transport.
protocol WireEndPoint: EndPoint {
    /// The request body as a JSON object; nil for a call that has none.
    /// - Throws: An encoding error when the body cannot be serialised.
    func bodyObject() throws -> [String: Any]?
}

// MARK: - WireCodec

/// Translates ``AIRequest`` onto one family's wire, and the reply back.
///
/// BR-AIL-17. A codec is the ONLY thing that knows a body shape, and it
/// declares what it supports, so a request field it cannot carry throws
/// ``AIKitError/unsupported(field:family:)`` before anything is sent. It knows
/// nothing about transports, retries, fallback or provenance — that is
/// ``HTTPAIProvider``'s half — and nothing about routes, which belong to the
/// family's `EndPoint` enum (BR-AIL-16).
protocol WireCodec: Sendable {

    /// The family this codec speaks for.
    var family: AIProviderFamily { get }

    /// What this codec's wire can express.
    var capabilities: WireCodecCapabilities { get }

    /// The wire body for one request.
    ///
    /// - Throws: ``AIKitError/unsupported(field:family:)`` for anything this
    ///   wire cannot hold. Nothing is sent when this throws.
    func body(
        for request: AIRequest,
        model: String,
        stream: Bool
    ) throws(AIKitError) -> any Encodable & Sendable

    /// The endpoint carrying that body, built from the family's route enum.
    ///
    /// - Throws: ``AIKitError/unsupported(field:family:)``, as `body` does.
    func endpoint(
        for request: AIRequest,
        entry: AIProviderEntry,
        credential: String?,
        stream: Bool
    ) throws(AIKitError) -> any EndPoint

    /// Reads a complete 2xx reply.
    ///
    /// - Throws: ``AIKitError/emptyReply(target:)`` when the reply carried no
    ///   text, ``AIKitError/requestFailed(_:)`` when a 200 carried an error
    ///   body (BR-AIL-10), ``AIKitError/replyUndecodable(target:reason:)``
    ///   when it did not decode.
    func response(from data: Data, context: WireReplyContext) throws(AIKitError) -> AIResponse

    /// Reads one line of a streamed reply.
    ///
    /// Returns nil for a line that is not payload: an SSE `event:` label, a
    /// keep-alive, a blank separator, or an event this family sends that
    /// carries no text (a prompt-processing progress tick).
    ///
    /// - Throws: ``AIKitError/replyUndecodable(target:reason:)`` for a `data:`
    ///   payload that is malformed, so a caller learns about it instead of
    ///   seeing a silent gap.
    func chunk(fromLine line: String, context: WireReplyContext) throws(AIKitError) -> AIChunk?
}

// MARK: - The shared refusal

extension WireCodec {

    /// Refuses everything this codec's wire cannot carry, before a byte is
    /// sent (BR-AIL-17).
    ///
    /// One implementation for every family: each family declares its
    /// ``capabilities`` and this reads them, so "declared it supports tools"
    /// and "actually sends tools" cannot drift apart. A codec calls this first
    /// in `body(for:model:stream:)`.
    ///
    /// - Throws: ``AIKitError/unsupported(field:family:)`` naming the field.
    func refuseWhatItCannotCarry(_ request: AIRequest, streaming: Bool) throws(AIKitError) {
        if streaming, !capabilities.streaming {
            throw .unsupported(field: "stream", family: family)
        }

        let systemPrompt = request.systemPrompt?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if !systemPrompt.isEmpty, !capabilities.systemPrompt {
            throw .unsupported(field: "systemPrompt", family: family)
        }

        if let tools = request.tools, !tools.isEmpty, !capabilities.tools {
            throw .unsupported(field: "tools", family: family)
        }

        if let format = request.responseFormat, !capabilities.responseFormats.contains(format) {
            throw .unsupported(field: "responseFormat=\(format.rawValue)", family: family)
        }

        for input in request.inputs where !capabilities.inputKinds.contains(input.kind) {
            throw .unsupported(field: "inputs[].kind=\(input.kind.rawValue)", family: family)
        }

        for message in request.messages {
            guard capabilities.roles.contains(message.role) else {
                throw .unsupported(field: "messages[].role=\(message.role.rawValue)", family: family)
            }
            // A tool call's id is the server's, and the tool result that
            // follows must echo it. Minting one with `UUID()` — which
            // `OpenAIEndPoint` did — hands the server an id it never issued,
            // so the loop it belongs to cannot be matched up. Refuse instead
            // (AIKit#18, `OpenAIEndPoint.swift:115`).
            for call in message.toolCalls ?? [] where (call.id ?? "").isEmpty {
                throw .unsupported(field: "messages[].toolCalls[].id is missing", family: family)
            }
            if message.role == .tool, (message.toolCallId ?? "").isEmpty {
                throw .unsupported(field: "messages[].toolCallId is missing", family: family)
            }
            if !message.images.isNilOrEmpty, !capabilities.inputKinds.contains(.image) {
                throw .unsupported(field: "messages[].images", family: family)
            }
        }
    }

    /// Returns `endpoint` once its body has been proved to encode.
    ///
    /// Called by every codec before it hands an endpoint back. Without it a
    /// malformed tool schema — a `parametersSchema` that is not JSON — would
    /// travel as `body == nil` and be POSTed with NO BODY AT ALL, and the
    /// server's complaint about a missing field would read as the model
    /// misbehaving.
    ///
    /// - Throws: ``AIKitError/requestFailed(_:)`` naming the family and the
    ///   cause. Nothing is sent.
    func proved(_ endpoint: some WireEndPoint) throws(AIKitError) -> any EndPoint {
        do {
            _ = try endpoint.bodyObject()
        } catch {
            throw .requestFailed(
                "The \(family.displayName) request body could not be encoded, so nothing was sent: \(error)"
            )
        }
        return endpoint
    }

    /// The text payload of an SSE line, or nil when the line is not one.
    ///
    /// Shared because all four families' streams are `data:`-prefixed: the
    /// `event:` label beside them repeats a `type` the payload already
    /// carries, so the payload alone is enough and nothing has to track which
    /// event it is inside.
    func ssePayload(of line: String) -> String? {
        let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.hasPrefix("data:") else { return nil }
        let payload = trimmed.dropFirst("data:".count).trimmingCharacters(in: .whitespaces)
        return payload.isEmpty ? nil : payload
    }

    /// Decodes one JSON value, reporting a failure against the target.
    func decode<T: Decodable>(
        _ type: T.Type,
        from data: Data,
        context: WireReplyContext
    ) throws(AIKitError) -> T {
        do {
            return try JSONDecoder().decode(type, from: data)
        } catch {
            throw .replyUndecodable(target: context.target, reason: "\(error)")
        }
    }
}

// MARK: - Helpers

extension Optional where Wrapped == [Data] {
    /// True when there is nothing there, either way of having nothing.
    var isNilOrEmpty: Bool { self?.isEmpty ?? true }
}
