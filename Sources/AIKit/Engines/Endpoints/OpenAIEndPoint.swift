import Foundation
import NetKit

/// Every call AIKit makes to an OpenAI-shaped API, one case per call.
///
/// The server, the bearer token and the way a body becomes JSON are written
/// once, on the enum; a case contributes only its path, its method and the
/// body it carries. Anthropic and LM Studio's native API are different APIs and
/// have their own enums (`AnthropicEndPoint`, `LMStudioEndPoint`).
enum OpenAIEndPoint: EndPoint {

    // MARK: - End points declaration

    /// `/v1/chat/completions`. Carries the typed body. The one place a
    /// body is rendered from an `AIRequest` is
    /// `ChatCompletionRequest(request:model:stream:)`; `OpenAIProvider` and the
    /// wire-standard dispatch both reach it through `AIWireStandard.endPoint`.
    case chat(server: AIWireServer, apiKey: String?, body: ChatCompletionRequest)
    /// `/v1/responses`: the prompt is `input`, the system prompt `instructions`.
    case responses(server: AIWireServer, apiKey: String?, model: String, request: AIRequest, stream: Bool)
    /// `/v1/models`.
    case models(server: AIWireServer, apiKey: String?)

    // MARK: - Properties

    private var server: AIWireServer {
        switch self {
        case .chat(let server, _, _), .responses(let server, _, _, _, _), .models(let server, _):
            server
        }
    }

    private var apiKey: String? {
        switch self {
        case .chat(_, let apiKey, _), .responses(_, let apiKey, _, _, _), .models(_, let apiKey):
            apiKey
        }
    }

    var scheme: String { server.scheme }
    var host: String { server.host }
    var port: Int? { server.port }
    var apiPath: String { "" }
    var apiFilePath: String { "" }

    var path: String {
        switch self {
        case .chat: AIWireStandard.openAIChat.path
        case .responses: AIWireStandard.openAIResponses.path
        case .models: "/v1/models"
        }
    }

    var method: RequestMethod {
        switch self {
        case .chat, .responses: .POST
        case .models: .GET
        }
    }

    var header: [String: String]? {
        var headers = ["Content-Type": "application/json"]
        if let apiKey, !apiKey.isEmpty {
            headers["Authorization"] = "Bearer \(apiKey)"
        }
        return headers
    }

    var queryParams: [String: Any]? { nil }

    var body: [String: Any]? {
        // Safety: whoever built the endpoint ran `bodyObject()`, which throws
        // before a request that cannot be encoded is ever sent.
        (try? bodyObject())
    }

    /// The request body as a JSON object; nil for a call that has none.
    /// - Throws: An encoding error when the body cannot be serialised. The
    ///   caller propagates it rather than sending a bodyless request.
    func bodyObject() throws -> [String: Any]? {
        switch self {
        case .chat(_, _, let body):
            return try encodeWireJSONObject(body, label: "ChatCompletionRequest")
        case .responses(_, _, let model, let request, let stream):
            let body = ResponsesRequest(
                model: model,
                input: request.messages.map { .init(role: $0.role.rawValue, content: $0.content) },
                // This standard's own field for a system prompt. It is not sent as
                // a pseudo-message: the reply echoes `instructions` back verbatim,
                // so the server reads it there.
                instructions: request.systemPrompt.flatMap { $0.isEmpty ? nil : $0 },
                maxOutputTokens: request.maxTokens,
                temperature: request.temperature,
                stream: stream
            )
            return try encodeWireJSONObject(body, label: AIWireStandard.openAIResponses.rawValue)
        case .models:
            return nil
        }
    }
}

// MARK: - Chat request

extension ChatCompletionRequest {
    /// Renders a typed request onto the chat-completions shape.
    init(request: AIRequest, model: String, stream: Bool) {
        var messages: [ChatCompletionRequest.ChatMessage] = []
        if let systemPrompt = request.systemPrompt, !systemPrompt.isEmpty {
            messages.append(.init(role: "system", content: systemPrompt))
        }
        for message in request.messages {
            let wireToolCalls: [ChatCompletionRequest.ToolCallRequest]? = message.toolCalls?.map { tc in
                .init(
                    id: tc.id ?? UUID().uuidString,
                    type: "function",
                    function: .init(name: tc.name, arguments: tc.arguments)
                )
            }
            let wireContent: String? = message.content.isEmpty ? nil : message.content
            messages.append(.init(
                role: message.role.rawValue,
                content: wireContent,
                toolCalls: wireToolCalls,
                toolCallId: message.toolCallId
            ))
        }

        let tools: [ChatCompletionRequest.ToolDefinition]? = request.tools?.map { tool in
            .init(
                type: "function",
                function: .init(
                    name: tool.name,
                    description: tool.description,
                    parameters: JSONFragment(rawJSON: tool.parametersSchema)
                )
            )
        }

        self.init(
            model: model,
            messages: messages,
            temperature: request.temperature,
            maxTokens: request.maxTokens,
            stream: stream,
            tools: tools,
            responseFormat: request.responseFormat.map { .init(type: $0 == .json ? "json_object" : "text") }
        )
    }
}

// MARK: - Chat reply

/// One reply from `/v1/chat/completions`, in either shape the surface answers with.
///
/// A completion is the normal shape. The other one is a 200 carrying an error
/// body, which an OpenAI-compatible server returns for a request it accepted
/// and then could not serve — a model name it has not loaded, a `/v1` left in
/// the server root. NetKit decodes a 2xx body straight into the type the caller
/// asks for, so asking for `ChatCompletionResponseDTO` alone turns that answer
/// into a `DecodingError` and loses the server's own words. BR-AIL-10 requires
/// those words to reach the caller, so the reply type admits both shapes and
/// `OpenAIProvider` decides which one is a failure.
enum ChatCompletionReply: Decodable, Sendable {
    /// The server answered with choices.
    case completion(ChatCompletionResponseDTO)
    /// The server answered 200 and named what went wrong.
    case serverError(message: String)

    /// Error bodies are read by the shared `wireErrorMessage`, which tests for
    /// an absent or null `error` explicitly and throws on a malformed one, so
    /// a decode failure is never mistaken for "no error, so a completion".
    init(from decoder: Decoder) throws {
        if let message = try wireErrorMessage(in: decoder) {
            self = .serverError(message: message)
            return
        }
        self = .completion(try ChatCompletionResponseDTO(from: decoder))
    }

    /// Reads this reply into the typed response.
    /// - Parameters:
    ///   - requestedModel: Used only when the reply names no model.
    ///   - requestURL: The URL the reply came from, for an error that names it.
    func response(
        requestedModel: String,
        providerID: String,
        latencyMs: Int,
        requestURL: String
    ) throws -> AIResponse {
        let completion: ChatCompletionResponseDTO
        switch self {
        case .completion(let value):
            completion = value
        case .serverError(let message):
            // BR-AIL-10: the server's own words, with the URL they came from.
            throw AIKitError.requestFailed("\(message) — from \(requestURL)")
        }

        guard let choice = completion.choices.first else {
            throw AIKitError.requestFailed(AIWireStandard.emptyReplyMessage(.openAIChat, requestURL: requestURL))
        }

        return AIResponse(
            content: choice.message.content ?? "",
            // The reply names the model that answered; `requestedModel` is only
            // the fallback, because this surface is measured to ignore an
            // unknown model and answer with whatever is loaded (BR-AIL-06).
            model: completion.model ?? requestedModel,
            tokensUsed: TokenUsage(
                prompt: completion.usage?.promptTokens,
                completion: completion.usage?.completionTokens
            ),
            latencyMs: latencyMs,
            toolCalls: choice.message.toolCalls?.map {
                AIToolCall(id: $0.id, name: $0.function.name, arguments: $0.function.arguments)
            },
            providerID: providerID
        )
    }
}

// MARK: - Responses reply

extension ResponsesReply {
    /// Reads this reply into the typed response.
    func response(
        requestedModel: String,
        providerID: String,
        latencyMs: Int,
        requestURL: String
    ) throws -> AIResponse {
        let response: ResponsesResponseDTO
        switch self {
        case .response(let value):
            response = value
        case .serverError(let message):
            // BR-AIL-10. Reached only when the error object actually decoded:
            // this envelope sends `"error": null` on success, and a null is not
            // an error body.
            throw AIKitError.requestFailed("\(message) — from \(requestURL)")
        }

        let text = response.text
        guard !text.isEmpty else {
            throw AIKitError.requestFailed(AIWireStandard.emptyReplyMessage(.openAIResponses, requestURL: requestURL))
        }

        return AIResponse(
            content: text,
            model: response.model ?? requestedModel,
            tokensUsed: TokenUsage(
                prompt: response.usage?.inputTokens,
                completion: response.usage?.outputTokens
            ),
            latencyMs: latencyMs,
            providerID: providerID
        )
    }
}
