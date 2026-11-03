import Foundation
import Logging
import NetKit

private let endpointLogger = Logger(label: "aikit.endpoints.chat-completion")

/// EndPoint for OpenAI-compatible /v1/chat/completions.
struct OpenAIChatCompletionEndPoint: EndPoint {
    let host: String
    let port: Int?
    let scheme: String
    let apiKey: String?
    let requestBody: ChatCompletionRequest
    /// JSON-encoded body, verified at init to be a JSON object.
    ///
    /// `Data` is `Sendable`; `[String: Any]` is not. The body is encoded once
    /// here so encoding failures reach the caller as a thrown error rather than
    /// silently producing a bodyless request.
    private let _bodyData: Data

    /// - Throws: An encoding error when `requestBody` cannot be serialised to
    ///   JSON. The caller propagates this rather than sending a bodyless request.
    init(
        host: String,
        port: Int?,
        scheme: String,
        apiKey: String?,
        requestBody: ChatCompletionRequest
    ) throws {
        self.host = host
        self.port = port
        self.scheme = scheme
        self.apiKey = apiKey
        self.requestBody = requestBody
        let data = try JSONEncoder().encode(requestBody)
        guard try JSONSerialization.jsonObject(with: data) is [String: Any] else {
            throw EncodingError.invalidValue(
                requestBody,
                .init(codingPath: [], debugDescription: "ChatCompletionRequest did not encode to a JSON object")
            )
        }
        self._bodyData = data
    }

    var apiPath: String { "" }
    var apiFilePath: String { "" }
    /// Read from the standard, not written again here. `OpenAIChatWireFamily`
    /// declares the same path through `AIWireStandard.openAIChat`, and a path
    /// spelled in two places is a path that drifts in one of them — so both
    /// readers share this constant and a test asserts the agreement.
    var path: String { AIWireStandard.openAIChat.path }
    var method: RequestMethod { .POST }

    var header: [String: String]? {
        var headers = ["Content-Type": "application/json"]
        if let apiKey, !apiKey.isEmpty {
            headers["Authorization"] = "Bearer \(apiKey)"
        }
        return headers
    }

    var queryParams: [String: Any]? { nil }

    var body: [String: Any]? {
        // Safety: _bodyData was verified to be a JSON object at init time.
        (try? JSONSerialization.jsonObject(with: _bodyData)) as? [String: Any]
    }
}

// MARK: - Reply

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
}

// MARK: - Wire family

/// `.openAIChat` as a declared standard.
///
/// The request and reply shapes are the ones `OpenAIProvider` already sends and
/// reads (`ChatCompletionRequest`, `ChatCompletionReply`): this family routes
/// them through the shared declaration shape so the standard can be *selected*
/// by `kind`, without a second encoder for a format AIKit already speaks.
enum OpenAIChatWireFamily: AIWireFamily {
    static var standard: AIWireStandard { .openAIChat }

    static func requestBody(for request: AIRequest, model: String, stream: Bool) -> ChatCompletionRequest {
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

        return ChatCompletionRequest(
            model: model,
            messages: messages,
            temperature: request.temperature,
            maxTokens: request.maxTokens,
            stream: stream,
            tools: tools,
            responseFormat: request.responseFormat.map { .init(type: $0 == .json ? "json_object" : "text") }
        )
    }

    static func response(
        from reply: ChatCompletionReply,
        requestedModel: String,
        providerID: String,
        latencyMs: Int,
        requestURL: String
    ) throws -> AIResponse {
        let completion: ChatCompletionResponseDTO
        switch reply {
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
                AIToolCall(name: $0.function.name, arguments: $0.function.arguments)
            },
            providerID: providerID
        )
    }
}
