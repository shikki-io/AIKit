import Foundation
import NetKit

/// AIProvider that talks to any OpenAI-compatible API.
///
/// One provider serves every server that speaks `/v1/chat/completions`: OpenAI
/// itself, LM Studio, Ollama, any proxy in front of them. `LMStudioProvider`
/// is a thin `AgentProviding` adapter over this type rather than a second
/// implementation of the same wire format.
public struct OpenAIProvider: AIProvider, Sendable {
    public let id: String
    public let displayName: String
    public let capabilities: AICapabilities

    let baseURL: String
    let modelName: String
    let apiKey: String?
    let networkService: any NetworkProtocol

    /// The message `complete` throws when a 200 reply carries no choices.
    ///
    /// `LMStudioProvider` matches on this exact value to keep its own
    /// `.emptyContent` case, so thrower and matcher read one constant instead
    /// of two copies of a sentence that could drift apart.
    static let emptyReplyMessage = "Empty response — no choices returned"

    /// Create a provider for one model on one OpenAI-compatible server.
    /// - Parameters:
    ///   - id: The `AIProvider.id` this provider reports, and the value every
    ///     answer carries in `AIResponse.providerID`.
    ///   - displayName: Human-readable name.
    ///   - capabilities: What this model can do, as the caller knows it.
    ///   - baseURL: Server root, for example `http://127.0.0.1:1234`. The
    ///     endpoint owns the path (BR-AIL-09).
    ///   - modelName: Model name as the server knows it.
    ///   - apiKey: Bearer token, when the server wants one.
    ///   - networkService: The transport. Inject a mock in tests (BR-AIL-12).
    public init(
        id: String,
        displayName: String,
        capabilities: AICapabilities,
        baseURL: String,
        modelName: String,
        apiKey: String? = nil,
        networkService: any NetworkProtocol
    ) {
        self.id = id
        self.displayName = displayName
        self.capabilities = capabilities
        self.baseURL = baseURL
        self.modelName = modelName
        self.apiKey = apiKey
        self.networkService = networkService
    }

    public var status: AIProviderStatus {
        get async { .ready }
    }

    public func complete(request: AIRequest) async throws -> AIResponse {
        let startTime = ContinuousClock.now

        let chatRequest = buildChatRequest(from: request, stream: false)
        let endpoint = makeEndpoint(for: chatRequest)

        // A transport failure travels as the type it was thrown as — `URLError`,
        // `NetKit.NetworkError` — because a caller deciding whether to fall back
        // needs to tell a refused connection from a server that answered
        // (BR-AIL-07, W5). Flattening it into a string here would erase that.
        let reply: ChatCompletionReply = try await networkService.sendRequest(endpoint: endpoint)

        let response: ChatCompletionResponseDTO
        switch reply {
        case .completion(let completion):
            response = completion
        case .serverError(let message):
            // BR-AIL-10: the server answered 200 and said why it could not
            // serve the request. Its own words travel, with the URL they came
            // from — a `/v1` left in a server root reads as a bad reply
            // otherwise, and the URL is what names that mistake.
            throw AIKitError.requestFailed("\(message) — from \(requestTarget(endpoint))")
        }

        let elapsed = startTime.duration(to: .now)
        let latencyMs = Int(elapsed.components.seconds * 1000 + elapsed.components.attoseconds / 1_000_000_000_000_000)

        guard let firstChoice = response.choices.first else {
            throw AIKitError.requestFailed(Self.emptyReplyMessage)
        }

        let toolCalls = firstChoice.message.toolCalls?.map { call in
            AIToolCall(id: call.id, name: call.function.name, arguments: call.function.arguments)
        }

        return AIResponse(
            content: firstChoice.message.content ?? "",
            model: modelName,
            tokensUsed: TokenUsage(
                prompt: response.usage?.promptTokens ?? 0,
                completion: response.usage?.completionTokens ?? 0
            ),
            latencyMs: latencyMs,
            toolCalls: toolCalls,
            providerID: id
        )
    }

    public func stream(request: AIRequest) async throws -> AsyncThrowingStream<AIChunk, Error> {
        let chatRequest = buildChatRequest(from: request, stream: true)
        let endpoint = makeEndpoint(for: chatRequest)
        let urlRequest = networkService.createRequest(endPoint: endpoint)

        return AsyncThrowingStream { continuation in
            let delegate = SSEStreamDelegate(continuation: continuation)
            let session = URLSession(configuration: .default, delegate: delegate, delegateQueue: nil)
            let task = session.dataTask(with: urlRequest)
            continuation.onTermination = { @Sendable _ in
                task.cancel()
                session.invalidateAndCancel()
            }
            task.resume()
        }
    }

    // MARK: - Private

    private func buildChatRequest(from request: AIRequest, stream: Bool) -> ChatCompletionRequest {
        var messages: [ChatCompletionRequest.ChatMessage] = []

        if let systemPrompt = request.systemPrompt {
            messages.append(.init(role: "system", content: systemPrompt))
        }

        for msg in request.messages {
            let wireToolCalls: [ChatCompletionRequest.ToolCallRequest]? = msg.toolCalls?.map { tc in
                .init(
                    id: tc.id ?? UUID().uuidString,
                    type: "function",
                    function: .init(name: tc.name, arguments: tc.arguments)
                )
            }
            // A tool-call-only assistant message has empty content in AIMessage
            // but requires nil (not an empty string) in the wire format so the
            // server does not interpret it as a content-and-tool-calls message.
            let wireContent: String? = msg.content.isEmpty ? nil : msg.content
            messages.append(.init(
                role: msg.role.rawValue,
                content: wireContent,
                toolCalls: wireToolCalls,
                toolCallId: msg.toolCallId
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

        let responseFormat: ChatCompletionRequest.ResponseFormatDTO? = request.responseFormat.map { fmt in
            .init(type: fmt.rawValue == "json" ? "json_object" : "text")
        }

        return ChatCompletionRequest(
            model: modelName,
            messages: messages,
            temperature: request.temperature,
            maxTokens: request.maxTokens,
            stream: stream,
            tools: tools,
            responseFormat: responseFormat
        )
    }

    /// The URL a request to `endpoint` actually goes to, for an error message
    /// that has to name where it came from.
    private func requestTarget(_ endpoint: OpenAIChatCompletionEndPoint) -> String {
        networkService.createRequest(endPoint: endpoint).url?.absoluteString
            ?? "\(baseURL)\(endpoint.path)"
    }

    private func makeEndpoint(for chatRequest: ChatCompletionRequest) -> OpenAIChatCompletionEndPoint {
        let parsed = URLComponents(string: baseURL)
        return OpenAIChatCompletionEndPoint(
            host: parsed?.host ?? "127.0.0.1",
            port: parsed?.port,
            scheme: parsed?.scheme ?? "http",
            apiKey: apiKey,
            requestBody: chatRequest
        )
    }
}

// MARK: - SSE Stream Delegate

/// URLSession delegate that receives SSE data incrementally and yields AIChunks.
private final class SSEStreamDelegate: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    private let continuation: AsyncThrowingStream<AIChunk, Error>.Continuation
    private var buffer = Data()

    init(continuation: AsyncThrowingStream<AIChunk, Error>.Continuation) {
        self.continuation = continuation
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        buffer.append(data)
        processBuffer()
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: (any Error)?) {
        // Process any remaining data in the buffer.
        processBuffer()
        if let error {
            continuation.finish(throwing: AIKitError.requestFailed(error.localizedDescription))
        } else {
            continuation.finish()
        }
    }

    private func processBuffer() {
        guard let text = String(data: buffer, encoding: .utf8) else { return }
        let lines = text.components(separatedBy: "\n")

        // Keep incomplete last line in buffer.
        if !text.hasSuffix("\n") {
            if let lastLine = lines.last {
                buffer = Data(lastLine.utf8)
            } else {
                buffer = Data()
            }
            for line in lines.dropLast() {
                processLine(line)
            }
        } else {
            buffer = Data()
            for line in lines {
                processLine(line)
            }
        }
    }

    private func processLine(_ line: String) {
        let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.hasPrefix("data: ") else { return }

        let payload = String(trimmed.dropFirst(6))
        if payload == "[DONE]" {
            continuation.yield(AIChunk(delta: "", isComplete: true))
            return
        }

        guard let data = payload.data(using: .utf8),
            let chunk = try? JSONDecoder().decode(ChatCompletionChunkDTO.self, from: data),
            let firstChoice = chunk.choices.first
        else { return }

        let delta = firstChoice.delta.content ?? ""
        let isComplete = firstChoice.finishReason != nil
        if !delta.isEmpty || isComplete {
            continuation.yield(AIChunk(delta: delta, isComplete: isComplete))
        }
    }
}

// MARK: - Timeout-carrying transport

/// A `NetworkProtocol` that puts a caller's timeout on the request it builds.
///
/// NetKit's `createRequest` leaves `URLRequest.timeoutInterval` at its
/// 60-second default, and `sendRequest` runs on `URLSession.shared`, so a
/// provider cannot otherwise wait longer than a minute. A local model loading
/// cold needs more than a minute, and a request that dies at 60 seconds reads
/// as a dead server rather than a slow one — the misreading W5's fallback rule
/// exists to avoid. `createRequest(endPoint:)` is a `NetworkProtocol`
/// requirement and not only an extension default, so the override here is the
/// one the inherited `sendRequest` calls.
///
/// Public so callers outside AIKit can create an `OpenAIProvider` with a
/// per-call timeout, which is needed for migrating any caller that passes a
/// `timeout` parameter (e.g. `AgentProvider.LocalProvider.execute`).
public struct TimeoutNetworkService: NetworkProtocol {
    /// Seconds a single request may take.
    public let timeoutInterval: TimeInterval
    /// NetKit's own request builder, reached through a type that does not
    /// override it. Delegating keeps the URL, header and body construction in
    /// one place instead of copying it here to drift.
    private let base = NetworkService()

    public init(timeoutInterval: TimeInterval) {
        self.timeoutInterval = timeoutInterval
    }

    public func createRequest(endPoint: EndPoint) -> URLRequest {
        var request = base.createRequest(endPoint: endPoint)
        request.timeoutInterval = timeoutInterval
        return request
    }
}
