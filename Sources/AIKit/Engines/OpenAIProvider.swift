import Foundation
import NetKit

/// AIProvider that talks to any OpenAI-compatible API.
public struct OpenAIProvider: AIProvider, Sendable {
    public let id: String
    public let displayName: String
    public let capabilities: AICapabilities

    let baseURL: String
    let modelName: String
    let apiKey: String?
    let networkService: any NetworkProtocol

    public var status: AIProviderStatus {
        get async { .ready }
    }

    public func complete(request: AIRequest) async throws -> AIResponse {
        let startTime = ContinuousClock.now

        let chatRequest = buildChatRequest(from: request, stream: false)
        let endpoint = makeEndpoint(for: chatRequest)

        let response: ChatCompletionResponseDTO = try await networkService.sendRequest(endpoint: endpoint)

        let elapsed = startTime.duration(to: .now)
        let latencyMs = Int(elapsed.components.seconds * 1000 + elapsed.components.attoseconds / 1_000_000_000_000_000)

        guard let firstChoice = response.choices.first else {
            throw AIKitError.requestFailed("Empty response — no choices returned")
        }

        let toolCalls = firstChoice.message.toolCalls?.map { call in
            AIToolCall(name: call.function.name, arguments: call.function.arguments)
        }

        return AIResponse(
            content: firstChoice.message.content ?? "",
            model: modelName,
            tokensUsed: TokenUsage(
                prompt: response.usage?.promptTokens ?? 0,
                completion: response.usage?.completionTokens ?? 0
            ),
            latencyMs: latencyMs,
            toolCalls: toolCalls
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
            messages.append(.init(role: msg.role.rawValue, content: msg.content))
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
