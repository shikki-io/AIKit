import Foundation
import Logging
import NetKit

private let providerLogger = Logger(label: "aikit.engines.openai-provider")

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
        // One system for OpenAI: the same request builder, endpoint and reply
        // mapper the wire-standard dispatch uses. A transport failure travels
        // as the type it was thrown as — `URLError`, `NetKit.NetworkError` —
        // because a caller deciding whether to fall back needs to tell a
        // refused connection from a server that answered (BR-AIL-07, W5).
        try await AIWireStandard.openAIChat.complete(
            request: request,
            target: target,
            networkService: networkService
        )
    }

    public func stream(request: AIRequest) async throws -> AsyncThrowingStream<AIChunk, Error> {
        let endpoint = try AIWireStandard.openAIChat.endPoint(target: target, request: request, stream: true)
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

    /// Where this provider's requests go. The root is parsed leniently — a
    /// scheme and host that are missing default — exactly as before.
    private var target: AIWireTarget {
        let parsed = URLComponents(string: baseURL)
        return AIWireTarget(
            server: AIWireServer(
                scheme: parsed?.scheme ?? "http",
                host: parsed?.host ?? "127.0.0.1",
                port: parsed?.port
            ),
            apiKey: apiKey,
            model: modelName,
            providerID: id
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

        guard let data = payload.data(using: .utf8) else {
            continuation.finish(throwing: AIKitError.requestFailed("SSE chunk is not valid UTF-8"))
            return
        }
        let chunk: ChatCompletionChunkDTO
        do {
            chunk = try JSONDecoder().decode(ChatCompletionChunkDTO.self, from: data)
        } catch {
            // keep-alive and [DONE] were already handled above; a remaining
            // decode failure is genuinely malformed — finish the stream with an
            // error so the caller learns about it instead of seeing a silent gap.
            continuation.finish(
                throwing: AIKitError.requestFailed("SSE chunk decode failed: \(error.localizedDescription)")
            )
            return
        }
        guard let firstChoice = chunk.choices.first else { return }

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
