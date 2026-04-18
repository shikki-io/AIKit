/// Universal AI provider protocol — any model, any backend.
public protocol AIProvider: Sendable {
    /// Unique identifier for this provider instance.
    var id: String { get }
    /// Human-readable name.
    var displayName: String { get }
    /// What this provider can do.
    var capabilities: AICapabilities { get }
    /// Current provider status.
    var status: AIProviderStatus { get async }

    /// Send a completion request and get a full response.
    func complete(request: AIRequest) async throws -> AIResponse

    /// Stream a completion request as chunks.
    func stream(request: AIRequest) async throws -> AsyncThrowingStream<AIChunk, Error>
}

/// Default streaming implementation that wraps complete() for providers
/// that do not support native streaming.
extension AIProvider {
    public func stream(request: AIRequest) async throws -> AsyncThrowingStream<AIChunk, Error> {
        let response = try await complete(request: request)
        return AsyncThrowingStream { continuation in
            continuation.yield(AIChunk(delta: response.content, isComplete: true))
            continuation.finish()
        }
    }
}
