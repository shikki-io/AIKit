import Foundation

/// Configurable mock AI provider with call tracking and canned responses.
public final class MockAIProvider: AIProvider, @unchecked Sendable {
    public let id: String
    public let displayName: String
    public let capabilities: AICapabilities
    private let _status: AIProviderStatus
    private var cannedResponses: [AIResponse]
    private var responseIndex: Int = 0
    private var _completeCallCount: Int = 0
    private var _streamCallCount: Int = 0
    public var shouldThrow: Error?

    public var completeCallCount: Int { _completeCallCount }
    public var streamCallCount: Int { _streamCallCount }

    public var status: AIProviderStatus {
        get async { _status }
    }

    public init(
        id: String = "mock",
        displayName: String = "Mock Provider",
        capabilities: AICapabilities = .textGeneration,
        status: AIProviderStatus = .ready,
        responses: [AIResponse] = []
    ) {
        self.id = id
        self.displayName = displayName
        self.capabilities = capabilities
        self._status = status
        self.cannedResponses = responses
    }

    public func complete(request: AIRequest) async throws -> AIResponse {
        _completeCallCount += 1
        if let error = shouldThrow {
            throw error
        }
        guard !cannedResponses.isEmpty else {
            return AIResponse(
                content: "mock response",
                model: "mock-model",
                tokensUsed: TokenUsage(prompt: 10, completion: 5),
                latencyMs: 1
            )
        }
        let index = min(responseIndex, cannedResponses.count - 1)
        responseIndex += 1
        return cannedResponses[index]
    }

    public func stream(request: AIRequest) async throws -> AsyncThrowingStream<AIChunk, Error> {
        _streamCallCount += 1
        if let error = shouldThrow {
            throw error
        }
        let response = try await complete(request: request)
        return AsyncThrowingStream { continuation in
            continuation.yield(AIChunk(delta: response.content, isComplete: true))
            continuation.finish()
        }
    }
}
