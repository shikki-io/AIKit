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
    private var _receivedRequests: [AIRequest] = []
    public var shouldThrow: Error?

    public var completeCallCount: Int { _completeCallCount }
    public var streamCallCount: Int { _streamCallCount }

    /// Every request this mock was asked to serve, in order.
    /// Lets a test prove that what the caller put in a request — an audio
    /// input, a system prompt — actually reached the provider.
    public var receivedRequests: [AIRequest] { _receivedRequests }

    /// The most recent request, or nil if the mock was never called.
    public var lastRequest: AIRequest? { _receivedRequests.last }

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

    /// Serves the next canned response, stamped with this mock's own `id` when
    /// the canned value does not already name a provider.
    ///
    /// A real provider always names itself, so a mock that answered anonymously
    /// would let a routing test pass against a response no real provider can
    /// produce. A test that needs a different author sets `providerID` on the
    /// canned response and that value is kept.
    public func complete(request: AIRequest) async throws -> AIResponse {
        _completeCallCount += 1
        _receivedRequests.append(request)
        if let error = shouldThrow {
            throw error
        }
        guard !cannedResponses.isEmpty else {
            return AIResponse(
                content: "mock response",
                model: "mock-model",
                tokensUsed: TokenUsage(prompt: 10, completion: 5),
                latencyMs: 1,
                providerID: id
            )
        }
        let index = min(responseIndex, cannedResponses.count - 1)
        responseIndex += 1
        var response = cannedResponses[index]
        if response.providerID.isEmpty {
            response.providerID = id
        }
        return response
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
