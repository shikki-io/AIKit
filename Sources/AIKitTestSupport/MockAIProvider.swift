import AIKit
import Foundation
import Logging

private let mockLogger = Logger(label: "aikit.testsupport.mock")

/// Says, once per process, that a test double answered a real request.
///
/// Once rather than per call: a mock serves hundreds of requests in a suite
/// and a line per call would bury the run's real output, while the fact worth
/// knowing — "a test double is answering in this process" — is true from the
/// first one. In production the first line is the alarm; there is no second
/// state to report.
private actor MockUsageAnnouncement {
    static let shared = MockUsageAnnouncement()
    private var announced = false

    func announceOnce(_ body: @Sendable () -> Void) {
        guard !announced else { return }
        announced = true
        body()
    }
}

/// Configurable mock AI provider with call tracking and canned responses.
///
/// **A test double. It reaches no model.** It returns canned text, counts the
/// calls it was given and keeps the requests it received, so a test can prove
/// what a caller actually sent. Nothing it answers came from inference.
///
/// Operator, AIKit#18 r4500328752: "Can you explicit with a log you are using
/// a mock and it's not supposed to be used for running ai provider when we are
/// in prod". Two answers, and the structural one matters more than the log:
///
///  * **Structural (W26, T-AIL-51):** this type lives in the
///    `AIKitTestSupport` product, not in `AIKit`. An app that depends on
///    `AIKit` alone cannot name it, so production cannot construct it by
///    accident — the mistake is now a link error at the dependency, visible in
///    `Package.swift` review, rather than a runtime surprise.
///  * **Observable:** the first request this type serves in a process logs at
///    `warning` and says what it is. A build that somehow reached here anyway
///    announces itself in the host's log instead of quietly shipping canned
///    text as a model's answer.
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
        mockLogger.debug("MockAIProvider built", metadata: ["provider.id": .string(id)])
    }

    /// Logs, on the first served request in this process, that a test double
    /// is answering. See the note on the type.
    private func announceMockUse(_ path: String) async {
        let providerID = id
        await MockUsageAnnouncement.shared.announceOnce {
            mockLogger.warning(
                """
                MockAIProvider is serving AI requests in this process. It is a TEST DOUBLE from \
                AIKitTestSupport: it returns canned text and reaches no model, so any answer it \
                gives is fabricated. If this line appears outside a test run, a mock provider was \
                registered in production — look at which AIKit assembly was built and remove the \
                AIKitTestSupport dependency from the shipping target.
                """,
                metadata: [
                    "provider.id": .string(providerID),
                    "provider.path": .string(path),
                ]
            )
        }
    }

    /// Serves the next canned response, stamped with this mock's own `id` when
    /// the canned value does not already name a provider.
    ///
    /// A real provider always names itself, so a mock that answered anonymously
    /// would let a routing test pass against a response no real provider can
    /// produce. A test that needs a different author sets `providerID` on the
    /// canned response and that value is kept.
    public func complete(request: AIRequest) async throws -> AIResponse {
        await announceMockUse("complete")
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
        await announceMockUse("stream")
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
