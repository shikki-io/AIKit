import Testing
@testable import AIKit

@Suite("Mocks")
struct MockTests {

    @Test("MockAIProvider returns canned responses")
    func cannedResponses() async throws {
        let r1 = AIResponse(
            content: "first", model: "m",
            tokensUsed: TokenUsage(prompt: 1, completion: 1), latencyMs: 1
        )
        let r2 = AIResponse(
            content: "second", model: "m",
            tokensUsed: TokenUsage(prompt: 2, completion: 2), latencyMs: 2
        )
        let mock = MockAIProvider(responses: [r1, r2])

        let request = AIRequest(messages: [AIMessage(role: .user, content: "hi")])
        let first = try await mock.complete(request: request)
        #expect(first.content == "first")

        let second = try await mock.complete(request: request)
        #expect(second.content == "second")

        // Beyond responses array, stays on last
        let third = try await mock.complete(request: request)
        #expect(third.content == "second")
    }

    @Test("MockAIProvider tracks call counts")
    func callTracking() async throws {
        let mock = MockAIProvider()
        let request = AIRequest(messages: [AIMessage(role: .user, content: "hi")])

        #expect(mock.completeCallCount == 0)
        _ = try await mock.complete(request: request)
        #expect(mock.completeCallCount == 1)
        _ = try await mock.complete(request: request)
        #expect(mock.completeCallCount == 2)
    }

    @Test("MockAIProvider returns default response when no canned responses")
    func defaultResponse() async throws {
        let mock = MockAIProvider()
        let request = AIRequest(messages: [AIMessage(role: .user, content: "hi")])
        let response = try await mock.complete(request: request)
        #expect(response.content == "mock response")
    }

    @Test("MockAIProvider throws when shouldThrow is set")
    func throwsError() async {
        let mock = MockAIProvider()
        mock.shouldThrow = AIKitError.requestFailed("test error")
        let request = AIRequest(messages: [AIMessage(role: .user, content: "hi")])

        do {
            _ = try await mock.complete(request: request)
            Issue.record("Expected error")
        } catch let error as AIKitError {
            #expect(error == .requestFailed("test error"))
        } catch {
            Issue.record("Unexpected error type: \(error)")
        }
    }

    @Test("MockAIProvider status matches initialization")
    func status() async {
        let ready = MockAIProvider(status: .ready)
        let loading = MockAIProvider(status: .loading(progress: 0.5))

        let readyStatus = await ready.status
        #expect(readyStatus == .ready)

        let loadingStatus = await loading.status
        #expect(loadingStatus == .loading(progress: 0.5))
    }

    @Test("MockRuntimeEngine loads and unloads models")
    func engineLoadUnload() async throws {
        let engine = MockRuntimeEngine()
        let descriptor = ModelDescriptor(
            id: ModelIdentifier(provider: "test", modelId: "model-1"),
            name: "Test Model",
            author: "Test",
            description: "A test model",
            capabilities: .textGeneration,
            format: .gguf,
            parameters: "4B",
            sizeBytes: 1_000_000,
            architecture: "llama",
            domain: .llm
        )

        #expect(engine.loadedModels().isEmpty)

        let provider = try await engine.loadModel(descriptor)
        #expect(provider.id == "test/model-1")
        #expect(engine.loadedModels().count == 1)

        try await engine.unloadModel(descriptor.id)
        #expect(engine.loadedModels().isEmpty)
    }
}
