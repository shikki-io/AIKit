import Testing
@testable import AIKit

@Suite("ProviderRouter")
struct ProviderRouterTests {

    @Test("Routes to provider with matching capabilities")
    func routeMatching() async {
        let textProvider = MockAIProvider(
            id: "text", capabilities: [.textGeneration, .toolUse]
        )
        let visionProvider = MockAIProvider(
            id: "vision", capabilities: [.vision, .ocr]
        )
        let router = ProviderRouter(providers: [textProvider, visionProvider])

        let found = await router.route(capabilities: .textGeneration)
        #expect(found?.id == "text")

        let foundVision = await router.route(capabilities: .vision)
        #expect(foundVision?.id == "vision")
    }

    @Test("Returns nil for unmatched capabilities")
    func routeNoMatch() async {
        let provider = MockAIProvider(id: "text", capabilities: .textGeneration)
        let router = ProviderRouter(providers: [provider])

        let found = await router.route(capabilities: .imageGeneration)
        #expect(found == nil)
    }

    @Test("Skips providers that are not ready")
    func skipsNonReady() async {
        let loading = MockAIProvider(
            id: "loading", capabilities: .textGeneration, status: .loading(progress: 0.5)
        )
        let ready = MockAIProvider(
            id: "ready", capabilities: .textGeneration, status: .ready
        )
        let router = ProviderRouter(providers: [loading, ready])

        let found = await router.route(capabilities: .textGeneration)
        #expect(found?.id == "ready")
    }

    @Test("Fallback chain tries all providers")
    func fallbackChain() async throws {
        let failing = MockAIProvider(id: "fail", capabilities: .textGeneration)
        failing.shouldThrow = AIKitError.requestFailed("down")

        let expected = AIResponse(
            content: "success",
            model: "m",
            tokensUsed: TokenUsage(prompt: 1, completion: 1),
            latencyMs: 1
        )
        let working = MockAIProvider(
            id: "work", capabilities: .textGeneration, responses: [expected]
        )

        let router = ProviderRouter(providers: [])
        let request = AIRequest(messages: [AIMessage(role: .user, content: "hi")])
        let response = try await router.routeWithFallback(
            request: request, providers: [failing, working]
        )
        #expect(response.content == "success")
        #expect(failing.completeCallCount == 1)
        #expect(working.completeCallCount == 1)
    }

    @Test("Fallback throws allProvidersFailed when none succeed")
    func fallbackAllFail() async {
        let a = MockAIProvider(id: "a")
        a.shouldThrow = AIKitError.requestFailed("err")
        let b = MockAIProvider(id: "b")
        b.shouldThrow = AIKitError.requestFailed("err")

        let router = ProviderRouter(providers: [])
        let request = AIRequest(messages: [AIMessage(role: .user, content: "hi")])

        do {
            _ = try await router.routeWithFallback(request: request, providers: [a, b])
            Issue.record("Expected allProvidersFailed error")
        } catch let error as AIKitError {
            #expect(error == .allProvidersFailed)
        } catch {
            Issue.record("Unexpected error type: \(error)")
        }
    }
}
