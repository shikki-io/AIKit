import Foundation
import Testing
@testable import AIKit

@Suite("SmartRouter")
struct SmartRouterTests {
    private let context = UsageContext(app: "brainy", task: "summary")

    @Test("routeSmart picks model with best quality score")
    func smartRouteByQuality() async {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("aikit-router-\(UUID().uuidString)")
        let store = PerformanceStore(storePath: tempDir.appendingPathComponent("perf.json"))

        let modelA = ModelIdentifier(provider: "a", modelId: "fast-model")
        let modelB = ModelIdentifier(provider: "b", modelId: "quality-model")

        store.record(model: modelA, context: context, latencyMs: 50, tokensPerSecond: 100.0, qualityScore: 0.5)
        store.record(model: modelB, context: context, latencyMs: 500, tokensPerSecond: 10.0, qualityScore: 0.95)

        let providerA = MockAIProvider(id: "a/fast-model", capabilities: .textGeneration)
        let providerB = MockAIProvider(id: "b/quality-model", capabilities: .textGeneration)

        let router = ProviderRouter(providers: [providerA, providerB], performanceStore: store)
        let request = AIRequest(messages: [AIMessage(role: .user, content: "test")])

        let selected = await router.routeSmart(request: request, context: context)
        #expect(selected?.id == "b/quality-model")
    }

    @Test("routeSmart falls back to fastest when no quality data")
    func smartRouteFallbackFastest() async {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("aikit-router-\(UUID().uuidString)")
        let store = PerformanceStore(storePath: tempDir.appendingPathComponent("perf.json"))

        let modelA = ModelIdentifier(provider: "a", modelId: "slow-model")
        let modelB = ModelIdentifier(provider: "b", modelId: "fast-model")

        store.record(model: modelA, context: context, latencyMs: 500, tokensPerSecond: 10.0)
        store.record(model: modelB, context: context, latencyMs: 50, tokensPerSecond: 100.0)

        let providerA = MockAIProvider(id: "a/slow-model", capabilities: .textGeneration)
        let providerB = MockAIProvider(id: "b/fast-model", capabilities: .textGeneration)

        let router = ProviderRouter(providers: [providerA, providerB], performanceStore: store)
        let request = AIRequest(messages: [AIMessage(role: .user, content: "test")])

        let selected = await router.routeSmart(request: request, context: context)
        #expect(selected?.id == "b/fast-model")
    }

    @Test("routeSmart works with empty performance store")
    func smartRouteEmptyStore() async {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("aikit-router-\(UUID().uuidString)")
        let store = PerformanceStore(storePath: tempDir.appendingPathComponent("perf.json"))

        let provider = MockAIProvider(id: "default", capabilities: .textGeneration)
        let router = ProviderRouter(providers: [provider], performanceStore: store)
        let request = AIRequest(messages: [AIMessage(role: .user, content: "test")])

        let selected = await router.routeSmart(request: request, context: context)
        #expect(selected?.id == "default")
    }

    @Test("routeSmart with no performance store returns first ready")
    func smartRouteNoStore() async {
        let providerA = MockAIProvider(id: "a", capabilities: .textGeneration)
        let providerB = MockAIProvider(id: "b", capabilities: .textGeneration)

        let router = ProviderRouter(providers: [providerA, providerB])
        let request = AIRequest(messages: [AIMessage(role: .user, content: "test")])

        let selected = await router.routeSmart(request: request, context: context)
        #expect(selected?.id == "a")
    }

    @Test("routeSmart skips non-ready providers")
    func smartRouteSkipsNonReady() async {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("aikit-router-\(UUID().uuidString)")
        let store = PerformanceStore(storePath: tempDir.appendingPathComponent("perf.json"))

        let model = ModelIdentifier(provider: "best", modelId: "model")
        store.record(model: model, context: context, latencyMs: 50, tokensPerSecond: 100.0, qualityScore: 0.99)

        let bestButDown = MockAIProvider(
            id: "best/model", capabilities: .textGeneration, status: .error("down")
        )
        let fallback = MockAIProvider(id: "fallback", capabilities: .textGeneration)

        let router = ProviderRouter(providers: [bestButDown, fallback], performanceStore: store)
        let request = AIRequest(messages: [AIMessage(role: .user, content: "test")])

        let selected = await router.routeSmart(request: request, context: context)
        // Best model is down, falls back to first ready.
        #expect(selected?.id == "fallback")
    }
}
