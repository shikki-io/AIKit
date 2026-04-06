import Foundation
import Testing
@testable import AIKit

@Suite("PerformanceStore")
struct PerformanceStoreTests {
    private func makeTempStore() -> PerformanceStore {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("aikit-tests-\(UUID().uuidString)")
        let storePath = tempDir.appendingPathComponent("performance.json")
        return PerformanceStore(storePath: storePath)
    }

    private let model = ModelIdentifier(provider: "lmstudio", modelId: "nemotron-3b")
    private let context = UsageContext(app: "brainy", task: "article-summary")

    @Test("Record inference updates stats")
    func recordCreatesEntry() {
        let store = makeTempStore()

        store.record(model: model, context: context, latencyMs: 200, tokensPerSecond: 45.0, qualityScore: 0.8)

        let stat = store.stats(for: model, context: context)
        #expect(stat != nil)
        #expect(stat?.avgLatencyMs == 200)
        #expect(stat?.avgTokensPerSecond == 45.0)
        #expect(stat?.qualityScore == 0.8)
        #expect(stat?.totalInvocations == 1)
    }

    @Test("Stats accumulate with running averages")
    func statsAccumulate() {
        let store = makeTempStore()

        store.record(model: model, context: context, latencyMs: 200, tokensPerSecond: 40.0, qualityScore: 0.8)
        store.record(model: model, context: context, latencyMs: 300, tokensPerSecond: 60.0, qualityScore: 0.6)

        let stat = store.stats(for: model, context: context)!
        #expect(stat.avgLatencyMs == 250)
        #expect(stat.avgTokensPerSecond == 50.0)
        #expect(stat.qualityScore! == 0.7)
        #expect(stat.totalInvocations == 2)
    }

    @Test("bestModel returns highest quality score")
    func bestModelByQuality() {
        let store = makeTempStore()
        let modelA = ModelIdentifier(provider: "a", modelId: "fast")
        let modelB = ModelIdentifier(provider: "b", modelId: "quality")

        store.record(model: modelA, context: context, latencyMs: 100, tokensPerSecond: 80.0, qualityScore: 0.6)
        store.record(model: modelB, context: context, latencyMs: 500, tokensPerSecond: 20.0, qualityScore: 0.95)

        let best = store.bestModel(for: context)
        #expect(best == modelB)
    }

    @Test("bestModel falls back to fastest when no quality data")
    func bestModelFallsBackToFastest() {
        let store = makeTempStore()
        let slow = ModelIdentifier(provider: "a", modelId: "slow")
        let fast = ModelIdentifier(provider: "b", modelId: "fast")

        store.record(model: slow, context: context, latencyMs: 500, tokensPerSecond: 10.0)
        store.record(model: fast, context: context, latencyMs: 100, tokensPerSecond: 80.0)

        let best = store.bestModel(for: context)
        #expect(best == fast)
    }

    @Test("Clear removes all stats")
    func clearRemovesAll() {
        let store = makeTempStore()

        store.record(model: model, context: context, latencyMs: 200, tokensPerSecond: 45.0)
        #expect(store.stats(for: model, context: context) != nil)

        store.clear()
        #expect(store.stats(for: model, context: context) == nil)
        #expect(store.bestModel(for: context) == nil)
    }

    @Test("Persistence round-trip: save and reload")
    func persistenceRoundTrip() {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("aikit-tests-\(UUID().uuidString)")
        let storePath = tempDir.appendingPathComponent("performance.json")

        // Write data.
        let store1 = PerformanceStore(storePath: storePath)
        store1.record(model: model, context: context, latencyMs: 150, tokensPerSecond: 55.0, qualityScore: 0.9)

        // Reload from same path.
        let store2 = PerformanceStore(storePath: storePath)
        let stat = store2.stats(for: model, context: context)
        #expect(stat != nil)
        #expect(stat?.avgLatencyMs == 150)
        #expect(stat?.avgTokensPerSecond == 55.0)
        #expect(stat?.qualityScore == 0.9)
        #expect(stat?.totalInvocations == 1)
    }

    @Test("allStats returns entries across contexts")
    func allStatsForModel() {
        let store = makeTempStore()
        let ctx1 = UsageContext(app: "brainy", task: "summary")
        let ctx2 = UsageContext(app: "brainy", task: "translate")

        store.record(model: model, context: ctx1, latencyMs: 100, tokensPerSecond: 50.0)
        store.record(model: model, context: ctx2, latencyMs: 200, tokensPerSecond: 30.0)

        let all = store.allStats(for: model)
        #expect(all.count == 2)
        #expect(all[ctx1]?.avgLatencyMs == 100)
        #expect(all[ctx2]?.avgLatencyMs == 200)
    }

    @Test("bestModel returns nil for unknown context")
    func bestModelNilForUnknown() {
        let store = makeTempStore()
        let unknown = UsageContext(app: "nonexistent", task: "none")
        #expect(store.bestModel(for: unknown) == nil)
    }
}
