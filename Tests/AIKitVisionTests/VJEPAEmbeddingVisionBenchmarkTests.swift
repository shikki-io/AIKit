// MARK: - VJEPAEmbeddingVisionBenchmarkTests
// Phase 3 gate — 10-task corpus runs end-to-end through the arena,
// metrics round-trip via SQLite, adaptive router updates its verdict.
//
// Spec §5 Phase 3 gate: "Battle framework runs end-to-end on 10 test
// tasks. Adaptive router correctly switches provider based on accumulated
// battle results. Latency overhead of battle framework < 50 ms per
// evaluation."

#if canImport(SQLite3)
import Foundation
import Testing
@testable import AIKitVision

@Suite("VJEPAEmbeddingVisionBenchmark")
struct VJEPAEmbeddingVisionBenchmarkTests {

    // MARK: Corpus shape

    @Test("starter corpus ships 10 tasks across 3 categories")
    func starterCorpusShape() {
        let corpus = VisionBenchmark.starterCorpus()
        #expect(corpus.count == 10)
        let categories = Set(corpus.map(\.task.category))
        #expect(categories == ["ui-identification", "error-detection", "action-performed"])
        let ids = corpus.map(\.task.taskId)
        #expect(ids == ["t01", "t02", "t03", "t04", "t05", "t06", "t07", "t08", "t09", "t10"])
    }

    @Test("starter corpus is deterministic across calls")
    func corpusDeterministic() {
        let a = VisionBenchmark.starterCorpus()
        let b = VisionBenchmark.starterCorpus()
        #expect(a.count == b.count)
        for i in 0..<a.count {
            #expect(a[i].task.taskId == b[i].task.taskId)
            #expect(a[i].labelSignal == b[i].labelSignal)
            // Frame tensor is bit-identical
            #expect(a[i].task.frames?.data == b[i].task.frames?.data)
        }
    }

    // MARK: Provider contracts

    @Test("VJEPARetrievalProvider recovers ground truth for starter corpus")
    func vjepaRetrievalProviderAccurate() async throws {
        let provider = VJEPARetrievalProvider.forStarterCorpus()
        for item in VisionBenchmark.starterCorpus() {
            let ans = try await provider.answer(task: item.task)
            #expect(ans.label == item.task.groundTruthLabel)
            #expect(ans.tokenCostUSD == 0)
            #expect(ans.energyWh > 0) // BR-22 on-device energy present
        }
    }

    @Test("MockLLMFrameDescriptionProvider bills tokens but no energy (BR-22)")
    func llmProviderBillsTokens() async throws {
        let provider = MockLLMFrameDescriptionProvider.forStarterCorpus()
        let corpus = VisionBenchmark.starterCorpus()
        let ans = try await provider.answer(task: corpus[0].task)
        #expect(ans.label == corpus[0].task.groundTruthLabel)
        #expect(ans.tokenCostUSD > 0)
        #expect(ans.energyWh == 0)
    }

    // MARK: Phase 3 gate — end-to-end 10-task run

    @Test("arena runs the full 10-task corpus end-to-end under the 50 ms gate")
    func phase3GateEndToEnd() async throws {
        let vjepa = VJEPARetrievalProvider.forStarterCorpus()
        let llm = MockLLMFrameDescriptionProvider.forStarterCorpus()
        let arena = BattleArena(providerA: vjepa, providerB: llm)

        let tasks = VisionBenchmark.starterCorpus().map(\.task)
        let results = try await arena.runAll(tasks)
        #expect(results.count == 10)
        // Phase 3 gate — arena overhead < 50 ms per battle.
        for r in results {
            #expect(r.overheadMs < 50, "overhead \(r.overheadMs) ms exceeds 50 ms gate for \(r.taskId)")
        }
    }

    // MARK: Phase 3 gate — metrics round-trip through SQLite

    @Test("battle metrics survive a SQLite round-trip in the enclave")
    func metricsRoundTripSQLite() async throws {
        let vjepa = VJEPARetrievalProvider.forStarterCorpus()
        let llm = MockLLMFrameDescriptionProvider.forStarterCorpus()
        let arena = BattleArena(providerA: vjepa, providerB: llm)
        let store = try BattleStore(path: ":memory:")

        let tasks = VisionBenchmark.starterCorpus().map(\.task)
        let results = try await arena.runAll(tasks)
        try store.insertAll(results)

        #expect(try store.count() == 10)
        let fetched = try store.all()
        #expect(Set(fetched.map(\.taskId)) == Set(results.map(\.taskId)))
    }

    // MARK: Phase 3 gate — adaptive router switches on synthetic win pattern

    @Test("adaptive router switches provider based on corpus battle results")
    func phase3GateAdaptiveRouterSwitches() async throws {
        // V-JEPA wins on cost-adjusted score (same relevance, zero token cost).
        // Router should converge to `.providerA` on each of the 3 categories.
        let vjepa = VJEPARetrievalProvider.forStarterCorpus()
        let llm = MockLLMFrameDescriptionProvider.forStarterCorpus()
        let arena = BattleArena(providerA: vjepa, providerB: llm)
        let router = AdaptiveVisionRouter(minBattlesForDecision: 3, decisionThreshold: 0.3)

        let tasks = VisionBenchmark.starterCorpus().map(\.task)
        let results = try await arena.runAll(tasks)
        router.recordAll(results)

        #expect(router.route(for: "ui-identification") == .providerA)
        #expect(router.route(for: "error-detection") == .providerA)
        #expect(router.route(for: "action-performed") == .providerA)
        #expect(router.route(for: "never-seen") == .undecided)
    }

    @Test("router flips to provider B when LLM dominates raw accuracy without cost")
    func routerFlipsOnZeroCostLLM() async throws {
        // Simulate an LLM with zero token cost (e.g. local LLaVA) where
        // V-JEPA retrieval mis-answers. Router should flip to B.
        struct WrongVJEPA: VisionProvider {
            let name: String = "wrong-vjepa"
            func answer(task: BattleTask) async throws -> ProviderAnswer {
                ProviderAnswer(label: "off", tokenCostUSD: 0, energyWh: 0.002, memoryMB: 180)
            }
        }
        let vjepa = WrongVJEPA()
        // Build an LLM with zero token billing so raw accuracy wins.
        var responses: [String: String] = [:]
        for item in VisionBenchmark.starterCorpus() {
            responses[item.task.question] = item.task.groundTruthLabel
        }
        let llm = MockLLMFrameDescriptionProvider(
            name: "free-llm",
            client: MockLLMClient(
                responses: responses,
                inputTokensPerCall: 0,
                outputTokensPerCall: 0
            )
        )
        let arena = BattleArena(providerA: vjepa, providerB: llm)
        let router = AdaptiveVisionRouter(minBattlesForDecision: 3, decisionThreshold: 0.3)

        let tasks = VisionBenchmark.starterCorpus().map(\.task)
        let results = try await arena.runAll(tasks)
        router.recordAll(results)

        #expect(router.route(for: "ui-identification") == .providerB)
        #expect(router.route(for: "error-detection") == .providerB)
    }

    // MARK: Cost telemetry emitted per task (BR-22)

    @Test("every battle emits non-zero cost telemetry on at least one provider (BR-22)")
    func costTelemetryPresent() async throws {
        let vjepa = VJEPARetrievalProvider.forStarterCorpus()
        let llm = MockLLMFrameDescriptionProvider.forStarterCorpus()
        let arena = BattleArena(providerA: vjepa, providerB: llm)
        let tasks = VisionBenchmark.starterCorpus().map(\.task)
        let results = try await arena.runAll(tasks)
        for r in results {
            let anyCost =
                r.providerA.tokenCostUSD > 0 || r.providerA.energyWh > 0 ||
                r.providerB.tokenCostUSD > 0 || r.providerB.energyWh > 0
            #expect(anyCost, "no cost telemetry for \(r.taskId)")
        }
    }
}

#endif
