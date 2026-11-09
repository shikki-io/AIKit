// MARK: - VJEPAEmbeddingBattleArenaTests
// TDDP step 21 (RED) + step 22 (GREEN): BattleArena runs two providers on
// the same task and records metrics.
//
// Spec §5 Phase 3 deliverables 1 + 5 + BR-22.

import Foundation
import Testing
@testable import AIKitVision

// MARK: - Test doubles

private struct StaticProvider: VisionProvider {
    let name: String
    let label: String
    let tokenCostUSD: Double
    let energyWh: Double
    let memoryMB: Int
    let sleepMs: Int

    func answer(task: BattleTask) async throws -> ProviderAnswer {
        if sleepMs > 0 {
            try await Task.sleep(nanoseconds: UInt64(sleepMs) * 1_000_000)
        }
        return ProviderAnswer(
            label: label,
            tokenCostUSD: tokenCostUSD,
            energyWh: energyWh,
            memoryMB: memoryMB
        )
    }
}

@Suite("VJEPAEmbeddingBattleArena")
struct VJEPAEmbeddingBattleArenaTests {

    private static func makeTask() -> BattleTask {
        BattleTask(
            taskId: "task-001",
            category: "ui-element-identified",
            question: "Which button was clicked?",
            groundTruthLabel: "submit"
        )
    }

    // MARK: Step 21 — arena records metrics for both providers

    @Test("BattleArena runs both providers and records per-provider measurements")
    func runsBothProviders() async throws {
        let a = StaticProvider(
            name: "vjepa-test",
            label: "submit",
            tokenCostUSD: 0,
            energyWh: 0.002,
            memoryMB: 120,
            sleepMs: 0
        )
        let b = StaticProvider(
            name: "mock-llm",
            label: "cancel",
            tokenCostUSD: 0.01,
            energyWh: 0,
            memoryMB: 5,
            sleepMs: 0
        )
        let arena = BattleArena(providerA: a, providerB: b)
        let metrics = try await arena.run(task: Self.makeTask())

        #expect(metrics.providerA.providerName == "vjepa-test")
        #expect(metrics.providerB.providerName == "mock-llm")
        #expect(metrics.providerA.relevance == 1.0)
        #expect(metrics.providerB.relevance == 0.0)
        #expect(metrics.providerA.energyWh == 0.002)
        #expect(metrics.providerB.tokenCostUSD == 0.01)
        #expect(metrics.taskCategory == "ui-element-identified")
    }

    // MARK: BR-22 — winner uses cost-adjusted score, not raw accuracy

    @Test("winner is decided by cost-adjusted score (BR-22)")
    func costAdjustedWinner() async throws {
        // Both providers are exactly right — but B has token cost.
        // Under BR-22, A wins on cost-adjusted score.
        let a = StaticProvider(
            name: "vjepa",
            label: "submit",
            tokenCostUSD: 0,
            energyWh: 0.001,
            memoryMB: 100,
            sleepMs: 0
        )
        let b = StaticProvider(
            name: "llm",
            label: "submit",
            tokenCostUSD: 0.02,
            energyWh: 0,
            memoryMB: 5,
            sleepMs: 0
        )
        let arena = BattleArena(providerA: a, providerB: b)
        let metrics = try await arena.run(task: Self.makeTask())
        #expect(metrics.winner == .providerA)
        #expect(metrics.scoreA > metrics.scoreB)
    }

    @Test("raw-accuracy winner can still lose on BR-22 if cost dominates")
    func costFlipsRawAccuracyWinner() async throws {
        // B is slightly more accurate (0.5 vs 0) but burns lots of tokens.
        // Default policy: costWeight=5.0 USD => 0.5 token USD costs B 2.5
        // in score, so A's 0 relevance still beats B's 0.5 - 2.5 = -2.0.
        let a = StaticProvider(
            name: "vjepa",
            label: "wrong",
            tokenCostUSD: 0,
            energyWh: 0,
            memoryMB: 100,
            sleepMs: 0
        )
        let b = StaticProvider(
            name: "llm",
            label: "submit-button",
            tokenCostUSD: 0.5,
            energyWh: 0,
            memoryMB: 5,
            sleepMs: 0
        )
        let task = BattleTask(
            taskId: "task-002",
            category: "ui-element-identified",
            question: "Which button?",
            groundTruthLabel: "submit"
        )
        let arena = BattleArena(providerA: a, providerB: b)
        let metrics = try await arena.run(task: task)
        #expect(metrics.winner == .providerA)
        #expect(metrics.scoreA > metrics.scoreB)
    }

    // MARK: Latency + overhead

    @Test("BattleArena overhead stays under 50 ms (Phase 3 gate)")
    func overheadBounded() async throws {
        let a = StaticProvider(
            name: "vjepa",
            label: "submit",
            tokenCostUSD: 0,
            energyWh: 0,
            memoryMB: 0,
            sleepMs: 0
        )
        let b = StaticProvider(
            name: "llm",
            label: "submit",
            tokenCostUSD: 0,
            energyWh: 0,
            memoryMB: 0,
            sleepMs: 0
        )
        let arena = BattleArena(providerA: a, providerB: b)
        let metrics = try await arena.run(task: Self.makeTask())
        #expect(metrics.overheadMs < 50)
    }

    @Test("BattleArena records per-provider latency")
    func perProviderLatency() async throws {
        let a = StaticProvider(
            name: "slow",
            label: "submit",
            tokenCostUSD: 0,
            energyWh: 0,
            memoryMB: 0,
            sleepMs: 20
        )
        let b = StaticProvider(
            name: "fast",
            label: "submit",
            tokenCostUSD: 0,
            energyWh: 0,
            memoryMB: 0,
            sleepMs: 0
        )
        let arena = BattleArena(providerA: a, providerB: b)
        let metrics = try await arena.run(task: Self.makeTask())
        #expect(metrics.providerA.latencyMs >= 10)
        #expect(metrics.providerB.latencyMs < metrics.providerA.latencyMs + 20)
    }

    // MARK: Draw + runAll

    @Test("identical measurements produce a draw")
    func drawOnTie() async throws {
        let a = StaticProvider(
            name: "a",
            label: "submit",
            tokenCostUSD: 0,
            energyWh: 0,
            memoryMB: 0,
            sleepMs: 0
        )
        let b = StaticProvider(
            name: "b",
            label: "submit",
            tokenCostUSD: 0,
            energyWh: 0,
            memoryMB: 0,
            sleepMs: 0
        )
        let arena = BattleArena(providerA: a, providerB: b)
        let metrics = try await arena.run(task: Self.makeTask())
        #expect(metrics.winner == .draw)
    }

    @Test("BattleArena.runAll returns metrics in order for a batch")
    func runAllOrder() async throws {
        let a = StaticProvider(
            name: "a",
            label: "submit",
            tokenCostUSD: 0,
            energyWh: 0,
            memoryMB: 0,
            sleepMs: 0
        )
        let b = StaticProvider(
            name: "b",
            label: "cancel",
            tokenCostUSD: 0,
            energyWh: 0,
            memoryMB: 0,
            sleepMs: 0
        )
        let arena = BattleArena(providerA: a, providerB: b)
        let tasks: [BattleTask] = (0..<3).map { i in
            BattleTask(
                taskId: "task-\(i)",
                category: "ui",
                question: "?",
                groundTruthLabel: "submit"
            )
        }
        let results = try await arena.runAll(tasks)
        #expect(results.count == 3)
        #expect(results.map(\.taskId) == ["task-0", "task-1", "task-2"])
    }
}
