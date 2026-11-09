// MARK: - VJEPAEmbeddingAdaptiveRouterTests
// TDDP step 23 (RED) + step 24 (GREEN): adaptive router switches provider
// after N battles show a clear winner.
//
// Spec §5 Phase 3 deliverable 6 + BR-22.

import Foundation
import Testing
@testable import AIKitVision

// MARK: - Helpers

private func makeMetrics(
    taskId: String,
    category: String,
    winner: BattleWinner,
    scoreA: Double = 1.0,
    scoreB: Double = 0.0
) -> BattleMetrics {
    BattleMetrics(
        battleId: UUID().uuidString,
        taskId: taskId,
        taskCategory: category,
        providerA: ProviderMeasurement(
            providerName: "a",
            relevance: scoreA,
            latencyMs: 10,
            memoryMB: 100,
            tokenCostUSD: 0,
            energyWh: 0.001,
            answer: "a"
        ),
        providerB: ProviderMeasurement(
            providerName: "b",
            relevance: scoreB,
            latencyMs: 5,
            memoryMB: 5,
            tokenCostUSD: 0.01,
            energyWh: 0,
            answer: "b"
        ),
        winner: winner,
        scoreA: scoreA,
        scoreB: scoreB,
        policy: .default,
        overheadMs: 1,
        timestamp: Date()
    )
}

@Suite("VJEPAEmbeddingAdaptiveRouter")
struct VJEPAEmbeddingAdaptiveRouterTests {

    // MARK: Step 23 — router is undecided before threshold

    @Test("router returns undecided when fewer battles than minBattlesForDecision")
    func undecidedBelowThreshold() {
        let router = AdaptiveVisionRouter(minBattlesForDecision: 3, decisionThreshold: 0.3)
        router.record(makeMetrics(taskId: "t1", category: "error-detection", winner: .providerA))
        router.record(makeMetrics(taskId: "t2", category: "error-detection", winner: .providerA))
        #expect(router.route(for: "error-detection") == .undecided)
    }

    // MARK: Step 24 — router switches after N clear wins

    @Test("router switches to provider A after N battles show clear winner")
    func switchesToA() {
        let router = AdaptiveVisionRouter(minBattlesForDecision: 3, decisionThreshold: 0.3)
        for i in 0..<5 {
            router.record(makeMetrics(taskId: "t\(i)", category: "error-detection", winner: .providerA))
        }
        #expect(router.route(for: "error-detection") == .providerA)
    }

    @Test("router switches to provider B after N battles show clear winner")
    func switchesToB() {
        let router = AdaptiveVisionRouter(minBattlesForDecision: 3, decisionThreshold: 0.3)
        for i in 0..<5 {
            router.record(makeMetrics(taskId: "t\(i)", category: "ui-identification", winner: .providerB))
        }
        #expect(router.route(for: "ui-identification") == .providerB)
    }

    @Test("router stays undecided when wins are split close to 50/50")
    func undecidedOnSplit() {
        let router = AdaptiveVisionRouter(minBattlesForDecision: 3, decisionThreshold: 0.3)
        router.record(makeMetrics(taskId: "t1", category: "split", winner: .providerA))
        router.record(makeMetrics(taskId: "t2", category: "split", winner: .providerB))
        router.record(makeMetrics(taskId: "t3", category: "split", winner: .providerA))
        router.record(makeMetrics(taskId: "t4", category: "split", winner: .providerB))
        #expect(router.route(for: "split") == .undecided)
    }

    // MARK: Per-category isolation

    @Test("router tracks win rate per category independently")
    func perCategoryIsolation() {
        let router = AdaptiveVisionRouter(minBattlesForDecision: 3, decisionThreshold: 0.3)
        for i in 0..<5 {
            router.record(makeMetrics(taskId: "a\(i)", category: "error-detection", winner: .providerA))
        }
        for i in 0..<5 {
            router.record(makeMetrics(taskId: "b\(i)", category: "ui-identification", winner: .providerB))
        }
        #expect(router.route(for: "error-detection") == .providerA)
        #expect(router.route(for: "ui-identification") == .providerB)
        #expect(router.route(for: "unseen-category") == .undecided)
    }

    // MARK: Snapshot API

    @Test("winRates returns per-category summary")
    func winRatesSnapshot() throws {
        let router = AdaptiveVisionRouter()
        for i in 0..<3 {
            router.record(makeMetrics(
                taskId: "t\(i)",
                category: "error-detection",
                winner: .providerA,
                scoreA: 1.0,
                scoreB: 0.2
            ))
        }
        router.record(makeMetrics(taskId: "d1", category: "error-detection", winner: .draw))

        let rates = router.winRates()
        #expect(rates.count == 1)
        let rate = try #require(rates.first)
        #expect(rate.category == "error-detection")
        #expect(rate.totalBattles == 4)
        #expect(rate.winsA == 3)
        #expect(rate.draws == 1)
        #expect(abs(rate.meanScoreA - 1.0) < 1e-6 || rate.meanScoreA > 0.7)
        #expect(rate.winRateA == 0.75)
    }

    // MARK: Threshold config

    @Test("custom decisionThreshold makes router more conservative")
    func conservativeThreshold() {
        // 60/40 split with threshold 0.3 → undecided; with threshold 0.15 → A.
        let batch: [BattleMetrics] = (0..<10).map { i in
            makeMetrics(
                taskId: "t\(i)",
                category: "c",
                winner: i < 6 ? .providerA : .providerB
            )
        }
        let strict = AdaptiveVisionRouter(minBattlesForDecision: 3, decisionThreshold: 0.3)
        strict.recordAll(batch)
        #expect(strict.route(for: "c") == .undecided)

        let lax = AdaptiveVisionRouter(minBattlesForDecision: 3, decisionThreshold: 0.15)
        lax.recordAll(batch)
        #expect(lax.route(for: "c") == .providerA)
    }

    // MARK: Regression — recordAll order is irrelevant

    @Test("recordAll produces the same route as sequential record")
    func batchAndSequentialAgree() {
        let batch: [BattleMetrics] = (0..<5).map { i in
            makeMetrics(taskId: "t\(i)", category: "x", winner: .providerA)
        }
        let r1 = AdaptiveVisionRouter()
        batch.forEach { r1.record($0) }
        let r2 = AdaptiveVisionRouter()
        r2.recordAll(batch)
        #expect(r1.route(for: "x") == r2.route(for: "x"))
    }
}
