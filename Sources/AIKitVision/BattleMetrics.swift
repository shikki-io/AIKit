// MARK: - BattleMetrics
// Per-battle telemetry captured by the auto-battle evaluation system
// (spec §5 Phase 3 deliverable 5 + BR-22 cost-per-task).
//
// Each battle run captures, for both Provider A and Provider B:
//   - `relevance` (0...1) — agreement with ground truth label
//   - `latency` in milliseconds
//   - `memory` peak in megabytes
//   - `tokenCost` in USD (cloud LLM billing units)
//   - `energyCost` in watt-hours (on-device compute estimate, BR-22)
//
// A single `BattleMetrics` value pairs the two provider measurements with
// the chosen winner under a cost-adjusted scoring function.
//
// Cost-adjusted score (BR-22):
//
//     score = relevance
//           - costWeightUSD * tokenCost
//           - energyWeightPerWh * energyCost
//
// Defaults live in `BattleScoringPolicy`. The adaptive router picks the
// provider with the higher cost-adjusted score, NOT raw accuracy.

import Foundation

// MARK: - Per-provider measurement

public struct ProviderMeasurement: Sendable, Equatable, Codable {
    /// Human-readable provider identifier (e.g. "vjepa-vitb16", "mock-llm").
    public let providerName: String

    /// Relevance score: 0 means wrong, 1 means matches ground truth.
    public let relevance: Double

    /// Wall-clock latency in milliseconds for the full provider call.
    public let latencyMs: Int

    /// Peak resident memory in megabytes during the provider call.
    public let memoryMB: Int

    /// Token billing cost in USD. 0 for pure on-device providers.
    public let tokenCostUSD: Double

    /// Estimated on-device compute energy cost in watt-hours (BR-22).
    public let energyWh: Double

    /// Free-form string label the provider returned (for debugging / audit).
    public let answer: String

    public init(
        providerName: String,
        relevance: Double,
        latencyMs: Int,
        memoryMB: Int,
        tokenCostUSD: Double,
        energyWh: Double,
        answer: String
    ) {
        self.providerName = providerName
        self.relevance = relevance
        self.latencyMs = latencyMs
        self.memoryMB = memoryMB
        self.tokenCostUSD = tokenCostUSD
        self.energyWh = energyWh
        self.answer = answer
    }
}

// MARK: - Scoring policy (BR-22)

/// Weights applied when converting raw measurements into a cost-adjusted
/// score. Values are per-unit cost penalties subtracted from relevance.
public struct BattleScoringPolicy: Sendable, Equatable, Codable {
    /// Cost penalty per USD of token spend.
    public let costWeightUSD: Double
    /// Cost penalty per watt-hour of on-device energy.
    public let energyWeightPerWh: Double

    public init(costWeightUSD: Double = 5.0, energyWeightPerWh: Double = 0.01) {
        self.costWeightUSD = costWeightUSD
        self.energyWeightPerWh = energyWeightPerWh
    }

    public static let `default` = BattleScoringPolicy()

    /// Cost-adjusted score for a single provider measurement.
    public func score(_ m: ProviderMeasurement) -> Double {
        m.relevance
            - costWeightUSD * m.tokenCostUSD
            - energyWeightPerWh * m.energyWh
    }
}

// MARK: - Winner

public enum BattleWinner: String, Sendable, Equatable, Codable {
    case providerA
    case providerB
    case draw
}

// MARK: - BattleMetrics

/// Full metric bundle recorded for a single battle (one task, two providers).
public struct BattleMetrics: Sendable, Equatable, Codable {

    public let battleId: String
    public let taskId: String
    public let taskCategory: String

    public let providerA: ProviderMeasurement
    public let providerB: ProviderMeasurement

    public let winner: BattleWinner
    public let scoreA: Double
    public let scoreB: Double
    public let policy: BattleScoringPolicy

    /// Arena overhead (wall-clock time in ms spent inside the arena runner
    /// minus the two provider calls). Gate target: < 50 ms (spec Phase 3).
    public let overheadMs: Int

    public let timestamp: Date

    public init(
        battleId: String,
        taskId: String,
        taskCategory: String,
        providerA: ProviderMeasurement,
        providerB: ProviderMeasurement,
        winner: BattleWinner,
        scoreA: Double,
        scoreB: Double,
        policy: BattleScoringPolicy,
        overheadMs: Int,
        timestamp: Date
    ) {
        self.battleId = battleId
        self.taskId = taskId
        self.taskCategory = taskCategory
        self.providerA = providerA
        self.providerB = providerB
        self.winner = winner
        self.scoreA = scoreA
        self.scoreB = scoreB
        self.policy = policy
        self.overheadMs = overheadMs
        self.timestamp = timestamp
    }

    // MARK: Winner selection (BR-22)

    /// Compute the winner under a scoring policy. `draw` on ties.
    public static func decideWinner(
        a: ProviderMeasurement,
        b: ProviderMeasurement,
        policy: BattleScoringPolicy = .default
    ) -> BattleDecision {
        let sa = policy.score(a)
        let sb = policy.score(b)
        if sa > sb { return BattleDecision(winner: .providerA, scoreA: sa, scoreB: sb) }
        if sb > sa { return BattleDecision(winner: .providerB, scoreA: sa, scoreB: sb) }
        return BattleDecision(winner: .draw, scoreA: sa, scoreB: sb)
    }
}

/// The outcome of one scoring comparison.
///
/// A named type rather than the `(winner:scoreA:scoreB:)` tuple this used to
/// return: three members is past SwiftLint's `large_tuple` cap, and the two
/// scores are easy to transpose at a call site when they are positional.
public struct BattleDecision: Sendable, Equatable {
    public let winner: BattleWinner
    public let scoreA: Double
    public let scoreB: Double

    public init(winner: BattleWinner, scoreA: Double, scoreB: Double) {
        self.winner = winner
        self.scoreA = scoreA
        self.scoreB = scoreB
    }
}
