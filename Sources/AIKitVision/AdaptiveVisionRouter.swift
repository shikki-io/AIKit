// MARK: - AdaptiveVisionRouter
// Spec §5 Phase 3 deliverable 6 — self-improvement loop.
//
// Tracks per-task-category win rate across battles and, once a category has
// accumulated at least `minBattlesForDecision` rounds AND the winner holds a
// lead of at least `decisionThreshold`, routes future requests in that
// category to the winning provider.
//
// "Winner" uses BR-22 cost-adjusted scoring (already materialised in
// `BattleMetrics.winner`), NOT raw accuracy. The router is a read-side
// projection over a stream of `BattleMetrics` records — it does not run
// battles itself.

import Foundation

// MARK: - Route decision

public enum VisionRoute: Equatable, Sendable {
    /// Not enough data yet — caller should fall back to its default provider.
    case undecided
    /// Category has accumulated a clear winner; caller should prefer A.
    case providerA
    /// Category has accumulated a clear winner; caller should prefer B.
    case providerB
}

// MARK: - Category stats

public struct CategoryWinRate: Sendable, Equatable, Codable {
    public let category: String
    public let totalBattles: Int
    public let winsA: Int
    public let winsB: Int
    public let draws: Int
    /// Mean cost-adjusted score for provider A over the category's battles.
    public let meanScoreA: Double
    public let meanScoreB: Double

    public var winRateA: Double {
        totalBattles == 0 ? 0 : Double(winsA) / Double(totalBattles)
    }

    public var winRateB: Double {
        totalBattles == 0 ? 0 : Double(winsB) / Double(totalBattles)
    }

    public init(
        category: String,
        totalBattles: Int,
        winsA: Int,
        winsB: Int,
        draws: Int,
        meanScoreA: Double,
        meanScoreB: Double
    ) {
        self.category = category
        self.totalBattles = totalBattles
        self.winsA = winsA
        self.winsB = winsB
        self.draws = draws
        self.meanScoreA = meanScoreA
        self.meanScoreB = meanScoreB
    }
}

// MARK: - Router

/// Threshold-based adaptive router over battle outcomes.
///
/// Thread-safe — all mutation goes through an `NSLock`-guarded internal
/// state. The router is `@unchecked Sendable` for the same reason as
/// `EmbeddingStore`: synchronous locking keeps call sites straightforward
/// and matches the existing workspace pattern.
public final class AdaptiveVisionRouter: @unchecked Sendable {

    // MARK: Config

    /// Minimum total battles per category before the router is willing to
    /// hand down a verdict. Defaults to 3 — low enough to stay responsive
    /// in unit tests and the Phase 3 gate; production tuning lives on the
    /// caller.
    public let minBattlesForDecision: Int

    /// Minimum win-rate delta (A minus B, or B minus A) for the router to
    /// emit a non-`undecided` route. Default 0.3 — 65% vs 35%. Lower values
    /// make the router more eager, higher values more conservative.
    public let decisionThreshold: Double

    // MARK: State

    private let lock = NSLock()
    private var perCategory: [String: CategoryAccumulator] = [:]

    // MARK: Init

    public init(minBattlesForDecision: Int = 3, decisionThreshold: Double = 0.3) {
        self.minBattlesForDecision = minBattlesForDecision
        self.decisionThreshold = decisionThreshold
    }

    // MARK: Record

    /// Record a single battle outcome into the per-category accumulator.
    public func record(_ metrics: BattleMetrics) {
        lock.lock(); defer { lock.unlock() }
        var acc = perCategory[metrics.taskCategory, default: CategoryAccumulator()]
        acc.add(metrics)
        perCategory[metrics.taskCategory] = acc
    }

    /// Record a batch at once. Convenience for the Phase 3 gate test that
    /// replays a corpus.
    public func recordAll(_ batch: [BattleMetrics]) {
        for m in batch { record(m) }
    }

    // MARK: Query

    /// Current route for the given task category.
    public func route(for category: String) -> VisionRoute {
        lock.lock(); defer { lock.unlock() }
        guard let acc = perCategory[category],
              acc.totalBattles >= minBattlesForDecision
        else {
            return .undecided
        }
        let delta = acc.winRateA - acc.winRateB
        if delta >= decisionThreshold { return .providerA }
        if -delta >= decisionThreshold { return .providerB }
        return .undecided
    }

    /// Snapshot of the per-category win rates. Useful for `shi vision battles`
    /// summary output and the Phase 3 gate assertion.
    public func winRates() -> [CategoryWinRate] {
        lock.lock(); defer { lock.unlock() }
        return perCategory
            .map { $0.value.snapshot(category: $0.key) }
            .sorted { $0.category < $1.category }
    }

    public func winRate(for category: String) -> CategoryWinRate? {
        lock.lock(); defer { lock.unlock() }
        guard let acc = perCategory[category] else { return nil }
        return acc.snapshot(category: category)
    }
}

// MARK: - Internal accumulator

private struct CategoryAccumulator {
    var totalBattles: Int = 0
    var winsA: Int = 0
    var winsB: Int = 0
    var draws: Int = 0
    var sumScoreA: Double = 0
    var sumScoreB: Double = 0

    var winRateA: Double {
        totalBattles == 0 ? 0 : Double(winsA) / Double(totalBattles)
    }

    var winRateB: Double {
        totalBattles == 0 ? 0 : Double(winsB) / Double(totalBattles)
    }

    mutating func add(_ m: BattleMetrics) {
        totalBattles += 1
        sumScoreA += m.scoreA
        sumScoreB += m.scoreB
        switch m.winner {
        case .providerA: winsA += 1
        case .providerB: winsB += 1
        case .draw:      draws  += 1
        }
    }

    func snapshot(category: String) -> CategoryWinRate {
        let meanA = totalBattles == 0 ? 0 : sumScoreA / Double(totalBattles)
        let meanB = totalBattles == 0 ? 0 : sumScoreB / Double(totalBattles)
        return CategoryWinRate(
            category: category,
            totalBattles: totalBattles,
            winsA: winsA,
            winsB: winsB,
            draws: draws,
            meanScoreA: meanA,
            meanScoreB: meanB
        )
    }
}
