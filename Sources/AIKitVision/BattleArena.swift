// MARK: - BattleArena
// Auto-battle evaluation framework (spec §5 Phase 3 deliverables 1, 3, 4, 5).
//
// The arena runs a single evaluation task through two `VisionProvider`
// implementations, collects `ProviderMeasurement` for each, picks a winner
// under the `BattleScoringPolicy` (cost-adjusted, BR-22), and returns a
// `BattleMetrics` bundle.
//
// Design constraints:
//   - Provider-agnostic (AD-06): the arena is generic over any two
//     `VisionProvider` values and never references specific providers
//     by name. This keeps V-JEPA vs any-LLM, DINOv2 vs V-JEPA, etc.
//     all expressible through the same arena.
//   - Cost-per-task telemetry (BR-22): every measurement includes a
//     `tokenCostUSD` and `energyWh` field so the adaptive router can pick
//     the cost-efficient winner, not the raw-accuracy winner.
//   - Overhead bounded (< 50 ms / Phase 3 gate): the arena's own bookkeeping
//     is timed and reported on every metrics bundle.
//
// Ground truth scoring is delegated to a pluggable `RelevanceJudge`. Default
// is a simple label-match judge that returns 1.0 on exact match (case
// insensitive), partial overlap otherwise. Benchmarks can swap in a
// stricter judge per task category.

import Foundation

// MARK: - Task + context

/// One evaluation task fed into the arena.
///
/// `videoURL` may be `nil` for tasks driven by synthetic frames (unit tests
/// and the embedded 10-task starter corpus).
public struct BattleTask: Sendable, Equatable {
    public let taskId: String
    public let category: String
    public let question: String
    public let groundTruthLabel: String
    public let videoURL: URL?
    public let frames: VideoFrameTensor?

    public init(
        taskId: String,
        category: String,
        question: String,
        groundTruthLabel: String,
        videoURL: URL? = nil,
        frames: VideoFrameTensor? = nil
    ) {
        self.taskId = taskId
        self.category = category
        self.question = question
        self.groundTruthLabel = groundTruthLabel
        self.videoURL = videoURL
        self.frames = frames
    }
}

// MARK: - VisionProvider protocol (AD-06)

/// Contract every provider must honour to participate in the battle arena.
///
/// Providers are async because real implementations may hit the network
/// (LLM) or disk (load weights). The arena wraps every call in a latency
/// timer and a relevance judge.
public protocol VisionProvider: Sendable {
    /// Human-readable identifier (e.g. "vjepa-vitb16", "mock-llm-haiku").
    var name: String { get }

    /// Run the provider against a single task, returning the raw answer
    /// plus billing telemetry. The arena decorates this with relevance
    /// scoring and overhead bookkeeping.
    func answer(task: BattleTask) async throws -> ProviderAnswer
}

/// Raw answer + billing telemetry returned by a provider.
public struct ProviderAnswer: Sendable, Equatable, Codable {
    public let label: String
    public let tokenCostUSD: Double
    public let energyWh: Double
    public let memoryMB: Int

    public init(label: String, tokenCostUSD: Double, energyWh: Double, memoryMB: Int) {
        self.label = label
        self.tokenCostUSD = tokenCostUSD
        self.energyWh = energyWh
        self.memoryMB = memoryMB
    }
}

// MARK: - Relevance judge

/// Scores a provider's raw answer against the task's ground truth label.
public protocol RelevanceJudge: Sendable {
    func score(answer: String, groundTruth: String) -> Double
}

/// Default judge: case-insensitive trimmed exact match → 1.0,
/// substring match → 0.5, otherwise 0. Keeps the arena deterministic in CI.
public struct LabelMatchJudge: RelevanceJudge {
    public init() {}

    public func score(answer: String, groundTruth: String) -> Double {
        let a = answer.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let g = groundTruth.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !a.isEmpty, !g.isEmpty else { return 0 }
        if a == g { return 1.0 }
        if a.contains(g) || g.contains(a) { return 0.5 }
        return 0
    }
}

// MARK: - Arena

/// Runs two providers on the same task and records metrics.
public struct BattleArena: Sendable {

    public let providerA: any VisionProvider
    public let providerB: any VisionProvider
    public let judge: any RelevanceJudge
    public let policy: BattleScoringPolicy

    public init(
        providerA: any VisionProvider,
        providerB: any VisionProvider,
        judge: any RelevanceJudge = LabelMatchJudge(),
        policy: BattleScoringPolicy = .default
    ) {
        self.providerA = providerA
        self.providerB = providerB
        self.judge = judge
        self.policy = policy
    }

    /// Run a single battle. Emits a fully-populated `BattleMetrics` bundle
    /// including cost telemetry (BR-22) and arena overhead (Phase 3 gate).
    public func run(task: BattleTask, now: Date = Date()) async throws -> BattleMetrics {
        let arenaStart = Date()

        // Provider A
        let aStart = Date()
        let aAnswer = try await providerA.answer(task: task)
        let aLatency = Int(Date().timeIntervalSince(aStart) * 1000)
        let aMeasurement = ProviderMeasurement(
            providerName: providerA.name,
            relevance: judge.score(answer: aAnswer.label, groundTruth: task.groundTruthLabel),
            latencyMs: aLatency,
            memoryMB: aAnswer.memoryMB,
            tokenCostUSD: aAnswer.tokenCostUSD,
            energyWh: aAnswer.energyWh,
            answer: aAnswer.label
        )

        // Provider B
        let bStart = Date()
        let bAnswer = try await providerB.answer(task: task)
        let bLatency = Int(Date().timeIntervalSince(bStart) * 1000)
        let bMeasurement = ProviderMeasurement(
            providerName: providerB.name,
            relevance: judge.score(answer: bAnswer.label, groundTruth: task.groundTruthLabel),
            latencyMs: bLatency,
            memoryMB: bAnswer.memoryMB,
            tokenCostUSD: bAnswer.tokenCostUSD,
            energyWh: bAnswer.energyWh,
            answer: bAnswer.label
        )

        let decision = BattleMetrics.decideWinner(a: aMeasurement, b: bMeasurement, policy: policy)

        // Arena overhead = total wall-clock minus both provider latencies.
        let totalMs = Int(Date().timeIntervalSince(arenaStart) * 1000)
        let overheadMs = max(0, totalMs - aLatency - bLatency)

        return BattleMetrics(
            battleId: UUID().uuidString,
            taskId: task.taskId,
            taskCategory: task.category,
            providerA: aMeasurement,
            providerB: bMeasurement,
            winner: decision.winner,
            scoreA: decision.scoreA,
            scoreB: decision.scoreB,
            policy: policy,
            overheadMs: overheadMs,
            timestamp: now
        )
    }

    /// Convenience: run a sequence of tasks and return the metrics list.
    public func runAll(_ tasks: [BattleTask], now: Date = Date()) async throws -> [BattleMetrics] {
        var out: [BattleMetrics] = []
        out.reserveCapacity(tasks.count)
        for task in tasks {
            out.append(try await run(task: task, now: now))
        }
        return out
    }
}
