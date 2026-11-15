import AIKitTestSupport
import Foundation
import Testing

@testable import AIKit

// MARK: - ProviderRouterFanOutTests (T-AIL-59)
//
// W32 of spec ai-layer-aikit-core-2026-10-31, BR-AIL-22: *"fans one request out
// to N providers in parallel and gets every answer back"*. Four properties, one
// suite:
//
//   * every provider is asked AT ONCE, under ONE deadline;
//   * a slow provider times out without holding up the rest;
//   * cancelling the caller stops every branch;
//   * outcomes keep the order the providers were GIVEN, and each names its
//     provider.
//
// The parallelism proof is a CONCURRENCY COUNTER, not a stopwatch. A test that
// asserts "N calls finished faster than N × delay" is measuring this Mac under
// load — the failure W17 of this spec spent a wave fixing — so each stub here
// registers its arrival and waits for the others: the peak in-flight count can
// only reach N if the branches really did overlap, and that assertion does not
// move when the machine is busy.
//
// No provider is real (BR-AIL-12).

@Suite("ProviderRouter — one request, N providers, in parallel")
struct ProviderRouterFanOutTests {

    // MARK: - Fixtures

    private func request(model: ModelIdentifier? = nil) -> AIRequest {
        AIRequest(
            messages: [AIMessage(role: .user, content: "ask everyone")],
            model: model
        )
    }

    private func router() -> ProviderRouter {
        // A fan-out takes its providers as an argument: the caller has usually
        // just built the list with `candidates(for:policy:)`.
        ProviderRouter(providers: [])
    }

    /// A budget no stub in this suite needs, for the cases where the deadline
    /// is not what is under test.
    private let generous: Duration = .seconds(30)

    /// Whether this branch ended at the shared deadline.
    ///
    /// Matched on the CASE rather than compared to `.timedOut(budget: x)`: the
    /// budget a branch reports is measured inside `fanOut`, microseconds after
    /// the caller named it, so an equality against the caller's literal would
    /// be asserting that two clock reads returned the same instant.
    private static func timedOut(_ outcome: AIProviderOutcome) -> Bool {
        if case .timedOut = outcome.verdict { return true }
        return false
    }

    /// The budget a timed-out branch reported, for the cases that compare the
    /// figure across branches.
    private static func reportedBudget(_ outcome: AIProviderOutcome) -> Duration? {
        guard case .timedOut(let budget) = outcome.verdict else { return nil }
        return budget
    }

    // MARK: - All at once, under one deadline

    @Test("every provider is asked at the same time, not one after another")
    func everyProviderIsAskedInParallel() async throws {
        let party = 4
        let gate = InFlightGate(party: party)
        let providers = (0..<party).map { index in
            ParallelProbeProvider(id: "probe-\(index)", gate: gate)
        }

        let outcomes = try await router().fanOut(
            request(), to: providers, within: generous)

        // Each stub returns only once the gate has seen all four in flight, so
        // four answers is itself the proof: a serial dispatch could never get
        // past the first one.
        #expect(outcomes.count == party)
        #expect(outcomes.allSatisfy { $0.isAnswer })
        #expect(await gate.peak == party)
        #expect(outcomes.map(\.providerID) == providers.map(\.id))
    }

    @Test("one deadline covers the whole fan-out, not one deadline each")
    func oneDeadlineIsSharedByEveryBranch() async throws {
        // FOUR providers that never answer, under a 500 ms budget. One shared
        // deadline costs 500 ms whatever N is; spending the budget per branch
        // in turn would cost 2 s. The bound is 1.5 s — comfortably above the
        // shared figure and below the serial one.
        //
        // The structural assertion below is the one that does not move when
        // this Mac is busy: four branches reporting ONE budget figure can only
        // come from one measurement. The stopwatch is the second opinion, with
        // 3× slack, because every branch here is asleep and costs no CPU.
        let budget = Duration.milliseconds(500)
        let providers = (0..<4).map { NeverAnsweringProvider(id: "stuck-\($0)") }

        let started = ContinuousClock.now
        let outcomes = try await router().fanOut(request(), to: providers, within: budget)
        let elapsed = ContinuousClock.now - started

        #expect(outcomes.count == 4)
        #expect(outcomes.allSatisfy { Self.timedOut($0) })
        #expect(Set(outcomes.compactMap(Self.reportedBudget)).count == 1)
        #expect(elapsed < .milliseconds(1500), "elapsed \(elapsed) for a \(budget) budget")
    }

    @Test("a slow provider times out and the fast ones still answer")
    func aSlowProviderDoesNotBlockTheRest() async throws {
        // The budget is 2 s and the slow provider needs 60: the deadline has to
        // arrive AFTER two mock calls and long before the slow one, and the
        // suite runs 700-odd tests in parallel, so the window for "a mock got
        // scheduled" is seconds rather than the 300 ms a first draft used — a
        // tighter budget measures how busy this machine is, which is the
        // mistake W17 of this spec spent a wave undoing.
        let slow = DelayedProvider(id: "cold-mlx", delay: .seconds(60))
        let fastA = MockAIProvider(id: "lmstudio-local", capabilities: .textGeneration)
        let fastB = MockAIProvider(id: "anthropic-cloud", capabilities: .textGeneration)

        let started = ContinuousClock.now
        let outcomes = try await router().fanOut(
            request(), to: [slow, fastA, fastB], within: .seconds(2))
        let elapsed = ContinuousClock.now - started

        #expect(outcomes.map(\.providerID) == ["cold-mlx", "lmstudio-local", "anthropic-cloud"])
        #expect(Self.timedOut(outcomes[0]))
        #expect(outcomes[1].isAnswer)
        #expect(outcomes[2].isAnswer)
        // A timed-out branch still reads as evidence rather than as silence.
        #expect(outcomes[0].cause?.contains("did not answer") == true)
        // And the fan-out did not wait for the 60 s provider.
        #expect(elapsed < .seconds(10), "elapsed \(elapsed)")
    }

    @Test("a deadline already spent asks nobody")
    func aSpentBudgetAsksNobody() async throws {
        let provider = MockAIProvider(id: "never-asked", capabilities: .textGeneration)

        let outcomes = try await router().fanOut(
            request(),
            to: [provider],
            deadline: ContinuousClock.now.advanced(by: .seconds(-1))
        )

        #expect(outcomes.count == 1)
        #expect(!outcomes[0].isAnswer)
        // The point: a caller that passed a spent budget sees that it was
        // spent, instead of N requests leaving the host anyway.
        #expect(provider.completeCallCount == 0)
    }

    @Test("an empty fan-out asks nobody and reports no failure")
    func anEmptyFanOutIsNotAFailure() async throws {
        let outcomes = try await router().fanOut(request(), to: [], within: generous)

        // Nothing was asked, so nothing failed — the same distinction
        // `AIProviderAttempt.summary` makes for an empty fallback list.
        #expect(outcomes.isEmpty)
    }

    // MARK: - The order is the caller's

    @Test("outcomes keep the order the providers were given, not the order they finished")
    func outcomesKeepTheInputOrder() async throws {
        let finishes = FinishLog()
        // Given slowest first, so completion order is the reverse of input
        // order and a method that appended results as they arrived would fail
        // this.
        let providers = [
            DelayedProvider(id: "slowest", delay: .milliseconds(240), finishes: finishes),
            DelayedProvider(id: "middle", delay: .milliseconds(120), finishes: finishes),
            DelayedProvider(id: "fastest", delay: .milliseconds(10), finishes: finishes),
        ]

        let outcomes = try await router().fanOut(request(), to: providers, within: generous)

        #expect(outcomes.map(\.providerID) == ["slowest", "middle", "fastest"])
        #expect(outcomes.allSatisfy { $0.isAnswer })
        // The finishing order really was different: the 240 ms branch finished
        // last. Asserted on the slowest alone rather than on the whole
        // sequence, so a busy machine reordering the two short branches does
        // not fail a test about the result order.
        #expect(await finishes.order.last == "slowest")
    }

    @Test("the same provider asked twice gets two outcomes, both in place")
    func duplicateProvidersEachGetTheirOwnOutcome() async throws {
        // Indexing by position, not by id: a caller comparing one provider at
        // two temperatures passes it twice, and a result keyed by id would
        // collapse the two.
        let provider = MockAIProvider(id: "twice", capabilities: .textGeneration)

        let outcomes = try await router().fanOut(
            request(), to: [provider, provider], within: generous)

        #expect(outcomes.map(\.providerID) == ["twice", "twice"])
        #expect(provider.completeCallCount == 2)
    }

    // MARK: - Every outcome names its provider (BR-AIL-06)

    @Test("each answer names its provider, its model and its latency")
    func eachOutcomeNamesItsProviderAndModel() async throws {
        let local = MockAIProvider(
            id: "mlx/local",
            capabilities: .textGeneration,
            responses: [
                AIResponse(
                    content: "from the device",
                    model: "gemma-4-4b",
                    tokensUsed: TokenUsage(prompt: 2, completion: 3),
                    latencyMs: 9,
                    providerID: "mlx/local"
                ),
            ]
        )
        let cloud = MockAIProvider(
            id: "anthropic-cloud",
            capabilities: .textGeneration,
            responses: [
                AIResponse(
                    content: "from the server",
                    model: "claude-sonnet-5-5",
                    tokensUsed: TokenUsage(prompt: 2, completion: 4),
                    latencyMs: 410,
                    providerID: "anthropic-cloud"
                ),
            ]
        )

        let outcomes = try await router().fanOut(
            request(), to: [local, cloud], within: generous)

        #expect(outcomes.map(\.providerID) == ["mlx/local", "anthropic-cloud"])
        #expect(outcomes.map(\.model) == ["gemma-4-4b", "claude-sonnet-5-5"])
        #expect(outcomes.compactMap(\.response).map(\.content) == ["from the device", "from the server"])
        #expect(outcomes.allSatisfy { $0.latency >= .zero })
        // The log line a caller prints names who answered.
        #expect(outcomes[0].description.contains("mlx/local"))
        #expect(outcomes[0].description.contains("gemma-4-4b"))
    }

    @Test("a branch that never answered still names the model it was asked for")
    func aFailedBranchNamesTheRequestedModel() async throws {
        let stuck = NeverAnsweringProvider(id: "stuck")

        let outcomes = try await router().fanOut(
            request(model: ModelIdentifier(provider: "lmstudio", modelId: "devstral-small-2507")),
            to: [stuck],
            within: .milliseconds(120)
        )

        #expect(outcomes[0].model == "lmstudio/devstral-small-2507")
        #expect(outcomes[0].response == nil)
    }

    @Test("a branch with no answer and no requested model names no model rather than guessing")
    func aFailedBranchWithNoModelNamesNone() async throws {
        let stuck = NeverAnsweringProvider(id: "stuck")

        let outcomes = try await router().fanOut(
            request(), to: [stuck], within: .milliseconds(120))

        #expect(outcomes[0].model == nil)
    }

    // MARK: - Every failure keeps its cause (BR-AIL-07)

    @Test("a provider that throws keeps its cause, and its type when AIKit owns it")
    func aFailureKeepsItsCauseAndItsType() async throws {
        let typed = MockAIProvider(id: "mlx/local", capabilities: .textGeneration)
        typed.shouldThrow = AIKitError.engineUnavailable("MLX requires macOS on Apple Silicon")
        let foreign = MockAIProvider(id: "lmstudio-local", capabilities: .textGeneration)
        foreign.shouldThrow = URLError(.cannotConnectToHost)

        let outcomes = try await router().fanOut(
            request(), to: [typed, foreign], within: generous)

        #expect(outcomes[0].error == .engineUnavailable("MLX requires macOS on Apple Silicon"))
        #expect(outcomes[0].cause?.contains("MLX requires macOS") == true)
        // A URLError is not an AIKitError, so the typed slot is empty — and the
        // sentence is still there, which is what an operator reads.
        #expect(outcomes[1].error == nil)
        #expect(outcomes[1].cause?.isEmpty == false)
    }

    @Test("a fan-out where nobody answered builds one error naming every provider")
    func everyBranchFailingBuildsOneAllProvidersFailed() async throws {
        // The bridge to the evidence vocabulary the kit already speaks
        // (BR-AIL-07): a caller does not invent a second way to report "nobody
        // answered".
        let refused = MockAIProvider(id: "lmstudio-local", capabilities: .textGeneration)
        refused.shouldThrow = AIKitError.requestFailed("connection refused")
        let limited = MockAIProvider(id: "openai-cloud", capabilities: .textGeneration)
        limited.shouldThrow = AIKitError.requestFailed("HTTP 429")

        let outcomes = try await router().fanOut(
            request(), to: [refused, limited], within: generous)

        #expect(!outcomes.contains { $0.isAnswer })
        let error = AIKitError.allProvidersFailed(outcomes.compactMap(\.attempt))
        let message = error.errorDescription ?? ""
        #expect(message.contains("lmstudio-local"))
        #expect(message.contains("openai-cloud"))
        #expect(message.contains("2 provider(s) failed"))
    }

    @Test("an answered branch has no attempt to report")
    func anAnsweringBranchIsNotAnAttempt() async throws {
        let answering = MockAIProvider(id: "answers", capabilities: .textGeneration)

        let outcomes = try await router().fanOut(request(), to: [answering], within: generous)

        #expect(outcomes[0].attempt == nil)
        #expect(outcomes[0].cause == nil)
    }

    // MARK: - Cancellation stops every branch

    @Test("cancelling the caller stops every branch and throws CancellationError")
    func cancellationStopsEveryBranch() async throws {
        let observed = CancellationLog()
        let providers = (0..<3).map {
            CancellationObservingProvider(id: "watcher-\($0)", log: observed)
        }
        let router = router()
        let requested = request()

        let task = Task {
            try await router.fanOut(requested, to: providers, within: .seconds(30))
        }

        // Cancel only once every branch has actually started, so this asserts
        // cancellation DURING the fan-out rather than before it.
        try await observed.waitUntilStarted(3)
        task.cancel()

        await #expect(throws: CancellationError.self) { try await task.value }
        // Every branch saw it, not just the first: that is what structured
        // concurrency buys and what an unstructured `Task {}` per provider
        // would not.
        #expect(await observed.cancelled.count == 3)
    }

    @Test("a caller already cancelled when the fan-out starts asks nobody")
    func alreadyCancelledAsksNobody() async throws {
        let provider = MockAIProvider(id: "never-asked", capabilities: .textGeneration)
        let router = router()
        let requested = request()
        let gate = GoSignal()

        let task = Task {
            // Held at a NON-throwing gate until the test has cancelled, so the
            // cancellation is certainly in force by the time `fanOut` is
            // entered. A `Task.sleep` here would throw first and the test would
            // prove nothing about `fanOut`'s own check.
            await gate.waitForGo()
            return try await router.fanOut(requested, to: [provider], within: .seconds(30))
        }
        task.cancel()
        await gate.go()

        await #expect(throws: CancellationError.self) { try await task.value }
        #expect(provider.completeCallCount == 0)
    }
}

// MARK: - Test doubles

/// A barrier that counts how many branches were in flight at once.
///
/// `peak` is the whole proof of parallelism: it can only reach `party` if every
/// branch was inside `complete` at the same moment.
private actor InFlightGate {
    private let party: Int
    private var current = 0
    private(set) var peak = 0

    init(party: Int) {
        self.party = party
    }

    func enter() {
        current += 1
        peak = max(peak, current)
    }

    func leave() {
        current -= 1
    }

    var allArrived: Bool { peak >= party }
}

/// Answers only once every sibling has arrived.
///
/// Polling rather than a continuation: a `withCheckedContinuation` that nobody
/// resumes would hang the suite forever when the property under test is broken,
/// and a test that hangs reports nothing. This one gives up after a bounded
/// number of polls and lets the assertion fail with a readable `peak`.
private struct ParallelProbeProvider: AIProvider {
    let id: String
    let gate: InFlightGate

    var displayName: String { id }
    var capabilities: AICapabilities { .textGeneration }
    var status: AIProviderStatus { get async { .ready } }

    func complete(request: AIRequest) async throws -> AIResponse {
        await gate.enter()
        var polls = 0
        while polls < 600 {
            if await gate.allArrived { break }
            try await Task.sleep(for: .milliseconds(5))
            polls += 1
        }
        await gate.leave()
        return AIResponse(
            content: "arrived with the others",
            model: "fixture",
            tokensUsed: TokenUsage(prompt: 1, completion: 1),
            latencyMs: 1,
            providerID: id
        )
    }
}

/// Records which providers finished, in the order they did.
private actor FinishLog {
    private(set) var order: [String] = []

    func record(_ id: String) { order.append(id) }
}

/// Answers after a delay. The delay is a `Task.sleep`, so it honours
/// cancellation the way every provider in this package does.
private struct DelayedProvider: AIProvider {
    let id: String
    let delay: Duration
    var finishes: FinishLog?

    var displayName: String { id }
    var capabilities: AICapabilities { .textGeneration }
    var status: AIProviderStatus { get async { .ready } }

    func complete(request: AIRequest) async throws -> AIResponse {
        try await Task.sleep(for: delay)
        await finishes?.record(id)
        return AIResponse(
            content: "answered after \(delay)",
            model: "fixture",
            tokensUsed: TokenUsage(prompt: 1, completion: 1),
            latencyMs: 1,
            providerID: id
        )
    }
}

/// Never answers inside any budget a test sets.
private struct NeverAnsweringProvider: AIProvider {
    let id: String

    var displayName: String { id }
    var capabilities: AICapabilities { .textGeneration }
    var status: AIProviderStatus { get async { .ready } }

    func complete(request: AIRequest) async throws -> AIResponse {
        try await Task.sleep(for: .seconds(600))
        return AIResponse(
            content: "unreachable",
            model: "fixture",
            tokensUsed: TokenUsage(prompt: 1, completion: 1),
            latencyMs: 1,
            providerID: id
        )
    }
}

/// A gate a cancelled task can still wait at.
///
/// `Task.yield()` does not throw when the task is cancelled, which is the whole
/// point: the waiting task stays parked until the test releases it and THEN
/// runs the code under test with the cancellation already in force.
private actor GoSignal {
    private var released = false

    func go() { released = true }

    func waitForGo() async {
        // Bounded, so a mistake in this helper fails the assertion instead of
        // hanging the suite.
        var spins = 0
        while !released, spins < 100_000 {
            await Task.yield()
            spins += 1
        }
    }
}

/// Who started, and who saw the cancellation.
private actor CancellationLog {
    private(set) var started: [String] = []
    private(set) var cancelled: [String] = []

    func recordStart(_ id: String) { started.append(id) }
    func recordCancellation(_ id: String) { cancelled.append(id) }

    /// Waits until `count` branches have started, so a test cancels DURING the
    /// fan-out. Bounded, so a broken fan-out fails rather than hangs.
    func waitUntilStarted(_ count: Int) async throws {
        var polls = 0
        while started.count < count, polls < 600 {
            try await Task.sleep(for: .milliseconds(5))
            polls += 1
        }
    }
}

/// Reports whether its own branch was cancelled.
private struct CancellationObservingProvider: AIProvider {
    let id: String
    let log: CancellationLog

    var displayName: String { id }
    var capabilities: AICapabilities { .textGeneration }
    var status: AIProviderStatus { get async { .ready } }

    func complete(request: AIRequest) async throws -> AIResponse {
        await log.recordStart(id)
        do {
            try await Task.sleep(for: .seconds(600))
        } catch {
            await log.recordCancellation(id)
            throw error
        }
        return AIResponse(
            content: "unreachable",
            model: "fixture",
            tokensUsed: TokenUsage(prompt: 1, completion: 1),
            latencyMs: 1,
            providerID: id
        )
    }
}
