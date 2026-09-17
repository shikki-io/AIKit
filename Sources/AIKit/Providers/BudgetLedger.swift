import Foundation

// MARK: - BudgetLedger
//
// W4 of spec `provider-kind-cli-subprocess`.
//
// A single actor that fuses two orthogonal admission-control concerns for the
// `CLISubprocessProvider` (and any other provider that plugs into it):
//
//   1. **Concurrency cap** — per-`AIProviderID` in-flight subprocess count.
//      An external subprocess (`claude -p`, `codex …`, custom scripts) is
//      expensive; unbounded fan-out will fork-bomb the operator's laptop.
//      Callers `await ledger.acquireSlot(for:)` to obtain a `SlotPermit`;
//      the ledger suspends further callers when the cap is reached and
//      resumes them (FIFO) as permits are released.
//
//   2. **Budget deduction** — per-`AIProviderID` token spend against the
//      caller's hard-stop limits. Callers `deduct(providerID:totalTokens:)` after
//      a successful run; the ledger tracks cumulative spend and answers
//      `remainingTokens(for:)` / `isExhausted(for:)`. On acquisition, the
//      ledger refuses to hand out a fresh permit for a provider that is
//      already at or beyond its hard-stop threshold — fail fast rather
//      than launch a subprocess we can't afford to record.
//
// The ledger DOES NOT persist and holds no configuration of its own: it is a
// session-scoped admission gate over limits the caller resolved. In shikki the
// durable month-to-date record stays with `AIDispatchStore` / `TokenBudgetWatchdog`,
// which also own loading `ai-budgets.toml`. A caller that wants both calls both.
//
// Sendable is trivially satisfied: the type is an `actor`; all state is
// isolated to the actor's serial executor.

public actor BudgetLedger {

    // MARK: - Errors

    public enum LedgerError: Error, Sendable, Equatable {
        /// The provider's cumulative spend is at or beyond the hard-stop
        /// threshold of its `AIBudgetEntry`. Callers should route the task
        /// to a fallback provider (typically the local one).
        case budgetExhausted(provider: AIProviderID, spent: Int, cap: Int)
    }

    // MARK: - SlotPermit

    /// Opaque handle proving the bearer holds a live concurrency slot on the
    /// ledger. Return it to `release(_:)` when the subprocess has exited
    /// (success or failure). The permit carries the `providerID` so the
    /// ledger can route the release back to the correct waiter queue.
    public struct SlotPermit: Sendable, Equatable {
        public let providerID: AIProviderID
        /// Monotonic id — lets tests assert individual permits without
        /// relying on address identity (which `struct` doesn't have).
        public let id: UInt64

        fileprivate init(providerID: AIProviderID, id: UInt64) {
            self.providerID = providerID
            self.id = id
        }
    }

    // MARK: - State

    /// Hard-stop token count per provider. A provider without an entry is
    /// unbounded. The caller resolves these from its own configuration —
    /// shikki applies `hardStopAtPct` to the monthly cap in `AIBudgetConfig`.
    private let hardStopTokens: [AIProviderID: Int]

    /// Per-provider concurrency cap. `nil` entry = unbounded parallelism for
    /// that provider (e.g. local models with cheap warm processes).
    private var concurrencyCaps: [AIProviderID: Int]

    /// Cumulative tokens spent in the current session, keyed by provider.
    private var spentTokens: [AIProviderID: Int] = [:]

    /// In-flight slot count, keyed by provider.
    private var activeSlots: [AIProviderID: Int] = [:]

    /// FIFO wait queues, keyed by provider. A `nil` entry means no waiters.
    private var waiters: [AIProviderID: [CheckedContinuation<Void, Never>]] = [:]

    /// Monotonic permit id counter. Used only for permit identity.
    private var nextPermitID: UInt64 = 0

    // MARK: - Init

    /// Create a ledger.
    /// - Parameters:
    ///   - budgetConfig: monthly-cap configuration. Defaults to `.default`
    ///     (no caps — all providers unbounded).
    ///   - concurrencyCaps: per-provider in-flight subprocess cap. Providers
    ///     omitted from this map are unbounded.
    public init(
        hardStopTokens: [AIProviderID: Int] = [:],
        concurrencyCaps: [AIProviderID: Int] = [:]
    ) {
        for (_, cap) in concurrencyCaps {
            precondition(cap > 0, "BudgetLedger concurrency cap must be > 0")
        }
        self.hardStopTokens = hardStopTokens
        self.concurrencyCaps = concurrencyCaps
    }

    // MARK: - Concurrency cap

    /// Acquire a concurrency slot for `providerID`.
    ///
    /// Suspends if the provider's in-flight count is at its cap; wakes in
    /// FIFO order as `release(_:)` frees a slot.
    ///
    /// Throws `LedgerError.budgetExhausted` immediately if the provider's
    /// cumulative spend is already at or beyond its hard-stop threshold — no
    /// slot is taken and no continuation is enqueued.
    public func acquireSlot(for providerID: AIProviderID) async throws -> SlotPermit {
        try checkNotExhausted(providerID)

        let cap = concurrencyCaps[providerID]
        let current = activeSlots[providerID, default: 0]

        if cap == nil || current < cap! {
            activeSlots[providerID] = current + 1
            return mintPermit(for: providerID)
        }

        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            waiters[providerID, default: []].append(continuation)
        }
        // Woken by `release(_:)`, which has already transferred the slot to us.
        return mintPermit(for: providerID)
    }

    /// Return a slot to the pool. If a waiter is queued, the slot is handed
    /// to the oldest waiter (their `withCheckedContinuation` resumes) and
    /// `activeSlots` stays the same. Otherwise `activeSlots` is decremented.
    public func release(_ permit: SlotPermit) {
        let providerID = permit.providerID
        if var queue = waiters[providerID], !queue.isEmpty {
            let next = queue.removeFirst()
            if queue.isEmpty {
                waiters.removeValue(forKey: providerID)
            } else {
                waiters[providerID] = queue
            }
            next.resume()
            return
        }
        let current = activeSlots[providerID, default: 0]
        if current <= 1 {
            activeSlots.removeValue(forKey: providerID)
        } else {
            activeSlots[providerID] = current - 1
        }
    }

    /// Convenience: run `body` while holding a slot for `providerID`, releasing
    /// even if `body` throws. Returns whatever `body` returns.
    public func withSlot<T: Sendable>(
        for providerID: AIProviderID,
        _ body: (SlotPermit) async throws -> T
    ) async throws -> T {
        let permit = try await acquireSlot(for: providerID)
        do {
            let value = try await body(permit)
            release(permit)
            return value
        } catch {
            release(permit)
            throw error
        }
    }

    // MARK: - Budget deduction

    /// Deduct `totalTokens` from `providerID`'s session spend.
    /// A failed dispatch still counts — the caller paid for the tokens either
    /// way. Callers that want to exclude failures pre-filter before calling.
    public func deduct(providerID: AIProviderID, totalTokens: Int) {
        spentTokens[providerID, default: 0] += totalTokens
    }

    /// Cumulative tokens spent this session for `providerID`.
    public func spent(for providerID: AIProviderID) -> Int {
        spentTokens[providerID, default: 0]
    }

    /// Remaining tokens before the hard-stop threshold. `nil` = unbounded
    /// (no entry for this provider).
    /// Returns `0` when spend has reached or crossed the hard-stop.
    public func remainingTokens(for providerID: AIProviderID) -> Int? {
        guard let hardStop = hardStopThreshold(for: providerID) else { return nil }
        let used = spentTokens[providerID, default: 0]
        return max(0, hardStop - used)
    }

    /// Whether the provider is at or past its hard-stop threshold.
    public func isExhausted(for providerID: AIProviderID) -> Bool {
        guard let hardStop = hardStopThreshold(for: providerID) else { return false }
        return spentTokens[providerID, default: 0] >= hardStop
    }

    /// Number of in-flight slots currently held for `providerID`.
    public func activeSlotCount(for providerID: AIProviderID) -> Int {
        activeSlots[providerID, default: 0]
    }

    /// Number of tasks currently suspended waiting for a slot for `providerID`.
    public func waiterCount(for providerID: AIProviderID) -> Int {
        waiters[providerID]?.count ?? 0
    }

    // MARK: - Private helpers

    private func mintPermit(for providerID: AIProviderID) -> SlotPermit {
        nextPermitID &+= 1
        return SlotPermit(providerID: providerID, id: nextPermitID)
    }

    private func checkNotExhausted(_ providerID: AIProviderID) throws {
        guard let hardStop = hardStopThreshold(for: providerID) else { return }
        let used = spentTokens[providerID, default: 0]
        if used >= hardStop {
            throw LedgerError.budgetExhausted(
                provider: providerID,
                spent: used,
                cap: hardStop
            )
        }
    }

    /// The effective hard-stop token count for `providerID`, or `nil` if
    /// unbounded.
    private func hardStopThreshold(for providerID: AIProviderID) -> Int? {
        hardStopTokens[providerID]
    }
}
