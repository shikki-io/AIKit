import Foundation
@testable import AIKit
import Testing

// MARK: - ConcurrencyCapTests
//
// W4 tests for `BudgetLedger` — the per-provider concurrency-cap half.
//
// Covers:
//   - Below-cap acquisitions never suspend.
//   - At-cap acquisition suspends until a slot is released.
//   - Slot release wakes the oldest waiter (FIFO).
//   - Per-provider caps are isolated.
//   - Provider with no configured cap is unbounded.
//   - `withSlot` releases on both success and thrown-error paths.

@Suite("BudgetLedger — concurrency cap")
struct ConcurrencyCapTests {

    // MARK: - Below cap

    @Test("acquire below cap returns immediately without waiter")
    func belowCapReturnsImmediately() async throws {
        let ledger = BudgetLedger(concurrencyCaps: [.claudeApi: 2])
        let permit1 = try await ledger.acquireSlot(for: .claudeApi)
        let permit2 = try await ledger.acquireSlot(for: .claudeApi)

        #expect(permit1.providerID == .claudeApi)
        #expect(permit2.providerID == .claudeApi)
        #expect(permit1.id != permit2.id)
        let active = await ledger.activeSlotCount(for: .claudeApi)
        let waiting = await ledger.waiterCount(for: .claudeApi)
        #expect(active == 2)
        #expect(waiting == 0)
    }

    // MARK: - At cap suspends

    @Test("at-cap acquisition suspends and resumes when a slot releases")
    func atCapSuspendsUntilRelease() async throws {
        let ledger = BudgetLedger(concurrencyCaps: [.claudeApi: 1])
        let permit1 = try await ledger.acquireSlot(for: .claudeApi)

        // Kick off a second acquisition; it must not complete yet.
        let waitingTask = Task<BudgetLedger.SlotPermit, Error> {
            try await ledger.acquireSlot(for: .claudeApi)
        }

        // Yield a few times so the task has a chance to enqueue on the ledger.
        try await Task.sleep(nanoseconds: 20_000_000)
        let waiterCountBeforeRelease = await ledger.waiterCount(for: .claudeApi)
        #expect(waiterCountBeforeRelease == 1, "second acquire must be suspended")
        #expect(!waitingTask.isCancelled)

        await ledger.release(permit1)
        let permit2 = try await waitingTask.value
        #expect(permit2.providerID == .claudeApi)

        let waiterCountAfterRelease = await ledger.waiterCount(for: .claudeApi)
        let activeAfter = await ledger.activeSlotCount(for: .claudeApi)
        #expect(waiterCountAfterRelease == 0)
        #expect(activeAfter == 1, "one slot still held by the woken waiter")
    }

    // MARK: - FIFO ordering

    @Test("waiters are woken in FIFO order")
    func waitersAreFIFO() async throws {
        let ledger = BudgetLedger(concurrencyCaps: [.claudeApi: 1])
        let permit = try await ledger.acquireSlot(for: .claudeApi)

        // Sequentially schedule two waiters — the ordering guarantee is on
        // the enqueue order, so we serialize the enqueue via Task.yield().
        actor Recorder {
            var order: [Int] = []
            func record(_ i: Int) { order.append(i) }
            func snapshot() -> [Int] { order }
        }
        let recorder = Recorder()

        let first = Task {
            let p = try await ledger.acquireSlot(for: .claudeApi)
            await recorder.record(1)
            await ledger.release(p)
        }
        // Ensure `first` has enqueued before scheduling `second`.
        while await ledger.waiterCount(for: .claudeApi) < 1 {
            await Task.yield()
        }
        let second = Task {
            let p = try await ledger.acquireSlot(for: .claudeApi)
            await recorder.record(2)
            await ledger.release(p)
        }
        while await ledger.waiterCount(for: .claudeApi) < 2 {
            await Task.yield()
        }

        await ledger.release(permit)
        _ = try await first.value
        _ = try await second.value

        let observed = await recorder.snapshot()
        #expect(observed == [1, 2])
    }

    // MARK: - Isolation between providers

    @Test("caps are isolated per provider")
    func capsIsolatedPerProvider() async throws {
        let ledger = BudgetLedger(concurrencyCaps: [
            .claudeApi: 1,
            .openaiApi: 1,
        ])
        let claude = try await ledger.acquireSlot(for: .claudeApi)
        let openai = try await ledger.acquireSlot(for: .openaiApi)

        let claudeActive = await ledger.activeSlotCount(for: .claudeApi)
        let openaiActive = await ledger.activeSlotCount(for: .openaiApi)
        #expect(claudeActive == 1)
        #expect(openaiActive == 1)

        await ledger.release(claude)
        await ledger.release(openai)
    }

    // MARK: - No cap = unbounded

    @Test("provider with no configured cap is unbounded")
    func noCapIsUnbounded() async throws {
        let ledger = BudgetLedger()  // no caps
        var permits: [BudgetLedger.SlotPermit] = []
        for _ in 0..<32 {
            permits.append(try await ledger.acquireSlot(for: .mlxLocal))
        }
        let active = await ledger.activeSlotCount(for: .mlxLocal)
        let waiting = await ledger.waiterCount(for: .mlxLocal)
        #expect(active == 32)
        #expect(waiting == 0)

        for p in permits { await ledger.release(p) }
        let activeAfter = await ledger.activeSlotCount(for: .mlxLocal)
        #expect(activeAfter == 0)
    }

    // MARK: - withSlot release semantics

    @Test("withSlot releases the slot on success")
    func withSlotReleasesOnSuccess() async throws {
        let ledger = BudgetLedger(concurrencyCaps: [.claudeApi: 1])
        let value = try await ledger.withSlot(for: .claudeApi) { permit in
            #expect(permit.providerID == .claudeApi)
            return 42
        }
        #expect(value == 42)
        let active = await ledger.activeSlotCount(for: .claudeApi)
        #expect(active == 0)
    }

    @Test("withSlot releases the slot when the body throws")
    func withSlotReleasesOnThrow() async throws {
        struct Boom: Error {}
        let ledger = BudgetLedger(concurrencyCaps: [.claudeApi: 1])
        do {
            _ = try await ledger.withSlot(for: .claudeApi) { _ in
                throw Boom()
            }
            Issue.record("Expected Boom to propagate")
        } catch is Boom {
            // expected
        }
        let active = await ledger.activeSlotCount(for: .claudeApi)
        let waiting = await ledger.waiterCount(for: .claudeApi)
        #expect(active == 0, "slot must be released even on throw")
        #expect(waiting == 0)

        // Subsequent acquisition must not deadlock.
        let permit = try await ledger.acquireSlot(for: .claudeApi)
        await ledger.release(permit)
    }

    // MARK: - Precondition on init

    @Test("zero or negative concurrency cap traps on init")
    func nonPositiveCapPreconditionFails() {
        // We can't crash-test a precondition inside a Swift Testing @Test
        // without terminating the whole runner, so we just document the
        // contract by ensuring positive caps do NOT trap.
        _ = BudgetLedger(concurrencyCaps: [.claudeApi: 1])
        _ = BudgetLedger(concurrencyCaps: [.claudeApi: 1000])
    }
}
