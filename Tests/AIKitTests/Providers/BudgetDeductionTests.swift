import Foundation
@testable import AIKit
import Testing

// MARK: - BudgetDeductionTests
//
// W4 tests for `BudgetLedger` — the per-provider token-deduction half.
//
// Covers:
//   - Deductions accumulate across calls.
//   - Providers without a hard-stop entry are unbounded (`remainingTokens == nil`).
//   - The caller resolves the hard stop (shikki: `cap * hardStopAtPct / 100`).
//   - `remainingTokens` clamps to `0` (never negative) once the hard-stop is crossed.
//   - `isExhausted` flips at the hard-stop threshold.
//   - `acquireSlot` throws `.budgetExhausted` for an exhausted provider.
//   - Per-provider spends are isolated.

@Suite("BudgetLedger — budget deduction")
struct BudgetDeductionTests {

    // MARK: - Fixtures

    /// Hard-stop token count, the way shikki resolves it: cap × hardStopPct.
    private static func limits(
        cap: Int?, hardStopPct: Int = 100, provider: AIProviderID = .claudeApi
    ) -> [AIProviderID: Int] {
        guard let cap else { return [:] }
        return [provider: (cap * hardStopPct) / 100]
    }

    // MARK: - Deduction math

    @Test("deduct sums tokensIn + tokensOut into spent")
    func deductSumsInAndOut() async {
        let ledger = BudgetLedger(hardStopTokens: Self.limits(cap: 10000))
        await ledger.deduct(providerID: .claudeApi, totalTokens: 1000)
        let spent = await ledger.spent(for: .claudeApi)
        #expect(spent == 1000)
    }

    @Test("deductions accumulate across multiple calls")
    func deductionsAccumulate() async {
        let ledger = BudgetLedger(hardStopTokens: Self.limits(cap: 10000))
        await ledger.deduct(providerID: .claudeApi, totalTokens: 200)
        await ledger.deduct(providerID: .claudeApi, totalTokens: 900)
        await ledger.deduct(providerID: .claudeApi, totalTokens: 500)
        let spent = await ledger.spent(for: .claudeApi)
        #expect(spent == 1600)
    }

    @Test("remainingTokens reflects deducted spend")
    func remainingReflectsSpend() async {
        let ledger = BudgetLedger(hardStopTokens: Self.limits(cap: 10000))
        await ledger.deduct(providerID: .claudeApi, totalTokens: 2000)
        let remaining = await ledger.remainingTokens(for: .claudeApi)
        #expect(remaining == 8000)
    }

    // MARK: - Unbounded providers

    @Test("provider without a budget entry is unbounded")
    func noEntryIsUnbounded() async {
        let ledger = BudgetLedger()  // no entries
        await ledger.deduct(providerID: .claudeApi, totalTokens: 1999998)
        let remaining = await ledger.remainingTokens(for: .claudeApi)
        let exhausted = await ledger.isExhausted(for: .claudeApi)
        #expect(remaining == nil)
        #expect(!exhausted)
    }

    @Test("entry with nil monthlyTokenCap is unbounded")
    func nilCapIsUnbounded() async {
        let ledger = BudgetLedger(hardStopTokens: Self.limits(cap: nil))
        await ledger.deduct(providerID: .claudeApi, totalTokens: 1000000)
        let remaining = await ledger.remainingTokens(for: .claudeApi)
        let exhausted = await ledger.isExhausted(for: .claudeApi)
        #expect(remaining == nil)
        #expect(!exhausted)
    }

    // MARK: - Hard-stop threshold

    @Test("hardStopAtPct scales the effective threshold below cap")
    func hardStopScalesThreshold() async {
        // cap 10_000, hard-stop 80% → threshold 8_000.
        let ledger = BudgetLedger(hardStopTokens: Self.limits(cap: 10000, hardStopPct: 80))
        await ledger.deduct(providerID: .claudeApi, totalTokens: 7000)  // spent 7_000
        let remainingBefore = await ledger.remainingTokens(for: .claudeApi)
        let exhaustedBefore = await ledger.isExhausted(for: .claudeApi)
        #expect(remainingBefore == 1000)
        #expect(!exhaustedBefore)

        await ledger.deduct(providerID: .claudeApi, totalTokens: 1000)  // spent 8_000
        let remainingAt = await ledger.remainingTokens(for: .claudeApi)
        let exhaustedAt = await ledger.isExhausted(for: .claudeApi)
        #expect(remainingAt == 0)
        #expect(exhaustedAt)
    }

    @Test("remainingTokens clamps to zero when spend crosses the threshold")
    func remainingClampsToZeroPastThreshold() async {
        let ledger = BudgetLedger(hardStopTokens: Self.limits(cap: 5000))
        await ledger.deduct(providerID: .claudeApi, totalTokens: 12000)  // way over
        let remaining = await ledger.remainingTokens(for: .claudeApi)
        #expect(remaining == 0)
    }

    // MARK: - acquireSlot gating

    @Test("acquireSlot throws budgetExhausted once provider hits its cap")
    func acquireThrowsWhenExhausted() async throws {
        let ledger = BudgetLedger(
            hardStopTokens: Self.limits(cap: 1000),
            concurrencyCaps: [.claudeApi: 4]
        )
        // Under cap: acquire OK.
        let permit = try await ledger.acquireSlot(for: .claudeApi)
        await ledger.release(permit)

        // Drive spend past the hard-stop.
        await ledger.deduct(providerID: .claudeApi, totalTokens: 1100)

        do {
            _ = try await ledger.acquireSlot(for: .claudeApi)
            Issue.record("Expected LedgerError.budgetExhausted")
        } catch BudgetLedger.LedgerError.budgetExhausted(let provider, let spent, let cap) {
            #expect(provider == .claudeApi)
            #expect(spent == 1100)
            #expect(cap == 1000)
        }
    }

    @Test("acquireSlot on unbounded provider never throws budgetExhausted")
    func acquireNeverThrowsForUnbounded() async throws {
        let ledger = BudgetLedger()
        await ledger.deduct(providerID: .mlxLocal, totalTokens: 1999998)
        let permit = try await ledger.acquireSlot(for: .mlxLocal)
        #expect(permit.providerID == .mlxLocal)
        await ledger.release(permit)
    }

    // MARK: - Per-provider isolation

    @Test("spends are isolated per provider")
    func spendsAreIsolated() async {
        let ledger = BudgetLedger(hardStopTokens: [.claudeApi: 10000, .openaiApi: 5000])
        await ledger.deduct(providerID: .claudeApi, totalTokens: 6000)
        await ledger.deduct(providerID: .openaiApi, totalTokens: 1000)

        let claudeSpent = await ledger.spent(for: .claudeApi)
        let openaiSpent = await ledger.spent(for: .openaiApi)
        let claudeRemaining = await ledger.remainingTokens(for: .claudeApi)
        let openaiRemaining = await ledger.remainingTokens(for: .openaiApi)

        #expect(claudeSpent == 6000)
        #expect(openaiSpent == 1000)
        #expect(claudeRemaining == 4000)
        #expect(openaiRemaining == 4000)
    }

    // MARK: - Zero-token deduction

    @Test("deducting zero tokens leaves the spend unchanged")
    func zeroTokensContributeNothing() async {
        let ledger = BudgetLedger(hardStopTokens: Self.limits(cap: 10000))
        await ledger.deduct(providerID: .claudeApi, totalTokens: 0)
        let spent = await ledger.spent(for: .claudeApi)
        #expect(spent == 0)
    }
}
