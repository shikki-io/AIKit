import Foundation
import Testing

@testable import AIKit

// MARK: - FallbackProviderChainTests
//
// The chain's own rules (LMStudio connection/rate-limit errors, URLError network
// failures) plus `extraFallbackEligible`, the seam a caller uses for error types
// this package does not know — shikki passes its `SpecPipelineError` rule there.
//
// T-AIL-47, W22 of spec ai-layer-aikit-core-2026-10-31: one deadline for the
// whole chain, cancellation between attempts, and every attempt naming the
// provider by its own `id` rather than by its Swift type.

@Suite("FallbackProviderChain — fallback eligibility and ordering")
struct FallbackProviderChainTests {

    /// A backend that answers, throws, or takes its time.
    private struct StubProvider: AgentProviding {
        let id: String
        let output: String?
        let error: (any Error)?
        /// Seconds this provider sleeps before answering or throwing.
        let delay: TimeInterval
        /// Set to the `timeout` each call was handed.
        let observed: TimeoutRecorder?

        init(
            id: String = "stub",
            output: String? = nil,
            error: (any Error)? = nil,
            delay: TimeInterval = 0,
            observed: TimeoutRecorder? = nil
        ) {
            self.id = id
            self.output = output
            self.error = error
            self.delay = delay
            self.observed = observed
        }

        func run(prompt: String, timeout: TimeInterval) async throws -> String {
            observed?.record(id: id, timeout: timeout)
            if delay > 0 {
                try await Task.sleep(for: .seconds(delay))
            }
            if let error { throw error }
            return output ?? ""
        }
    }

    /// What timeout each provider was handed, in call order.
    private final class TimeoutRecorder: @unchecked Sendable {
        private let lock = NSLock()
        private var calls: [(id: String, timeout: TimeInterval)] = []

        func record(id: String, timeout: TimeInterval) {
            lock.lock()
            defer { lock.unlock() }
            calls.append((id, timeout))
        }

        var observed: [(id: String, timeout: TimeInterval)] {
            lock.lock()
            defer { lock.unlock() }
            return calls
        }
    }

    private struct UnknownError: Error {}

    // MARK: - Eligibility

    @Test("a rate-limited provider falls back to the next one")
    func rateLimitedFallsBack() async throws {
        let chain = FallbackProviderChain(providers: [
            StubProvider(id: "local", error: LMStudioProvider.LMStudioError.rateLimited),
            StubProvider(id: "cloud", output: "second"),
        ])
        let result = try await chain.run(prompt: "p", timeout: 5)
        #expect(result == "second")
    }

    @Test("a network failure falls back to the next provider")
    func urlErrorFallsBack() async throws {
        let chain = FallbackProviderChain(providers: [
            StubProvider(id: "local", error: URLError(.cannotConnectToHost)),
            StubProvider(id: "cloud", output: "second"),
        ])
        let result = try await chain.run(prompt: "p", timeout: 5)
        #expect(result == "second")
    }

    @Test("an unknown error type propagates instead of falling back")
    func unknownErrorPropagates() async {
        let chain = FallbackProviderChain(providers: [
            StubProvider(id: "local", error: UnknownError()),
            StubProvider(id: "cloud", output: "second"),
        ])
        await #expect(throws: UnknownError.self) {
            try await chain.run(prompt: "p", timeout: 5)
        }
    }

    @Test("extraFallbackEligible makes a caller's own error type fall back")
    func callerRuleFallsBack() async throws {
        let chain = FallbackProviderChain(
            providers: [
                StubProvider(id: "local", error: UnknownError()),
                StubProvider(id: "cloud", output: "second"),
            ],
            extraFallbackEligible: { $0 is UnknownError }
        )
        let result = try await chain.run(prompt: "p", timeout: 5)
        #expect(result == "second")
    }

    @Test("an empty chain reports that nothing was asked")
    func emptyChainThrows() async throws {
        // One error type for one chain. `ChainError.noProviders` is gone: a
        // caller had to catch TWO error types to learn the same thing, and
        // `allProvidersFailed([])` already says "no provider was tried" rather
        // than claiming a provider failed.
        let chain = FallbackProviderChain(providers: [])
        do {
            _ = try await chain.run(prompt: "p", timeout: 5)
            Issue.record("Expected allProvidersFailed")
        } catch let error as AIKitError {
            guard case .allProvidersFailed(let attempts) = error else {
                Issue.record("Expected allProvidersFailed, got \(error)")
                return
            }
            #expect(attempts.isEmpty)
            #expect(error.localizedDescription.contains("No provider was tried"))
        }
    }

    @Test("the first healthy provider wins and later ones are never called")
    func firstHealthyWins() async throws {
        let chain = FallbackProviderChain(providers: [
            StubProvider(id: "local", output: "first"),
            StubProvider(id: "cloud", error: UnknownError()),
        ])
        let result = try await chain.run(prompt: "p", timeout: 5)
        #expect(result == "first")
    }

    // MARK: - T-AIL-47: attempts carry provider.id

    @Test("every attempt names the provider by its settings id, not by its type")
    func attemptsCarryTheProviderID() async throws {
        // What this replaces recorded `String(describing: type(of: provider))`,
        // so an operator running a local LM Studio and the Mac across the desk
        // read "LMStudioProvider failed; LMStudioProvider failed" and could not
        // tell which server was down (AIKit#18, FallbackProviderChain.swift:28).
        let chain = FallbackProviderChain(providers: [
            StubProvider(id: "lmstudio-local", error: LMStudioProvider.LMStudioError.connectionRefused("down")),
            StubProvider(id: "lmstudio-desk", error: LMStudioProvider.LMStudioError.rateLimited),
        ])

        do {
            _ = try await chain.run(prompt: "p", timeout: 5)
            Issue.record("Expected allProvidersFailed")
        } catch let error as AIKitError {
            guard case .allProvidersFailed(let attempts) = error else {
                Issue.record("Expected allProvidersFailed, got \(error)")
                return
            }
            #expect(attempts.map(\.providerID) == ["lmstudio-local", "lmstudio-desk"])
            #expect(!attempts.contains { $0.providerID.contains("StubProvider") })
        }
    }

    @Test("a real provider's id is the id its settings entry carries")
    func aRealProviderNamesItsEntry() throws {
        // The requirement has no default implementation on purpose, so each
        // conformer states its id — and these two are the ones a chain holds.
        let lmStudio = try LMStudioProvider(id: "lmstudio-desk", baseURL: fixtureServerRoot, model: "m")
        #expect(lmStudio.id == "lmstudio-desk")

        let nested = FallbackProviderChain(providers: [], id: "local-first")
        #expect(nested.id == "local-first")

        // And a row this provider cannot serve is refused where it is built,
        // rather than carried as an empty address.
        #expect(throws: AIKitError.self) {
            try LMStudioProvider(
                entry: AIProviderEntry(
                    id: "claude-code",
                    family: .cliSubprocess,
                    address: nil,
                    model: "sonnet",
                    options: .cliSubprocess(AICLISubprocessOptions(binary: "claude"))
                )
            )
        }
    }

    // MARK: - T-AIL-47: one shared deadline

    @Test("the timeout is the whole chain's budget, not each provider's")
    func slowProvidersShareOneDeadline() async throws {
        // Three providers that each take 0.2s used to cost 3 × the caller's
        // timeout, because each attempt was handed the FULL value: a 300-second
        // ceiling became fifteen minutes with nothing in the error saying why.
        let recorder = TimeoutRecorder()
        let chain = FallbackProviderChain(providers: [
            StubProvider(id: "a", error: URLError(.timedOut), delay: 0.2, observed: recorder),
            StubProvider(id: "b", error: URLError(.timedOut), delay: 0.2, observed: recorder),
            StubProvider(id: "c", output: "answer", delay: 0, observed: recorder),
        ])

        let result = try await chain.run(prompt: "p", timeout: 5)
        #expect(result == "answer")

        let observed = recorder.observed
        #expect(observed.map(\.id) == ["a", "b", "c"])
        // Each attempt gets what is LEFT, so the budget only ever shrinks.
        #expect(observed[0].timeout <= 5)
        #expect(observed[1].timeout < observed[0].timeout)
        #expect(observed[2].timeout < observed[1].timeout)
        // And none of them was handed more than the caller allowed the chain.
        #expect(observed.allSatisfy { $0.timeout <= 5 })
    }

    @Test("a chain whose budget is already spent says so and asks nobody")
    func anExhaustedBudgetAsksNobody() async throws {
        let recorder = TimeoutRecorder()
        let chain = FallbackProviderChain(providers: [
            StubProvider(id: "slow", error: URLError(.timedOut), delay: 0.3, observed: recorder),
            StubProvider(id: "never-asked", output: "unreachable", observed: recorder),
        ])

        do {
            _ = try await chain.run(prompt: "p", timeout: 0.1)
            Issue.record("Expected allProvidersFailed")
        } catch let error as AIKitError {
            guard case .allProvidersFailed(let attempts) = error else {
                Issue.record("Expected allProvidersFailed, got \(error)")
                return
            }
            // The second provider is RECORDED as not asked, with the reason —
            // rather than silently skipped, which would read as a chain of one.
            #expect(attempts.map(\.providerID) == ["slow", "never-asked"])
            #expect(attempts[1].cause.contains("not asked"))
        }
        #expect(recorder.observed.map(\.id) == ["slow"], "the second provider was never run")
    }

    // MARK: - T-AIL-47: cancellation

    @Test("a cancelled task stops the chain between attempts")
    func cancellationStopsTheWalk() async throws {
        let recorder = TimeoutRecorder()
        let chain = FallbackProviderChain(providers: [
            StubProvider(id: "first", error: URLError(.timedOut), delay: 0.2, observed: recorder),
            StubProvider(id: "second", output: "late", delay: 0, observed: recorder),
        ])

        let task = Task {
            try await chain.run(prompt: "p", timeout: 30)
        }
        // Cancel while the first provider is still sleeping, so the check
        // between attempts is the one that fires.
        try await Task.sleep(for: .milliseconds(50))
        task.cancel()

        await #expect(throws: CancellationError.self) {
            try await task.value
        }
        // Without the check the chain kept walking after the caller had gone,
        // holding a subprocess slot and a budget for an answer nobody reads.
        #expect(!recorder.observed.map(\.id).contains("second"))
    }
}
