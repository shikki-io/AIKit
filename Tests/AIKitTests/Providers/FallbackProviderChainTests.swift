import Foundation
import Testing
@testable import AIKit

// MARK: - FallbackProviderChainTests
//
// The chain's own rules (LMStudio connection/rate-limit errors, URLError network
// failures) plus `extraFallbackEligible`, the seam a caller uses for error types
// this package does not know — shikki passes its `SpecPipelineError` rule there.

@Suite("FallbackProviderChain — fallback eligibility and ordering")
struct FallbackProviderChainTests {

    private struct StubProvider: AgentProviding {
        let output: String?
        let error: (any Error)?
        func run(prompt: String, timeout: TimeInterval) async throws -> String {
            if let error { throw error }
            return output ?? ""
        }
    }

    private struct UnknownError: Error {}

    @Test("a rate-limited provider falls back to the next one")
    func rateLimitedFallsBack() async throws {
        let chain = FallbackProviderChain(providers: [
            StubProvider(output: nil, error: LMStudioProvider.LMStudioError.rateLimited),
            StubProvider(output: "second", error: nil),
        ])
        let result = try await chain.run(prompt: "p", timeout: 5)
        #expect(result == "second")
    }

    @Test("a network failure falls back to the next provider")
    func urlErrorFallsBack() async throws {
        let chain = FallbackProviderChain(providers: [
            StubProvider(output: nil, error: URLError(.cannotConnectToHost)),
            StubProvider(output: "second", error: nil),
        ])
        let result = try await chain.run(prompt: "p", timeout: 5)
        #expect(result == "second")
    }

    @Test("an unknown error type propagates instead of falling back")
    func unknownErrorPropagates() async {
        let chain = FallbackProviderChain(providers: [
            StubProvider(output: nil, error: UnknownError()),
            StubProvider(output: "second", error: nil),
        ])
        await #expect(throws: UnknownError.self) {
            try await chain.run(prompt: "p", timeout: 5)
        }
    }

    @Test("extraFallbackEligible makes a caller's own error type fall back")
    func callerRuleFallsBack() async throws {
        let chain = FallbackProviderChain(
            providers: [
                StubProvider(output: nil, error: UnknownError()),
                StubProvider(output: "second", error: nil),
            ],
            extraFallbackEligible: { $0 is UnknownError }
        )
        let result = try await chain.run(prompt: "p", timeout: 5)
        #expect(result == "second")
    }

    @Test("an empty chain throws noProviders")
    func emptyChainThrows() async {
        let chain = FallbackProviderChain(providers: [])
        await #expect(throws: FallbackProviderChain.ChainError.self) {
            try await chain.run(prompt: "p", timeout: 5)
        }
    }

    @Test("the first healthy provider wins and later ones are never called")
    func firstHealthyWins() async throws {
        let chain = FallbackProviderChain(providers: [
            StubProvider(output: "first", error: nil),
            StubProvider(output: nil, error: UnknownError()),
        ])
        let result = try await chain.run(prompt: "p", timeout: 5)
        #expect(result == "first")
    }
}
