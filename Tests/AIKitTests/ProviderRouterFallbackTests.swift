import Foundation
import Testing
@testable import AIKit

// MARK: - ProviderRouterFallbackTests
//
// A fallback that answers must say who answered, and a fallback that fails must
// say what each provider said (BR-AIL-06, BR-AIL-07). `routeWithFallback` used
// to do neither: `catch { continue }` dropped every cause and threw a bare
// `allProvidersFailed`.
//
// No provider here is real (BR-AIL-12): `MockAIProvider` throws what it is told
// to throw.

@Suite("ProviderRouter — fallback keeps its evidence")
struct ProviderRouterFallbackTests {

    private func router() -> ProviderRouter {
        // The fallback list is the method's argument, not the router's own.
        ProviderRouter(providers: [])
    }

    private func request() -> AIRequest {
        AIRequest(messages: [AIMessage(role: .user, content: "hi")])
    }

    // MARK: - T-AIL-08 — the answer names who answered

    @Test("the response names the provider that answered, not the one asked first")
    func namesWhoAnswered() async throws {
        let down = MockAIProvider(id: "lmstudio-local", capabilities: .textGeneration)
        down.shouldThrow = URLError(.cannotConnectToHost)
        let answering = MockAIProvider(id: "openai-cloud", capabilities: .textGeneration)

        let response = try await router().routeWithFallback(
            request: request(), providers: [down, answering]
        )

        #expect(response.providerID == "openai-cloud")
        #expect(down.completeCallCount == 1)
        #expect(answering.completeCallCount == 1)
    }

    @Test("the answering provider's own model survives the fallback")
    func namesTheModelThatAnswered() async throws {
        let down = MockAIProvider(id: "first", capabilities: .textGeneration)
        down.shouldThrow = URLError(.cannotConnectToHost)
        let canned = AIResponse(
            content: "answered",
            model: "devstral-small-2507",
            tokensUsed: TokenUsage(prompt: 3, completion: 4),
            latencyMs: 12,
            providerID: "second"
        )
        let answering = MockAIProvider(
            id: "second", capabilities: .textGeneration, responses: [canned]
        )

        let response = try await router().routeWithFallback(
            request: request(), providers: [down, answering]
        )

        #expect(response.providerID == "second")
        #expect(response.model == "devstral-small-2507")
        #expect(response.content == "answered")
    }

    @Test("a first provider that answers is the one named, and the rest are never asked")
    func firstProviderWinsAndNamesItself() async throws {
        let first = MockAIProvider(id: "first", capabilities: .textGeneration)
        let second = MockAIProvider(id: "second", capabilities: .textGeneration)

        let response = try await router().routeWithFallback(
            request: request(), providers: [first, second]
        )

        #expect(response.providerID == "first")
        #expect(second.completeCallCount == 0)
    }

    // MARK: - T-AIL-09 — every cause is kept

    @Test("a total failure lists every provider and its cause")
    func keepsEveryCause() async throws {
        let refused = MockAIProvider(id: "lmstudio-local", capabilities: .textGeneration)
        refused.shouldThrow = LMStudioProvider.LMStudioError.connectionRefused("127.0.0.1:1234")
        let limited = MockAIProvider(id: "openai-cloud", capabilities: .textGeneration)
        limited.shouldThrow = LMStudioProvider.LMStudioError.rateLimited
        let missing = MockAIProvider(id: "mlx/local", capabilities: .textGeneration)
        missing.shouldThrow = AIKitError.engineUnavailable("MLX requires macOS on Apple Silicon")

        await #expect {
            try await router().routeWithFallback(
                request: request(), providers: [refused, limited, missing]
            )
        } throws: { error in
            guard let aiKitError = error as? AIKitError,
                  case .allProvidersFailed(let attempts) = aiKitError
            else {
                return false
            }
            // Every provider asked, in order, each with its own cause.
            #expect(attempts.count == 3)
            #expect(attempts.map(\.providerID) == ["lmstudio-local", "openai-cloud", "mlx/local"])
            #expect(attempts[0].cause.contains("127.0.0.1:1234"))
            #expect(attempts[2].cause.contains("MLX requires macOS on Apple Silicon"))
            return true
        }
    }

    @Test("the error message names each provider, so a log line carries the evidence")
    func summaryNamesEveryProvider() async throws {
        let a = MockAIProvider(id: "alpha", capabilities: .textGeneration)
        a.shouldThrow = AIKitError.requestFailed("HTTP 500 — from http://h:1/v1/chat/completions")
        let b = MockAIProvider(id: "beta", capabilities: .textGeneration)
        b.shouldThrow = URLError(.timedOut)

        await #expect {
            try await router().routeWithFallback(request: request(), providers: [a, b])
        } throws: { error in
            let message = (error as? AIKitError)?.errorDescription ?? ""
            #expect(message.contains("alpha"))
            #expect(message.contains("beta"))
            #expect(message.contains("HTTP 500"))
            // A bare case is what this replaced; the count proves it is not back.
            #expect(message.contains("2 provider(s) failed"))
            return true
        }
    }

    @Test("an empty fallback list says nothing was tried, not that everything failed")
    func emptyListIsNotAProviderFailure() async throws {
        await #expect {
            try await router().routeWithFallback(request: request(), providers: [])
        } throws: { error in
            guard let aiKitError = error as? AIKitError,
                  case .allProvidersFailed(let attempts) = aiKitError
            else {
                return false
            }
            #expect(attempts.isEmpty)
            #expect(
                (error as? AIKitError)?.errorDescription
                    == "No provider was tried: the fallback list was empty"
            )
            return true
        }
    }

    // MARK: - Stopping instead of burning the list

    @Test("an error no other provider can fix is rethrown as itself")
    func nonEligibleFirstErrorKeepsItsType() async throws {
        let refusing = MockAIProvider(id: "only", capabilities: .textGeneration)
        refusing.shouldThrow = AIKitError.streamingNotSupported
        let never = MockAIProvider(id: "never", capabilities: .textGeneration)

        await #expect(throws: AIKitError.streamingNotSupported) {
            try await router().routeWithFallback(
                request: request(), providers: [refusing, never]
            )
        }
        // The walk stopped: trying the rest could not have helped.
        #expect(never.completeCallCount == 0)
    }

    @Test("a non-recoverable failure after a recoverable one still carries both causes")
    func nonEligibleErrorAfterAnAttemptKeepsTheEarlierCause() async throws {
        let down = MockAIProvider(id: "down", capabilities: .textGeneration)
        down.shouldThrow = URLError(.cannotConnectToHost)
        let refusing = MockAIProvider(id: "refusing", capabilities: .textGeneration)
        refusing.shouldThrow = AIKitError.streamingNotSupported
        let never = MockAIProvider(id: "never", capabilities: .textGeneration)

        await #expect {
            try await router().routeWithFallback(
                request: request(), providers: [down, refusing, never]
            )
        } throws: { error in
            guard let aiKitError = error as? AIKitError,
                  case .allProvidersFailed(let attempts) = aiKitError
            else {
                return false
            }
            // The first provider's cause would be lost if the second error
            // simply propagated.
            #expect(attempts.map(\.providerID) == ["down", "refusing"])
            return true
        }
        #expect(never.completeCallCount == 0)
    }

    // MARK: - One eligibility rule, shared with the chain

    @Test("the router and the chain agree on what falls back")
    func oneRuleForBothCallers() {
        // Same inputs through both doors: the chain's own check now delegates to
        // the rule the router uses, so drift between them cannot return.
        let errors: [any Error] = [
            URLError(.cannotConnectToHost),
            URLError(.badURL),
            LMStudioProvider.LMStudioError.rateLimited,
            LMStudioProvider.LMStudioError.emptyContent,
            AIKitError.engineUnavailable("no MLX here"),
            AIKitError.streamingNotSupported,
        ]

        for error in errors {
            #expect(
                FallbackEligibility.allows(error)
                    == FallbackProviderChain.isFallbackEligible(error),
                "disagreement on \(error)"
            )
        }
    }

    @Test("a server that answered is not retried elsewhere")
    func aRealReplyIsNotRetried() {
        // The distinction the rule is built on: the server was reached and said
        // something, so asking a different server changes the answer rather than
        // recovering it.
        #expect(!FallbackEligibility.allows(LMStudioProvider.LMStudioError.emptyContent))
        #expect(!FallbackEligibility.allows(LMStudioProvider.LMStudioError.httpError(400, "bad")))
        #expect(FallbackEligibility.allows(LMStudioProvider.LMStudioError.connectionRefused("h")))
    }

    @Test("an unreachable or absent provider yields to the next one")
    func availabilityFailuresFallBack() {
        #expect(FallbackEligibility.allows(AIKitError.engineUnavailable("MLX needs macOS")))
        #expect(FallbackEligibility.allows(AIKitError.noProviderAvailable(.voiceToText)))
        #expect(FallbackEligibility.allows(AIKitError.modelNotFound(
            ModelIdentifier(provider: "openai", modelId: "absent")
        )))
        #expect(!FallbackEligibility.allows(AIKitError.downloadFailed("HTTP 404")))
        #expect(!FallbackEligibility.allows(AIKitError.allProvidersFailed([])))
    }

    @Test("an error type AIKit does not know is not retried")
    func unknownErrorIsNotEligible() {
        struct SomeCallerError: Error {}
        #expect(!FallbackEligibility.allows(SomeCallerError()))
    }

    // MARK: - A slow local model is not a down one

    @Test("a provider that is merely slow is never cut off by the router")
    func noRouterSideDeadline() async throws {
        // A cold MLX model loads weights before its first token. The router
        // imposes no deadline, so a slow answer is still an answer: the timeout
        // is the caller's input.
        let slow = SlowProvider(id: "cold-mlx", delay: .milliseconds(120))

        let response = try await router().routeWithFallback(
            request: request(), providers: [slow]
        )

        #expect(response.providerID == "cold-mlx")
    }

    private struct SlowProvider: AIProvider {
        let id: String
        let delay: Duration
        var displayName: String { id }
        var capabilities: AICapabilities { .textGeneration }
        var status: AIProviderStatus { get async { .ready } }

        func complete(request: AIRequest) async throws -> AIResponse {
            try await Task.sleep(for: delay)
            return AIResponse(
                content: "loaded, then answered",
                model: "cold-model",
                tokensUsed: TokenUsage(prompt: 1, completion: 1),
                latencyMs: 120,
                providerID: id
            )
        }
    }
}
