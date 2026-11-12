import Foundation
import Logging

// MARK: - FallbackProviderChain

/// Wraps multiple ``AgentProviding`` implementations in priority order.
///
/// If a provider fails for a reason the next one could plausibly fix — a
/// refused connection, a rate limit, a 5xx — the next one is asked.
/// `FallbackEligibility` is the one rule that decides, shared with
/// `ProviderRouter.routeWithFallback`.
///
/// Usage — LOCAL FIRST, which is the fleet's default order (BR-AIL-11, and the
/// operator's `providers.toml` is where that order actually lives):
///
/// ```swift
/// let chain = FallbackProviderChain(providers: [
///     lmStudioProvider,  // primary — local, free, no data leaves the host
///     claudeProvider,    // fallback — cloud
/// ])
/// let result = try await chain.run(prompt: "...", timeout: 300)
/// ```
///
/// The doc example used to put the cloud provider first and label it "primary"
/// (AIKit#18 r4500440905). A doc example is read as the recommended shape, and
/// that one recommended sending every prompt off the machine before trying the
/// model already running on it.
///
/// `timeout` is the budget for the WHOLE chain, not for each provider in turn.
/// Three slow providers used to cost three full timeouts — a 300-second ceiling
/// became fifteen minutes with no caller able to see why — so the deadline is
/// computed once and each attempt gets what is left of it.
public struct FallbackProviderChain: AgentProviding, Sendable {
    private let providers: [any AgentProviding]
    private let logger: Logger
    /// Caller-supplied rule for error types this package does not know about.
    /// Consulted after the built-in LMStudio / URLError rules.
    private let extraFallbackEligible: @Sendable (any Error) -> Bool

    // MARK: - AgentProviding

    /// What this chain is called. A chain is itself a backend, so a chain
    /// nested in a chain names itself in the outer one's attempt list.
    public let id: String

    // MARK: - Init

    public init(
        providers: [any AgentProviding],
        id: String = "fallback-chain",
        logger: Logger = Logger(label: "aikit.fallback-chain"),
        extraFallbackEligible: @escaping @Sendable (any Error) -> Bool = { _ in false }
    ) {
        self.providers = providers
        self.id = id
        self.logger = logger
        self.extraFallbackEligible = extraFallbackEligible
    }

    // MARK: - run

    /// Asks each provider in turn, within one shared deadline.
    ///
    /// - Parameters:
    ///   - prompt: the prompt.
    ///   - timeout: seconds the WHOLE chain may take.
    /// - Throws: the provider's own error when the first attempt fails for a
    ///   reason no other provider can fix; ``AIKitError/allProvidersFailed(_:)``
    ///   otherwise, carrying one ``AIProviderAttempt`` per provider asked, each
    ///   naming the provider by its own `id` (BR-AIL-07);
    ///   `CancellationError` when the calling task was cancelled.
    public func run(prompt: String, timeout: TimeInterval) async throws -> String {
        // An empty chain is NOT "all providers failed": nothing was asked.
        // `AIProviderAttempt.summary` says exactly that for an empty list, so
        // the one error case covers it and the second case this type used to
        // declare (`ChainError.noProviders`) is gone — two error types for one
        // chain meant a caller had to catch both to learn the same thing
        // (AIKit#18, panel finding on `FallbackProviderChain.swift:63`).
        guard !providers.isEmpty else {
            throw AIKitError.allProvidersFailed([])
        }

        let deadline = ContinuousClock.now.advanced(by: .seconds(timeout))
        var attempts: [AIProviderAttempt] = []

        for (index, provider) in providers.enumerated() {
            // Between attempts, not inside them: a cancelled task must not
            // start asking the next provider. Without this the chain kept
            // walking its whole list after the caller had gone away, holding a
            // subprocess slot and a budget for an answer nobody would read.
            try Task.checkCancellation()

            let remaining = ContinuousClock.now.duration(to: deadline)
            guard remaining > .zero else {
                attempts.append(
                    AIProviderAttempt(
                        providerID: provider.id,
                        cause: "not asked: the chain's \(Int(timeout))s budget was already spent"
                    )
                )
                // Every provider after this one is in the same position, and
                // saying so once per provider is noise. Record this one and
                // stop.
                break
            }

            do {
                logger.info(
                    "Trying provider \(index + 1)/\(providers.count)",
                    metadata: [
                        "provider": "\(provider.id)",
                        "remaining": "\(Self.seconds(of: remaining))s",
                    ])
                return try await provider.run(prompt: prompt, timeout: Self.seconds(of: remaining))
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                attempts.append(AIProviderAttempt(providerID: provider.id, error: error))
                guard isFallbackEligible(error) else {
                    logger.error(
                        "Provider \(provider.id) failed (non-recoverable)",
                        metadata: ["error": "\(error)"]
                    )
                    // Rethrow the real error when it is all that happened, so
                    // the caller keeps the typed cause; wrap once earlier
                    // providers have already failed, so their causes survive.
                    if attempts.count == 1 {
                        throw error
                    }
                    throw AIKitError.allProvidersFailed(attempts)
                }
                logger.warning(
                    "Provider \(provider.id) failed (fallback-eligible), trying next",
                    metadata: ["error": "\(error)"]
                )
            }
        }

        // All providers exhausted. Every cause, not just the last one: a chain
        // whose primary was rate-limited and whose fallback refused the
        // connection used to report only the refusal (BR-AIL-07).
        throw AIKitError.allProvidersFailed(attempts)
    }

    // MARK: - Fallback Eligibility

    /// Determines whether an error should trigger fallback to the next provider.
    ///
    /// Delegates to `FallbackEligibility`, the one rule this chain now shares
    /// with `ProviderRouter.routeWithFallback`. The rule used to live here and
    /// the router had none at all, so the same error fell back in one path and
    /// not the other. Rate limits (429) and connection errors stay eligible.
    static func isFallbackEligible(_ error: any Error) -> Bool {
        FallbackEligibility.allows(error)
    }

    /// Instance check: the built-in rules, then the caller's own.
    /// A caller whose providers throw their own error types — shikki's
    /// `SpecPipelineError`, for one — supplies `extraFallbackEligible`.
    func isFallbackEligible(_ error: any Error) -> Bool {
        Self.isFallbackEligible(error) || extraFallbackEligible(error)
    }

    /// A `Duration` as the seconds a `TimeInterval` API takes.
    static func seconds(of duration: Duration) -> TimeInterval {
        let components = duration.components
        return TimeInterval(components.seconds) + TimeInterval(components.attoseconds) / 1e18
    }
}
