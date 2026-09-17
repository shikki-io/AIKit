import Foundation
import Logging

// MARK: - FallbackProviderChain

/// Wraps multiple AgentProviding implementations in priority order.
/// If the primary provider fails with a rate limit (429) or connection error,
/// automatically falls back to the next provider in the chain.
///
/// Usage:
/// ```swift
/// let chain = FallbackProviderChain(providers: [
///     claudeProvider,    // primary — cloud
///     lmStudioProvider,  // fallback — local
/// ])
/// let result = try await chain.run(prompt: "...", timeout: 300)
/// ```
public struct FallbackProviderChain: AgentProviding, Sendable {
    private let providers: [any AgentProviding]
    private let logger: Logger
    /// Caller-supplied rule for error types this package does not know about.
    /// Consulted after the built-in LMStudio / URLError rules.
    private let extraFallbackEligible: @Sendable (any Error) -> Bool

    // MARK: - Errors

    public enum ChainError: Error, Sendable {
        /// All providers in the chain failed. Contains the last error encountered.
        case allProvidersFailed(String)
        /// No providers were configured in the chain.
        case noProviders
    }

    // MARK: - Init

    public init(
        providers: [any AgentProviding],
        logger: Logger = Logger(label: "aikit.fallback-chain"),
        extraFallbackEligible: @escaping @Sendable (any Error) -> Bool = { _ in false }
    ) {
        self.providers = providers
        self.logger = logger
        self.extraFallbackEligible = extraFallbackEligible
    }

    // MARK: - AgentProviding

    public func run(prompt: String, timeout: TimeInterval) async throws -> String {
        guard !providers.isEmpty else {
            throw ChainError.noProviders
        }

        var lastError: (any Error)?

        for (index, provider) in providers.enumerated() {
            let providerName = String(describing: type(of: provider))
            do {
                logger.info("Trying provider \(index + 1)/\(providers.count)", metadata: [
                    "provider": "\(providerName)"
                ])
                let result = try await provider.run(prompt: prompt, timeout: timeout)
                return result
            } catch {
                lastError = error
                if isFallbackEligible(error) {
                    logger.warning(
                        "Provider \(providerName) failed (fallback-eligible), trying next",
                        metadata: ["error": "\(error)"]
                    )
                    continue
                } else {
                    // Non-fallback-eligible errors propagate immediately
                    logger.error(
                        "Provider \(providerName) failed (non-recoverable)",
                        metadata: ["error": "\(error)"]
                    )
                    throw error
                }
            }
        }

        // All providers exhausted
        let message = lastError.map { String(describing: $0) } ?? "unknown"
        throw ChainError.allProvidersFailed(
            "All \(providers.count) providers failed. Last error: \(message)"
        )
    }

    // MARK: - Fallback Eligibility

    /// Determines whether an error should trigger fallback to the next provider.
    /// Rate limits (429) and connection errors are fallback-eligible.
    static func isFallbackEligible(_ error: any Error) -> Bool {
        // LM Studio specific errors
        if let lmError = error as? LMStudioProvider.LMStudioError {
            switch lmError {
            case .connectionRefused, .rateLimited:
                return true
            case .invalidResponse, .httpError, .emptyContent:
                return false
            }
        }

        // URLError network issues
        if let urlError = error as? URLError {
            switch urlError.code {
            case .cannotConnectToHost, .networkConnectionLost, .timedOut, .cannotFindHost:
                return true
            default:
                return false
            }
        }

        return false
    }

    /// Instance check: the built-in rules, then the caller's own.
    /// A caller whose providers throw their own error types — shikki's
    /// `SpecPipelineError`, for one — supplies `extraFallbackEligible`.
    func isFallbackEligible(_ error: any Error) -> Bool {
        Self.isFallbackEligible(error) || extraFallbackEligible(error)
    }
}
