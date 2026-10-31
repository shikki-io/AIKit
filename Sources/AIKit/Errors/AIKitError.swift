import Foundation

/// Errors thrown by AIKit operations.
public enum AIKitError: Error, LocalizedError, Equatable {
    case noProviderAvailable(AICapabilities)
    case modelNotFound(ModelIdentifier)
    case downloadFailed(String)
    case engineUnavailable(String)
    case requestFailed(String)
    case streamingNotSupported
    /// Every provider in a fallback list was tried and none answered.
    ///
    /// Carries one `AIProviderAttempt` per provider that was asked, in the order
    /// they were tried. A bare case here used to discard every cause — the
    /// request simply "failed", and the reason a local server refused the
    /// connection or a cloud one rate-limited was gone (BR-AIL-07).
    case allProvidersFailed([AIProviderAttempt])

    public var errorDescription: String? {
        switch self {
        case .noProviderAvailable(let capabilities):
            "No provider available for capabilities: \(capabilities)"
        case .modelNotFound(let id):
            "Model not found: \(id)"
        case .downloadFailed(let reason):
            "Download failed: \(reason)"
        case .engineUnavailable(let engine):
            "Engine unavailable: \(engine)"
        case .requestFailed(let reason):
            "Request failed: \(reason)"
        case .streamingNotSupported:
            "Streaming is not supported by this provider"
        case .allProvidersFailed(let attempts):
            AIProviderAttempt.summary(of: attempts)
        }
    }
}

// MARK: - AIProviderAttempt

/// One provider's failed turn in a fallback list: who was asked, and why it did
/// not answer.
///
/// The cause is kept as text rather than as the error value itself. An
/// `any Error` existential is neither `Equatable` nor `Sendable`, and
/// `AIKitError` is both — carrying the live error would cost the whole error
/// family conformances that callers and tests already rely on. The description
/// is what an operator reads in a log or a CLI line, so nothing diagnostic is
/// lost; a caller that must branch on a type catches it at the provider instead.
public struct AIProviderAttempt: Sendable, Equatable, CustomStringConvertible {
    /// The `AIProvider.id` that was asked.
    public let providerID: String
    /// Why it did not answer, in the error's own words.
    public let cause: String

    public init(providerID: String, cause: String) {
        self.providerID = providerID
        self.cause = cause
    }

    /// Records an attempt from the error a provider threw.
    public init(providerID: String, error: any Error) {
        self.init(providerID: providerID, cause: Self.describe(error))
    }

    public var description: String { "\(providerID) — \(cause)" }

    /// The error's own sentence when it writes one, its value otherwise.
    ///
    /// `localizedDescription` on a plain `struct` error yields Foundation's
    /// "The operation couldn't be completed" boilerplate, which names nothing.
    /// A `LocalizedError` — every `AIKitError` — does name itself.
    static func describe(_ error: any Error) -> String {
        if let described = (error as? any LocalizedError)?.errorDescription {
            return described
        }
        return String(describing: error)
    }

    /// One line naming every provider that was asked and what it said.
    static func summary(of attempts: [AIProviderAttempt]) -> String {
        guard !attempts.isEmpty else {
            // Not the same failure: nothing was asked, so no provider failed.
            // Reporting "all providers failed" for an empty list would send a
            // reader looking for a provider fault that never happened.
            return "No provider was tried: the fallback list was empty"
        }
        let causes = attempts.map(\.description).joined(separator: "; ")
        return "All \(attempts.count) provider(s) failed: \(causes)"
    }
}

// MARK: - FallbackEligibility

/// Whether a failed attempt may be retried on the next provider.
///
/// The one rule, consulted by both `ProviderRouter.routeWithFallback` and
/// `FallbackProviderChain`. The two were written apart and drifted: the chain
/// weighed each error, the router retried on anything with `catch { continue }`.
///
/// The question each case answers is the same one the chain always asked: could
/// the *next* provider plausibly serve this request? A server that is down,
/// rate-limiting, or missing the model is a fact about that provider, so the
/// next one gets a turn. A reply the server really did produce, or a refusal
/// that will repeat everywhere, is not worth asking again.
public enum FallbackEligibility {

    /// Whether `error` means "ask the next provider".
    ///
    /// Unknown error types are not eligible. A caller whose providers throw
    /// their own types widens the rule through
    /// `FallbackProviderChain.extraFallbackEligible` rather than by editing
    /// this one.
    public static func allows(_ error: any Error) -> Bool {
        if let aiKitError = error as? AIKitError {
            return allows(aiKitError)
        }

        // LM Studio, reached as an `AgentProviding` backend.
        if let lmError = error as? LMStudioProvider.LMStudioError {
            switch lmError {
            case .connectionRefused, .rateLimited:
                return true
            case .invalidResponse, .httpError, .emptyContent:
                // The server answered. Its answer is the news, not a reason to
                // ask elsewhere.
                return false
            }
        }

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

    private static func allows(_ error: AIKitError) -> Bool {
        switch error {
        case .engineUnavailable:
            // This engine cannot run here — an MLX provider on iOS, Core ML off
            // an Apple platform. Exactly what a fallback list exists for.
            true

        case .noProviderAvailable:
            // The provider refuses the capability itself: `whisper.cpp` is not
            // installed, so the Core ML transcriber behind it gets the request.
            true

        case .modelNotFound:
            // Absent from this provider's registry; the next one may hold it.
            true

        case .requestFailed:
            // The catch-all, and the honest reading is that it is two errors
            // under one name: a server that could not be reached or replied with
            // an error body, and a request that was malformed before any server
            // saw it ("No user message found", "carries 0 bytes of audio").
            // Eligible, because the unreachable-server reading is the one a
            // fallback list is built for, and because the alternative — refusing
            // to fall back when a local server is simply down — is the failure
            // the operator actually hits. The cost is that a malformed request
            // is now retried on every provider; the attempt list is what keeps
            // that visible, printing the same cause once per provider instead of
            // hiding it behind a bare "all providers failed".
            true

        case .downloadFailed:
            // A registry transfer, not a completion attempt. Retrying the
            // request elsewhere does not finish a download.
            false

        case .streamingNotSupported:
            // A refusal that repeats: the caller asked a provider to stream and
            // must see that, not a different provider's answer.
            false

        case .allProvidersFailed:
            // A nested fallback already exhausted its own list and is carrying
            // its own evidence. Spending another list on it would bury that.
            false
        }
    }
}
