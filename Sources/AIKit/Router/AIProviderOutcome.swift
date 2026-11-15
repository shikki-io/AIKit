import Foundation

// MARK: - AIProviderOutcome

/// What ONE provider did with a request that was fanned out to several
/// (BR-AIL-22).
///
/// W32 of spec ai-layer-aikit-core-2026-10-31. Operator, 2026-11-12: *"shikki
/// gonna have the possibility to target every ai providers to orchestrate with
/// the public api … dispatch on different ai provider in //"*. A fan-out has no
/// single answer and no single error, so it cannot return `AIResponse` and it
/// must not throw: one branch answering and another timing out is a NORMAL
/// result, and a caller comparing two models needs both halves of it.
///
/// Every outcome names its provider, because that is the same rule a single
/// answer obeys (BR-AIL-06): a response that cannot name its author hides a
/// misrouted request, and in a fan-out the author is the whole point — the
/// caller asked four providers precisely to compare who said what.
///
/// A failure keeps its cause (BR-AIL-07) rather than being dropped, which is
/// the fan-out form of the `catch { continue }` this spec removed from
/// `routeWithFallback`.
public struct AIProviderOutcome: Sendable, Equatable {

    // MARK: - Verdict

    /// The three things a branch can end as.
    ///
    /// `timedOut` is a case of its own rather than an ``AIKitError``: the
    /// provider did not fail, it did not finish inside the budget the CALLER
    /// set, and the two read differently in a log. There is no AIKitError case
    /// for "the caller's deadline arrived" and inventing one would make a
    /// caller's own choice look like a server fault.
    public enum Verdict: Sendable, Equatable {

        /// The provider answered. The response names itself and its model.
        case answered(AIResponse)

        /// The provider threw.
        ///
        /// `cause` is the error's own sentence and is always present — it is
        /// what an operator reads. `error` is the typed value when the provider
        /// threw an ``AIKitError``, and `nil` when it threw something else (a
        /// `URLError`, a caller's own type): carrying `any Error` would cost
        /// this type `Equatable` and `Sendable`, exactly as it would
        /// ``AIProviderAttempt``, so the shape matches that precedent.
        case failed(cause: String, error: AIKitError?)

        /// The shared deadline arrived before this provider answered.
        ///
        /// Carries the budget the whole fan-out was given, so the line reads
        /// "did not answer inside 2s" rather than naming a per-branch number
        /// nobody set.
        case timedOut(budget: Duration)
    }

    // MARK: - Properties

    /// The ``AIProvider/id`` that was asked.
    public let providerID: String

    /// The model behind the answer, or the model that was asked for when
    /// nothing answered.
    ///
    /// `nil` only when neither names one: a failed branch whose request carried
    /// no ``AIRequest/model`` has no model to report, and reporting the
    /// provider's id in its place would invent a fact.
    public let model: String?

    /// How long this branch took, measured on a `ContinuousClock` — so it is
    /// wall time that does not move when the system clock does.
    ///
    /// Measured for a failure and a timeout too: "refused after 3 ms" and
    /// "refused after 9 s" are different news about the same refusal.
    public let latency: Duration

    /// What happened.
    public let verdict: Verdict

    // MARK: - Init

    public init(providerID: String, model: String?, latency: Duration, verdict: Verdict) {
        self.providerID = providerID
        self.model = model
        self.latency = latency
        self.verdict = verdict
    }

    // MARK: - Reading an outcome

    /// The answer, when this branch produced one.
    public var response: AIResponse? {
        guard case .answered(let response) = verdict else { return nil }
        return response
    }

    /// Whether this branch answered.
    public var isAnswer: Bool { response != nil }

    /// Why this branch produced no answer, in the error's own words, or `nil`
    /// when it did answer.
    public var cause: String? {
        switch verdict {
        case .answered:
            nil
        case .failed(let cause, _):
            cause
        case .timedOut(let budget):
            "did not answer inside the fan-out's \(budget) budget"
        }
    }

    /// The typed error, when the provider threw an ``AIKitError``.
    ///
    /// `nil` for an answer, for a timeout, and for a provider that threw a type
    /// AIKit does not own — in which case ``cause`` still carries its sentence.
    public var error: AIKitError? {
        guard case .failed(_, let error) = verdict else { return nil }
        return error
    }

    /// This outcome as a fallback ATTEMPT, or `nil` when it answered.
    ///
    /// The bridge to the evidence vocabulary the rest of the kit already
    /// speaks: a caller whose whole fan-out came back empty builds
    /// ``AIKitError/allProvidersFailed(_:)`` out of these and gets one error
    /// naming every provider and its cause (BR-AIL-07), instead of inventing a
    /// second way to report the same thing.
    ///
    /// ```swift
    /// let outcomes = try await router.fanOut(request, to: providers, within: .seconds(30))
    /// guard outcomes.contains(where: \.isAnswer) else {
    ///     throw AIKitError.allProvidersFailed(outcomes.compactMap(\.attempt))
    /// }
    /// ```
    public var attempt: AIProviderAttempt? {
        guard let cause else { return nil }
        return AIProviderAttempt(providerID: providerID, cause: cause)
    }
}

// MARK: - CustomStringConvertible

extension AIProviderOutcome: CustomStringConvertible {

    /// One line per branch, for a log or a CLI table.
    public var description: String {
        let who = model.map { "\(providerID) (\($0))" } ?? providerID
        switch verdict {
        case .answered(let response):
            return "\(who) answered in \(latency): \(response.content.count) characters"
        case .failed(let cause, _):
            return "\(who) failed after \(latency): \(cause)"
        case .timedOut(let budget):
            return "\(who) did not answer inside \(budget)"
        }
    }
}
