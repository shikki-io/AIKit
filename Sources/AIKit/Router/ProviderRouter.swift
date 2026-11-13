/// Routes AI requests to the best available provider.
public struct ProviderRouter: Sendable {
    private let providers: [any AIProvider]
    private let performanceStore: PerformanceStore?

    public init(
        providers: [any AIProvider],
        performanceStore: PerformanceStore? = nil
    ) {
        self.providers = providers
        self.performanceStore = performanceStore
    }

    /// Select best provider for the given capabilities.
    /// Returns the first provider whose capabilities are a superset of the requested ones.
    public func route(capabilities: AICapabilities) async -> (any AIProvider)? {
        for provider in providers where provider.capabilities.contains(capabilities) {
            let status = await provider.status
            if case .ready = status {
                return provider
            }
        }
        return nil
    }

    /// Pick the best provider based on historical performance data.
    /// Falls back to capability-based routing if no performance data exists.
    ///
    /// Synthesis preference (G7 / local-first profile):
    /// When the request requires `.synthesis`, ready MLX providers (id prefix "mlx/")
    /// are tried first — this routes Synthesizer traffic to the local MLX LLM
    /// before falling back to cloud providers.
    public func routeSmart(
        request: AIRequest,
        context: UsageContext,
        providers candidateProviders: [any AIProvider]? = nil
    ) async -> (any AIProvider)? {
        let candidates = candidateProviders ?? providers

        // Synthesis fast-path: if the model is an MLX LLM provider and is ready,
        // prefer it unconditionally — local-first, zero cost.
        if request.model?.provider == "mlx" {
            for provider in candidates where provider.id.hasPrefix("mlx/") {
                if provider.capabilities.contains(.synthesis) {
                    let status = await provider.status
                    if case .ready = status {
                        return provider
                    }
                }
            }
        }

        guard let store = performanceStore else {
            // No performance data — fall back to first ready provider.
            return await firstReady(from: candidates)
        }

        // Check if there's a best model for this context.
        if let bestModelId = store.bestModel(for: context) {
            // Find the provider that matches the best model.
            for provider in candidates {
                if provider.id.contains(bestModelId.modelId) || provider.id == bestModelId.description {
                    let status = await provider.status
                    if case .ready = status {
                        return provider
                    }
                }
            }
        }

        // No performance match — return first ready provider.
        return await firstReady(from: candidates)
    }

    private func firstReady(from candidates: [any AIProvider]) async -> (any AIProvider)? {
        for provider in candidates {
            let status = await provider.status
            if case .ready = status {
                return provider
            }
        }
        return nil
    }

    /// Try providers in order until one succeeds, keeping every failure.
    ///
    /// The answer carries its own `providerID`, so a response that came from the
    /// second provider says so (BR-AIL-06). When nobody answers, the thrown
    /// `AIKitError.allProvidersFailed` lists each provider and its cause
    /// (BR-AIL-07) — this method used to swallow all of them in
    /// `catch { continue }` and throw a bare case.
    ///
    /// Whether a failure moves on to the next provider is
    /// `FallbackEligibility`'s single rule, shared with
    /// `FallbackProviderChain`. An error that no other provider can fix stops
    /// the walk rather than burning the rest of the list.
    ///
    /// No deadline is imposed here: a timeout belongs to the caller, because a
    /// local model loading weights for the first time is slow, not down, and a
    /// router-side deadline would read it as down.
    ///
    /// - Throws: the provider's own error when the first attempt fails for a
    ///   reason no other provider can fix; `AIKitError.allProvidersFailed`
    ///   otherwise, carrying one attempt per provider asked.
    public func routeWithFallback(
        request: AIRequest,
        providers fallbackProviders: [any AIProvider]
    ) async throws -> AIResponse {
        var attempts: [AIProviderAttempt] = []

        for provider in fallbackProviders {
            // Between attempts (W22): a cancelled caller must not start asking
            // the next provider. The chain this shares its eligibility rule
            // with does the same, so the two paths now stop at the same point.
            try Task.checkCancellation()

            do {
                return try await provider.complete(request: request)
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                attempts.append(AIProviderAttempt(providerID: provider.id, error: error))

                guard FallbackEligibility.allows(error) else {
                    // Trying the rest cannot help. Rethrow the real error when it
                    // is all that happened, so the caller keeps the typed cause;
                    // wrap it once earlier providers have already failed, so
                    // their causes are not dropped on the way out.
                    if attempts.count == 1 {
                        throw error
                    }
                    throw AIKitError.allProvidersFailed(attempts)
                }
            }
        }

        throw AIKitError.allProvidersFailed(attempts)
    }

    /// Try THIS router's own providers in order until one succeeds.
    ///
    /// The order is the order the router was built with, which for a router
    /// resolved out of ``AIKitAssembly`` is the order of the `[[providers]]`
    /// rows (BR-AIL-11, W22): local first is the operator's file, not a rule in
    /// code. A caller that had to pass the list back in was re-deciding an
    /// order settings had already decided.
    ///
    /// - Throws: as ``routeWithFallback(request:providers:)``.
    public func routeWithFallback(request: AIRequest) async throws -> AIResponse {
        try await routeWithFallback(request: request, providers: providers)
    }

    /// The providers this router holds, in the order it will try them.
    public var orderedProviders: [any AIProvider] { providers }

    // MARK: - W32 — orchestration by capability, device and topic (BR-AIL-22)

    /// The providers that qualify for this request under this policy, in the
    /// order to ask them — and every provider that does not, with its reason.
    ///
    /// BR-AIL-22, operator 2026-11-12: *"shikki gonna have the possibility to
    /// target every ai providers to orchestrate with the public api … in
    /// function of the topics and the capabilities of the current device"*.
    /// Orchestration is part of the talk door, not something each caller
    /// rebuilds: before this, selecting by capability meant
    /// ``route(capabilities:)``, which returns ONE provider and keeps no record
    /// of the ones it walked past, so a caller that wanted several had to
    /// re-implement the filter and a caller that wanted to know why a provider
    /// was absent could not find out.
    ///
    /// Three filters, in this order, and the order is not arbitrary:
    ///  1. **Capability** — the policy's own, union what the request implies
    ///     (``AIRoutingPolicy/requiredCapabilities(for:)``). Pure.
    ///  2. **Device fit** — a LOCAL provider is kept only if the device that
    ///     would hold its weights can (``AILocalModelPlacement``). Pure.
    ///  3. **Readiness** — `await provider.status` must be `.ready`.
    ///
    /// The two pure filters run first so that a provider already out is never
    /// asked for its status: on an HTTP provider that is a network round trip,
    /// and spending one to learn whether a model this device cannot load is
    /// awake is a cost with no answer attached.
    ///
    /// Then the topic ranking REORDERS the survivors, stably: the ids the topic
    /// names come first in the named order, and the rest keep the order they
    /// were given in — which for a router built by ``AIKitAssembly`` is the
    /// order of the `[[providers]]` rows, the operator's own (BR-AIL-11).
    ///
    /// - Parameters:
    ///   - request: the request to be served. Read for the capabilities it
    ///     implies; nothing is sent.
    ///   - policy: capabilities, topic ranking and device filter, as data.
    ///   - candidateProviders: the pool to filter, or `nil` for this router's
    ///     own list.
    /// - Returns: the selection and the rejections. Never throws: "nobody
    ///   qualified" is an answer with reasons in it, and a thrown
    ///   ``AIKitError/noProviderAvailable(_:)`` would carry only the
    ///   capabilities and lose every reason.
    public func candidates(
        for request: AIRequest,
        policy: AIRoutingPolicy,
        providers candidateProviders: [any AIProvider]? = nil
    ) async -> AIRoutingPlan {
        let pool = candidateProviders ?? providers
        let required = policy.requiredCapabilities(for: request)

        var kept: [any AIProvider] = []
        var rejected: [AIProviderRejection] = []

        for provider in pool {
            guard provider.capabilities.contains(required) else {
                rejected.append(
                    AIProviderRejection(
                        providerID: provider.id,
                        rejection: .missingCapabilities(
                            required: required, offered: provider.capabilities
                        )
                    ))
                continue
            }

            if let deviceRejection = policy.deviceRejection(forProviderID: provider.id) {
                rejected.append(
                    AIProviderRejection(providerID: provider.id, rejection: deviceRejection))
                continue
            }

            let status = await provider.status
            guard case .ready = status else {
                rejected.append(
                    AIProviderRejection(providerID: provider.id, rejection: .notReady(status)))
                continue
            }

            kept.append(provider)
        }

        return AIRoutingPlan(selected: Self.ranked(kept, by: policy), rejected: rejected)
    }

    /// `providers` reordered by the policy's topic ranking, stably.
    ///
    /// `sorted(by:)` is not documented as stable in the Swift standard library,
    /// so the original index is part of the comparison rather than trusted:
    /// without it, two unranked providers could swap places between runs and a
    /// caller's configured order would become non-deterministic for no reason
    /// anyone asked for.
    private static func ranked(
        _ providers: [any AIProvider], by policy: AIRoutingPolicy
    ) -> [any AIProvider] {
        guard policy.ranksByTopic else { return providers }
        return providers.enumerated()
            .sorted { left, right in
                let leftRank = policy.rank(ofProviderID: left.element.id) ?? Int.max
                let rightRank = policy.rank(ofProviderID: right.element.id) ?? Int.max
                if leftRank != rightRank { return leftRank < rightRank }
                return left.offset < right.offset
            }
            .map(\.element)
    }

    /// Asks EVERY provider the same request at once, and returns what each one
    /// said.
    ///
    /// BR-AIL-22: *"fans one request out to N providers in parallel and gets
    /// every answer back"*. One deadline for the whole fan-out, cancellation
    /// propagated to every branch, and outcomes in the order the providers were
    /// GIVEN — not the order they finished, because a caller comparing four
    /// models reads the list against the list it passed, and a result order
    /// that depends on who was fastest is a result it cannot index.
    ///
    /// Each answer names its provider and model (BR-AIL-06) and each failure
    /// keeps its cause (BR-AIL-07): see ``AIProviderOutcome``.
    ///
    /// ## This is not a fallback
    ///
    /// ``routeWithFallback(request:providers:)`` asks providers in turn until
    /// one answers, and spends nothing on the rest. This asks all of them and
    /// pays for all of them. Use it to compare answers or to seat a panel; do
    /// not use it for reliability, where the chain is both cheaper and the
    /// thing that keeps an ordered, typed failure.
    ///
    /// ## What the deadline does, honestly
    ///
    /// At the deadline each unfinished branch is CANCELLED and reported
    /// `.timedOut`. Structured concurrency means this method cannot return
    /// before its children do, so a provider that ignores cancellation delays
    /// the whole call until it returns — the outcome is still `.timedOut`, but
    /// the wall time is the stubborn provider's. Every provider in this package
    /// awaits either a `URLSession` byte stream or `Task.sleep`, both of which
    /// honour cancellation; a caller injecting its own must do the same.
    ///
    /// - Parameters:
    ///   - request: the one request every provider is asked.
    ///   - fanOutProviders: who to ask. An empty list returns an empty result —
    ///     nothing was asked, so nothing failed.
    ///   - deadline: the instant the whole fan-out must be done by, shared by
    ///     every branch.
    /// - Returns: one outcome per provider, in the order given.
    /// - Throws: `CancellationError`, and only that — when the CALLING task is
    ///   cancelled. A provider's own error is an outcome, not a throw.
    public func fanOut(
        _ request: AIRequest,
        to fanOutProviders: [any AIProvider],
        deadline: ContinuousClock.Instant
    ) async throws -> [AIProviderOutcome] {
        guard !fanOutProviders.isEmpty else { return [] }
        // Before anything is spent: a cancelled caller must not open N
        // connections for an answer nobody will read. The same check
        // `routeWithFallback` and `FallbackProviderChain` make between
        // attempts, made here before the only moment this method has.
        try Task.checkCancellation()

        let model = request.model?.description
        // Measured ONCE, so every branch reports the same figure: a per-branch
        // "time it had left" would make N outcomes carry N different budgets
        // for one shared deadline, which is the opposite of what this method
        // promises.
        let budget = ContinuousClock.now.duration(to: deadline)

        guard budget > .zero else {
            // A deadline already in the past. Reported rather than asked, so a
            // caller that passed a spent budget sees that it was spent instead
            // of N requests leaving the host anyway.
            return fanOutProviders.map { provider in
                AIProviderOutcome(
                    providerID: provider.id,
                    model: model,
                    latency: .zero,
                    verdict: .timedOut(budget: .zero)
                )
            }
        }

        return try await withThrowingTaskGroup(of: (Int, AIProviderOutcome).self) { group in
            for (index, provider) in fanOutProviders.enumerated() {
                group.addTask {
                    (
                        index,
                        try await Self.ask(
                            provider,
                            request: request,
                            deadline: deadline,
                            budget: budget,
                            requestedModel: model
                        )
                    )
                }
            }

            // Indexed, not appended: `group.next()` yields in completion order,
            // and the contract is the order the caller gave.
            var outcomes = [AIProviderOutcome?](repeating: nil, count: fanOutProviders.count)
            for try await (index, outcome) in group {
                outcomes[index] = outcome
            }
            // Total by construction — one task per index, every task yields its
            // own index — so a nil here would be a bug in this method and not a
            // provider's doing.
            return outcomes.compactMap { $0 }
        }
    }

    /// ``fanOut(_:to:deadline:)`` with the deadline stated as a budget from
    /// now.
    ///
    /// The form most callers want: a budget is what a caller has ("30 seconds
    /// for this panel"), an instant is what a shared deadline IS, and
    /// converting at the boundary keeps both honest instead of re-deriving the
    /// instant inside each branch.
    public func fanOut(
        _ request: AIRequest,
        to fanOutProviders: [any AIProvider],
        within budget: Duration
    ) async throws -> [AIProviderOutcome] {
        try await fanOut(
            request,
            to: fanOutProviders,
            deadline: ContinuousClock.now.advanced(by: budget)
        )
    }

    /// One branch: ask this provider, racing its answer against the shared
    /// deadline.
    ///
    /// - Throws: `CancellationError`, and only that — when this branch's task
    ///   was cancelled, which means the fan-out as a whole is over. Throwing
    ///   rather than returning an outcome is what propagates the cancellation:
    ///   the outer group rethrows it to the caller and cancels every sibling,
    ///   so "cancellation stops every branch" is structural rather than a thing
    ///   each branch remembers to do.
    private static func ask(
        _ provider: any AIProvider,
        request: AIRequest,
        deadline: ContinuousClock.Instant,
        budget: Duration,
        requestedModel: String?
    ) async throws -> AIProviderOutcome {
        let clock = ContinuousClock()
        let started = clock.now

        func outcome(_ verdict: AIProviderOutcome.Verdict, model: String?) -> AIProviderOutcome {
            AIProviderOutcome(
                providerID: provider.id,
                model: model,
                latency: started.duration(to: clock.now),
                verdict: verdict
            )
        }

        // The race, run in a NON-throwing group carrying a `Result`.
        //
        // Each child catches for itself, so the only value that leaves the race
        // is the one that finished FIRST: the loser is cancelled by
        // `cancelAll()` and its result is never read. A throwing group would
        // leave the loser's error to the group's own discard rules, which is a
        // behaviour to look up rather than to read here.
        //
        // `AIResponse?`: the sleeping child returns nil, which is how the
        // deadline wins the race without throwing — a thrown timeout would be
        // indistinguishable from the provider's own failure.
        let first = await withTaskGroup(of: Result<AIResponse?, any Error>.self) { race in
            race.addTask {
                do {
                    return .success(try await provider.complete(request: request))
                } catch {
                    return .failure(error)
                }
            }
            race.addTask {
                do {
                    try await Task.sleep(until: deadline, clock: clock)
                    return .success(nil)
                } catch {
                    return .failure(error)
                }
            }
            // Total: a group with two children yields at least one result.
            let first = await race.next() ?? .success(nil)
            race.cancelAll()
            return first
        }

        do {
            guard let answered = try first.get() else {
                return outcome(.timedOut(budget: budget), model: requestedModel)
            }
            return outcome(.answered(answered), model: answered.model)
        } catch where Task.isCancelled {
            // The fan-out is over: the caller went away. Not this provider's
            // failure, so it is not reported as one — it is rethrown, and the
            // outer group carries it to the caller while cancelling the
            // siblings.
            //
            // Keyed on `Task.isCancelled` rather than on the error being a
            // `CancellationError`, because both halves of that test are
            // unreliable alone: a cancelled `URLSession` throws `URLError`
            // (`.cancelled`), not `CancellationError`, and a provider is free
            // to throw `CancellationError` for its own reasons — which, with no
            // cancellation in force, is a provider failure like any other and
            // falls to the next clause.
            throw CancellationError()
        } catch {
            return outcome(
                .failed(
                    cause: AIProviderAttempt.describe(error),
                    error: error as? AIKitError
                ),
                model: requestedModel
            )
        }
    }
}

/// User preferences for provider selection.
public struct ProviderPreferences: Sendable, Codable, Equatable {
    /// Prefer local models over API.
    public var preferLocal: Bool
    /// Maximum acceptable latency (ms). 0 = no limit.
    public var maxLatencyMs: Int
    /// Budget: max cost per request (USD). 0 = free only.
    public var maxCostPerRequest: Double
    /// Required capabilities.
    public var requiredCapabilities: AICapabilities

    public init(
        preferLocal: Bool = true,
        maxLatencyMs: Int = 0,
        maxCostPerRequest: Double = 0,
        requiredCapabilities: AICapabilities = []
    ) {
        self.preferLocal = preferLocal
        self.maxLatencyMs = maxLatencyMs
        self.maxCostPerRequest = maxCostPerRequest
        self.requiredCapabilities = requiredCapabilities
    }
}
