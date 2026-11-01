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
        for provider in providers {
            if provider.capabilities.contains(capabilities) {
                let status = await provider.status
                if case .ready = status {
                    return provider
                }
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
            do {
                return try await provider.complete(request: request)
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
