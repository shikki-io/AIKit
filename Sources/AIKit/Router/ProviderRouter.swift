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
    public func routeSmart(
        request: AIRequest,
        context: UsageContext,
        providers candidateProviders: [any AIProvider]? = nil
    ) async -> (any AIProvider)? {
        let candidates = candidateProviders ?? providers

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

    /// Try providers in order until one succeeds. Throws allProvidersFailed if none work.
    public func routeWithFallback(
        request: AIRequest,
        providers fallbackProviders: [any AIProvider]
    ) async throws -> AIResponse {
        for provider in fallbackProviders {
            do {
                return try await provider.complete(request: request)
            } catch {
                continue
            }
        }
        throw AIKitError.allProvidersFailed
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
