import Foundation

/// Persists performance statistics per model+context pair.
/// Enables smart routing by tracking latency, throughput, and quality scores.
public final class PerformanceStore: @unchecked Sendable {
    private let storePath: URL
    private let lock = NSLock()
    private var entries: [String: PerformanceStat] = [:]

    /// Create a performance store.
    /// - Parameter storePath: File URL for persistence. Defaults to `~/.aikit/performance.json`.
    public init(storePath: URL? = nil) {
        self.storePath = storePath ?? Self.defaultStorePath()
        load()
    }

    // MARK: - Public API

    /// Record a completed inference.
    public func record(
        model: ModelIdentifier,
        context: UsageContext,
        latencyMs: Int,
        tokensPerSecond: Double,
        qualityScore: Double? = nil
    ) {
        let key = Self.key(model: model, context: context)
        lock.lock()
        defer { lock.unlock() }

        if var existing = entries[key] {
            let count = existing.totalInvocations
            existing.avgLatencyMs = (existing.avgLatencyMs * count + latencyMs) / (count + 1)
            existing.avgTokensPerSecond = (existing.avgTokensPerSecond * Double(count) + tokensPerSecond) / Double(count + 1)
            if let newScore = qualityScore {
                if let oldScore = existing.qualityScore {
                    existing.qualityScore = (oldScore * Double(count) + newScore) / Double(count + 1)
                } else {
                    existing.qualityScore = newScore
                }
            }
            existing.totalInvocations = count + 1
            existing.lastUsedAt = Date()
            entries[key] = existing
        } else {
            entries[key] = PerformanceStat(
                avgLatencyMs: latencyMs,
                avgTokensPerSecond: tokensPerSecond,
                qualityScore: qualityScore,
                totalInvocations: 1,
                lastUsedAt: Date()
            )
        }
        save()
    }

    /// Get stats for a model in a specific context.
    public func stats(for model: ModelIdentifier, context: UsageContext) -> PerformanceStat? {
        let key = Self.key(model: model, context: context)
        lock.lock()
        defer { lock.unlock() }
        return entries[key]
    }

    /// Get best model for a context (highest quality score, or fastest if no quality data).
    public func bestModel(for context: UsageContext) -> ModelIdentifier? {
        lock.lock()
        defer { lock.unlock() }

        let contextSuffix = "|\(Self.contextKey(context))"
        let matching = entries.filter { $0.key.hasSuffix(contextSuffix) }

        guard !matching.isEmpty else { return nil }

        // Prefer highest quality score. Fall back to fastest (highest tokens/sec).
        let hasQuality = matching.filter { $0.value.qualityScore != nil }

        let bestEntry: (key: String, value: PerformanceStat)?
        if !hasQuality.isEmpty {
            bestEntry = hasQuality.max { ($0.value.qualityScore ?? 0) < ($1.value.qualityScore ?? 0) }
        } else {
            bestEntry = matching.max { $0.value.avgTokensPerSecond < $1.value.avgTokensPerSecond }
        }

        guard let entry = bestEntry else { return nil }
        return Self.modelIdentifier(from: entry.key)
    }

    /// Get all stats for a model across all contexts.
    public func allStats(for model: ModelIdentifier) -> [UsageContext: PerformanceStat] {
        let modelPrefix = "\(model.provider)/\(model.modelId)|"
        lock.lock()
        defer { lock.unlock() }

        var result: [UsageContext: PerformanceStat] = [:]
        for (key, stat) in entries where key.hasPrefix(modelPrefix) {
            if let context = Self.usageContext(from: key) {
                result[context] = stat
            }
        }
        return result
    }

    /// Clear all stats.
    public func clear() {
        lock.lock()
        defer { lock.unlock() }
        entries.removeAll()
        save()
    }

    // MARK: - Persistence

    private func load() {
        guard FileManager.default.fileExists(atPath: storePath.path) else { return }
        do {
            let data = try Data(contentsOf: storePath)
            entries = try JSONDecoder().decode([String: PerformanceStat].self, from: data)
        } catch {
            entries = [:]
        }
    }

    private func save() {
        do {
            let dir = storePath.deletingLastPathComponent()
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            let data = try JSONEncoder().encode(entries)
            try data.write(to: storePath, options: .atomic)
        } catch {
            // Best-effort persistence — don't crash.
        }
    }

    // MARK: - Key Helpers

    static func key(model: ModelIdentifier, context: UsageContext) -> String {
        "\(model.provider)/\(model.modelId)|\(contextKey(context))"
    }

    static func contextKey(_ context: UsageContext) -> String {
        "\(context.app):\(context.task)"
    }

    static func modelIdentifier(from key: String) -> ModelIdentifier? {
        let parts = key.split(separator: "|", maxSplits: 1)
        guard let modelPart = parts.first else { return nil }
        let modelParts = modelPart.split(separator: "/", maxSplits: 1)
        guard modelParts.count == 2 else { return nil }
        return ModelIdentifier(provider: String(modelParts[0]), modelId: String(modelParts[1]))
    }

    static func usageContext(from key: String) -> UsageContext? {
        let parts = key.split(separator: "|", maxSplits: 1)
        guard parts.count == 2 else { return nil }
        let contextParts = parts[1].split(separator: ":", maxSplits: 1)
        guard contextParts.count == 2 else { return nil }
        return UsageContext(app: String(contextParts[0]), task: String(contextParts[1]))
    }

    private static func defaultStorePath() -> URL {
        #if os(macOS)
        let home = FileManager.default.homeDirectoryForCurrentUser
        return home.appendingPathComponent(".aikit/performance.json")
        #else
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        return docs.appendingPathComponent("aikit-performance.json")
        #endif
    }
}
