import Foundation

/// Describes the context in which a model is used — enables per-task performance tracking.
public struct UsageContext: Sendable, Codable, Hashable {
    /// Application name, e.g. "brainy", "flsh".
    public var app: String
    /// Task type, e.g. "manga-translation", "article-summary", "voice-note".
    public var task: String
    /// Input content language.
    public var inputLanguage: String?
    /// Output content language.
    public var outputLanguage: String?

    public init(
        app: String,
        task: String,
        inputLanguage: String? = nil,
        outputLanguage: String? = nil
    ) {
        self.app = app
        self.task = task
        self.inputLanguage = inputLanguage
        self.outputLanguage = outputLanguage
    }
}

/// Aggregated performance statistics for a model in a given usage context.
public struct PerformanceStat: Sendable, Codable, Equatable {
    public var avgLatencyMs: Int
    public var avgTokensPerSecond: Double
    public var qualityScore: Double?
    public var totalInvocations: Int
    public var lastUsedAt: Date

    public init(
        avgLatencyMs: Int,
        avgTokensPerSecond: Double,
        qualityScore: Double? = nil,
        totalInvocations: Int,
        lastUsedAt: Date
    ) {
        self.avgLatencyMs = avgLatencyMs
        self.avgTokensPerSecond = avgTokensPerSecond
        self.qualityScore = qualityScore
        self.totalInvocations = totalInvocations
        self.lastUsedAt = lastUsedAt
    }
}
