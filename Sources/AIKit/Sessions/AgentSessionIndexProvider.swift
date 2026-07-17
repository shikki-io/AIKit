// Migrated from shikki ShiKit (PR #1401 review, third pass): the AI-provider
// SPM owns the session-store contract AND the claude implementation. Origin:
// shi-crash-resilience-ssot-2026-07-17 W1; shikki consumes via re-export once
// its AIKit pin bumps (extract-before-consume bridge).

import Foundation

// MARK: - AgentSessionIndexProvider
//
// Provider-agnostic session-store abstraction (PR #1401 review, moved out of
// the claude-specific file on the second review pass). FINAL HOME: the AIKit
// SPM (FJ-Studios/AIKit) next to the other provider abstractions — migrates
// with the extraction epic (plan 34bd228b); parked in ShiKit/Agent/ (the
// provider-resolution area, next to LaunchProviderResolver) until then.

/// Provider-agnostic surface for "what agent sessions can I resume on disk".
/// `ClaudeSessionIndexProvider` is the claude-code conformer; other model
/// runtimes (gemma, KIMI, …) implement their own layout scan behind the same
/// contract so ResumeCommand never hardcodes a vendor.
public protocol AgentSessionIndexProvider: Sendable {
    /// Human/provider identifier ("claude-code", "gemma", …).
    var providerID: String { get }
    /// Scan the provider's session store, newest-first.
    func scanSessions(limit: Int?) -> [AgentSessionEntry]
    /// The shell invocation that resumes a given session id.
    func resumeInvocation(sessionID: String) -> [String]
}

/// claude-code conformer — thin instance facade over the static scanner.
public struct ClaudeSessionIndexProvider: AgentSessionIndexProvider {
    public let providerID = "claude-code"
    public init() {}

    public func scanSessions(limit: Int?) -> [AgentSessionEntry] {
        ClaudeSessionIndex.scan(limit: limit)
    }

    public func resumeInvocation(sessionID: String) -> [String] {
        ["claude", "--resume", sessionID]
    }
}
