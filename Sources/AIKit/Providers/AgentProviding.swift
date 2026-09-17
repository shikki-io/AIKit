import Foundation

// MARK: - AgentProviding
//
// The provider-agnostic seam: give a prompt, get raw output, or throw.
// Moved from shikki (`SpecPipeline.swift`, BR-SP-01) with CLISubprocessProvider,
// FallbackProviderChain and LMStudioProvider, which implement it.

/// Protocol for agent invocation (AI-provider agnostic).
public protocol AgentProviding: Sendable {
    /// Run a prompt through the agent and return the raw output.
    func run(prompt: String, timeout: TimeInterval) async throws -> String
}
