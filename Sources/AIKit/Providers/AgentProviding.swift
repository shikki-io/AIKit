import Foundation

// MARK: - AgentProviding
//
// The provider-agnostic seam: give a prompt, get raw output, or throw.
// Moved from shikki (`SpecPipeline.swift`, BR-SP-01) with CLISubprocessProvider,
// FallbackProviderChain and LMStudioProvider, which implement it.

/// Protocol for agent invocation (AI-provider agnostic).
protocol AgentProviding: Sendable {

    /// What this backend is called, as settings name it.
    ///
    /// A REQUIREMENT with no default on purpose (W22 of spec
    /// ai-layer-aikit-core-2026-10-31). `FallbackProviderChain` used to record
    /// a failed attempt as `String(describing: type(of: provider))`, so an
    /// operator with two LM Studio entries — a local one and the Mac on the
    /// desk — read "LMStudioProvider failed; LMStudioProvider failed" and
    /// could not tell which server was down (AIKit#18, panel finding on
    /// `FallbackProviderChain.swift:28`). A default here would have put that
    /// type name back under a new name, so conformers state their own id and
    /// it matches ``AIProvider/id`` for the same backend.
    var id: String { get }

    /// Run a prompt through the agent and return the raw output.
    func run(prompt: String, timeout: TimeInterval) async throws -> String
}
