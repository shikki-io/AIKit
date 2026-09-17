import Foundation

// MARK: - AIProviderID
//
// The provider-id vocabulary shared by every CLI/HTTP provider in this package.
// Moved from shikki (`ShiAIRoutingTypes.swift`, spec
// shi-multi-ai-provider-routing-with-local-fallback-2026-06-20) so the plumbing
// that keys on it — BudgetLedger, CLISubprocessProvider — lives here too.
// `rawValue` is the wire-level provider id (kebab-case) and is unchanged, so
// persisted values stay readable across the move.
//
// Policy that classifies these ids — sovereignty tier, capability tier, privacy
// class, task profiles — stays in shikki as extensions on this type.

/// Stable identifier for an AI provider in the routing layer.
///
/// `rawValue` matches the spec's wire-level provider id (kebab-case).
public enum AIProviderID: String, CaseIterable, Codable, Sendable, Hashable {
    // MARK: Cloud
    case claudeCode = "claude-code"
    case claudeApi = "claude-api"
    case openaiApi = "openai-api"
    case mistralLaPlateforme = "mistral-la-plateforme"
    case groq = "groq"
    case together = "together"
    case fireworks = "fireworks"
    case deepinfra = "deepinfra"
    // MARK: Cloud (CN-jurisdiction) — PR #1668 review: a sovereignty model that
    // names EU and US but leaves the Chinese market out is not a model of the
    // market; these are the API-served families operators actually run.
    case deepseekApi = "deepseek-api"
    case qwenDashScope = "qwen-dashscope"
    case moonshotKimi = "moonshot-kimi"
    case zhipuGlm = "zhipu-glm"

    // MARK: Local (Swift-native preferred per swift-empire memory)
    case mlxLocal = "mlx-local"
    case llamaSwift = "llama-swift"

    // MARK: Local (Swift-bridged HTTP)
    case ollamaLocal = "ollama-local"
    case mistralRsLocal = "mistralrs-local"
    case llamacppLocal = "llamacpp-local"
}
