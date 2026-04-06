import Foundation

/// Uniquely identifies a model across providers.
public struct ModelIdentifier: Sendable, Codable, Hashable {
    /// Provider key, e.g. "lmstudio", "ollama", "huggingface", "openai".
    public var provider: String
    /// Model ID within the provider, e.g. "nvidia/nemotron-3-nano-4b" or "gpt-5.4-mini".
    public var modelId: String

    public init(provider: String, modelId: String) {
        self.provider = provider
        self.modelId = modelId
    }
}

extension ModelIdentifier: CustomStringConvertible {
    public var description: String {
        "\(provider)/\(modelId)"
    }
}
