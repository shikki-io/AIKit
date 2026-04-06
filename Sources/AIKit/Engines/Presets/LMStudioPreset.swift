import NetKit

public extension OpenAICompatibleEngine {
    /// Pre-configured for LM Studio local server.
    static func lmStudio(
        baseURL: String = "http://127.0.0.1:1234",
        networkService: (any NetworkProtocol)? = nil
    ) -> OpenAICompatibleEngine {
        OpenAICompatibleEngine(
            id: "lmstudio",
            displayName: "LM Studio",
            baseURL: baseURL,
            supportedFormats: [.gguf, .mlx, .api],
            networkService: networkService
        )
    }

    /// Pre-configured for Ollama.
    static func ollama(
        baseURL: String = "http://127.0.0.1:11434",
        networkService: (any NetworkProtocol)? = nil
    ) -> OpenAICompatibleEngine {
        OpenAICompatibleEngine(
            id: "ollama",
            displayName: "Ollama",
            baseURL: baseURL,
            supportedFormats: [.gguf, .api],
            networkService: networkService
        )
    }

    /// Pre-configured for OpenAI API.
    static func openAI(
        apiKey: String,
        networkService: (any NetworkProtocol)? = nil
    ) -> OpenAICompatibleEngine {
        OpenAICompatibleEngine(
            id: "openai",
            displayName: "OpenAI",
            baseURL: "https://api.openai.com",
            apiKey: apiKey,
            networkService: networkService
        )
    }

    /// Pre-configured for Anthropic Claude (via OpenAI-compatible proxy).
    static func anthropic(
        apiKey: String,
        baseURL: String = "https://api.anthropic.com/v1",
        networkService: (any NetworkProtocol)? = nil
    ) -> OpenAICompatibleEngine {
        OpenAICompatibleEngine(
            id: "anthropic",
            displayName: "Anthropic",
            baseURL: baseURL,
            apiKey: apiKey,
            networkService: networkService
        )
    }
}
