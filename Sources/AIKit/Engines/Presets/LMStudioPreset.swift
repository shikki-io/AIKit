import NetKit

// Presets for the servers this fleet actually runs. Each one is a server ROOT:
// the route belongs to the endpoint enum (BR-AIL-09), and `AIServerAddress`
// refuses an address carrying a path — which is why the Anthropic preset no
// longer ends in `/v1` (AIKit#18, panel finding on `LMStudioPreset.swift:49`).
//
// Every factory throws for the same reason: the address is validated once,
// where it is written, instead of being defaulted to the local machine at call
// time (BR-AIL-18).

extension OpenAICompatibleEngine {
    /// Pre-configured for LM Studio local server.
    public static func lmStudio(
        baseURL: String = "http://127.0.0.1:1234",
        transport: (any AIHTTPTransport)? = nil
    ) throws(AIKitError) -> OpenAICompatibleEngine {
        try OpenAICompatibleEngine(
            id: "lmstudio",
            displayName: "LM Studio",
            baseURL: baseURL,
            supportedFormats: [.gguf, .mlx, .api],
            transport: transport
        )
    }

    /// Pre-configured for Ollama.
    public static func ollama(
        baseURL: String = "http://127.0.0.1:11434",
        transport: (any AIHTTPTransport)? = nil
    ) throws(AIKitError) -> OpenAICompatibleEngine {
        try OpenAICompatibleEngine(
            id: "ollama",
            displayName: "Ollama",
            baseURL: baseURL,
            supportedFormats: [.gguf, .api],
            transport: transport
        )
    }

    /// Pre-configured for OpenAI API.
    public static func openAI(
        apiKey: String,
        transport: (any AIHTTPTransport)? = nil
    ) throws(AIKitError) -> OpenAICompatibleEngine {
        try OpenAICompatibleEngine(
            id: "openai",
            displayName: "OpenAI",
            baseURL: "https://api.openai.com",
            apiKey: apiKey,
            transport: transport
        )
    }

    /// Pre-configured for Anthropic Claude (via OpenAI-compatible proxy).
    public static func anthropic(
        apiKey: String,
        baseURL: String = "https://api.anthropic.com",
        transport: (any AIHTTPTransport)? = nil
    ) throws(AIKitError) -> OpenAICompatibleEngine {
        try OpenAICompatibleEngine(
            id: "anthropic",
            displayName: "Anthropic",
            baseURL: baseURL,
            apiKey: apiKey,
            transport: transport
        )
    }
}
