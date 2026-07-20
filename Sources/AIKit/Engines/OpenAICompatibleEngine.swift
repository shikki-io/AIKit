import Foundation
import NetKit

/// OpenAI-compatible engine — works with OpenAI, LM Studio, Ollama, and any
/// server exposing /v1/chat/completions and /v1/models endpoints.
public final class OpenAICompatibleEngine: RuntimeEngine, @unchecked Sendable {
    public let id: String
    public let displayName: String
    public let supportedFormats: [ModelFormat]

    private let baseURL: String
    private let apiKey: String?
    private let networkService: any NetworkProtocol
    private var _loadedModels: [ModelIdentifier] = []

    public init(
        id: String = "openai-compatible",
        displayName: String = "OpenAI Compatible",
        baseURL: String,
        apiKey: String? = nil,
        supportedFormats: [ModelFormat] = [.api],
        networkService: (any NetworkProtocol)? = nil
    ) {
        self.id = id
        self.displayName = displayName
        self.baseURL = baseURL
        self.apiKey = apiKey
        self.supportedFormats = supportedFormats
        self.networkService = networkService ?? NetworkService()
    }

    public var isAvailable: Bool { true }

    /// Discover models from /v1/models endpoint.
    public func discoverModels() async throws -> [ModelDescriptor] {
        let parsed = URLComponents(string: baseURL)
        let endpoint = OpenAIModelsListEndPoint(
            host: parsed?.host ?? "127.0.0.1",
            port: parsed?.port,
            scheme: parsed?.scheme ?? "http",
            apiKey: apiKey
        )

        let response: ModelsListResponseDTO = try await networkService.sendRequest(endpoint: endpoint)

        return response.data.map { model in
            let modelId = ModelIdentifier(provider: id, modelId: model.id)
            let capabilities = inferCapabilities(from: model.id)
            return ModelDescriptor(
                id: modelId,
                name: model.id,
                author: model.ownedBy ?? "unknown",
                description: "Model \(model.id) via \(displayName)",
                capabilities: capabilities,
                format: .api,
                parameters: "unknown",
                sizeBytes: 0,
                architecture: "unknown",
                domain: inferDomain(from: model.id),
                isLocal: baseURL.contains("127.0.0.1") || baseURL.contains("localhost"),
                tags: []
            )
        }
    }

    public func loadModel(_ descriptor: ModelDescriptor) async throws -> any AIProvider {
        let provider = OpenAIProvider(
            id: "\(id)/\(descriptor.id.modelId)",
            displayName: "\(displayName) — \(descriptor.name)",
            capabilities: descriptor.capabilities,
            baseURL: baseURL,
            modelName: descriptor.id.modelId,
            apiKey: apiKey,
            networkService: networkService
        )
        if !_loadedModels.contains(descriptor.id) {
            _loadedModels.append(descriptor.id)
        }
        return provider
    }

    public func unloadModel(_ id: ModelIdentifier) async throws {
        _loadedModels.removeAll { $0 == id }
    }

    public func loadedModels() -> [ModelIdentifier] {
        _loadedModels
    }

    // MARK: - Heuristics

    private func inferCapabilities(from modelId: String) -> AICapabilities {
        let lower = modelId.lowercased()
        var caps: AICapabilities = .textGeneration

        if lower.contains("vision") || lower.contains("llava") {
            caps.insert(.vision)
        }
        if lower.contains("embed") {
            caps = .embedding
        }
        if lower.contains("tool") || lower.contains("function") || lower.contains("gpt") {
            caps.insert(.toolUse)
        }
        return caps
    }

    private func inferDomain(from modelId: String) -> ModelDomain {
        let lower = modelId.lowercased()
        if lower.contains("embed") { return .embedding }
        if lower.contains("whisper") || lower.contains("tts") { return .voice }
        if lower.contains("vision") || lower.contains("llava") { return .vision }
        return .llm
    }
}
