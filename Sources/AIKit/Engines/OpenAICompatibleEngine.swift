import Foundation
import NetKit

/// Discovers what an OpenAI-compatible server serves, and builds a provider per
/// model.
///
/// A `RuntimeEngine`, not a provider: the chat half it used to carry is
/// ``HTTPAIProvider`` with the `openai-chat` codec now, so what remains here is
/// the model catalogue — `GET /v1/models`, and one provider per descriptor.
/// Works with OpenAI, LM Studio, Ollama, and any server exposing that surface.
///
/// The address is an ``AIServerAddress``, validated once in `init`. What that
/// replaces was the defect: `URLComponents(string: baseURL)?.scheme ?? "http"`
/// with `?.host ?? "127.0.0.1"`, which turned any unparseable configuration
/// into a request against the local machine, bearer key included
/// (AIKit#18 r4500281951).
final class OpenAICompatibleEngine: RuntimeEngine, @unchecked Sendable {
    let id: String
    let displayName: String
    let supportedFormats: [ModelFormat]

    /// The server root. Validated, never defaulted (BR-AIL-18).
    let address: AIServerAddress
    private let apiKey: String?
    /// The ONE transport: discovery sends on it, and so does every provider
    /// this engine builds.
    ///
    /// It is an ``AIHTTPTransport`` rather than a bare `NetworkProtocol`
    /// because a provider needs the status-and-stream surface. A first draft of
    /// this wave took `NetworkProtocol` and fell back to the production
    /// transport when it was not one — and the tests, which injected NetKit's
    /// `MockNetworkService`, silently talked to the operator's live LM Studio
    /// and to api.openai.com. A seam that can be half-injected is not a seam
    /// (BR-AIL-12).
    private let transport: any AIHTTPTransport
    private var _loadedModels: [ModelIdentifier] = []

    /// - Throws: ``AIKitError/invalidServerAddress(value:reason:)`` when
    ///   `baseURL` is not a server root — naming the value, so a configuration
    ///   mistake is reported where it was written rather than sent somewhere.
    init(
        id: String = "openai-compatible",
        displayName: String = "OpenAI Compatible",
        baseURL: String,
        apiKey: String? = nil,
        supportedFormats: [ModelFormat] = [.api],
        transport: any AIHTTPTransport
    ) throws(AIKitError) {
        self.id = id
        self.displayName = displayName
        self.address = try AIServerAddress(validating: baseURL)
        self.apiKey = apiKey
        self.supportedFormats = supportedFormats
        self.transport = transport
    }

    var isAvailable: Bool { true }

    /// Discover models from `GET /v1/models`.
    func discoverModels() async throws -> [ModelDescriptor] {
        let endpoint = OpenAIEndPoint.models(address: address, credential: apiKey)
        let response: ModelsListResponseDTO = try await transport.sendRequest(endpoint: endpoint)

        return response.data.map { model in
            let modelId = ModelIdentifier(provider: id, modelId: model.id)
            return ModelDescriptor(
                id: modelId,
                name: model.id,
                author: model.ownedBy ?? "unknown",
                description: "Model \(model.id) via \(displayName)",
                capabilities: inferCapabilities(from: model.id),
                format: .api,
                parameters: "unknown",
                sizeBytes: 0,
                architecture: "unknown",
                domain: inferDomain(from: model.id),
                // The model does not leave this network, which is what a
                // reader of `isLocal` is asking. Read from
                // ``AIServerAddress/reach``, not from a substring test on the
                // host: `host == "127.0.0.1" || host == "localhost"` called
                // the Mac on the desk at `192.168.1.42` — or at
                // `studio.local` — a cloud model (AIKit#18, panel finding on
                // `OpenAICompatibleEngine.swift:62`).
                isLocal: address.reach != .remote,
                tags: []
            )
        }
    }

    func loadModel(_ descriptor: ModelDescriptor) async throws -> any AIProvider {
        let entry = AIProviderEntry(
            id: "\(id)/\(descriptor.id.modelId)",
            family: .openAIChat,
            address: address,
            model: descriptor.id.modelId
        )
        let provider = HTTPAIProvider(
            entry: entry,
            codec: OpenAIChatCodec(),
            network: transport,
            credential: apiKey,
            displayName: "\(displayName) — \(descriptor.name)",
            capabilities: descriptor.capabilities
        )
        if !_loadedModels.contains(descriptor.id) {
            _loadedModels.append(descriptor.id)
        }
        return provider
    }

    func unloadModel(_ id: ModelIdentifier) async throws {
        _loadedModels.removeAll { $0 == id }
    }

    func loadedModels() -> [ModelIdentifier] {
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
