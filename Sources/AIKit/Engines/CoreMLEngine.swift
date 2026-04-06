#if canImport(CoreML)
import Foundation
import CoreML

/// Runtime engine for CoreML compiled models (.mlmodel/.mlpackage).
/// Best for vision tasks on Neural Engine (ANE): OCR, layout detection, inpainting.
public final class CoreMLEngine: RuntimeEngine, @unchecked Sendable {
    public let id = "coreml"
    public let displayName = "Apple CoreML"
    public let supportedFormats: [ModelFormat] = [.coreml]

    /// Directory containing .mlmodelc files.
    let modelsDirectory: URL

    /// Currently loaded model identifiers.
    private var _loadedModels: [ModelIdentifier] = []

    public init(modelsDirectory: URL? = nil) {
        if let dir = modelsDirectory {
            self.modelsDirectory = dir
        } else {
            #if os(macOS)
            let home = FileManager.default.homeDirectoryForCurrentUser
            self.modelsDirectory = home.appendingPathComponent(".aikit/coreml-models", isDirectory: true)
            #else
            let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
            self.modelsDirectory = docs.appendingPathComponent("aikit-coreml-models")
            #endif
        }
    }

    public var isAvailable: Bool {
        // CoreML is available on all Apple platforms.
        true
    }

    // MARK: - RuntimeEngine

    public func loadModel(_ descriptor: ModelDescriptor) async throws -> any AIProvider {
        // Resolve model path — either from descriptor or models directory.
        let modelURL: URL
        if let localPath = descriptor.localPath {
            modelURL = localPath
        } else {
            modelURL = modelsDirectory.appendingPathComponent("\(descriptor.id.modelId).mlmodelc")
        }

        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw AIKitError.modelNotFound(descriptor.id)
        }

        let compiledURL: URL
        if modelURL.pathExtension == "mlmodelc" {
            compiledURL = modelURL
        } else {
            compiledURL = try await MLModel.compileModel(at: modelURL)
        }

        let config = MLModelConfiguration()
        config.computeUnits = .all // Use ANE + GPU + CPU as needed.
        let model = try MLModel(contentsOf: compiledURL, configuration: config)

        let provider = CoreMLProvider(
            id: "\(id)/\(descriptor.id.modelId)",
            displayName: "\(displayName) — \(descriptor.name)",
            capabilities: descriptor.capabilities,
            model: model
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
}

// MARK: - CoreMLProvider

/// AIProvider that wraps a loaded CoreML model for inference.
struct CoreMLProvider: AIProvider, @unchecked Sendable {
    let id: String
    let displayName: String
    let capabilities: AICapabilities

    private let model: MLModel

    init(id: String, displayName: String, capabilities: AICapabilities, model: MLModel) {
        self.id = id
        self.displayName = displayName
        self.capabilities = capabilities
        self.model = model
    }

    var status: AIProviderStatus {
        get async { .ready }
    }

    func complete(request: AIRequest) async throws -> AIResponse {
        let startTime = ContinuousClock.now

        // CoreML models have model-specific I/O. For now, pass the user message
        // as a generic text input and return the model output as text.
        // Real implementations will specialize per model type (vision, OCR, etc.).
        guard let prompt = request.messages.first(where: { $0.role == .user })?.content else {
            throw AIKitError.requestFailed("No user message found for CoreML inference")
        }

        // Generic feature provider with a "text" input.
        let input = try MLDictionaryFeatureProvider(dictionary: ["text": prompt as NSString])
        let output = try await model.prediction(from: input)

        // Extract first output feature as string.
        var resultText = ""
        for name in output.featureNames {
            if let value = output.featureValue(for: name)?.stringValue {
                resultText = value
                break
            }
        }

        let elapsed = startTime.duration(to: .now)
        let latencyMs = Int(elapsed.components.seconds * 1000 + elapsed.components.attoseconds / 1_000_000_000_000_000)

        return AIResponse(
            content: resultText,
            model: id,
            tokensUsed: TokenUsage(prompt: 0, completion: 0),
            latencyMs: latencyMs
        )
    }
}

#else

import Foundation

/// Linux stub — CoreML is not available outside Apple platforms.
public final class CoreMLEngine: RuntimeEngine, @unchecked Sendable {
    public let id = "coreml"
    public let displayName = "Apple CoreML"
    public let supportedFormats: [ModelFormat] = [.coreml]

    public init(modelsDirectory: URL? = nil) {}

    public var isAvailable: Bool { false }

    public func loadModel(_ descriptor: ModelDescriptor) async throws -> any AIProvider {
        throw AIKitError.engineUnavailable("CoreML is only available on Apple platforms")
    }

    public func unloadModel(_ id: ModelIdentifier) async throws {
        throw AIKitError.engineUnavailable("CoreML is only available on Apple platforms")
    }

    public func loadedModels() -> [ModelIdentifier] { [] }
}

#endif
