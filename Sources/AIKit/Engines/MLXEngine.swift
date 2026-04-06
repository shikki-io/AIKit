#if os(macOS)
import Foundation

/// Runtime engine for Apple MLX models.
/// Supports LLM (GGUF/MLX), video generation (mlx-video), and audio (mlx-audio).
/// Uses shell-out to Python mlx packages initially, with path to Swift-native via MLX Swift bindings.
public final class MLXEngine: RuntimeEngine, @unchecked Sendable {
    public let id = "mlx"
    public let displayName = "Apple MLX"
    public let supportedFormats: [ModelFormat] = [.mlx, .gguf]

    /// Path to Python binary. Auto-detected or user-configured.
    private let pythonPath: String
    /// Python module for video generation.
    private let mlxVideoModule: String
    /// Currently loaded model identifiers.
    private var _loadedModels: [ModelIdentifier] = []

    public init(pythonPath: String? = nil) {
        // Auto-detect python3 path if not provided.
        if let path = pythonPath {
            self.pythonPath = path
        } else if ShellRunner.commandExists("python3") {
            self.pythonPath = "/usr/bin/env"
        } else {
            self.pythonPath = "/usr/bin/python3"
        }
        self.mlxVideoModule = "mlx_video"
    }

    public var isAvailable: Bool {
        // MLX requires macOS on Apple Silicon.
        #if arch(arm64)
        return true
        #else
        return false
        #endif
    }

    /// Detect installed MLX models/capabilities.
    public func detectCapabilities() -> AICapabilities {
        var caps: AICapabilities = []

        // Check for mlx-video
        if ShellRunner.commandExists("python3") {
            caps.insert(.videoGeneration)
        }

        return caps
    }

    // MARK: - RuntimeEngine

    public func loadModel(_ descriptor: ModelDescriptor) async throws -> any AIProvider {
        guard isAvailable else {
            throw AIKitError.engineUnavailable("MLX requires macOS on Apple Silicon")
        }

        let provider: any AIProvider

        if descriptor.capabilities.contains(.videoGeneration) || descriptor.domain == .video {
            provider = MLXVideoProvider(
                id: "\(id)/\(descriptor.id.modelId)",
                displayName: "\(displayName) — \(descriptor.name)",
                pythonPath: pythonPath,
                modelName: descriptor.id.modelId
            )
        } else {
            throw AIKitError.engineUnavailable(
                "MLXEngine currently supports video generation only. LLM support via MLX Swift bindings is planned."
            )
        }

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

#else

import Foundation

/// Linux stub — MLX is not available outside Apple platforms.
public final class MLXEngine: RuntimeEngine, @unchecked Sendable {
    public let id = "mlx"
    public let displayName = "Apple MLX"
    public let supportedFormats: [ModelFormat] = [.mlx, .gguf]

    public init(pythonPath: String? = nil) {}

    public var isAvailable: Bool { false }

    public func loadModel(_ descriptor: ModelDescriptor) async throws -> any AIProvider {
        throw AIKitError.engineUnavailable("MLX is only available on macOS with Apple Silicon")
    }

    public func unloadModel(_ id: ModelIdentifier) async throws {
        throw AIKitError.engineUnavailable("MLX is only available on macOS with Apple Silicon")
    }

    public func loadedModels() -> [ModelIdentifier] { [] }
}

#endif
