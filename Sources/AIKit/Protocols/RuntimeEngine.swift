/// A runtime that can load and run AI models.
public protocol RuntimeEngine: Sendable {
    /// Unique identifier for this engine.
    var id: String { get }
    /// Human-readable name.
    var displayName: String { get }
    /// Model formats this engine can run.
    var supportedFormats: [ModelFormat] { get }
    /// Whether this engine is available on the current platform.
    var isAvailable: Bool { get }

    /// Load a model and return a provider that can serve requests.
    func loadModel(_ descriptor: ModelDescriptor) async throws -> any AIProvider
    /// Unload a model from memory.
    func unloadModel(_ id: ModelIdentifier) async throws
    /// List currently loaded models.
    func loadedModels() -> [ModelIdentifier]
}
