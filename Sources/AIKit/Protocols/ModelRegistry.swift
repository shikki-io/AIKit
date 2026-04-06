/// Discovers, downloads, and manages AI models.
public protocol ModelRegistry: Sendable {
    /// List all known models (local + available for download).
    func listModels(filter: ModelFilter?) async throws -> [ModelDescriptor]
    /// Get a specific model by ID.
    func getModel(id: ModelIdentifier) async throws -> ModelDescriptor
    /// Download a model to local storage.
    func download(id: ModelIdentifier, onProgress: @Sendable @escaping (Double) -> Void) async throws
    /// Delete a local model.
    func deleteLocal(id: ModelIdentifier) async throws
    /// Search model registries (e.g. HuggingFace).
    func search(query: String, format: ModelFormat?) async throws -> [ModelDescriptor]
}
