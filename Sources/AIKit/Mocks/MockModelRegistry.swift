import Foundation

/// In-memory mock model registry for testing.
public final class MockModelRegistry: ModelRegistry, @unchecked Sendable {
    public var models: [ModelDescriptor]

    public init(models: [ModelDescriptor] = []) {
        self.models = models
    }

    public func listModels(filter: ModelFilter?) async throws -> [ModelDescriptor] {
        guard let filter else { return models }
        return models.filter { filter.matches($0) }
    }

    public func getModel(id: ModelIdentifier) async throws -> ModelDescriptor {
        guard let model = models.first(where: { $0.id == id }) else {
            throw AIKitError.modelNotFound(id)
        }
        return model
    }

    public func download(id: ModelIdentifier, onProgress: @Sendable @escaping (Double) -> Void) async throws {
        guard let index = models.firstIndex(where: { $0.id == id }) else {
            throw AIKitError.modelNotFound(id)
        }
        onProgress(0.5)
        onProgress(1.0)
        models[index].isLocal = true
    }

    public func deleteLocal(id: ModelIdentifier) async throws {
        guard let index = models.firstIndex(where: { $0.id == id }) else {
            throw AIKitError.modelNotFound(id)
        }
        models[index].isLocal = false
        models[index].localPath = nil
    }

    public func search(query: String, format: ModelFormat?) async throws -> [ModelDescriptor] {
        models.filter { model in
            let matchesQuery = model.name.localizedCaseInsensitiveContains(query)
                || model.description.localizedCaseInsensitiveContains(query)
            let matchesFormat = format == nil || model.format == format
            return matchesQuery && matchesFormat
        }
    }
}
