import Foundation

/// Production model registry combining local storage, HuggingFace search, and download.
public final class DefaultModelRegistry: ModelRegistry, @unchecked Sendable {
    private let localStore: LocalModelStore
    private let hfClient: HuggingFaceClient
    private let downloader: ModelDownloader

    public init(
        localStore: LocalModelStore = LocalModelStore(),
        hfClient: HuggingFaceClient = HuggingFaceClient(),
        downloader: ModelDownloader = ModelDownloader()
    ) {
        self.localStore = localStore
        self.hfClient = hfClient
        self.downloader = downloader
    }

    public func listModels(filter: ModelFilter?) async throws -> [ModelDescriptor] {
        let localModels = localStore.listLocal().map { $0.descriptor }

        guard let filter else { return localModels }
        return localModels.filter { filter.matches($0) }
    }

    public func getModel(id: ModelIdentifier) async throws -> ModelDescriptor {
        // Check local first
        if let local = localStore.listLocal().first(where: { $0.descriptor.id == id }) {
            return local.descriptor
        }

        // Try HuggingFace
        if id.provider == "huggingface" {
            let info = try await hfClient.getModel(id: id.modelId)
            return info.toDescriptor(format: .gguf)
        }

        throw AIKitError.modelNotFound(id)
    }

    public func download(
        id: ModelIdentifier,
        onProgress: @Sendable @escaping (Double) -> Void
    ) async throws {
        // Resolve model info
        let info: HFModelInfo
        if id.provider == "huggingface" {
            info = try await hfClient.getModel(id: id.modelId)
        } else {
            throw AIKitError.downloadFailed("Download only supported for HuggingFace models")
        }

        // Find a GGUF file to download (convention: look for *.gguf)
        let filename = "\(info.modelId.components(separatedBy: "/").last ?? "model").gguf"
        let downloadURL = hfClient.downloadURL(repoId: info.modelId, filename: filename)
        let destinationPath = localStore.modelPath(for: id)
            .appendingPathComponent(filename)

        try await downloader.download(from: downloadURL, to: destinationPath) { progress in
            onProgress(progress.fraction)
        }

        let descriptor = info.toDescriptor(format: .gguf)
        var localDescriptor = descriptor
        localDescriptor.isLocal = true
        localDescriptor.localPath = destinationPath
        try localStore.register(localDescriptor, at: destinationPath)
    }

    public func deleteLocal(id: ModelIdentifier) async throws {
        try localStore.delete(id)
    }

    public func search(query: String, format: ModelFormat?) async throws -> [ModelDescriptor] {
        let filter: HFSearchFilter?
        if let format {
            filter = HFSearchFilter(tags: [format.rawValue], sort: "downloads", direction: "-1")
        } else {
            filter = HFSearchFilter(sort: "downloads", direction: "-1")
        }

        let results = try await hfClient.search(query: query, filter: filter)
        return results.map { $0.toDescriptor(format: format ?? .gguf) }
    }
}
