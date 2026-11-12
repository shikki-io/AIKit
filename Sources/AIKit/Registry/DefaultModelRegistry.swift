import CoreKit
import Foundation

/// Production model registry combining local storage, HuggingFace search, and download.
public final class DefaultModelRegistry: ModelRegistry, @unchecked Sendable {
    private let localStore: LocalModelStore
    private let hfClient: HuggingFaceClient
    private let downloader: ModelDownloader

    /// - Parameter localStore: injected, with no default. The manifest's
    ///   location comes from the host's path layer (W23 of spec
    ///   ai-layer-aikit-core-2026-10-31), so a kit cannot build one for itself.
    ///
    /// The search client and the downloader are NOT parameters here: the
    /// Hugging Face endpoint stays internal BEHIND the catalogue (W27,
    /// BR-AIL-21, door 3). A caller asks the catalogue for models; which
    /// registry it searches and how it fetches bytes is the catalogue's own
    /// business, and a test reaches the init below through `@testable` to
    /// inject both.
    public convenience init(localStore: LocalModelStore) {
        self.init(
            localStore: localStore,
            hfClient: HuggingFaceClient(),
            downloader: ModelDownloader()
        )
    }

    /// The full init. Internal because `HuggingFaceClient` and
    /// `ModelDownloader` are.
    init(
        localStore: LocalModelStore,
        hfClient: HuggingFaceClient = HuggingFaceClient(),
        downloader: ModelDownloader = ModelDownloader()
    ) {
        self.localStore = localStore
        self.hfClient = hfClient
        self.downloader = downloader
    }

    public func listModels(filter: ModelFilter?) async throws -> [ModelDescriptor] {
        let localModels = try localStore.listLocal().map { $0.descriptor }

        guard let filter else { return localModels }
        return localModels.filter { filter.matches($0) }
    }

    public func getModel(id: ModelIdentifier) async throws -> ModelDescriptor {
        // Check local first
        if let local = try localStore.listLocal().first(where: { $0.descriptor.id == id }) {
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

        // Find a GGUF file to download (convention: look for *.gguf). The
        // repository id is a POSIX wire path, so its last component is read as
        // one — `VirtualPath` parses and validates it, and no separator is
        // spelled here.
        let stem = (try? VirtualPath(parsing: info.modelId))?.last?.rawValue ?? "model"
        let filename = "\(stem).gguf"
        let downloadURL = try hfClient.downloadURL(repoId: info.modelId, filename: filename)
        // Resolved by the path layer, not appended to whatever this type held.
        let destination = try localStore.fileURL(for: id, named: filename)

        try await downloader.download(from: downloadURL, to: destination) { progress in
            onProgress(progress.fraction)
        }

        let descriptor = info.toDescriptor(format: .gguf)
        var localDescriptor = descriptor
        localDescriptor.isLocal = true
        localDescriptor.localPath = destination.url
        try localStore.register(localDescriptor, at: destination.url)
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
