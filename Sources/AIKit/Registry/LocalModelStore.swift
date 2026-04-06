import Foundation

/// A single model stored on disk.
public struct LocalModel: Sendable, Codable, Equatable {
    public let descriptor: ModelDescriptor
    public let localPath: URL
    public let downloadedAt: Date
    public let sizeOnDisk: Int64

    public init(
        descriptor: ModelDescriptor,
        localPath: URL,
        downloadedAt: Date = Date(),
        sizeOnDisk: Int64
    ) {
        self.descriptor = descriptor
        self.localPath = localPath
        self.downloadedAt = downloadedAt
        self.sizeOnDisk = sizeOnDisk
    }
}

/// Manages the local model storage directory (~/.aikit/models/).
public final class LocalModelStore: @unchecked Sendable {
    private let basePath: URL
    private let manifestURL: URL
    private let fileManager: FileManager

    public init(basePath: URL? = nil) {
        #if os(macOS)
        let defaultBase = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".aikit")
            .appendingPathComponent("models")
        #else
        let defaultBase = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
            .appendingPathComponent("aikit-models")
        #endif
        let base = basePath ?? defaultBase
        self.basePath = base
        self.manifestURL = base.appendingPathComponent("manifest.json")
        self.fileManager = FileManager.default
    }

    /// List all locally downloaded models.
    public func listLocal() -> [LocalModel] {
        loadManifest()
    }

    /// Get the expected local path for a model.
    public func modelPath(for id: ModelIdentifier) -> URL {
        basePath
            .appendingPathComponent(id.provider)
            .appendingPathComponent(id.modelId.replacingOccurrences(of: "/", with: "_"))
    }

    /// Check if a model is downloaded.
    public func isDownloaded(_ id: ModelIdentifier) -> Bool {
        loadManifest().contains { $0.descriptor.id == id }
    }

    /// Record a downloaded model in the manifest.
    public func register(_ descriptor: ModelDescriptor, at path: URL) throws {
        ensureDirectoryExists()
        var models = loadManifest()

        // Replace if already present
        models.removeAll { $0.descriptor.id == descriptor.id }

        let sizeOnDisk = fileSizeAt(path)
        let local = LocalModel(
            descriptor: descriptor,
            localPath: path,
            downloadedAt: Date(),
            sizeOnDisk: sizeOnDisk
        )
        models.append(local)
        try saveManifest(models)
    }

    /// Delete a local model from disk and manifest.
    public func delete(_ id: ModelIdentifier) throws {
        var models = loadManifest()
        guard let index = models.firstIndex(where: { $0.descriptor.id == id }) else {
            throw AIKitError.modelNotFound(id)
        }

        let localPath = models[index].localPath
        if fileManager.fileExists(atPath: localPath.path) {
            try fileManager.removeItem(at: localPath)
        }

        models.remove(at: index)
        try saveManifest(models)
    }

    /// Total disk usage of all downloaded models.
    public func totalSizeBytes() -> Int64 {
        loadManifest().reduce(0) { $0 + $1.sizeOnDisk }
    }

    // MARK: - Private

    private func ensureDirectoryExists() {
        if !fileManager.fileExists(atPath: basePath.path) {
            try? fileManager.createDirectory(at: basePath, withIntermediateDirectories: true)
        }
    }

    private func loadManifest() -> [LocalModel] {
        guard let data = try? Data(contentsOf: manifestURL) else { return [] }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return (try? decoder.decode([LocalModel].self, from: data)) ?? []
    }

    private func saveManifest(_ models: [LocalModel]) throws {
        ensureDirectoryExists()
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(models)
        try data.write(to: manifestURL, options: .atomic)
    }

    private func fileSizeAt(_ url: URL) -> Int64 {
        guard let attrs = try? fileManager.attributesOfItem(atPath: url.path),
              let size = attrs[.size] as? Int64 else {
            return 0
        }
        return size
    }
}
