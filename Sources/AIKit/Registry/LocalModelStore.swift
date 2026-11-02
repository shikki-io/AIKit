import Foundation
import Logging

private let storeLogger = Logger(label: "aikit.registry.local-model-store")

/// Errors the manifest layer throws.
public enum LocalModelStoreError: Error, Equatable {
    /// The manifest file exists but could not be decoded.
    ///
    /// A corrupt manifest must not be silently treated as empty: returning `[]`
    /// would let the next write overwrite every recorded model. The caller must
    /// decide whether to surface the error or recover.
    case corruptManifest
}

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
        // One definition of the root, in `ModelStore` (W13 of spec
        // ai-layer-aikit-core-2026-10-31). The default used to be spelled out
        // again here, and two spellings of "the cache root" is exactly how the
        // same model came to sit in two directories at once.
        let base = basePath ?? ModelStore.defaultRoot
        self.basePath = base
        self.manifestURL = base.appendingPathComponent("manifest.json")
        self.fileManager = FileManager.default
    }

    /// List all locally downloaded models.
    ///
    /// - Throws: `LocalModelStoreError.corruptManifest` when the manifest file
    ///   exists but cannot be decoded. A missing manifest is not an error; it
    ///   means the store is fresh and the result is `[]`.
    public func listLocal() throws -> [LocalModel] {
        try loadManifest()
    }

    /// Get the expected local path for a model.
    public func modelPath(for id: ModelIdentifier) -> URL {
        basePath
            .appendingPathComponent(id.provider)
            .appendingPathComponent(id.modelId.replacingOccurrences(of: "/", with: "_"))
    }

    /// Check if a model is downloaded.
    ///
    /// - Throws: `LocalModelStoreError.corruptManifest` when the manifest exists
    ///   but cannot be decoded.
    public func isDownloaded(_ id: ModelIdentifier) throws -> Bool {
        try loadManifest().contains { $0.descriptor.id == id }
    }

    /// Record a downloaded model in the manifest.
    public func register(_ descriptor: ModelDescriptor, at path: URL) throws {
        ensureDirectoryExists()
        var models = try loadManifest()

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
        var models = try loadManifest()
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
    ///
    /// - Throws: `LocalModelStoreError.corruptManifest` when the manifest exists
    ///   but cannot be decoded.
    public func totalSizeBytes() throws -> Int64 {
        try loadManifest().reduce(0) { $0 + $1.sizeOnDisk }
    }

    // MARK: - Private

    private func ensureDirectoryExists() {
        if !fileManager.fileExists(atPath: basePath.path) {
            try? fileManager.createDirectory(at: basePath, withIntermediateDirectories: true)
        }
    }

    private func loadManifest() throws -> [LocalModel] {
        let data: Data
        do {
            data = try Data(contentsOf: manifestURL)
        } catch let cocoaError as CocoaError where cocoaError.code == .fileNoSuchFile {
            return []
        } catch {
            throw LocalModelStoreError.corruptManifest
        }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        do {
            return try decoder.decode([LocalModel].self, from: data)
        } catch {
            storeLogger.error("LocalModelStore: manifest decode failed — \(error)")
            throw LocalModelStoreError.corruptManifest
        }
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
            let size = attrs[.size] as? Int64
        else {
            return 0
        }
        return size
    }
}
