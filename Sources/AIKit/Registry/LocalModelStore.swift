import CoreKit
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

/// The manifest of what this device downloaded, and where each model sits.
///
/// W23 of spec ai-layer-aikit-core-2026-10-31 took this store's locations away
/// from it. AIKit#18 r4500478370 and r4500481612, on the two lines that are
/// gone:
///
/// > *"Move this in the constructors init and remove the fact the base path
/// > should be optional and this parameter should be name more baseUrl than
/// > basePath because it's an url"*
///
/// > *"This smell, it arrive like this hardcoded so it can drift and from my
/// > perspective it should be link to a protocol from CoreKit to manage SoT of
/// > the resource"*
///
/// Both are answered by the same change, and it is stronger than the rename
/// asked for: there is no base path parameter at all, optional or not, and no
/// URL to name. The store holds an injected CoreKit ``PathResolving`` and asks
/// it where each thing lives, from a ``VirtualPathOwning`` declaration —
/// ``LocalModelManifest`` for the manifest, ``LocalModelDirectory`` for one
/// model's folder. So:
///
/// - nothing here can drift from ``ModelStore``: both resolve through the one
///   layer, against the one ``AIKitNode/models`` node, and there is no second
///   spelling of the root to disagree with (the drift that put one 4 GB model
///   in two directories at once, measured 2026-10-31);
/// - the manifest is read and written by CoreKit's own typed document store
///   (``OwnedResourceStore``) rather than by a hand-rolled `JSONEncoder`-to-file
///   pair;
/// - the directory separator is never spelled here, so the same declaration
///   resolves on macOS, iOS, Linux and Windows.
///
/// `@unchecked Sendable` is unchanged and still earned the same way: every
/// member is a `let`, and the file I/O is `FileManager`'s own.
public final class LocalModelStore: @unchecked Sendable {

    /// The injected resolver. Prod and tests differ by its root and by nothing
    /// else (CoreKit BR-RP-03).
    private let paths: any PathResolving

    /// CoreKit's typed document store over ``LocalModelManifest``.
    ///
    /// The model is the OWNER's `Document` and not a parameter this type picks,
    /// so a reader and a writer of the manifest cannot disagree about what is
    /// in the file (CoreKit `VirtualDocumentOwning`).
    private let manifest: OwnedResourceStore<[LocalModel]>

    private let fileManager: FileManager

    /// - Parameter paths: the injected path layer. It must have AIKit's owners
    ///   registered (``AIKitPaths/owners``), or construction refuses by name
    ///   rather than writing into a second, invisible tree.
    /// - Throws: `VirtualPathError` when the manifest cannot be placed —
    ///   an unregistered owner, or a root the layer refused.
    public init(paths: any PathResolving) throws(VirtualPathError) {
        self.paths = paths
        self.manifest = try OwnedResourceStore(LocalModelManifest(), paths: paths)
        self.fileManager = FileManager.default
    }

    /// Where the manifest is, resolved. For a diagnostic that prints it.
    public var manifestURL: URL { manifest.url }

    /// List all locally downloaded models.
    ///
    /// - Throws: ``LocalModelStoreError/corruptManifest`` when the manifest file
    ///   exists but cannot be decoded. A missing manifest is not an error; it
    ///   means the store is fresh and the result is `[]`.
    public func listLocal() throws -> [LocalModel] {
        try loadManifest()
    }

    /// The expected local directory for a model: `<provider>/<model id>` below
    /// the `models` node, as the path layer resolves it.
    public func modelPath(for id: ModelIdentifier) throws(VirtualPathError) -> URL {
        try paths.url(for: LocalModelDirectory(id: id))
    }

    /// Where one file of a model is written, resolved by the path layer.
    ///
    /// Returns a ``ModelFileDestination`` and not a `URL`: a writer takes the
    /// resolved location as a TYPE, so no caller can compose one by appending
    /// to whatever it happened to hold.
    public func fileURL(
        for id: ModelIdentifier, named filename: String
    ) throws(VirtualPathError) -> ModelFileDestination {
        ModelFileDestination(
            resolved: try paths.url(for: LocalModelFile(id: id, filename: filename)))
    }

    /// Check if a model is downloaded.
    ///
    /// - Throws: ``LocalModelStoreError/corruptManifest`` when the manifest
    ///   exists but cannot be decoded.
    public func isDownloaded(_ id: ModelIdentifier) throws -> Bool {
        try loadManifest().contains { $0.descriptor.id == id }
    }

    /// Record a downloaded model in the manifest.
    ///
    /// The directory is created by the write, not by this call: resolution
    /// creates nothing, the writer creates what it needs (CoreKit BR-RP-01).
    public func register(_ descriptor: ModelDescriptor, at path: URL) throws {
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
    /// - Throws: ``LocalModelStoreError/corruptManifest`` when the manifest
    ///   exists but cannot be decoded.
    public func totalSizeBytes() throws -> Int64 {
        try loadManifest().reduce(0) { $0 + $1.sizeOnDisk }
    }

    // MARK: - Private

    /// The manifest, or `[]` when it was never written.
    ///
    /// A missing document is `nil` from ``OwnedResourceStore``, never an error,
    /// which is the same distinction this store always drew: fresh is not
    /// corrupt. Anything else — unreadable, undecodable — is reported as
    /// ``LocalModelStoreError/corruptManifest`` with the underlying reason
    /// logged, because an empty list here would be overwritten on the next
    /// write and take every recorded model with it.
    private func loadManifest() throws -> [LocalModel] {
        do {
            return try manifest.load() ?? []
        } catch {
            storeLogger.error("LocalModelStore: manifest load failed — \(error)")
            throw LocalModelStoreError.corruptManifest
        }
    }

    private func saveManifest(_ models: [LocalModel]) throws {
        try manifest.save(models)
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
