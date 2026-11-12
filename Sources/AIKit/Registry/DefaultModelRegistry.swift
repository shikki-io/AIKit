import CoreKit
import Foundation

/// Production model registry combining local storage, HuggingFace search, and download.
public final class DefaultModelRegistry: ModelRegistry, @unchecked Sendable {
    private let localStore: LocalModelStore
    private let modelStore: ModelStore
    private let planner: ModelDownloadPlanner
    private let consent: ModelDownloadConsent
    private let hfClient: HuggingFaceClient
    private let downloader: ModelDownloader

    /// - Parameters:
    ///   - localStore: injected, with no default. The manifest's
    ///     location comes from the host's path layer (W23 of spec
    ///     ai-layer-aikit-core-2026-10-31), so a kit cannot build one for
    ///     itself.
    ///   - modelStore: the verified file store. Injected for the same reason
    ///     and over the SAME path layer: it is what verifies a downloaded file
    ///     against the digest the hub published, and what answers "is this file
    ///     already here" so a plan does not count bytes twice (W33, BR-AIL-23).
    ///     Two stores rather than one because they answer two questions — the
    ///     manifest records what this device has, the file store vouches for
    ///     bytes — and folding them would be a wave of its own.
    ///   - planner: the download policy. Defaults to the shipped one, so a host
    ///     that configures nothing still refuses a download that cannot fit.
    ///     A host reading `[models.download]` from its settings file passes a
    ///     planner built from it.
    ///   - consent: an answer the operator already gave, applied to every
    ///     download this registry makes. A per-download question belongs to the
    ///     interface asking it, which builds a registry with the answer it got.
    ///
    /// The search client and the downloader are NOT parameters here: the
    /// Hugging Face endpoint stays internal BEHIND the catalogue (W27,
    /// BR-AIL-21, door 3). A caller asks the catalogue for models; which
    /// registry it searches and how it fetches bytes is the catalogue's own
    /// business, and a test reaches the init below through `@testable` to
    /// inject both.
    public convenience init(
        localStore: LocalModelStore,
        modelStore: ModelStore,
        planner: ModelDownloadPlanner = ModelDownloadPlanner(),
        consent: ModelDownloadConsent = .none
    ) {
        self.init(
            localStore: localStore,
            modelStore: modelStore,
            planner: planner,
            consent: consent,
            hfClient: HuggingFaceClient(),
            downloader: ModelDownloader()
        )
    }

    /// The full init. Internal because `HuggingFaceClient` and
    /// `ModelDownloader` are.
    init(
        localStore: LocalModelStore,
        modelStore: ModelStore,
        planner: ModelDownloadPlanner = ModelDownloadPlanner(),
        consent: ModelDownloadConsent = .none,
        hfClient: HuggingFaceClient = HuggingFaceClient(),
        downloader: ModelDownloader = ModelDownloader()
    ) {
        self.localStore = localStore
        self.modelStore = modelStore
        self.planner = planner
        self.consent = consent
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

    /// Download a model, after planning the transfer (BR-AIL-23).
    ///
    /// The order is the rule: the repository's file listing is read first, so
    /// the PUBLISHED size is known; the plan then compares it with the free
    /// space on the destination volume and with the network path, and
    /// ``ModelDownloadPlan/authorize(consent:)`` throws before a byte moves.
    /// What this replaced started transferring immediately, with the size
    /// sitting unread in the same API it was already calling.
    ///
    /// The file that arrives is verified against the hub's digest BEFORE it
    /// enters the manifest: a corrupt or truncated download used to be
    /// registered on its HTTP status and failed much later, inside a model
    /// loader, as a parse error that named nothing.
    ///
    /// - Throws: ``AIKitError/insufficientSpace(model:neededBytes:freeBytes:)``
    ///   or ``AIKitError/downloadNeedsConsent(model:reason:network:)`` from the
    ///   plan; ``ModelStoreError/digestMismatch(file:algorithm:expected:actual:)``
    ///   when the bytes that arrived are not the published ones, after deleting
    ///   them; ``AIKitError/downloadFailed(_:)`` when the repository publishes
    ///   no single weights file to choose.
    public func download(
        id: ModelIdentifier,
        onProgress: @Sendable @escaping (Double) -> Void
    ) async throws {
        guard id.provider == "huggingface" else {
            throw AIKitError.downloadFailed("Download only supported for HuggingFace models")
        }

        let info = try await hfClient.getModel(id: id.modelId)
        // The tree, BEFORE any transfer: it carries each file's published size
        // and digest, which are the two things a plan and a verification need.
        // No listing, no plan — and a download nobody could size is exactly
        // what this wave removes.
        let published = try await hfClient.files(repoId: info.modelId)
        let file = try Self.weightsFile(among: published, repository: info.modelId)

        // Resolved by the path layer, not appended to whatever this type held.
        let destination = try localStore.fileURL(for: id, named: file.path)

        // A file already on disk that hashes to the published digest is not
        // fetched again, and is not counted in the plan's bytes either.
        let alreadyHeld: Set<String> =
            modelStore.isVerified(fileAt: destination.url, as: file) ? [file.path] : []

        let plan = await planner.plan(
            repository: info.modelId,
            files: [file],
            alreadyHeld: alreadyHeld,
            into: destination.url.deletingLastPathComponent()
        )
        try plan.authorize(consent: consent)

        if alreadyHeld.isEmpty {
            let downloadURL = try hfClient.downloadURL(repoId: info.modelId, filename: file.path)
            try await downloader.download(from: downloadURL, to: destination) { progress in
                onProgress(progress.fraction)
            }
            // Verified before it is registered: a mismatch deletes the bytes
            // and names the file and both digests.
            try modelStore.acceptDownloaded(file, at: destination)
        }

        var localDescriptor = info.toDescriptor(format: .gguf)
        localDescriptor.isLocal = true
        localDescriptor.localPath = destination.url
        // The published size, which the descriptor used to carry as 0 because
        // the model-detail endpoint does not report one and the tree was never
        // read.
        localDescriptor.sizeBytes = file.sizeBytes
        try localStore.register(localDescriptor, at: destination.url)
        onProgress(1.0)
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

    // MARK: - Choosing the file

    /// The extension this registry downloads. It serves the GGUF convention;
    /// a multi-file MLX snapshot is `AIKitMLX`'s downloader, not this one.
    static let weightsExtension = "gguf"

    /// The one weights file in a repository's listing.
    ///
    /// The convention this keeps is the one it replaced: `<repository
    /// name>.gguf`. What it no longer does is BUILD that name and request it
    /// blind — the name was composed from the repository id and fetched whether
    /// or not the repository published such a file, so a repository following a
    /// different convention produced a 404 reported as a download failure.
    ///
    /// Now the listing decides. An exact match wins; a repository publishing
    /// exactly one `.gguf` file under any name is that file; a repository
    /// publishing several is REFUSED, naming them. The last part is deliberate:
    /// picking "the largest" would quietly choose a quantisation on the
    /// operator's behalf, and which quantisation to run is their decision.
    static func weightsFile(
        among files: [ModelFile], repository: String
    ) throws(AIKitError) -> ModelFile {
        let candidates = files.filter {
            $0.path.lowercased().hasSuffix(".\(weightsExtension)")
        }

        // The repository id is a POSIX wire path, so its last component is read
        // as one — `VirtualPath` parses and validates it, and no separator is
        // spelled here.
        if let stem = (try? VirtualPath(parsing: repository))?.last?.rawValue,
            let exact = candidates.first(where: { $0.path == "\(stem).\(weightsExtension)" }) {
            return exact
        }

        switch candidates.count {
        case 0:
            throw .downloadFailed(
                "\(repository) publishes no .\(weightsExtension) file to download")
        case 1:
            return candidates[0]
        default:
            let names = candidates.map(\.path).sorted().joined(separator: ", ")
            throw .downloadFailed(
                """
                \(repository) publishes \(candidates.count) .\(weightsExtension) files and \
                none is named after the repository, so which one to download is the \
                caller's choice, not this registry's: \(names)
                """)
        }
    }
}
