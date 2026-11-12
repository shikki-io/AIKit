import CoreKit
import CryptoKit
import Foundation

// MARK: - The tree AIKit declares

/// The nodes AIKit owns in the host's virtual tree (CoreKit
/// ``VirtualNodeNaming``).
///
/// W23 of spec ai-layer-aikit-core-2026-10-31, AIKit#18 r4500361107:
/// *"from CoreKit I guess we are supposed to start to have an abstract system
/// with a default implementation … I want to be sure even for stores we use
/// also my protocol and implementation we have in our Core SPM"*.
///
/// One node, because AIKit has one place on disk: everything about models on
/// this device. Where that node actually SITS is not declared here and cannot
/// be — the host injects the root, and may serve this one node from somewhere
/// else entirely (`PathLayer`'s `nodeRoots`), which is how a model library on
/// an external disk becomes a settings value rather than a code change.
///
/// There is no `defaultRoot` any more. A default root is a second source of
/// truth for a location the host owns, and two spellings of "the cache root"
/// is exactly how one 4 GB model came to sit in two directories at once
/// (measured 2026-10-31: `mlx-community/Mistral-7B-Instruct-v0.3-4bit`'s
/// `model.safetensors`, 4077480215 bytes, inodes 939711379 and 939711382,
/// both hashing to `d7182d88…503d`).
///
/// For a host that wants the layout AIKit shipped before this wave, the wiring
/// is exact and loses nothing: mount the layer's root at the operator's
/// `.aikit` directory and this node resolves to the same `.aikit/models` every
/// existing download already sits in.
public enum AIKitNode: String, VirtualNodeNaming {
    /// Model weights, manifests and verification receipts.
    case models
}

// MARK: - Owners

/// The manifest ``LocalModelStore`` keeps: every model this device downloaded.
public struct LocalModelManifest: VirtualDocumentOwning {
    public typealias Document = [LocalModel]

    public static let placement = VirtualPlacement(node: AIKitNode.models, name: "manifest.json")

    public init() {}
}

/// One model's directory as ``LocalModelStore`` lays it out: the provider, then
/// the model id as a single folded name.
public struct LocalModelDirectory: VirtualPathOwning {
    public static let placement = VirtualPlacement(node: AIKitNode.models, name: "local-model")

    public let id: ModelIdentifier

    public init(id: ModelIdentifier) {
        self.id = id
    }

    public func location() throws(VirtualPathError) -> VirtualPath {
        VirtualPath(try PathSegment(id.provider), try ModelStore.folded(id.modelId))
    }
}

/// One file inside a model's ``LocalModelDirectory``, e.g. the single GGUF
/// `DefaultModelRegistry` downloads.
public struct LocalModelFile: VirtualPathOwning {
    public static let placement = VirtualPlacement(
        node: AIKitNode.models, name: "local-model-file")

    public let id: ModelIdentifier
    /// The file's name, relative to the model's own directory.
    public let filename: String

    public init(id: ModelIdentifier, filename: String) {
        self.id = id
        self.filename = filename
    }

    public func location() throws(VirtualPathError) -> VirtualPath {
        let relative = try VirtualPath(parsing: filename)
        guard !relative.isEmpty else { throw .emptySegment }
        return VirtualPath(try PathSegment(id.provider), try ModelStore.folded(id.modelId))
            .appending(relative)
    }
}

/// One model repository's own directory, e.g. `mlx-community/Mistral-7B` as
/// `mlx-community_Mistral-7B`.
public struct ModelRepositoryDirectory: VirtualPathOwning {
    public static let placement = VirtualPlacement(node: AIKitNode.models, name: "model-repository")

    public let repositoryID: String

    public init(repositoryID: String) {
        self.repositoryID = repositoryID
    }

    public func location() throws(VirtualPathError) -> VirtualPath {
        VirtualPath(try ModelStore.folded(repositoryID))
    }
}

/// One file inside a repository's directory, at the repository-relative path
/// the hub published.
///
/// That path comes from a remote index, so it is PARSED into validated
/// components rather than appended as a string: `../../../.ssh/authorized_keys`
/// is a legal thing for a hub to return and is not a legal place to write. A
/// ``PathSegment`` cannot hold `..` or a separator, so the refusal happens
/// here, before any URL exists, and names the component that offended.
public struct ModelRepositoryFile: VirtualPathOwning {
    public static let placement = VirtualPlacement(
        node: AIKitNode.models, name: "model-repository-file")

    public let repositoryID: String
    /// Repository-relative, POSIX, as the hub published it.
    public let path: String

    public init(repositoryID: String, path: String) {
        self.repositoryID = repositoryID
        self.path = path
    }

    public func location() throws(VirtualPathError) -> VirtualPath {
        // An empty path is not a file: `VirtualPath(parsing: "")` is the
        // legitimate empty path, which here would resolve to the repository's
        // own directory and let a listing name it as a file to write.
        let relative = try VirtualPath(parsing: path)
        guard !relative.isEmpty else { throw .emptySegment }
        return VirtualPath(try ModelStore.folded(repositoryID)).appending(relative)
    }
}

/// What ``ModelStore`` verified for one repository, so a second load does not
/// re-read gigabytes.
public struct ModelVerificationReceipt: VirtualDocumentOwning {
    public typealias Document = ModelStore.Receipt

    public static let placement = VirtualPlacement(
        node: AIKitNode.models, name: ".aikit-verified.json")

    public let repositoryID: String

    public init(repositoryID: String) {
        self.repositoryID = repositoryID
    }

    public func location() throws(VirtualPathError) -> VirtualPath {
        VirtualPath(try ModelStore.folded(repositoryID), Self.placement.name)
    }
}

// MARK: - Composition

/// What a host registers to let AIKit resolve its own locations.
///
/// An owner that was never added to the tree is refused BY NAME when it is
/// resolved (`VirtualPathError.unregisteredOwner`) rather than silently writing
/// into a second, invisible tree — so this list is the whole wiring contract,
/// and ``layer(root:nodeRoots:)`` is it in one line.
public enum AIKitPaths {

    /// Every owner kind AIKit resolves.
    public static let owners: [any VirtualPathOwning.Type] = [
        LocalModelManifest.self,
        LocalModelDirectory.self,
        LocalModelFile.self,
        ModelRepositoryDirectory.self,
        ModelRepositoryFile.self,
        ModelVerificationReceipt.self,
    ]

    /// The node vocabulary AIKit declares, for a host composing the whole tree
    /// through `VirtualNodeRegistry`.
    ///
    /// Computed, not stored: `VirtualNodeNaming` is not `Sendable`, so a stored
    /// array of its metatypes is a mutable global as far as the compiler is
    /// concerned. The value is a literal of two static things and costs
    /// nothing to rebuild.
    public static var nodes: [any VirtualNodeNaming.Type] { [AIKitNode.self] }

    /// A `PathLayer` over `root` with AIKit's owners registered.
    ///
    /// The SAME type production and tests use: a test differs by the root it
    /// passes and by nothing else (CoreKit BR-RP-03), which is what makes
    /// `ModelStoreTests.locationsComeFromThePathLayer` an assertion about the
    /// production composition rather than about a test double.
    ///
    /// - Parameters:
    ///   - root: the injected root; an absolute file URL.
    ///   - nodeRoots: an AIKit node served from somewhere other than below
    ///     `root` — a models directory on an external disk, say.
    public static func layer(
        root: URL,
        nodeRoots: [AIKitNode: URL] = [:]
    ) throws(VirtualPathError) -> PathLayer {
        try PathLayer(root: root, nodeRoots: nodeRoots, registry: VirtualPathRegistry(owners))
    }
}

// MARK: - A library another application owns

/// A directory some OTHER application fills with model weights, which AIKit may
/// READ and never writes to.
///
/// A Mac that runs LM Studio already holds the weights an on-device provider
/// wants; downloading them again is 4 GB of nothing. These are not AIKit's
/// nodes — they are outside the host's root, so they are injected as URLs and
/// no node declares them (a node whose base escaped the root would fail
/// CoreKit's `VirtualNodeNamingRatchet`, correctly).
///
/// The location is ``VirtualPath`` COMPONENTS, not a string with separators in
/// it (AIKit#18 r4500366145 and r4500367342: *"here it miss probably the case
/// of Linux & windows"*). One declaration therefore renders as
/// `~/.lmstudio/models` on macOS and Linux and as
/// `%USERPROFILE%\.lmstudio\models` on Windows, because
/// ``VirtualPath/fileURL(under:)`` builds a `URL` component by component and
/// the separator is never typed here at all.
public struct ExternalModelLibrary: Sendable, Equatable {

    /// The application that owns the directory, for a message a human reads.
    public let application: String
    /// Where it sits below the user's home directory, as components.
    public let path: VirtualPath

    public init(application: String, path: VirtualPath) {
        self.application = application
        self.path = path
    }

    /// This library's directory below `home`.
    public func url(under home: URL) -> URL {
        path.fileURL(under: home)
    }

    /// The libraries worth probing on the platform this binary was built for.
    ///
    /// Home-relative on every desktop platform, which is why one row serves
    /// all three: LM Studio's documented default model directory is
    /// `~/.lmstudio/models`, and HuggingFace's `HF_HOME` defaults to
    /// `~/.cache/huggingface` with the hub cache at `$HF_HOME/hub`.
    ///
    /// EMPTY on iOS and the other sandboxed platforms, and that is the whole
    /// enforcement of BR-AIL-14's "on iOS the only root is the app's own
    /// container": there is nothing else in the list to read.
    public static var knownForHost: [ExternalModelLibrary] {
        #if os(macOS)
            return [
                ExternalModelLibrary(application: "LM Studio", path: VirtualPath(".lmstudio", "models")),
                ExternalModelLibrary(
                    application: "HuggingFace hub cache",
                    path: VirtualPath(".cache", "huggingface", "hub")),
                // AIKit's own former second root, measured holding a duplicate
                // of the Mistral weights on 2026-10-31. Kept as a READ root so
                // that duplicate is reused instead of downloaded again.
                ExternalModelLibrary(
                    application: "AIKit (legacy cache)",
                    path: VirtualPath("Library", "Caches", "models")),
            ]
        #elseif os(Linux) || os(Windows)
            return [
                ExternalModelLibrary(application: "LM Studio", path: VirtualPath(".lmstudio", "models")),
                ExternalModelLibrary(
                    application: "HuggingFace hub cache",
                    path: VirtualPath(".cache", "huggingface", "hub")),
            ]
        #else
            return []
        #endif
    }

    /// ``knownForHost`` resolved below `home`, in order.
    ///
    /// The home directory is a parameter, so the one place that decides it is
    /// the host's composition root and a test never reads the operator's real
    /// one.
    public static func knownURLs(under home: URL) -> [URL] {
        knownForHost.map { $0.url(under: home) }
    }
}

// MARK: - Where a model file is written

/// A place inside the store where one model file may be written.
///
/// Only ``ModelStore`` mints one, and only by resolving an owner through the
/// injected path layer, so a writer cannot be handed a URL somebody composed:
/// the type IS the proof that the location came from the layer. This is what
/// replaced the `URL?` that `destination(for:ofRepository:)` used to return,
/// where `nil` meant "escaped" and said nothing about why.
public struct ModelFileDestination: Sendable, Equatable {

    /// The resolved file location.
    public let url: URL

    /// Internal, and deliberately NOT the memberwise signature: the argument
    /// label says where the URL came from, and a caller outside AIKit has no
    /// way to make one at all.
    init(resolved url: URL) {
        self.url = url
    }
}

// MARK: - A file the hub published

/// The checksum a hub published for one file, and the algorithm that produced
/// it.
///
/// Two algorithms, because HuggingFace publishes two. Measured on
/// `mlx-community/Mistral-7B-Instruct-v0.3-4bit` on 2026-10-31, from
/// `/api/models/<id>/tree/main`:
///
/// ```
/// model.safetensors   oid 92ae3de5…  lfs.oid d7182d88…503d  size 4077480215
/// config.json         oid 68b84bf6…  (no lfs object)        size 721
/// ```
///
/// `shasum -a 256` of the 4 GB weights file returns `d7182d88…503d`, so
/// `lfs.oid` IS the file's SHA-256. `git hash-object config.json` returns
/// `68b84bf6…`, so the tree's own `oid` on a non-LFS file is the git blob
/// SHA-1 — `sha1("blob <bytes>\0" + content)`.
///
/// Both are therefore verifiable, which matters: an LFS object is published
/// only for the large files (weights, `tokenizer.model`), and refusing every
/// file without one would refuse every model over its `config.json`.
public enum ModelFileDigest: Sendable, Equatable, Hashable {
    /// SHA-256, as published in a tree entry's `lfs.oid`.
    case sha256(String)
    /// Git blob SHA-1, as published in a non-LFS tree entry's `oid`.
    case gitBlobSHA1(String)

    /// The hex digest itself, lowercased.
    public var hex: String {
        switch self {
        case .sha256(let value), .gitBlobSHA1(let value):
            value.lowercased()
        }
    }

    /// The algorithm's name, for an error a human reads.
    public var algorithm: String {
        switch self {
        case .sha256: "sha256"
        case .gitBlobSHA1: "git-blob-sha1"
        }
    }
}

/// One file of a model repository, as the hub describes it.
public struct ModelFile: Sendable, Equatable {
    /// Path relative to the repository root, e.g. `model.safetensors`.
    public let path: String
    /// Size in bytes, as published.
    public let sizeBytes: Int64
    /// The published checksum, or `nil` when the hub published none.
    ///
    /// `nil` is not "fine": ``ModelStore`` refuses an unverifiable file rather
    /// than storing bytes it cannot vouch for.
    public let digest: ModelFileDigest?

    public init(path: String, sizeBytes: Int64, digest: ModelFileDigest?) {
        self.path = path
        self.sizeBytes = sizeBytes
        self.digest = digest
    }
}

// MARK: - Errors

/// Why a model file was refused.
///
/// Every case names the file, because "the download failed" sends an operator
/// looking at the network when the bytes on disk are the news.
public enum ModelStoreError: Error, LocalizedError, Equatable {
    /// The bytes on disk do not hash to what the hub published.
    ///
    /// Carries both digests: a reader who sees only "mismatch" cannot tell a
    /// truncated download from the wrong file entirely.
    case digestMismatch(file: String, algorithm: String, expected: String, actual: String)

    /// The hub published no checksum for this file, so it cannot be verified.
    case unverifiableFile(String)

    /// The file the hub listed never arrived.
    case fileMissing(String)

    /// The path the hub listed is not a path inside the model's own directory.
    case pathEscapesStore(String)

    /// The path layer refused to resolve a location, and why.
    ///
    /// Carries the `VirtualPathError` itself rather than its sentence, so a
    /// caller can act on the REASON — an owner missing from the registry is a
    /// wiring bug in the host, a forbidden character is bad data from a hub —
    /// instead of matching on prose.
    case unresolvableLocation(resource: String, reason: VirtualPathError)

    public var errorDescription: String? {
        switch self {
        case .digestMismatch(let file, let algorithm, let expected, let actual):
            """
            \(file) failed its \(algorithm) check: \
            expected \(expected), got \(actual). The file was deleted.
            """
        case .unverifiableFile(let file):
            """
            \(file) has no published checksum, so it cannot be verified. \
            An unverified file is not a verified one; refusing rather than \
            storing bytes AIKit cannot vouch for.
            """
        case .fileMissing(let file):
            "\(file) was listed by the hub but is not on disk."
        case .pathEscapesStore(let file):
            """
            \(file) is not a path inside the model's own directory and was \
            refused. A listing decides what to download, never where to write.
            """
        case .unresolvableLocation(let resource, let reason):
            "The path layer could not place \(resource): \(reason.description)"
        }
    }
}

// MARK: - The store

/// The one place that answers "do we have this model".
///
/// W13 of spec ai-layer-aikit-core-2026-10-31 built it. Two problems, measured
/// on this machine on 2026-10-31, motivate it:
///
/// 1. **One model, stored twice.** `model.safetensors` of
///    `mlx-community/Mistral-7B-Instruct-v0.3-4bit` is 4077480215 bytes at
///    inode 939711379 under `~/.cache/huggingface/hub/…/blobs/` and 4077480215
///    bytes at inode 939711382 under `~/Library/Caches/models/…`. Both hash to
///    `d7182d88…503d`: identical content, 4 GB spent twice, because two
///    downloaders each kept their own root.
/// 2. **A download proved nothing.** The 2.x `HubApi` compared a local hash to
///    the expected one in offline mode only, so a normal download was accepted
///    on its HTTP status alone.
///
/// So this type owns both answers. ``locate(_:ofRepository:onProgress:)`` finds
/// a file AIKit already has — in its own node or in an injected external
/// library such as LM Studio's — and vouches for it by checksum before reusing
/// it. ``adopt(_:as:)`` brings it in by **hard link**, never by copy, so a
/// shared file takes no second space. ``acceptDownloaded(_:at:onProgress:)``
/// hashes what arrived and deletes it on a mismatch.
///
/// W23 then took its LOCATIONS away from it (AIKit#18 r4500361107,
/// r4500366145, r4500367342, r4500373871). Every path this store answers is
/// resolved by the injected CoreKit ``PathResolving`` from a
/// ``VirtualPathOwning`` declaration — ``ModelRepositoryDirectory``,
/// ``ModelRepositoryFile``, ``ModelVerificationReceipt`` — so:
///
/// - the store names no directory and spells no separator, and the one place a
///   separator is named at all is ``PathSeparator`` in CoreKit;
/// - moving the root moves every location this store answers, with no code
///   change, which is what the host's settings do and what
///   `ModelStoreTests.locationsComeFromThePathLayer` asserts;
/// - a path that arrives from a hub is parsed into validated
///   ``PathSegment``s, so a traversal is unrepresentable rather than
///   detected.
///
/// It reaches no network and knows no MLX: it is given a `[ModelFile]` list and
/// works on files. That is why it lives in `AIKit` rather than in `AIKitMLX`,
/// and why a remote provider's consumer links it without linking MLX
/// (BR-AIL-14).
public struct ModelStore: Sendable {

    // MARK: - Where it lives

    /// The injected resolver. Prod and tests differ by its root and by nothing
    /// else (CoreKit BR-RP-03).
    public let paths: any PathResolving

    /// Directories AIKit may reuse a file FROM but never writes to, in probe
    /// order.
    ///
    /// Injected, with no default: which other applications' libraries may be
    /// read is the host's call, not a list compiled into a kit. See
    /// ``ExternalModelLibrary/knownURLs(under:)`` for the documented defaults a
    /// host can hand over in one line.
    public let externalLibraries: [URL]

    /// `FileManager.default`, read rather than stored: `FileManager` is not
    /// `Sendable`, and a stored one would cost this struct its `Sendable`
    /// conformance — which `MLXHubDownloader` needs, because `Downloader` is
    /// `Sendable`.
    private var fileManager: FileManager { .default }

    /// - Parameters:
    ///   - paths: the injected path layer. It must have AIKit's owners
    ///     registered (``AIKitPaths/owners``), or every resolution refuses by
    ///     name rather than guessing.
    ///   - externalLibraries: directories to reuse from, never written to.
    ///     Empty — the default — refuses reuse entirely.
    public init(paths: any PathResolving, externalLibraries: [URL] = []) {
        self.paths = paths
        self.externalLibraries = externalLibraries
    }

    /// The directory the `models` node is served from.
    ///
    /// Read from the layer on every access rather than stored, so a layer that
    /// serves the node from elsewhere is honoured here too.
    public var modelsDirectory: URL {
        paths.base(of: AIKitNode.models)
    }

    // MARK: - Layout

    /// A repository id as ONE directory name.
    ///
    /// The separator a repository id carries is POSIX on every platform (it is
    /// a wire identifier, not a local path), so it is folded to `_` and the
    /// result is validated as a single ``PathSegment``. A hostile id therefore
    /// cannot climb out: `../../etc` names a directory called `.._.._etc`, and
    /// anything a filesystem could not carry is refused here with its reason.
    static func folded(_ repositoryID: String) throws(VirtualPathError) -> PathSegment {
        try PathSegment(
            repositoryID.replacingOccurrences(
                of: PathSeparator.posix.rawValue, with: foldedSeparator))
    }

    /// What a repository id's separator becomes in a directory name.
    private static let foldedSeparator = "_"

    /// This store's directory for one repository id, e.g. `mlx-community/X`.
    ///
    /// Resolve-only: nothing is created, and the URL does not depend on what is
    /// already on disk. That last part used to be false and was measured as a
    /// test failure on a second `download` call: `appendingPathComponent`
    /// consults the file system and appends a trailing slash only when the path
    /// already exists as a directory, so one directory had two unequal `URL`s
    /// over its lifetime. ``VirtualPath/fileURL(under:)`` builds the components
    /// itself and cannot vary that way.
    public func directory(forRepository id: String) throws(ModelStoreError) -> URL {
        try resolve(ModelRepositoryDirectory(repositoryID: id), named: id)
    }

    /// Where `file` belongs inside this store.
    ///
    /// - Throws: ``ModelStoreError/pathEscapesStore(_:)`` when the hub's path is
    ///   not a path inside the model's own directory — a traversal, a separator
    ///   inside a component, an empty component;
    ///   ``ModelStoreError/unresolvableLocation(resource:reason:)`` for anything
    ///   else the layer refused, including an owner the host never registered.
    public func destination(
        for file: ModelFile, ofRepository id: String
    ) throws(ModelStoreError) -> ModelFileDestination {
        do {
            let url = try paths.url(for: ModelRepositoryFile(repositoryID: id, path: file.path))
            return ModelFileDestination(resolved: url)
        } catch {
            switch error {
            case .emptySegment, .traversalSegment, .separatorInSegment, .malformedWireForm,
                .escapesRoot:
                throw ModelStoreError.pathEscapesStore(file.path)
            default:
                throw ModelStoreError.unresolvableLocation(resource: file.path, reason: error)
            }
        }
    }

    /// Resolve one owner, naming what could not be placed.
    private func resolve<Owner: VirtualPathOwning>(
        _ owner: Owner, named resource: String
    ) throws(ModelStoreError) -> URL {
        do {
            return try paths.url(for: owner)
        } catch {
            throw ModelStoreError.unresolvableLocation(resource: resource, reason: error)
        }
    }

    // MARK: - Hashing

    /// The digest of the file at `url`, computed the way `expected` was.
    ///
    /// Streams the file in 4 MB chunks and reports `(bytesHashed, totalBytes)`
    /// as it goes. A 4 GB file takes seconds to read, and a caller with no
    /// progress to show has no way to tell that from a hang — which is the
    /// whole reason this reports at all.
    public func digest(
        ofFileAt url: URL,
        matching expected: ModelFileDigest,
        onProgress: (@Sendable (Int64, Int64) -> Void)? = nil
    ) throws -> String {
        let totalBytes = Self.sizeOfFile(at: url, using: fileManager) ?? 0

        switch expected {
        case .sha256:
            var hasher = SHA256()
            try Self.stream(url, totalBytes: totalBytes, onProgress: onProgress) {
                hasher.update(data: $0)
            }
            return Self.hex(hasher.finalize())

        case .gitBlobSHA1:
            // git hashes `"blob <byteCount>\0"` and then the content. Measured
            // against `git hash-object config.json` on 2026-10-31.
            var hasher = Insecure.SHA1()
            hasher.update(data: Data("blob \(totalBytes)\u{0}".utf8))
            try Self.stream(url, totalBytes: totalBytes, onProgress: onProgress) {
                hasher.update(data: $0)
            }
            return Self.hex(hasher.finalize())
        }
    }

    /// Whether the file at `url` is the one `file` describes.
    ///
    /// Never throws and never deletes: this is the question asked of a
    /// CANDIDATE, including a file in somebody else's directory. Size is
    /// checked first, so a wrong-sized candidate costs a `stat` instead of a
    /// 4 GB read. A file with no published digest answers `false`, because
    /// "we cannot tell" is not "yes".
    public func isVerified(
        fileAt url: URL,
        as file: ModelFile,
        onProgress: (@Sendable (Int64, Int64) -> Void)? = nil
    ) -> Bool {
        guard let digest = file.digest,
            let size = Self.sizeOfFile(at: url, using: fileManager), size == file.sizeBytes,
            let actual = try? self.digest(ofFileAt: url, matching: digest, onProgress: onProgress)
        else {
            return false
        }
        return actual == digest.hex
    }

    // MARK: - Reuse

    /// A file AIKit already holds that IS `file`, or `nil`.
    ///
    /// Searches this store's own directory for the repository first, then each
    /// external library. Candidates are probed, never enumerated: hashing every
    /// multi-gigabyte file under three roots to answer one question would cost
    /// more than the download it saves. The probes per library are
    ///
    /// - `<library>/blobs/<sha256>` — the HuggingFace cache names objects by
    ///   their SHA-256, so a hit here is exact before a byte is read;
    /// - `<library>/<repository id>/<path>` and
    ///   `<library>/<folded repository id>/<path>` — the layout LM Studio and
    ///   this store use;
    /// - `<library>/models--<org>--<name>/snapshots/*/<path>` — the
    ///   HuggingFace cache's snapshot view.
    ///
    /// Each candidate must match on size and then on digest. A file with no
    /// published digest is never located, so it is never reused.
    public func locate(
        _ file: ModelFile,
        ofRepository id: String,
        onProgress: (@Sendable (Int64, Int64) -> Void)? = nil
    ) throws(ModelStoreError) -> URL? {
        guard file.digest != nil else { return nil }

        return try candidates(for: file, ofRepository: id).first { candidate in
            isVerified(fileAt: candidate, as: file, onProgress: onProgress)
        }
    }

    /// Bring `source` into this store at `destination` without spending the
    /// bytes twice.
    ///
    /// A hard link, so the file has one copy on disk and two names: that is
    /// what "takes no second space" means, and it is why reuse is worth doing
    /// at all. A link cannot cross a volume, and an external disk holding a
    /// model library is an ordinary setup, so a cross-device failure falls back
    /// to a copy rather than refusing the model.
    ///
    /// - Returns: `true` when the file was linked, `false` when it had to be
    ///   copied. A caller that reports disk savings needs to know which.
    @discardableResult
    public func adopt(_ source: URL, as destination: ModelFileDestination) throws -> Bool {
        let target = destination.url
        try fileManager.createDirectory(
            at: target.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        if fileManager.fileExists(atPath: target.path) {
            try fileManager.removeItem(at: target)
        }

        // A hub cache entry is a relative symlink, and linking the link itself
        // leaves a path that resolves to nothing from its new directory.
        let file = source.resolvingSymlinksInPath()
        do {
            try fileManager.linkItem(at: file, to: target)
            return true
        } catch let error as NSError where error.isCrossDevice {
            try fileManager.copyItem(at: file, to: target)
            return false
        }
        // Any other link failure is news — a permission problem reported as a
        // copy would hide it.
    }

    // MARK: - Acceptance

    /// Accept a file that just arrived, or delete it and say why.
    ///
    /// This is the check the 2.x `HubApi` did not do outside offline mode: a
    /// download was trusted because the server answered 200. A truncated or
    /// substituted weights file then failed much later, inside MLX, as a shape
    /// or parse error that named nothing.
    ///
    /// - Throws: ``ModelStoreError/unverifiableFile(_:)`` when the hub published
    ///   no digest — an unverifiable file is refused rather than kept;
    ///   ``ModelStoreError/digestMismatch(file:algorithm:expected:actual:)``,
    ///   naming the file and both digests, after deleting the bytes, so the next
    ///   attempt starts clean rather than resuming onto a corrupt file;
    ///   ``ModelStoreError/fileMissing(_:)`` when nothing arrived.
    public func acceptDownloaded(
        _ file: ModelFile,
        at destination: ModelFileDestination,
        onProgress: (@Sendable (Int64, Int64) -> Void)? = nil
    ) throws {
        let url = destination.url
        guard let expected = file.digest else {
            try? fileManager.removeItem(at: url)
            throw ModelStoreError.unverifiableFile(file.path)
        }
        guard fileManager.fileExists(atPath: url.path) else {
            throw ModelStoreError.fileMissing(file.path)
        }

        let actual = try digest(ofFileAt: url, matching: expected, onProgress: onProgress)
        guard actual == expected.hex else {
            try? fileManager.removeItem(at: url)
            throw ModelStoreError.digestMismatch(
                file: file.path,
                algorithm: expected.algorithm,
                expected: expected.hex,
                actual: actual
            )
        }
    }

    // MARK: - Receipt

    /// What this store verified for one repository, so a second load does not
    /// re-read gigabytes.
    public struct Receipt: Sendable, Equatable, Codable {
        /// Repository-relative path to the digest that was checked.
        public var verified: [String: String]
        /// Repository-relative path to the size that was checked.
        public var sizes: [String: Int64]

        public init(verified: [String: String] = [:], sizes: [String: Int64] = [:]) {
            self.verified = verified
            self.sizes = sizes
        }

        /// Whether `file` was verified here, and still has the size it had.
        ///
        /// The honest limit: this trusts a path it hashed earlier, on the
        /// evidence that the size is unchanged. Re-hashing 4 GB on every load
        /// is the freeze a progress bar exists to prevent, and this store
        /// hashes in full at the two moments bytes change — a download and an
        /// adoption.
        func covers(_ file: ModelFile) -> Bool {
            guard let digest = file.digest else { return false }
            return verified[file.path] == digest.hex && sizes[file.path] == file.sizeBytes
        }
    }

    /// The receipt store for one repository: CoreKit's own typed document
    /// store at the location the owner declares, not a hand-rolled
    /// `JSONEncoder`-to-file pair (AIKit#18 r4500373871).
    private func receiptStore(
        forRepository id: String
    ) throws(ModelStoreError) -> OwnedResourceStore<Receipt> {
        do {
            return try OwnedResourceStore(
                ModelVerificationReceipt(repositoryID: id), paths: paths)
        } catch {
            throw ModelStoreError.unresolvableLocation(resource: id, reason: error)
        }
    }

    /// The receipt stored for a repository, or an empty one.
    ///
    /// A receipt that cannot be DECODED answers empty rather than throwing: it
    /// is evidence of past verification, so losing it costs a re-hash and
    /// nothing else. A receipt that cannot be PLACED does throw — that is the
    /// host's wiring, not a stale file.
    public func receipt(forRepository id: String) throws(ModelStoreError) -> Receipt {
        let store = try receiptStore(forRepository: id)
        do {
            return try store.load() ?? Receipt()
        } catch {
            return Receipt()
        }
    }

    /// Record that every file in `files` is present and verified.
    public func writeReceipt(_ files: [ModelFile], forRepository id: String) throws {
        var receipt = try receipt(forRepository: id)
        for file in files {
            guard let digest = file.digest else { continue }
            receipt.verified[file.path] = digest.hex
            receipt.sizes[file.path] = file.sizeBytes
        }
        try receiptStore(forRepository: id).save(receipt)
    }

    /// Which of `files` are already on disk, the size they should be, and
    /// covered by this store's receipt.
    ///
    /// One receipt read for the whole list: asking per file re-read the receipt
    /// from disk once per file.
    public func heldPaths(
        among files: [ModelFile], ofRepository id: String
    ) throws(ModelStoreError) -> Set<String> {
        let receipt = try receipt(forRepository: id)
        var held: Set<String> = []
        for file in files {
            let url = try destination(for: file, ofRepository: id).url
            guard let size = Self.sizeOfFile(at: url, using: fileManager),
                size == file.sizeBytes, receipt.covers(file)
            else { continue }
            held.insert(file.path)
        }
        return held
    }

    /// Whether EVERY file is already on disk, the size it should be, and
    /// covered by this store's receipt.
    ///
    /// When this is true the caller may return the directory without reaching
    /// the network at all — which is what makes a second load free. An empty
    /// list is never "held": nothing was asked, so nothing is satisfied.
    public func holdsVerified(
        _ files: [ModelFile], ofRepository id: String
    ) throws(ModelStoreError) -> Bool {
        guard !files.isEmpty else { return false }
        return try heldPaths(among: files, ofRepository: id).count == files.count
    }

    // MARK: - Private

    /// Probe paths that could already hold `file`, cheapest first.
    ///
    /// Every component here is a ``PathSegment`` or a ``VirtualPath`` parsed
    /// from data, so this method names no separator and cannot build a path
    /// that climbs. A library whose layout the id or the path cannot address at
    /// all contributes no candidate rather than a half-built URL.
    private func candidates(
        for file: ModelFile, ofRepository id: String
    ) throws(ModelStoreError) -> [URL] {
        var urls: [URL] = [try destination(for: file, ofRepository: id).url]

        // Both come from data, so both are parsed rather than appended. A
        // repository id or a hub path that is not addressable leaves the
        // external libraries unprobed — the store's own directory above is
        // still answered.
        guard let relative = try? VirtualPath(parsing: file.path),
            let folded = try? Self.folded(id),
            let wire = try? VirtualPath(parsing: id),
            let hubName = try? PathSegment(
                Self.hubPrefix
                    + id.replacingOccurrences(
                        of: PathSeparator.posix.rawValue, with: Self.hubSeparator))
        else {
            return urls
        }

        for library in externalLibraries {
            if case .sha256(let digest) = file.digest, let blob = try? PathSegment(digest.lowercased()) {
                // The HuggingFace cache stores each object as blobs/<sha256>,
                // so this probe is the exact answer before any read.
                urls.append(VirtualPath(Self.blobs, blob).fileURL(under: library))
            }
            urls.append(wire.appending(relative).fileURL(under: library))
            urls.append(VirtualPath(folded).appending(relative).fileURL(under: library))

            let hubRepo = VirtualPath(hubName).fileURL(under: library)
            if case .sha256(let digest) = file.digest, let blob = try? PathSegment(digest.lowercased()) {
                urls.append(VirtualPath(Self.blobs, blob).fileURL(under: hubRepo))
            }
            let snapshots = VirtualPath(Self.snapshots).fileURL(under: hubRepo)
            let revisions =
                (try? fileManager.contentsOfDirectory(
                    at: snapshots, includingPropertiesForKeys: nil)) ?? []
            for revision in revisions {
                urls.append(relative.fileURL(under: revision))
            }
        }

        return urls
    }

    /// How the HuggingFace cache names a repository's directory:
    /// `models--<org>--<name>`.
    private static let hubPrefix = "models--"
    private static let hubSeparator = "--"
    private static let blobs: PathSegment = "blobs"
    private static let snapshots: PathSegment = "snapshots"

    /// Byte count of a regular file, following a symlink.
    ///
    /// `attributesOfItem` does NOT follow a symlink, and the HuggingFace cache
    /// is symlinks into `blobs/` — measured: the snapshot entry for a 4 GB
    /// weights file reports 76 bytes, the length of the link target. Reading
    /// the resolved path is what makes a cache hit visible at all.
    static func sizeOfFile(at url: URL, using fileManager: FileManager) -> Int64? {
        let resolved = url.resolvingSymlinksInPath()
        guard let attributes = try? fileManager.attributesOfItem(atPath: resolved.path),
            attributes[.type] as? FileAttributeType == .typeRegular,
            let size = attributes[.size] as? Int64
        else {
            return nil
        }
        return size
    }

    private static let chunkBytes = 4 * 1024 * 1024

    /// Read `url` in chunks, reporting bytes as they are consumed.
    private static func stream(
        _ url: URL,
        totalBytes: Int64,
        onProgress: (@Sendable (Int64, Int64) -> Void)?,
        consume: (Data) -> Void
    ) throws {
        let handle = try FileHandle(forReadingFrom: url.resolvingSymlinksInPath())
        defer { try? handle.close() }

        var hashed: Int64 = 0
        onProgress?(0, totalBytes)
        while let chunk = try handle.read(upToCount: chunkBytes), !chunk.isEmpty {
            consume(chunk)
            hashed += Int64(chunk.count)
            onProgress?(hashed, totalBytes)
        }
    }

    private static func hex(_ digest: some Digest) -> String {
        digest.map { String(format: "%02x", $0) }.joined()
    }
}

// MARK: - Cross-device detection

extension NSError {
    /// Whether this error is `EXDEV`, "cross-device link".
    ///
    /// `FileManager.linkItem` wraps the POSIX error in a Cocoa error rather
    /// than surfacing it, so the underlying error has to be read. This is the
    /// one link failure worth a copy: a model library on an external disk is an
    /// ordinary setup and must still load.
    fileprivate var isCrossDevice: Bool {
        if domain == NSPOSIXErrorDomain, code == Int(EXDEV) { return true }
        guard let underlying = userInfo[NSUnderlyingErrorKey] as? NSError else { return false }
        return underlying.domain == NSPOSIXErrorDomain && underlying.code == Int(EXDEV)
    }
}
