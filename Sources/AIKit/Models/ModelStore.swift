import CryptoKit
import Foundation

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
    /// `nil` is not "fine": `ModelStore` refuses an unverifiable file rather
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

    /// The path the hub listed points outside the model's own directory.
    case pathEscapesStore(String)

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
            \(file) resolves outside the model's own directory and was refused. \
            A listing decides what to download, never where to write.
            """
        }
    }
}

// MARK: - The store

/// The one place that answers "do we have this model".
///
/// W13 of spec ai-layer-aikit-core-2026-10-31. Two problems, measured on this
/// machine on 2026-10-31, motivate it:
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
/// So this type owns both answers. `locate` finds a file AIKit already has —
/// in its own root or in a configured extra root such as `~/.lmstudio/models` —
/// and vouches for it by checksum before reusing it. `adopt` brings it in by
/// **hard link**, never by copy, so a shared file takes no second space.
/// `acceptDownloaded` hashes what arrived and deletes it on a mismatch.
///
/// It reaches no network and knows no MLX: it is given a `[ModelFile]` list and
/// works on files. That is why it lives in `AIKit` rather than in `AIKitMLX`,
/// and why a remote provider's consumer links it without linking MLX
/// (BR-AIL-14).
public struct ModelStore: Sendable {

    // MARK: - Roots

    /// The one root AIKit writes model weights under.
    ///
    /// Deliberately the same directory `LocalModelStore` has always used, which
    /// is what makes "one root" true rather than aspirational: the store that
    /// records a download and the store that answers "do we have it" cannot
    /// disagree if there is one definition. `LocalModelStore.init` reads this.
    ///
    /// On a non-macOS platform this is inside the app's own container, which is
    /// the only place an iOS app may write.
    public static var defaultRoot: URL {
        #if os(macOS)
            return FileManager.default.homeDirectoryForCurrentUser
                .appending(path: ".aikit", directoryHint: .isDirectory)
                .appending(path: "models", directoryHint: .isDirectory)
        #else
            let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
            let base = documents.first ?? FileManager.default.temporaryDirectory
            return base.appending(path: "aikit-models", directoryHint: .isDirectory)
        #endif
    }

    /// Directories AIKit may reuse a file FROM but never writes to.
    ///
    /// A Mac that runs LM Studio already holds the weights an on-device
    /// provider wants; downloading them again is 4 GB of nothing. The HF cache
    /// is here for the same reason, and it is the cheapest root to search:
    /// it stores each object at `blobs/<sha256>`, so a lookup is one
    /// `fileExists` rather than a hash of every candidate.
    ///
    /// Empty off macOS: an iOS app has no sibling directories to read, and
    /// BR-AIL-14's "the only root is the app's own container" is enforced by
    /// there being nothing else in this list.
    public static var defaultExtraRoots: [URL] {
        #if os(macOS)
            let home = FileManager.default.homeDirectoryForCurrentUser
            return [
                home.appending(path: ".lmstudio/models", directoryHint: .isDirectory),
                home.appending(path: ".cache/huggingface/hub", directoryHint: .isDirectory),
                home.appending(path: "Library/Caches/models", directoryHint: .isDirectory),
            ]
        #else
            return []
        #endif
    }

    /// Where this store writes.
    public let root: URL
    /// Where this store may read from, in order.
    public let extraRoots: [URL]

    /// `FileManager.default`, read rather than stored: `FileManager` is not
    /// `Sendable`, and a stored one would cost this struct its `Sendable`
    /// conformance — which `MLXHubDownloader` needs, because `Downloader` is
    /// `Sendable`.
    private var fileManager: FileManager { .default }

    /// - Parameters:
    ///   - root: the one directory weights are written under.
    ///   - extraRoots: directories to reuse from, never written to. Pass `[]`
    ///     to refuse reuse entirely.
    public init(
        root: URL = ModelStore.defaultRoot,
        extraRoots: [URL] = ModelStore.defaultExtraRoots
    ) {
        self.root = root
        self.extraRoots = extraRoots
    }

    // MARK: - Layout

    /// This store's directory for one repository id, e.g. `mlx-community/X`.
    ///
    /// The `/` becomes `_` so one repository is one directory, matching
    /// `LocalModelStore.modelPath(for:)`.
    /// `directoryHint:` is not cosmetic. `appendingPathComponent` consults the
    /// file system and appends a trailing slash only when the path already
    /// exists as a directory, so this method returned
    /// `…/mlx-community_X` before the first download and
    /// `…/mlx-community_X/` after it — two unequal `URL`s for one directory,
    /// which breaks any caller that compares or caches what it was handed.
    /// Measured as a test failure on the second `download` call.
    public func directory(forRepository id: String) -> URL {
        root.appending(
            path: id.replacingOccurrences(of: "/", with: "_"),
            directoryHint: .isDirectory
        )
    }

    /// Where `file` belongs inside this store, or `nil` when its path escapes
    /// the model's own directory.
    ///
    /// The path comes from a remote index. `../../../.ssh/authorized_keys` is a
    /// legal string for a hub to return and is not a legal place to write, so
    /// the resolved path is checked against the directory it must stay under.
    /// A repository id cannot escape either: its `/` becomes `_`, so
    /// `../../etc` names a directory called `.._.._etc`.
    public func destination(for file: ModelFile, ofRepository id: String) -> URL? {
        let directory = directory(forRepository: id)
        let candidate = directory
            .appending(path: file.path, directoryHint: .notDirectory)
            .standardizedFileURL
        let base = directory.standardizedFileURL.path
        guard candidate.path.hasPrefix(base.hasSuffix("/") ? base : base + "/") else { return nil }
        return candidate
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
        guard let digest = file.digest, let size = Self.sizeOfFile(at: url, using: fileManager), size == file.sizeBytes, let actual = try? self.digest(ofFileAt: url, matching: digest, onProgress: onProgress)
        else {
            return false
        }
        return actual == digest.hex
    }

    // MARK: - Reuse

    /// A file AIKit already holds that IS `file`, or `nil`.
    ///
    /// Searches this store's own directory for the repository first, then each
    /// extra root. Candidates are probed, never enumerated: hashing every
    /// multi-gigabyte file under three roots to answer one question would cost
    /// more than the download it saves. The probes per root are
    ///
    /// - `<root>/blobs/<sha256>` — the HuggingFace cache names objects by their
    ///   SHA-256, so a hit here is exact before a byte is read;
    /// - `<root>/<repository id>/<path>` and `<root>/<repository_id>/<path>` —
    ///   the layout LM Studio and this store use;
    /// - `<root>/models--<org>--<name>/snapshots/*/<path>` — the HuggingFace
    ///   cache's snapshot view.
    ///
    /// Each candidate must match on size and then on digest. A file with no
    /// published digest is never located, so it is never reused.
    public func locate(
        _ file: ModelFile,
        ofRepository id: String,
        onProgress: (@Sendable (Int64, Int64) -> Void)? = nil
    ) -> URL? {
        guard file.digest != nil else { return nil }

        return candidates(for: file, ofRepository: id).first { candidate in
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
    public func adopt(_ source: URL, as destination: URL) throws -> Bool {
        try fileManager.createDirectory(
            at: destination.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        if fileManager.fileExists(atPath: destination.path) {
            try fileManager.removeItem(at: destination)
        }

        do {
            try fileManager.linkItem(at: source, to: destination)
            return true
        } catch let error as NSError where error.isCrossDevice {
            try fileManager.copyItem(at: source, to: destination)
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
    /// - Throws: `ModelStoreError.unverifiableFile` when the hub published no
    ///   digest — an unverifiable file is refused rather than kept;
    ///   `ModelStoreError.digestMismatch`, naming the file and both digests,
    ///   after deleting the bytes, so the next attempt starts clean rather than
    ///   resuming onto a corrupt file;
    ///   `ModelStoreError.fileMissing` when nothing arrived.
    public func acceptDownloaded(
        _ file: ModelFile,
        at url: URL,
        onProgress: (@Sendable (Int64, Int64) -> Void)? = nil
    ) throws {
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

    private var receiptName: String { ".aikit-verified.json" }

    /// The receipt stored for a repository, or an empty one.
    public func receipt(forRepository id: String) -> Receipt {
        let url = directory(forRepository: id)
            .appending(path: receiptName, directoryHint: .notDirectory)
        guard let data = try? Data(contentsOf: url),
            let receipt = try? JSONDecoder().decode(Receipt.self, from: data)
        else {
            return Receipt()
        }
        return receipt
    }

    /// Record that every file in `files` is present and verified.
    public func writeReceipt(_ files: [ModelFile], forRepository id: String) throws {
        var receipt = receipt(forRepository: id)
        for file in files {
            guard let digest = file.digest else { continue }
            receipt.verified[file.path] = digest.hex
            receipt.sizes[file.path] = file.sizeBytes
        }

        let directory = directory(forRepository: id)
        try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(receipt).write(
            to: directory.appending(path: receiptName, directoryHint: .notDirectory),
            options: .atomic
        )
    }

    /// Which of `files` are already on disk, the size they should be, and
    /// covered by this store's receipt.
    ///
    /// One receipt read for the whole list: asking per file re-read the receipt
    /// from disk once per file.
    public func heldPaths(among files: [ModelFile], ofRepository id: String) -> Set<String> {
        let receipt = receipt(forRepository: id)
        let directory = directory(forRepository: id)

        return Set(
            files
                .filter { file in
                    let url = directory.appending(path: file.path, directoryHint: .notDirectory)
                    guard let size = Self.sizeOfFile(at: url, using: fileManager) else {
                        return false
                    }
                    return size == file.sizeBytes && receipt.covers(file)
                }
                .map(\.path)
        )
    }

    /// Whether EVERY file is already on disk, the size it should be, and
    /// covered by this store's receipt.
    ///
    /// When this is true the caller may return the directory without reaching
    /// the network at all — which is what makes a second load free. An empty
    /// list is never "held": nothing was asked, so nothing is satisfied.
    public func holdsVerified(_ files: [ModelFile], ofRepository id: String) -> Bool {
        guard !files.isEmpty else { return false }
        return heldPaths(among: files, ofRepository: id).count == files.count
    }

    // MARK: - Private

    /// Probe paths that could already hold `file`, cheapest first.
    private func candidates(for file: ModelFile, ofRepository id: String) -> [URL] {
        var urls: [URL] = [
            directory(forRepository: id).appending(path: file.path, directoryHint: .notDirectory)
        ]

        let underscored = id.replacingOccurrences(of: "/", with: "_")
        let hubName = "models--" + id.replacingOccurrences(of: "/", with: "--")

        for extraRoot in extraRoots {
            if case .sha256(let hex) = file.digest {
                // The HuggingFace cache stores each object as blobs/<sha256>,
                // so this probe is the exact answer before any read.
                urls.append(
                    extraRoot
                        .appending(path: "blobs", directoryHint: .isDirectory)
                        .appending(path: hex.lowercased(), directoryHint: .notDirectory)
                )
            }
            urls.append(
                extraRoot
                    .appending(path: id, directoryHint: .isDirectory)
                    .appending(path: file.path, directoryHint: .notDirectory)
            )
            urls.append(
                extraRoot
                    .appending(path: underscored, directoryHint: .isDirectory)
                    .appending(path: file.path, directoryHint: .notDirectory)
            )

            let hubRepo = extraRoot.appending(path: hubName, directoryHint: .isDirectory)
            if case .sha256(let hex) = file.digest {
                urls.append(
                    hubRepo
                        .appending(path: "blobs", directoryHint: .isDirectory)
                        .appending(path: hex.lowercased(), directoryHint: .notDirectory)
                )
            }
            let snapshots = hubRepo.appending(path: "snapshots", directoryHint: .isDirectory)
            let revisions =
                (try? fileManager.contentsOfDirectory(at: snapshots, includingPropertiesForKeys: nil))
                ?? []
            for revision in revisions {
                urls.append(revision.appending(path: file.path, directoryHint: .notDirectory))
            }
        }

        return urls
    }

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
