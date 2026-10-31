import CryptoKit
import Foundation
import Testing

@testable import AIKit

/// W13 of spec ai-layer-aikit-core-2026-10-31: one store answers "do we have
/// this model", reuses a file it can vouch for without spending the bytes
/// twice, and refuses one it cannot.
///
/// Every digest in here is computed from the bytes the test wrote, never
/// pasted: a hard-coded hash proves the test's own arithmetic, not the store's.
/// The one exception is `gitBlobDigestMatchesGitHashObject`, which pins the git
/// blob formula against a value `git hash-object` produced.
@Suite("ModelStore")
struct ModelStoreTests {

    // MARK: - Fixtures

    /// A sandbox with a store root and an extra root, both empty.
    ///
    /// A class so the suite can hold it in a `let` and let `deinit` remove the
    /// tree; the alternative is a `defer` in every test, which is the line that
    /// gets forgotten.
    private final class Sandbox {
        let base: URL
        let root: URL
        let extraRoot: URL
        let store: ModelStore

        init() throws {
            base = FileManager.default.temporaryDirectory
                .appendingPathComponent("aikit-modelstore-\(UUID().uuidString)")
            root = base.appendingPathComponent("store")
            extraRoot = base.appendingPathComponent("lmstudio")
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            try FileManager.default.createDirectory(
                at: extraRoot,
                withIntermediateDirectories: true
            )
            store = ModelStore(root: root, extraRoots: [extraRoot])
        }

        deinit {
            try? FileManager.default.removeItem(at: base)
        }
    }

    private static func sha256Hex(of data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// Write `data` at `url`, creating parents.
    private static func write(_ data: Data, to url: URL) throws {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try data.write(to: url)
    }

    private static func inode(of url: URL) throws -> UInt64 {
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        return (attributes[.systemFileNumber] as? UInt64) ?? 0
    }

    private static func linkCount(of url: URL) throws -> Int {
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        return (attributes[.referenceCount] as? Int) ?? 0
    }

    // MARK: - T-AIL-25

    @Test("A verified file in an extra root is linked, not downloaded, and takes no second space")
    func reusesAVerifiedFile() throws {
        let sandbox = try Sandbox()
        let repository = "mlx-community/Test-7B-4bit"

        // The operator already has these weights, under LM Studio's root.
        let weights = Data("the weights of a model that is already on disk".utf8)
        let existing = sandbox.extraRoot
            .appendingPathComponent(repository)
            .appendingPathComponent("model.safetensors")
        try Self.write(weights, to: existing)

        let file = ModelFile(
            path: "model.safetensors",
            sizeBytes: Int64(weights.count),
            digest: .sha256(Self.sha256Hex(of: weights))
        )

        // Located, because the checksum matches what the hub published.
        let located = sandbox.store.locate(file, ofRepository: repository)
        #expect(located?.resolvingSymlinksInPath() == existing.resolvingSymlinksInPath())

        // Adopted into the store's own root.
        let destination = sandbox.store
            .directory(forRepository: repository)
            .appendingPathComponent(file.path)
        let linked = try sandbox.store.adopt(#require(located), as: destination)

        #expect(linked, "a same-volume reuse must be a hard link, never a copy")
        #expect(try Self.inode(of: destination) == Self.inode(of: existing))
        #expect(try Self.linkCount(of: destination) == 2)
        #expect(try Data(contentsOf: destination) == weights)

        // The whole point: one copy of the bytes, two names.
        let bytesUnderBothRoots = try [existing, destination]
            .map { try Self.inode(of: $0) }
        #expect(Set(bytesUnderBothRoots).count == 1, "two inodes would mean the bytes were copied")
    }

    @Test("A file whose bytes differ is not located, however right its name looks")
    func doesNotReuseAFileThatOnlyLooksRight() throws {
        let sandbox = try Sandbox()
        let repository = "mlx-community/Test-7B-4bit"

        let published = Data("the real weights".utf8)
        let imposter = Data("different bytes!".utf8)
        #expect(published.count == imposter.count, "same size, so only the hash can tell them apart")

        try Self.write(
            imposter,
            to: sandbox.extraRoot
                .appendingPathComponent(repository)
                .appendingPathComponent("model.safetensors")
        )

        let file = ModelFile(
            path: "model.safetensors",
            sizeBytes: Int64(published.count),
            digest: .sha256(Self.sha256Hex(of: published))
        )

        #expect(sandbox.store.locate(file, ofRepository: repository) == nil)
    }

    @Test("A file with no published checksum is never reused")
    func neverReusesAnUnverifiableFile() throws {
        let sandbox = try Sandbox()
        let repository = "mlx-community/Test-7B-4bit"

        let bytes = Data("some bytes".utf8)
        try Self.write(
            bytes,
            to: sandbox.extraRoot
                .appendingPathComponent(repository)
                .appendingPathComponent("model.safetensors")
        )

        let file = ModelFile(
            path: "model.safetensors",
            sizeBytes: Int64(bytes.count),
            digest: nil
        )

        #expect(sandbox.store.locate(file, ofRepository: repository) == nil)
    }

    @Test("The HuggingFace cache's blobs/<sha256> layout is found before anything is read")
    func locatesAHubCacheBlob() throws {
        let sandbox = try Sandbox()
        let repository = "mlx-community/Test-7B-4bit"

        // The real cache names each object by its SHA-256 and symlinks the
        // snapshot entry at it — measured on this machine 2026-10-31.
        let weights = Data("weights kept in a content-addressed blob".utf8)
        let hex = Self.sha256Hex(of: weights)
        let blob = sandbox.extraRoot
            .appendingPathComponent("models--mlx-community--Test-7B-4bit")
            .appendingPathComponent("blobs")
            .appendingPathComponent(hex)
        try Self.write(weights, to: blob)

        let file = ModelFile(
            path: "model.safetensors",
            sizeBytes: Int64(weights.count),
            digest: .sha256(hex)
        )

        #expect(
            sandbox.store.locate(file, ofRepository: repository)?.resolvingSymlinksInPath()
                == blob.resolvingSymlinksInPath()
        )
    }

    @Test("A symlink into a blob reports the blob's size, not the link's")
    func followsASymlinkWhenSizingAFile() throws {
        let sandbox = try Sandbox()

        let weights = Data(repeating: 0xAB, count: 4096)
        let blob = sandbox.extraRoot.appendingPathComponent("blobs/deadbeef")
        try Self.write(weights, to: blob)

        let snapshot = sandbox.extraRoot.appendingPathComponent("snapshots/rev/model.safetensors")
        try FileManager.default.createDirectory(
            at: snapshot.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try FileManager.default.createSymbolicLink(at: snapshot, withDestinationURL: blob)

        // Unfollowed, this reports the length of the link target string — 76
        // bytes for the real 4 GB file measured on 2026-10-31.
        #expect(ModelStore.sizeOfFile(at: snapshot, using: .default) == 4096)
    }

    // MARK: - T-AIL-26

    @Test("A wrong hash deletes the file and throws, naming the file and both hashes")
    func mismatchIsRefused() throws {
        let sandbox = try Sandbox()
        let repository = "mlx-community/Test-7B-4bit"

        let published = Data("what the hub published".utf8)
        let arrived = Data("what actually arrived!".utf8)
        let expectedHex = Self.sha256Hex(of: published)
        let actualHex = Self.sha256Hex(of: arrived)

        let destination = sandbox.store
            .directory(forRepository: repository)
            .appendingPathComponent("model.safetensors")
        try Self.write(arrived, to: destination)

        let file = ModelFile(
            path: "model.safetensors",
            sizeBytes: Int64(published.count),
            digest: .sha256(expectedHex)
        )

        let error = #expect(throws: ModelStoreError.self) {
            try sandbox.store.acceptDownloaded(file, at: destination)
        }

        #expect(
            error
                == .digestMismatch(
                    file: "model.safetensors",
                    algorithm: "sha256",
                    expected: expectedHex,
                    actual: actualHex
                )
        )

        // Both hashes in the sentence, so a reader can tell a truncated
        // download from the wrong file entirely.
        let message = try #require(error?.errorDescription)
        #expect(message.contains("model.safetensors"))
        #expect(message.contains(expectedHex))
        #expect(message.contains(actualHex))

        // Deleted, so the next attempt starts clean instead of resuming onto
        // corrupt bytes.
        #expect(!FileManager.default.fileExists(atPath: destination.path))
    }

    @Test("A missing published checksum throws rather than keeping bytes AIKit cannot vouch for")
    func missingOidIsRefused() throws {
        let sandbox = try Sandbox()
        let repository = "mlx-community/Test-7B-4bit"

        let destination = sandbox.store
            .directory(forRepository: repository)
            .appendingPathComponent("model.safetensors")
        try Self.write(Data("bytes nobody vouched for".utf8), to: destination)

        let file = ModelFile(path: "model.safetensors", sizeBytes: 24, digest: nil)

        let error = #expect(throws: ModelStoreError.self) {
            try sandbox.store.acceptDownloaded(file, at: destination)
        }
        #expect(error == .unverifiableFile("model.safetensors"))
        #expect(try #require(error?.errorDescription).contains("model.safetensors"))
        #expect(!FileManager.default.fileExists(atPath: destination.path))
    }

    @Test("A matching download is accepted and left where it landed")
    func matchingDownloadIsAccepted() throws {
        let sandbox = try Sandbox()
        let repository = "mlx-community/Test-7B-4bit"

        let weights = Data(repeating: 0x42, count: 10 * 1024 * 1024)
        let destination = sandbox.store
            .directory(forRepository: repository)
            .appendingPathComponent("model.safetensors")
        try Self.write(weights, to: destination)

        let file = ModelFile(
            path: "model.safetensors",
            sizeBytes: Int64(weights.count),
            digest: .sha256(Self.sha256Hex(of: weights))
        )

        try sandbox.store.acceptDownloaded(file, at: destination)
        #expect(FileManager.default.fileExists(atPath: destination.path))
    }

    @Test("A file the hub listed but that never arrived is reported as missing, not as a mismatch")
    func absentFileIsReportedMissing() throws {
        let sandbox = try Sandbox()
        let file = ModelFile(path: "model.safetensors", sizeBytes: 10, digest: .sha256("abc"))

        let error = #expect(throws: ModelStoreError.self) {
            try sandbox.store.acceptDownloaded(
                file,
                at: sandbox.root.appendingPathComponent("nothing-here")
            )
        }
        #expect(error == .fileMissing("model.safetensors"))
    }

    // MARK: - Hashing

    @Test("Hashing streams the file and reports bytes as it goes")
    func hashingReportsItsOwnProgress() throws {
        let sandbox = try Sandbox()

        // Larger than one 4 MB chunk, so progress is reported more than twice.
        let bytes = Data(repeating: 0x7F, count: 9 * 1024 * 1024)
        let url = sandbox.root.appendingPathComponent("big.safetensors")
        try Self.write(bytes, to: url)

        let reports = Reports()
        let hex = try sandbox.store.digest(
            ofFileAt: url,
            matching: .sha256(Self.sha256Hex(of: bytes))
        ) { hashed, total in
            reports.append(hashed, total)
        }

        #expect(hex == Self.sha256Hex(of: bytes))

        let observed = reports.all
        #expect(observed.count >= 3, "a 9 MB file read in 4 MB chunks reports at least 3 times")
        #expect(observed.allSatisfy { $0.1 == Int64(bytes.count) }, "total is the file's size")
        #expect(observed.first?.0 == 0)
        #expect(observed.last?.0 == Int64(bytes.count))
        #expect(
            zip(observed, observed.dropFirst()).allSatisfy { $0.0 <= $1.0 },
            "bytes hashed only ever go up"
        )
    }

    @Test("The git blob digest is the one git computes")
    func gitBlobDigestMatchesGitHashObject() throws {
        let sandbox = try Sandbox()

        // `printf 'hello\n' | git hash-object --stdin` is this value, and it is
        // the formula HuggingFace's non-LFS tree `oid` follows: measured
        // 2026-10-31, `git hash-object config.json` returned the published
        // `68b84bf6877ff51d722ec9f076527ab29243d567`.
        let url = sandbox.root.appendingPathComponent("hello.txt")
        try Self.write(Data("hello\n".utf8), to: url)

        let hex = try sandbox.store.digest(
            ofFileAt: url,
            matching: .gitBlobSHA1("ce013625030ba8dba906f756967f9e9ca394464a")
        )
        #expect(hex == "ce013625030ba8dba906f756967f9e9ca394464a")
    }

    @Test("A non-LFS file is verified against the tree's own oid")
    func verifiesANonLFSFile() throws {
        let sandbox = try Sandbox()
        let repository = "mlx-community/Test-7B-4bit"

        let config = Data(#"{"model_type":"mistral"}"#.utf8)
        let url = sandbox.extraRoot
            .appendingPathComponent(repository)
            .appendingPathComponent("config.json")
        try Self.write(config, to: url)

        var hasher = Insecure.SHA1()
        hasher.update(data: Data("blob \(config.count)\u{0}".utf8))
        hasher.update(data: config)
        let oid = hasher.finalize().map { String(format: "%02x", $0) }.joined()

        let file = ModelFile(
            path: "config.json",
            sizeBytes: Int64(config.count),
            digest: .gitBlobSHA1(oid)
        )
        #expect(sandbox.store.locate(file, ofRepository: repository) != nil)
    }

    // MARK: - Receipt

    @Test("A receipt makes a second look-up free, and a changed size invalidates it")
    func receiptCoversWhatWasVerified() throws {
        let sandbox = try Sandbox()
        let repository = "mlx-community/Test-7B-4bit"

        let weights = Data("weights".utf8)
        let destination = sandbox.store
            .directory(forRepository: repository)
            .appendingPathComponent("model.safetensors")
        try Self.write(weights, to: destination)

        let file = ModelFile(
            path: "model.safetensors",
            sizeBytes: Int64(weights.count),
            digest: .sha256(Self.sha256Hex(of: weights))
        )

        #expect(!sandbox.store.holdsVerified([file], ofRepository: repository))
        try sandbox.store.writeReceipt([file], forRepository: repository)
        #expect(sandbox.store.holdsVerified([file], ofRepository: repository))

        // A file that grew is not the file that was verified.
        try Self.write(Data("weights and more".utf8), to: destination)
        #expect(!sandbox.store.holdsVerified([file], ofRepository: repository))
    }

    @Test("An empty file list is never 'already held'")
    func emptyListIsNotHeld() throws {
        let sandbox = try Sandbox()
        #expect(!sandbox.store.holdsVerified([], ofRepository: "any/model"))
    }

    // MARK: - Where a listed path may write

    @Test("A path from a listing may not resolve outside the model's own directory")
    func refusesAPathThatEscapes() throws {
        let sandbox = try Sandbox()
        let repository = "mlx-community/Test-7B-4bit"
        let directory = sandbox.store.directory(forRepository: repository)

        // An ordinary path, including a nested one, resolves inside.
        for path in ["model.safetensors", "subdir/extra.safetensors"] {
            let file = ModelFile(path: path, sizeBytes: 1, digest: nil)
            let destination = try #require(
                sandbox.store.destination(for: file, ofRepository: repository)
            )
            #expect(destination.path.hasPrefix(directory.standardizedFileURL.path))
        }

        // A listing decides WHAT to download, never WHERE to write it.
        for path in [
            "../escaped.safetensors",
            "../../../.ssh/authorized_keys",
            "subdir/../../escaped",
        ] {
            let file = ModelFile(path: path, sizeBytes: 1, digest: nil)
            #expect(
                sandbox.store.destination(for: file, ofRepository: repository) == nil,
                "\(path) resolves outside the model directory and must be refused"
            )
        }
    }

    @Test("A repository id cannot escape the root either")
    func repositoryIdCannotEscapeTheRoot() throws {
        let sandbox = try Sandbox()
        // The `/` becomes `_`, so this names a directory, not a parent walk.
        let directory = sandbox.store.directory(forRepository: "../../etc")
        #expect(directory.lastPathComponent == ".._.._etc")
        #expect(directory.standardizedFileURL.path.hasPrefix(sandbox.root.standardizedFileURL.path))
    }

    // MARK: - One root

    @Test("LocalModelStore and ModelStore agree on the root by construction")
    func oneRootForBothStores() {
        // Two spellings of "the cache root" is how one model came to sit in two
        // directories at once; this is the assertion that there is one.
        #expect(
            LocalModelStore().modelPath(for: ModelIdentifier(provider: "p", modelId: "m"))
                .path
                .hasPrefix(ModelStore.defaultRoot.path)
        )
    }

    @Test("On iOS the only root is the app's own container")
    func noExtraRootsOffMacOS() {
        #if os(macOS)
            #expect(!ModelStore.defaultExtraRoots.isEmpty, "a Mac may reuse LM Studio's models")
        #else
            #expect(
                ModelStore.defaultExtraRoots.isEmpty,
                "an iOS app may read nothing outside its container"
            )
        #endif
    }
}

// MARK: - Progress collector

/// Collects `(bytes, total)` reports from a `@Sendable` progress callback.
///
/// `Synchronization.Mutex` would be the modern answer and needs macOS 15; this
/// package targets macOS 14, so the lock is explicit.
private final class Reports: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: [(Int64, Int64)] = []

    func append(_ bytes: Int64, _ total: Int64) {
        lock.lock()
        defer { lock.unlock() }
        storage.append((bytes, total))
    }

    var all: [(Int64, Int64)] {
        lock.lock()
        defer { lock.unlock() }
        return storage
    }
}
