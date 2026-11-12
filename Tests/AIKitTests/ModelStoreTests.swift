import CoreKit
import CoreKitTestSupport
import CryptoKit
import Foundation
import Testing

@testable import AIKit

// MARK: - Shared fixture

/// A run-scoped root and the PRODUCTION ``PathLayer`` over it.
///
/// W23 of spec ai-layer-aikit-core-2026-10-31. The layer is built by
/// ``AIKitPaths/layer(root:nodeRoots:)`` — the same call a host makes — so
/// these suites exercise the production composition and differ from it by the
/// root alone (CoreKit BR-RP-03). The root is CoreKit's own `TestScratch`, so
/// nothing is written to `NSTemporaryDirectory()` or to the operator's home,
/// and two concurrent kagami runs cannot share a directory.
///
/// A class, so `deinit` sweeps the tree: a `defer` in every test is the line
/// that gets forgotten.
final class TestPathLayer {

    let scratch: TestScratch
    /// The injected root. Every AIKit location resolves below it.
    let root: URL
    let layer: PathLayer

    init(scope: String, nodeRoots: [AIKitNode: URL] = [:]) throws {
        scratch = TestScratch.forPackage()
        root = try scratch.makeScratchDir(scope: scope)
        layer = try AIKitPaths.layer(root: root, nodeRoots: nodeRoots)
    }

    deinit {
        try? scratch.removeScratchDir(root)
    }
}

/// W13 of spec ai-layer-aikit-core-2026-10-31: one store answers "do we have
/// this model", reuses a file it can vouch for without spending the bytes
/// twice, and refuses one it cannot. W23 took its locations away from it: every
/// path below is resolved by the injected CoreKit path layer from a
/// `VirtualPathOwning` declaration, and the store names no directory at all.
///
/// Every digest in here is computed from the bytes the test wrote, never
/// pasted: a hard-coded hash proves the test's own arithmetic, not the store's.
/// The one exception is `gitBlobDigestMatchesGitHashObject`, which pins the git
/// blob formula against a value `git hash-object` produced.
@Suite("ModelStore")
struct ModelStoreTests {

    // MARK: - Fixtures

    /// A sandbox with a resolved store root and one external library, both
    /// empty and both on the same volume (so a reuse can be a hard link).
    private final class Sandbox {
        let paths: TestPathLayer
        let extraRoot: URL
        let store: ModelStore

        /// Where the `models` node is served from — the store's own root.
        var root: URL { paths.layer.base(of: AIKitNode.models) }

        init(scope: String = "model-store") throws {
            let paths = try TestPathLayer(scope: scope)
            let extraRoot = paths.root.appendingPathComponent("lmstudio")
            let modelsRoot = paths.layer.base(of: AIKitNode.models)
            try FileManager.default.createDirectory(
                at: modelsRoot, withIntermediateDirectories: true)
            try FileManager.default.createDirectory(
                at: extraRoot, withIntermediateDirectories: true)
            self.paths = paths
            self.extraRoot = extraRoot
            self.store = ModelStore(paths: paths.layer, externalLibraries: [extraRoot])
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

    private static func file(_ path: String, _ data: Data) -> ModelFile {
        ModelFile(path: path, sizeBytes: Int64(data.count), digest: .sha256(sha256Hex(of: data)))
    }

    // MARK: - T-AIL-48

    @Test("Every location this store answers comes from the path layer's root")
    func locationsComeFromThePathLayer() throws {
        // Two layers, differing by their root and by NOTHING else.
        let first = try TestPathLayer(scope: "paths-first")
        let second = try TestPathLayer(scope: "paths-second")
        #expect(first.root != second.root)

        let repository = "mlx-community/Test-7B-4bit"
        let weights = Data("weights".utf8)
        let descriptor = ModelFile(
            path: "model.safetensors",
            sizeBytes: Int64(weights.count),
            digest: .sha256(Self.sha256Hex(of: weights))
        )
        let modelID = ModelIdentifier(provider: "huggingface", modelId: "nvidia/nemotron-3")

        /// Every location the two stores answer, in the same order.
        func locations(_ paths: PathLayer) throws -> [URL] {
            let store = ModelStore(paths: paths)
            let local = try LocalModelStore(paths: paths)
            return [
                store.modelsDirectory,
                try store.directory(forRepository: repository),
                try store.destination(for: descriptor, ofRepository: repository).url,
                local.manifestURL,
                try local.modelPath(for: modelID),
                try local.fileURL(for: modelID, named: "model.gguf").url,
            ]
        }

        let here = try locations(first.layer)
        let there = try locations(second.layer)

        #expect(here.count == there.count)
        for (mine, theirs) in zip(here, there) {
            // Moving the root moved this location, with no code change…
            #expect(mine != theirs, "\(mine.lastPathComponent) did not move with the root")
            // …and each one lands under the root it was resolved against.
            #expect(PathContainment.isContained(mine, in: first.root))
            #expect(PathContainment.isContained(theirs, in: second.root))
        }

        // The relative shape is identical: only the root differs.
        let relative = zip(here, there).map { mine, theirs in
            (
                mine.path.dropFirst(first.root.standardizedFileURL.path.count),
                theirs.path.dropFirst(second.root.standardizedFileURL.path.count)
            )
        }
        #expect(relative.allSatisfy { $0.0 == $0.1 })
    }

    @Test("Serving the models node from elsewhere moves every location too")
    func aNodeRootOverrideMovesEverything() throws {
        // What a settings value does: the root stays, one node is served from
        // another disk. No owner's declaration changes.
        let plain = try TestPathLayer(scope: "node-root-plain")
        let elsewhere = plain.root.appendingPathComponent("external-disk")
        let moved = try TestPathLayer(
            scope: "node-root-moved", nodeRoots: [.models: elsewhere])

        let store = ModelStore(paths: moved.layer)
        #expect(store.modelsDirectory.standardizedFileURL == elsewhere.standardizedFileURL)
        #expect(
            try store.directory(forRepository: "org/model").path
                .hasPrefix(elsewhere.standardizedFileURL.path)
        )
        #expect(
            ModelStore(paths: plain.layer).modelsDirectory.standardizedFileURL
                != elsewhere.standardizedFileURL
        )
    }

    @Test("The stores name no path of their own: no separator in any string literal")
    func theStoresSpellNoSeparator() throws {
        let packageRoot = TestPackagePaths.packageRoot(fromTestFile: #filePath)
        let sources = [
            "Sources/AIKit/Models/ModelStore.swift",
            "Sources/AIKit/Registry/LocalModelStore.swift",
            "Sources/AIKit/Registry/ModelDownloader.swift",
        ]

        for relative in sources {
            let url = relative.split(separator: PathSeparator.posix.character)
                .reduce(packageRoot) { $0.appendingPathComponent(String($1)) }
            let source = try String(contentsOf: url, encoding: .utf8)
            let offenders = SwiftSourceLiterals.all(in: source)
                .filter { literal in
                    PathSeparator.allCases.contains { literal.contains($0.rawValue) }
                }
            #expect(
                offenders.isEmpty,
                """
                \(relative) spells a path separator inside a string literal: \
                \(offenders). A location is components resolved by the path \
                layer, never a string a store typed.
                """
            )
        }
    }

    @Test("An owner the host never registered is refused by name, never guessed")
    func anUnregisteredOwnerIsRefused() throws {
        let scratch = TestScratch.forPackage()
        let root = try scratch.makeScratchDir(scope: "unregistered")
        defer { try? scratch.removeScratchDir(root) }

        // A layer with an EMPTY registry: the production layer type, composed
        // without AIKit's owners.
        let bare = try PathLayer(root: root)
        let store = ModelStore(paths: bare)

        let error = #expect(throws: ModelStoreError.self) {
            try store.directory(forRepository: "org/model")
        }
        guard case .unresolvableLocation(let resource, let reason) = try #require(error) else {
            Issue.record("expected an unresolvable location, got \(String(describing: error))")
            return
        }
        #expect(resource == "org/model")
        #expect(reason == .unregisteredOwner(String(reflecting: ModelRepositoryDirectory.self)))
        #expect(try #require(error?.errorDescription).contains("ModelRepositoryDirectory"))
    }

    @Test("AIKit's node tree is sound, and every owner is registered exactly once")
    func theNodeTreeIsSound() throws {
        let paths = try TestPathLayer(scope: "node-tree")

        // CoreKit's own ratchet: no cycle, no two cases naming one node, every
        // base under the root.
        #expect(VirtualNodeNamingRatchet.violations(of: AIKitNode.self, in: paths.layer).isEmpty)
        #expect(VirtualNodeNamingRatchet.violations(composing: AIKitPaths.nodes, in: paths.layer).isEmpty)

        // Registration is what makes resolution legal, so the list and the
        // registry must agree — a placement clash would have thrown above.
        #expect(paths.layer.registry.entries.count == AIKitPaths.owners.count)
        for owner in AIKitPaths.owners {
            #expect(paths.layer.registry.contains(owner), "\(owner) is not in the tree")
        }
        // All of AIKit's owners live on the one node it declares.
        #expect(
            paths.layer.registry.entries(on: AIKitNode.models).count == AIKitPaths.owners.count
        )
    }

    @Test("An external library is declared as components, so it renders on every platform")
    func externalLibrariesAreComponents() {
        let home = URL(fileURLWithPath: "/somewhere/home", isDirectory: true)

        for library in ExternalModelLibrary.knownForHost {
            #expect(!library.path.isEmpty, "\(library.application) declares no location")
            // Components, never a string with separators in it.
            #expect(
                library.path.segments.allSatisfy { segment in
                    PathSeparator.allCases.allSatisfy { !segment.rawValue.contains($0.rawValue) }
                }
            )
            #expect(library.url(under: home).path.hasPrefix(home.path))
        }

        #if os(macOS)
            let applications = ExternalModelLibrary.knownForHost.map(\.application)
            #expect(applications.contains("LM Studio"))
            #expect(ExternalModelLibrary.knownURLs(under: home).count == applications.count)
        #elseif os(Linux) || os(Windows)
            #expect(ExternalModelLibrary.knownForHost.count == 2)
        #else
            // BR-AIL-14: an iOS app may read nothing outside its container, and
            // the enforcement is that there is nothing in the list to read.
            #expect(ExternalModelLibrary.knownForHost.isEmpty)
        #endif
    }

    @Test("With no external library injected, nothing outside the store is reused")
    func noExternalLibraryMeansNoReuse() throws {
        let sandbox = try Sandbox()
        let repository = "mlx-community/Test-7B-4bit"

        let weights = Data("weights already on disk somewhere else".utf8)
        try Self.write(
            weights,
            to: sandbox.extraRoot
                .appendingPathComponent(repository)
                .appendingPathComponent("model.safetensors")
        )
        let file = Self.file("model.safetensors", weights)

        // The sandbox store was given the library, so it finds it…
        #expect(try sandbox.store.locate(file, ofRepository: repository) != nil)
        // …and a store given none does not, however reachable the bytes are.
        let closed = ModelStore(paths: sandbox.paths.layer)
        #expect(try closed.locate(file, ofRepository: repository) == nil)
    }

    // MARK: - T-AIL-25

    @Test("A verified file in an external library is linked, not downloaded, and takes no second space")
    func reusesAVerifiedFile() throws {
        let sandbox = try Sandbox()
        let repository = "mlx-community/Test-7B-4bit"

        // The operator already has these weights, under LM Studio's root.
        let weights = Data("the weights of a model that is already on disk".utf8)
        let existing = sandbox.extraRoot
            .appendingPathComponent(repository)
            .appendingPathComponent("model.safetensors")
        try Self.write(weights, to: existing)

        let file = Self.file("model.safetensors", weights)

        // Located, because the checksum matches what the hub published.
        let located = try sandbox.store.locate(file, ofRepository: repository)
        #expect(located?.resolvingSymlinksInPath() == existing.resolvingSymlinksInPath())

        // Adopted into the store's own root, at the place the layer resolved.
        let destination = try sandbox.store.destination(for: file, ofRepository: repository)
        let linked = try sandbox.store.adopt(#require(located), as: destination)

        #expect(linked, "a same-volume reuse must be a hard link, never a copy")
        #expect(try Self.inode(of: destination.url) == Self.inode(of: existing))
        #expect(try Self.linkCount(of: destination.url) == 2)
        #expect(try Data(contentsOf: destination.url) == weights)

        // The whole point: one copy of the bytes, two names.
        let bytesUnderBothRoots = try [existing, destination.url]
            .map { try Self.inode(of: $0) }
        #expect(Set(bytesUnderBothRoots).count == 1, "two inodes would mean the bytes were copied")
    }

    @Test("A file found through a relative symlink is adopted as the real file")
    func adoptsTheRealFileBehindARelativeSymlink() throws {
        let sandbox = try Sandbox()
        let repository = "mlx-community/Test-7B-4bit"

        // The HuggingFace cache: a snapshot entry is `../../blobs/<id>`, which
        // means nothing once the link itself moves to another directory.
        let bytes = Data("a small file the hub cache keeps behind a link".utf8)
        let cache = sandbox.extraRoot.appendingPathComponent("hub-cache")
        let blob = cache.appendingPathComponent("blobs").appendingPathComponent("abc")
        try Self.write(bytes, to: blob)
        let snapshot = cache.appendingPathComponent("snapshots").appendingPathComponent("rev")
        try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: true)
        let link = snapshot.appendingPathComponent("config.json")
        try FileManager.default.createSymbolicLink(
            atPath: link.path,
            withDestinationPath: "../../blobs/abc"
        )

        let destination = try sandbox.store.destination(
            for: Self.file("config.json", bytes),
            ofRepository: repository
        )
        try sandbox.store.adopt(link, as: destination)

        #expect(try Data(contentsOf: destination.url) == bytes)
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

        #expect(
            try sandbox.store.locate(
                Self.file("model.safetensors", published), ofRepository: repository) == nil
        )
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

        #expect(try sandbox.store.locate(file, ofRepository: repository) == nil)
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

        #expect(
            try sandbox.store.locate(
                Self.file("model.safetensors", weights), ofRepository: repository
            )?.resolvingSymlinksInPath() == blob.resolvingSymlinksInPath()
        )
    }

    @Test("A snapshot directory under the hub layout is probed for the file")
    func locatesAFileInAHubSnapshot() throws {
        let sandbox = try Sandbox()
        let repository = "mlx-community/Test-7B-4bit"

        // No LFS object, so the blobs/<sha256> probe cannot answer: the
        // snapshot walk is the one that finds it.
        let config = Data(#"{"model_type":"mistral"}"#.utf8)
        var hasher = Insecure.SHA1()
        hasher.update(data: Data("blob \(config.count)\u{0}".utf8))
        hasher.update(data: config)
        let oid = hasher.finalize().map { String(format: "%02x", $0) }.joined()

        let snapshot = sandbox.extraRoot
            .appendingPathComponent("models--mlx-community--Test-7B-4bit")
            .appendingPathComponent("snapshots")
            .appendingPathComponent("abc123")
        try Self.write(config, to: snapshot.appendingPathComponent("config.json"))

        let file = ModelFile(
            path: "config.json",
            sizeBytes: Int64(config.count),
            digest: .gitBlobSHA1(oid)
        )
        #expect(
            try sandbox.store.locate(file, ofRepository: repository)?.resolvingSymlinksInPath()
                == snapshot.appendingPathComponent("config.json").resolvingSymlinksInPath()
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

        let file = ModelFile(
            path: "model.safetensors",
            sizeBytes: Int64(published.count),
            digest: .sha256(expectedHex)
        )
        let destination = try sandbox.store.destination(for: file, ofRepository: repository)
        try Self.write(arrived, to: destination.url)

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
        #expect(!FileManager.default.fileExists(atPath: destination.url.path))
    }

    @Test("A missing published checksum throws rather than keeping bytes AIKit cannot vouch for")
    func missingOidIsRefused() throws {
        let sandbox = try Sandbox()
        let repository = "mlx-community/Test-7B-4bit"

        let file = ModelFile(path: "model.safetensors", sizeBytes: 24, digest: nil)
        let destination = try sandbox.store.destination(for: file, ofRepository: repository)
        try Self.write(Data("bytes nobody vouched for".utf8), to: destination.url)

        let error = #expect(throws: ModelStoreError.self) {
            try sandbox.store.acceptDownloaded(file, at: destination)
        }
        #expect(error == .unverifiableFile("model.safetensors"))
        #expect(try #require(error?.errorDescription).contains("model.safetensors"))
        #expect(!FileManager.default.fileExists(atPath: destination.url.path))
    }

    @Test("A matching download is accepted and left where it landed")
    func matchingDownloadIsAccepted() throws {
        let sandbox = try Sandbox()
        let repository = "mlx-community/Test-7B-4bit"

        let weights = Data(repeating: 0x42, count: 10 * 1024 * 1024)
        let file = Self.file("model.safetensors", weights)
        let destination = try sandbox.store.destination(for: file, ofRepository: repository)
        try Self.write(weights, to: destination.url)

        try sandbox.store.acceptDownloaded(file, at: destination)
        #expect(FileManager.default.fileExists(atPath: destination.url.path))
    }

    @Test("A file the hub listed but that never arrived is reported as missing, not as a mismatch")
    func absentFileIsReportedMissing() throws {
        let sandbox = try Sandbox()
        let file = ModelFile(path: "model.safetensors", sizeBytes: 10, digest: .sha256("abc"))

        let error = #expect(throws: ModelStoreError.self) {
            try sandbox.store.acceptDownloaded(
                file,
                at: try sandbox.store.destination(for: file, ofRepository: "org/never-downloaded")
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
        #expect(try sandbox.store.locate(file, ofRepository: repository) != nil)
    }

    // MARK: - Receipt

    @Test("A receipt makes a second look-up free, and a changed size invalidates it")
    func receiptCoversWhatWasVerified() throws {
        let sandbox = try Sandbox()
        let repository = "mlx-community/Test-7B-4bit"

        let weights = Data("weights".utf8)
        let file = Self.file("model.safetensors", weights)
        let destination = try sandbox.store.destination(for: file, ofRepository: repository)
        try Self.write(weights, to: destination.url)

        #expect(try !sandbox.store.holdsVerified([file], ofRepository: repository))
        try sandbox.store.writeReceipt([file], forRepository: repository)
        #expect(try sandbox.store.holdsVerified([file], ofRepository: repository))

        // A file that grew is not the file that was verified.
        try Self.write(Data("weights and more".utf8), to: destination.url)
        #expect(try !sandbox.store.holdsVerified([file], ofRepository: repository))
    }

    @Test("The receipt is stored where its owner declares, and read back by a second store")
    func theReceiptLivesAtItsOwnersLocation() throws {
        let sandbox = try Sandbox()
        let repository = "mlx-community/Test-7B-4bit"

        let weights = Data("weights".utf8)
        let file = Self.file("model.safetensors", weights)
        try Self.write(
            weights, to: try sandbox.store.destination(for: file, ofRepository: repository).url)
        try sandbox.store.writeReceipt([file], forRepository: repository)

        // Where the owner says, not where this type remembered writing it.
        let expected = try sandbox.paths.layer.url(
            for: ModelVerificationReceipt(repositoryID: repository))
        #expect(FileManager.default.fileExists(atPath: expected.path))
        #expect(
            expected.lastPathComponent
                == ModelVerificationReceipt.placement.name.rawValue
        )

        // A fresh store over the same layer reads it: the file is the SSoT, not
        // any instance's memory.
        let second = ModelStore(paths: sandbox.paths.layer)
        #expect(try second.holdsVerified([file], ofRepository: repository))
        #expect(try second.receipt(forRepository: repository).verified[file.path] == file.digest?.hex)
    }

    @Test("A receipt that cannot be decoded reads as empty, never as verified")
    func aCorruptReceiptIsNotVerification() throws {
        let sandbox = try Sandbox()
        let repository = "mlx-community/Test-7B-4bit"

        let weights = Data("weights".utf8)
        let file = Self.file("model.safetensors", weights)
        try Self.write(
            weights, to: try sandbox.store.destination(for: file, ofRepository: repository).url)

        let receiptURL = try sandbox.paths.layer.url(
            for: ModelVerificationReceipt(repositoryID: repository))
        try Self.write(Data("not valid json {{{{".utf8), to: receiptURL)

        // Losing the receipt costs a re-hash and nothing else, so it is not an
        // error — but it must never read as "already verified".
        #expect(try sandbox.store.receipt(forRepository: repository) == ModelStore.Receipt())
        #expect(try !sandbox.store.holdsVerified([file], ofRepository: repository))
    }

    @Test("An empty file list is never 'already held'")
    func emptyListIsNotHeld() throws {
        let sandbox = try Sandbox()
        #expect(try !sandbox.store.holdsVerified([], ofRepository: "any/model"))
    }

    // MARK: - Where a listed path may write

    @Test("A path from a listing may not resolve outside the model's own directory")
    func refusesAPathThatEscapes() throws {
        let sandbox = try Sandbox()
        let repository = "mlx-community/Test-7B-4bit"
        let directory = try sandbox.store.directory(forRepository: repository)

        // An ordinary path, including a nested one, resolves inside.
        for path in ["model.safetensors", "subdir/extra.safetensors"] {
            let file = ModelFile(path: path, sizeBytes: 1, digest: nil)
            let destination = try sandbox.store.destination(for: file, ofRepository: repository)
            #expect(destination.url.path.hasPrefix(directory.standardizedFileURL.path))
        }

        // A listing decides WHAT to download, never WHERE to write it. The
        // refusal names the path the hub sent, not a sanitised version of it.
        for path in [
            "../escaped.safetensors",
            "../../../.ssh/authorized_keys",
            "subdir/../../escaped",
            "/etc/passwd",
            "",
        ] {
            let file = ModelFile(path: path, sizeBytes: 1, digest: nil)
            let error = #expect(throws: ModelStoreError.self) {
                try sandbox.store.destination(for: file, ofRepository: repository)
            }
            #expect(
                error == .pathEscapesStore(path),
                "\(path) is not a path inside the model directory and must be refused"
            )
        }
    }

    @Test("A repository id cannot escape the root either")
    func repositoryIdCannotEscapeTheRoot() throws {
        let sandbox = try Sandbox()
        // The `/` becomes `_`, so this names a directory, not a parent walk.
        let directory = try sandbox.store.directory(forRepository: "../../etc")
        #expect(directory.lastPathComponent == ".._.._etc")
        #expect(directory.standardizedFileURL.path.hasPrefix(sandbox.root.standardizedFileURL.path))
    }

    // MARK: - One root

    @Test("LocalModelStore and ModelStore agree on the root by construction")
    func oneRootForBothStores() throws {
        // Two spellings of "the cache root" is how one model came to sit in two
        // directories at once. There is now ONE: the `models` node of the
        // layer, and neither store can name anything else.
        let sandbox = try Sandbox()
        let local = try LocalModelStore(paths: sandbox.paths.layer)
        let resolved = try local.modelPath(for: ModelIdentifier(provider: "p", modelId: "m"))

        #expect(resolved.path.hasPrefix(sandbox.store.modelsDirectory.standardizedFileURL.path))
        // Compared as PATHS, not URLs: `deletingLastPathComponent()` leaves a
        // trailing separator that `standardizedFileURL` keeps, so two URLs for
        // one directory come out unequal.
        #expect(
            local.manifestURL.deletingLastPathComponent().standardizedFileURL.path
                == sandbox.store.modelsDirectory.standardizedFileURL.path)
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

// MARK: - A source scanner for the literal ratchet

/// Every STRING LITERAL in a Swift source, with comments removed.
///
/// A ratchet on "the store spells no separator" has to look at literals and
/// nothing else: `/` is also the division operator (`size /= 1024` in
/// `ModelDownloader`), and a doc comment legitimately writes
/// `~/.lmstudio/models` when explaining what it no longer hardcodes. Grepping
/// the file would flag both and prove nothing.
///
/// Handles `//` and nested `/* */` comments, `"…"` with escapes, `"""…"""`
/// blocks and `#"…"#` raw literals.
///
/// Escape sequences are DECODED, not copied: an escaped backslash contributes
/// one backslash — which IS the Windows separator and has to be caught — and
/// every other escape contributes nothing. Copying them verbatim put a
/// backslash inside every INTERPOLATED literal, because `\(` and the
/// line-continuation `\` both start with one, and the check then reported
/// `"\(file) failed its \(algorithm) check"` as a path. Measured as three false
/// offenders on this suite's first run.
enum SwiftSourceLiterals {

    static func all(in source: String) -> [String] {
        var scanner = Scanner(characters: Array(source))
        return scanner.run()
    }

    /// One pass over the characters, one small step per state.
    ///
    /// Split this way because the single `while`-with-nested-`switch` form was
    /// one function of cyclomatic complexity 21, which SwiftLint fails at 10 —
    /// correctly: the state machine is readable as six short steps and is not
    /// readable as one.
    private struct Scanner {
        let characters: [Character]
        var index = 0
        var state = State.code
        var blockDepth = 0
        var current = ""
        var literals: [String] = []

        mutating func run() -> [String] {
            while index < characters.count {
                switch state {
                case .code: stepCode()
                case .lineComment: stepLineComment()
                case .blockComment: stepBlockComment()
                case .string: stepQuoted(closing: 1)
                case .multilineString: stepQuoted(closing: 3)
                case .rawString: stepRawString()
                }
            }
            return literals
        }

        // MARK: - Steps

        private mutating func stepCode() {
            if matches("//") {
                enter(.lineComment, consuming: 2)
            } else if matches("/*") {
                blockDepth = 1
                enter(.blockComment, consuming: 2)
            } else if matches("#\"") {
                open(.rawString, consuming: 2)
            } else if matches("\"\"\"") {
                open(.multilineString, consuming: 3)
            } else if matches("\"") {
                open(.string, consuming: 1)
            } else {
                index += 1
            }
        }

        private mutating func stepLineComment() {
            if characters[index] == "\n" { state = .code }
            index += 1
        }

        private mutating func stepBlockComment() {
            if matches("/*") {
                blockDepth += 1
                index += 2
            } else if matches("*/") {
                blockDepth -= 1
                index += 2
                if blockDepth == 0 { state = .code }
            } else {
                index += 1
            }
        }

        /// A `"…"` or `"""…"""` literal: escapes are decoded, the delimiter ends it.
        private mutating func stepQuoted(closing quoteCount: Int) {
            let delimiter = String(repeating: "\"", count: quoteCount)
            if matches(delimiter) {
                close(consuming: quoteCount)
            } else if characters[index] == "\\" {
                current.append(contentsOf: decodedEscape())
                index += 2
            } else {
                current.append(characters[index])
                index += 1
            }
        }

        /// A `#"…"#` literal: no escapes at all, by definition.
        private mutating func stepRawString() {
            if matches("\"#") {
                close(consuming: 2)
            } else {
                current.append(characters[index])
                index += 1
            }
        }

        // MARK: - Primitives

        private func matches(_ token: String) -> Bool {
            let expected = Array(token)
            guard index + expected.count <= characters.count else { return false }
            return Array(characters[index..<(index + expected.count)]) == expected
        }

        private mutating func enter(_ next: State, consuming count: Int) {
            state = next
            index += count
        }

        private mutating func open(_ next: State, consuming count: Int) {
            current = ""
            enter(next, consuming: count)
        }

        private mutating func close(consuming count: Int) {
            literals.append(current)
            enter(.code, consuming: count)
        }

        /// What the escape at ``index`` contributes to the literal's VALUE: one
        /// backslash for `\\`, nothing for any other escape.
        private func decodedEscape() -> [Character] {
            guard index + 1 < characters.count else { return [] }
            return characters[index + 1] == "\\" ? ["\\"] : []
        }
    }

    private enum State {
        case code, lineComment, blockComment, string, multilineString, rawString
    }
}
