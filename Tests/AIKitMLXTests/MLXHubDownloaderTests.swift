import AIKit
import CryptoKit
import Foundation
import MLXLMCommon
import Testing

@testable import AIKitMLX

/// W13 of spec ai-layer-aikit-core-2026-10-31: one cache root, every file
/// verified against the checksum the hub published, and progress measured in
/// bytes.
///
/// No test reaches a server (BR-AIL-12): the index and the transfer are the
/// injected closures `MLXHubDownloader` takes, and the stub counts what it was
/// asked to do.
@Suite("MLXHubDownloader")
struct MLXHubDownloaderTests {

    // MARK: - Fixtures

    /// A hub that exists only in memory, and remembers what it was asked.
    private final class StubHub: @unchecked Sendable {
        private let lock = NSLock()
        private var contents: [String: Data] = [:]
        private var indexCallCount = 0
        private var transferredPaths: [String] = []
        /// Thrown by `index` when set, to exercise the offline fall-back.
        var indexError: (any Error)?
        /// Bytes per `onBytes` report inside one transfer.
        var chunkBytes = 4 * 1024 * 1024

        init(contents: [String: Data]) {
            self.contents = contents
        }

        var indexCalls: Int {
            lock.withLock { indexCallCount }
        }

        var transferred: [String] {
            lock.withLock { transferredPaths }
        }

        /// What the hub publishes, with the digest each file really hashes to.
        var files: [ModelFile] {
            lock.withLock {
                contents
                    .map { path, data in
                        ModelFile(
                            path: path,
                            sizeBytes: Int64(data.count),
                            digest: .sha256(MLXHubDownloaderTests.sha256Hex(of: data))
                        )
                    }
                    .sorted { $0.path < $1.path }
            }
        }

        var index: MLXHubDownloader.FileIndex {
            // `NSLock.lock()` is unavailable from an async context; the scoped
            // `withLock` is the async-safe form.
            { [self] _, _ in
                let error = lock.withLock {
                    indexCallCount += 1
                    return indexError
                }
                if let error { throw error }
                return files
            }
        }

        var transfer: MLXHubDownloader.FileTransfer {
            { [self] _, _, path, destination, onBytes in
                let (stored, chunk) = lock.withLock {
                    transferredPaths.append(path)
                    return (contents[path], chunkBytes)
                }

                let data = try #require(stored, "the stub was asked for a file it does not have")
                try FileManager.default.createDirectory(
                    at: destination.deletingLastPathComponent(),
                    withIntermediateDirectories: true
                )

                // Report as a real transfer would: a running byte count, not
                // one call at the end.
                let total = Int64(data.count)
                var written: Int64 = 0
                while written < total {
                    written = min(written + Int64(chunk), total)
                    onBytes(written, total)
                }
                try data.write(to: destination)
            }
        }
    }

    /// A sandbox with a store root and an extra root, both empty.
    private final class Sandbox {
        let base: URL
        let root: URL
        let extraRoot: URL
        let store: ModelStore

        init() throws {
            base = FileManager.default.temporaryDirectory
                .appendingPathComponent("aikit-hubdownloader-\(UUID().uuidString)")
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

    static func sha256Hex(of data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private static let repository = "mlx-community/Test-7B-4bit"

    /// One small config and one file big enough to be reported in chunks.
    private static func sampleContents(weightBytes: Int = 12 * 1024 * 1024) -> [String: Data] {
        [
            "config.json": Data(#"{"model_type":"mistral"}"#.utf8),
            "model.safetensors": Data(repeating: 0x5A, count: weightBytes),
        ]
    }

    // MARK: - T-AIL-27

    @Test("A load writes under one root, and a second load downloads nothing")
    func oneCacheRoot() async throws {
        let sandbox = try Sandbox()
        let hub = StubHub(contents: Self.sampleContents(weightBytes: 1024))
        let downloader = MLXHubDownloader(
            store: sandbox.store,
            index: hub.index,
            transfer: hub.transfer
        )

        let directory = try await downloader.download(
            id: Self.repository,
            revision: nil,
            matching: [],
            useLatest: false,
            progressHandler: { _ in }
        )

        // One root: the directory is under the store's root and nowhere else.
        #expect(directory.path.hasPrefix(sandbox.root.path))
        #expect(directory == sandbox.store.directory(forRepository: Self.repository))
        for path in hub.files.map(\.path) {
            #expect(
                FileManager.default.fileExists(
                    atPath: directory.appendingPathComponent(path).path
                )
            )
        }
        #expect(hub.transferred.sorted() == ["config.json", "model.safetensors"])

        // Nothing landed in the extra root, which this store may only read.
        let extraContents = try FileManager.default.contentsOfDirectory(
            at: sandbox.extraRoot,
            includingPropertiesForKeys: nil
        )
        #expect(extraContents.isEmpty)

        let afterFirst = hub.transferred.count

        let second = try await downloader.download(
            id: Self.repository,
            revision: nil,
            matching: [],
            useLatest: false,
            progressHandler: { _ in }
        )

        #expect(second == directory)
        #expect(
            hub.transferred.count == afterFirst,
            "the second load must move no bytes: every file is already held and verified"
        )
    }

    @Test("A file already under an extra root is linked, never transferred")
    func reusesAnExtraRootInsteadOfDownloading() async throws {
        let sandbox = try Sandbox()
        let contents = Self.sampleContents(weightBytes: 2048)
        let hub = StubHub(contents: contents)

        // The operator already has the weights, under LM Studio's root.
        let weights = try #require(contents["model.safetensors"])
        let existing = sandbox.extraRoot
            .appendingPathComponent(Self.repository)
            .appendingPathComponent("model.safetensors")
        try FileManager.default.createDirectory(
            at: existing.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try weights.write(to: existing)

        let downloader = MLXHubDownloader(
            store: sandbox.store,
            index: hub.index,
            transfer: hub.transfer
        )
        let directory = try await downloader.download(
            id: Self.repository,
            revision: nil,
            matching: [],
            useLatest: false,
            progressHandler: { _ in }
        )

        #expect(
            hub.transferred == ["config.json"],
            "the weights were on disk already; only the config had to be fetched"
        )

        // Linked, not copied: one copy of the bytes under two names.
        let adopted = directory.appendingPathComponent("model.safetensors")
        let adoptedAttributes = try FileManager.default.attributesOfItem(atPath: adopted.path)
        let existingAttributes = try FileManager.default.attributesOfItem(atPath: existing.path)
        #expect(
            adoptedAttributes[.systemFileNumber] as? UInt64
                == existingAttributes[.systemFileNumber] as? UInt64
        )
        #expect(adoptedAttributes[.referenceCount] as? Int == 2)
    }

    @Test("Only the files the patterns select are fetched")
    func honoursDownloadPatterns() async throws {
        let sandbox = try Sandbox()
        let hub = StubHub(contents: Self.sampleContents(weightBytes: 512))
        let downloader = MLXHubDownloader(
            store: sandbox.store,
            index: hub.index,
            transfer: hub.transfer
        )

        _ = try await downloader.download(
            id: Self.repository,
            revision: nil,
            matching: ["*.safetensors"],
            useLatest: false,
            progressHandler: { _ in }
        )

        #expect(hub.transferred == ["model.safetensors"])
    }

    @Test("Glob matching: an empty list selects everything, and * crosses a slash")
    func globMatching() {
        #expect(MLXHubDownloader.matches("model.safetensors", []))
        #expect(MLXHubDownloader.matches("model.safetensors", ["*.safetensors"]))
        #expect(!MLXHubDownloader.matches("config.json", ["*.safetensors"]))
        // `*.json` is meant to reach a nested file too, which is why `fnmatch`
        // runs with no FNM_PATHNAME.
        #expect(MLXHubDownloader.matches("subdir/config.json", ["*.json"]))
        #expect(MLXHubDownloader.matches("config.json", ["*.safetensors", "*.json"]))
    }

    @Test("A model already on disk still loads when the hub cannot be reached")
    func fallsBackToTheReceiptWhenOffline() async throws {
        let sandbox = try Sandbox()
        let hub = StubHub(contents: Self.sampleContents(weightBytes: 1024))
        let downloader = MLXHubDownloader(
            store: sandbox.store,
            index: hub.index,
            transfer: hub.transfer
        )

        let directory = try await downloader.download(
            id: Self.repository,
            revision: nil,
            matching: [],
            useLatest: false,
            progressHandler: { _ in }
        )

        struct NoNetwork: Error {}
        hub.indexError = NoNetwork()

        let offline = try await downloader.download(
            id: Self.repository,
            revision: nil,
            matching: [],
            useLatest: false,
            progressHandler: { _ in }
        )
        #expect(offline == directory)
    }

    @Test("With nothing on disk, an unreachable hub is an error and not a silent empty directory")
    func offlineWithoutAReceiptThrows() async throws {
        let sandbox = try Sandbox()
        let hub = StubHub(contents: Self.sampleContents(weightBytes: 1024))
        struct NoNetwork: Error, Equatable {}
        hub.indexError = NoNetwork()

        let downloader = MLXHubDownloader(
            store: sandbox.store,
            index: hub.index,
            transfer: hub.transfer
        )

        await #expect(throws: NoNetwork.self) {
            _ = try await downloader.download(
                id: Self.repository,
                revision: nil,
                matching: [],
                useLatest: false,
                progressHandler: { _ in }
            )
        }
    }

    @Test("A transfer whose bytes do not match the published digest is deleted and named")
    func refusesACorruptTransfer() async throws {
        let sandbox = try Sandbox()
        let published = Data(repeating: 0x11, count: 4096)
        let hub = StubHub(contents: ["model.safetensors": published])

        // The hub lists the digest of `published`; the transfer delivers
        // something else, which is what a truncated or substituted download
        // looks like from here.
        let corrupt = Data(repeating: 0x22, count: 4096)
        let downloader = MLXHubDownloader(
            store: sandbox.store,
            index: hub.index,
            transfer: { _, _, _, destination, onBytes in
                try FileManager.default.createDirectory(
                    at: destination.deletingLastPathComponent(),
                    withIntermediateDirectories: true
                )
                try corrupt.write(to: destination)
                onBytes(Int64(corrupt.count), Int64(corrupt.count))
            }
        )

        let error = await #expect(throws: ModelStoreError.self) {
            _ = try await downloader.download(
                id: Self.repository,
                revision: nil,
                matching: [],
                useLatest: false,
                progressHandler: { _ in }
            )
        }

        #expect(
            error
                == .digestMismatch(
                    file: "model.safetensors",
                    algorithm: "sha256",
                    expected: Self.sha256Hex(of: published),
                    actual: Self.sha256Hex(of: corrupt)
                )
        )

        let landed = sandbox.store
            .directory(forRepository: Self.repository)
            .appendingPathComponent("model.safetensors")
        #expect(!FileManager.default.fileExists(atPath: landed.path))
    }

    @Test("A listing that names a path outside the model directory is refused by name")
    func refusesAnEscapingPath() async throws {
        let sandbox = try Sandbox()
        let payload = Data("not going there".utf8)
        let hub = StubHub(contents: ["../../escaped.safetensors": payload])

        let downloader = MLXHubDownloader(
            store: sandbox.store,
            index: hub.index,
            transfer: hub.transfer
        )

        let error = await #expect(throws: ModelStoreError.self) {
            _ = try await downloader.download(
                id: Self.repository,
                revision: nil,
                matching: [],
                useLatest: false,
                progressHandler: { _ in }
            )
        }
        #expect(error == .pathEscapesStore("../../escaped.safetensors"))
        #expect(hub.transferred.isEmpty, "refused before a byte was fetched")
    }

    // MARK: - T-AIL-28

    @Test("Progress is bytes across a large file, and hashing reports its own")
    func progressIsBytes() async throws {
        let sandbox = try Sandbox()
        let contents = Self.sampleContents(weightBytes: 12 * 1024 * 1024)
        let hub = StubHub(contents: contents)

        let downloadReports = Reports()
        let hashReports = HashReports()

        let downloader = MLXHubDownloader(
            store: sandbox.store,
            index: hub.index,
            transfer: hub.transfer,
            onHashProgress: { path, hashed, total in hashReports.append(path, hashed, total) }
        )

        _ = try await downloader.download(
            id: Self.repository,
            revision: nil,
            matching: [],
            useLatest: false,
            progressHandler: { progress in
                downloadReports.append(progress.completedUnitCount, progress.totalUnitCount)
            }
        )

        let expectedTotal = contents.values.reduce(Int64(0)) { $0 + Int64($1.count) }
        let observed = downloadReports.all
        #expect(!observed.isEmpty)

        // The unit is bytes, not a count of files: two files whose sizes differ
        // by four orders of magnitude cannot both be "half the work".
        #expect(
            observed.allSatisfy { $0.1 == expectedTotal },
            "every report's total is the byte total, measured \(expectedTotal)"
        )
        #expect(observed.first?.0 == 0)
        #expect(observed.last?.0 == expectedTotal)
        #expect(
            zip(observed, observed.dropFirst()).allSatisfy { $0.0 <= $1.0 },
            "bytes completed only ever go up"
        )

        // Reported DURING the 12 MB file, not only at its end: the stub reports
        // every 4 MB, so there are intermediate values strictly inside the file.
        let small = Int64(try #require(contents["config.json"]).count)
        let midFile = observed.filter { $0.0 > small && $0.0 < expectedTotal }
        #expect(
            midFile.count >= 2,
            "a 12 MB file reported in 4 MB chunks shows intermediate byte counts, saw \(midFile.count)"
        )

        // Hashing is its own progress: seconds of reading that move no bytes
        // over the network would otherwise render as a frozen download bar.
        let weightHashing = hashReports.all.filter { $0.path == "model.safetensors" }
        #expect(
            weightHashing.count >= 3,
            "a 12 MB file hashed in 4 MB chunks reports at least 3 times"
        )
        #expect(weightHashing.allSatisfy { $0.total == Int64(12 * 1024 * 1024) })
        #expect(weightHashing.first?.hashed == 0)
        #expect(weightHashing.last?.hashed == Int64(12 * 1024 * 1024))
        #expect(
            zip(weightHashing, weightHashing.dropFirst()).allSatisfy { $0.hashed <= $1.hashed },
            "bytes hashed only ever go up"
        )
    }

    @Test("Hashing progress is reported for a file that is reused rather than downloaded")
    func hashingReportsWhileVerifyingACandidate() async throws {
        let sandbox = try Sandbox()
        let weights = Data(repeating: 0x33, count: 9 * 1024 * 1024)
        let hub = StubHub(contents: ["model.safetensors": weights])

        let existing = sandbox.extraRoot
            .appendingPathComponent(Self.repository)
            .appendingPathComponent("model.safetensors")
        try FileManager.default.createDirectory(
            at: existing.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try weights.write(to: existing)

        let hashReports = HashReports()
        let downloader = MLXHubDownloader(
            store: sandbox.store,
            index: hub.index,
            transfer: hub.transfer,
            onHashProgress: { path, hashed, total in hashReports.append(path, hashed, total) }
        )

        _ = try await downloader.download(
            id: Self.repository,
            revision: nil,
            matching: [],
            useLatest: false,
            progressHandler: { _ in }
        )

        #expect(hub.transferred.isEmpty)
        let hashed = hashReports.all.filter { $0.path == "model.safetensors" }
        #expect(hashed.count >= 3, "verifying a 9 MB candidate reports as it reads")
        #expect(hashed.last?.hashed == Int64(weights.count))
    }
}

// MARK: - Collectors

/// Collects `(completed, total)` from a `@Sendable` progress callback.
///
/// `Synchronization.Mutex` would be the modern answer and needs macOS 15; this
/// package targets macOS 14, so the lock is explicit.
private final class Reports: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: [(Int64, Int64)] = []

    func append(_ completed: Int64, _ total: Int64) {
        lock.withLock { storage.append((completed, total)) }
    }

    var all: [(Int64, Int64)] {
        lock.withLock { storage }
    }
}

/// One hash-progress report.
struct HashReport: Sendable, Equatable {
    let path: String
    let hashed: Int64
    let total: Int64
}

/// Collects hash-progress reports from the callback.
private final class HashReports: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: [HashReport] = []

    func append(_ path: String, _ hashed: Int64, _ total: Int64) {
        lock.withLock { storage.append(HashReport(path: path, hashed: hashed, total: total)) }
    }

    var all: [HashReport] {
        lock.withLock { storage }
    }
}
