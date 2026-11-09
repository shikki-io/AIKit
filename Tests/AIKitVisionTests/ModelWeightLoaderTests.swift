// MARK: - ModelWeightLoaderTests
// TDDP step 11 (RED) + step 12 (GREEN): lazy download + resume + SHA-256.
//
// See features/shikki-vjepa-mlx-node.md §5 Phase 1 (BR-01, BR-07, BR-08, BR-09).

import Foundation
import Testing
@testable import AIKitVision

@Suite("ModelWeightLoader")
struct ModelWeightLoaderTests {

    // MARK: Setup

    /// Build a loader pointing at a temp cache directory + a stubbed HTTP fetcher.
    private func makeLoader(
        fetcher: StubHTTPFetcher
    ) -> (loader: ModelWeightLoader, cacheDir: URL) {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("vjepa-cache-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let loader = ModelWeightLoader(cacheRoot: dir, httpFetcher: fetcher)
        return (loader, dir)
    }

    // MARK: Cached path

    @Test("returns cached path and skips download when SHA-256 matches")
    func test_cachedSkipsDownload() async throws {
        let payload = Data("fake-weights".utf8)
        let sha = payload.sha256Hex()
        let descriptor = ModelWeightLoader.ModelDescriptor(
            modelId: "vjepa2-vitb16",
            fileName: "model.safetensors",
            remoteURL: URL(string: "https://huggingface.co/facebook/vjepa2-vitb16-v2.1/resolve/main/model.safetensors")!,
            expectedSHA256: sha,
            expectedBytes: Int64(payload.count)
        )
        let fetcher = StubHTTPFetcher(body: payload)
        let (loader, cacheDir) = makeLoader(fetcher: fetcher)

        // Pre-populate the cache with the exact expected bytes.
        let expectedPath = cacheDir
            .appendingPathComponent(descriptor.modelId)
            .appendingPathComponent(descriptor.fileName)
        try FileManager.default.createDirectory(
            at: expectedPath.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try payload.write(to: expectedPath)

        let resolved = try await loader.resolve(descriptor)
        #expect(resolved == expectedPath)
        #expect(fetcher.calls == 0)  // no download — cache hit
    }

    // MARK: Download when missing

    @Test("downloads weights when cache is empty and verifies SHA-256")
    func test_downloadsWhenMissing() async throws {
        let payload = Data(repeating: 0xAA, count: 4096)
        let sha = payload.sha256Hex()
        let descriptor = ModelWeightLoader.ModelDescriptor(
            modelId: "vjepa2-vitb16",
            fileName: "model.safetensors",
            remoteURL: URL(string: "https://huggingface.co/facebook/vjepa2-vitb16-v2.1/resolve/main/model.safetensors")!,
            expectedSHA256: sha,
            expectedBytes: Int64(payload.count)
        )
        let fetcher = StubHTTPFetcher(body: payload)
        let (loader, cacheDir) = makeLoader(fetcher: fetcher)
        let resolved = try await loader.resolve(descriptor)

        #expect(fetcher.calls == 1)
        #expect(FileManager.default.fileExists(atPath: resolved.path))
        let written = try Data(contentsOf: resolved)
        #expect(written == payload)
        // File must live under the cache root, not elsewhere.
        #expect(resolved.path.hasPrefix(cacheDir.path))
    }

    // MARK: SHA mismatch

    @Test("throws checksumMismatch when downloaded SHA-256 differs")
    func test_throwsOnChecksumMismatch() async throws {
        let payload = Data(repeating: 0x11, count: 128)
        let descriptor = ModelWeightLoader.ModelDescriptor(
            modelId: "vjepa2-vitb16",
            fileName: "model.safetensors",
            remoteURL: URL(string: "https://huggingface.co/fake")!,
            expectedSHA256: "deadbeef".padding(toLength: 64, withPad: "0", startingAt: 0),
            expectedBytes: Int64(payload.count)
        )
        let fetcher = StubHTTPFetcher(body: payload)
        let (loader, _) = makeLoader(fetcher: fetcher)
        do {
            _ = try await loader.resolve(descriptor)
            Issue.record("Expected checksum mismatch")
        } catch ModelWeightLoader.Error.checksumMismatch {
            // expected
        }
    }

    // MARK: Range resume

    @Test("resume uses partial download and Range header when a stub .part file exists")
    func test_rangeResume() async throws {
        let full = Data(repeating: 0x77, count: 8192)
        let partial = full.prefix(2048)
        let sha = full.sha256Hex()
        let descriptor = ModelWeightLoader.ModelDescriptor(
            modelId: "vjepa2-vitb16",
            fileName: "model.safetensors",
            remoteURL: URL(string: "https://huggingface.co/fake")!,
            expectedSHA256: sha,
            expectedBytes: Int64(full.count)
        )

        // Fetcher returns only the suffix when asked with Range: bytes=2048-.
        let fetcher = StubHTTPFetcher(body: full.suffix(from: 2048), respectsRange: true)
        let (loader, cacheDir) = makeLoader(fetcher: fetcher)

        // Pre-populate the partial file.
        let modelDir = cacheDir.appendingPathComponent(descriptor.modelId)
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
        let partPath = modelDir.appendingPathComponent(descriptor.fileName + ".part")
        try Data(partial).write(to: partPath)

        let resolved = try await loader.resolve(descriptor)
        let written = try Data(contentsOf: resolved)
        #expect(written == full)
        #expect(fetcher.calls == 1)
        #expect(fetcher.lastRangeStart == 2048)
    }
}

// MARK: - Stub HTTP fetcher

final class StubHTTPFetcher: ModelWeightHTTPFetcher, @unchecked Sendable {
    let body: Data
    let respectsRange: Bool
    private(set) var calls: Int = 0
    private(set) var lastRangeStart: Int64? = nil

    init<S: Sequence>(body: S, respectsRange: Bool = false) where S.Element == UInt8 {
        self.body = Data(body)
        self.respectsRange = respectsRange
    }

    func fetch(url: URL, rangeStart: Int64?) async throws -> AsyncByteStream {
        calls += 1
        lastRangeStart = rangeStart
        let slice: Data
        if let start = rangeStart, respectsRange {
            slice = body
        } else {
            slice = body
            // If rangeStart is passed but the fetcher does not respect it, still
            // return the full body — the loader's checksum check will catch
            // corruption. This exercises the fallback path as well.
            _ = rangeStart
        }
        return AsyncByteStream { continuation in
            Task {
                // Push in 1 KiB chunks to exercise the streaming path.
                var offset = 0
                let chunkSize = 1024
                while offset < slice.count {
                    let end = min(offset + chunkSize, slice.count)
                    continuation.yield(slice[offset..<end])
                    offset = end
                }
                continuation.finish()
            }
        }
    }
}

// MARK: - SHA helper for tests

private extension Data {
    func sha256Hex() -> String {
        return ModelWeightLoader.sha256Hex(of: self)
    }
}
