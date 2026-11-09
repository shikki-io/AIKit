// kagami-scope: exempt — AIKit (moved from Irori) declares its scope in .kagami/scopes.override.toml; file touched only for resolver-exempt marker (no behavior change).
// MARK: - DINOv2ModelCacheTests
// Spec: features/shikki-vjepa-mlx-node.md §14.3 — "Weight cache:
// ~/.shikki/models/dinov2-vits14/" — same XDG pattern as V-JEPA.
//
// Confirms the DINOv2 descriptor round-trips through the shared
// `ModelWeightLoader` cache (BR-07 path convention, same as VJEPA).

import Foundation
import Testing
@testable import AIKitVision

@Suite("DINOv2ModelCache")
struct DINOv2ModelCacheTests {

    @Test("DINOv2 descriptor uses the ~/.shikki/models/dinov2-vits14/ convention") // resolver-exempt: test description string
    func descriptorPath() {
        let d = DINOv2ModelDescriptor.vits14
        #expect(d.descriptor.modelId == "dinov2-vits14")
        // CoreML is shipped as an .mlpackage directory — the default cache
        // path ships the zipped blob (`<name>.mlpackage.zip`) so the shared
        // weight loader (file-oriented) can round-trip it.
        let fn = d.descriptor.fileName
        #expect(fn == "dinov2-vits14.mlpackage.zip" || fn == "dinov2-vits14.mlpackage")
    }

    @Test("cache round-trip: ModelWeightLoader resolves + re-reads DINOv2 blob")
    func cacheRoundTrip() async throws {
        let tmpRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("shikki-dinov2-cache-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: tmpRoot) }

        let payload = Data(repeating: 0x2A, count: 2048)
        let sha = ModelWeightLoader.sha256Hex(of: payload)

        let fetcher = StubHTTPFetcher(body: payload)
        let loader = ModelWeightLoader(cacheRoot: tmpRoot, httpFetcher: fetcher)

        let descriptor = ModelWeightLoader.ModelDescriptor(
            modelId: "dinov2-vits14",
            fileName: "dinov2-vits14.mlpackage.zip",
            remoteURL: URL(string: "https://example.invalid/dinov2-vits14.mlpackage.zip")!,
            expectedSHA256: sha,
            expectedBytes: Int64(payload.count)
        )

        let url = try await loader.resolve(descriptor)
        #expect(url.path.contains("/dinov2-vits14/"))
        #expect(FileManager.default.fileExists(atPath: url.path))
        let readBack = try Data(contentsOf: url)
        #expect(readBack == payload)

        // Second resolve is a cache hit — no extra fetch.
        let callsAfterFirst = fetcher.calls
        _ = try await loader.resolve(descriptor)
        #expect(fetcher.calls == callsAfterFirst, "second resolve should hit the cache")
    }
}
