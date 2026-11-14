// kagami-scope: exempt — AIKit (moved from Irori) declares its scope in .kagami/scopes.override.toml; file touched only for resolver-exempt marker (no behavior change).
// MARK: - VJEPAEmbeddingStoreTests
// TDDP step 17 (RED) + step 18 (GREEN): raw-SQL persistence of embeddings.
//
// Spec §5 Phase 2 deliverable 3 + BR-20 (local SQLite enclave at
// ~/.shikki/vision/embeddings.sqlite, NOT ShikkiDB). Tests use :memory:
// to stay hermetic and CI-stable.

#if canImport(SQLite3)
import Foundation
import Testing
@testable import AIKitVision

@Suite("VJEPAEmbeddingStore")
struct VJEPAEmbeddingStoreTests {

    // MARK: - Schema

    @Test("EmbeddingStore creates video_embeddings table with correct schema")
    func schema() throws {
        let store = try EmbeddingStore(path: ":memory:")
        let cols = try store.dumpSchema()
        // Required columns per spec §5 Phase 2 deliverable 3.
        for column in ["id", "video_id", "source_path", "embedding", "frame_count", "model_version", "created_at"] {
            #expect(cols.contains(column), "missing column: \(column)")
        }
    }

    // MARK: - Round-trip

    @Test("EmbeddingStore persists and retrieves a video embedding round-trip")
    func roundTrip() throws {
        let store = try EmbeddingStore(path: ":memory:")
        let embedding = VJEPAEmbedding(
            frameCount: 8,
            featureDim: 768,
            data: (0..<(8 * 768)).map { Float($0) * 1e-3 }
        )
        let id = try store.insert(
            videoId: "debug-session-001",
            sourcePath: "/Users/jeo/Desktop/debug-session.mov",
            embedding: embedding,
            modelVersion: "vjepa2-vitb16-v2.1"
        )
        let round = try #require(try store.fetch(id: id))
        #expect(round.videoId == "debug-session-001")
        #expect(round.sourcePath == "/Users/jeo/Desktop/debug-session.mov")
        #expect(round.modelVersion == "vjepa2-vitb16-v2.1")
        #expect(round.embedding.frameCount == 8)
        #expect(round.embedding.featureDim == 768)
        #expect(round.embedding.data.count == 8 * 768)
        // Float bit-exact after round-trip (BLOB is little-endian float32).
        for i in stride(from: 0, to: round.embedding.data.count, by: 97) {
            #expect(round.embedding.data[i] == embedding.data[i])
        }
    }

    @Test("EmbeddingStore fetch returns nil when id is absent")
    func fetchMissing() throws {
        let store = try EmbeddingStore(path: ":memory:")
        let result = try store.fetch(id: 99_999)
        #expect(result == nil)
    }

    @Test("EmbeddingStore.all returns every inserted row")
    func allRows() throws {
        let store = try EmbeddingStore(path: ":memory:")
        for i in 0..<5 {
            let emb = VJEPAEmbedding(
                frameCount: 2, featureDim: 4,
                data: [Float(i), 0.5, -0.25, 1.0, 2.0, 3.0, 4.0, 5.0]
            )
            _ = try store.insert(
                videoId: "clip-\(i)",
                sourcePath: "/tmp/clip-\(i).mov",
                embedding: emb,
                modelVersion: "vjepa2-vitb16-v2.1"
            )
        }
        let all = try store.all()
        #expect(all.count == 5)
    }

    @Test("EmbeddingStore.count reflects row count")
    func count() throws {
        let store = try EmbeddingStore(path: ":memory:")
        #expect(try store.count() == 0)
        let emb = VJEPAEmbedding(frameCount: 1, featureDim: 2, data: [1.0, 2.0])
        _ = try store.insert(videoId: "v1", sourcePath: "/tmp/a.mov", embedding: emb, modelVersion: "v2.1")
        _ = try store.insert(videoId: "v2", sourcePath: "/tmp/b.mov", embedding: emb, modelVersion: "v2.1")
        #expect(try store.count() == 2)
    }

    @Test("EmbeddingStore defaults to ~/.shikki/vision/embeddings.sqlite (BR-20)") // resolver-exempt: test description string
    func defaultPath() {
        let shikkiRoot = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".shikki")
        let expected = shikkiRoot.appendingPathComponent("vision/embeddings.sqlite").path
        #expect(EmbeddingStore.defaultPath == expected)
    }
}

#endif
