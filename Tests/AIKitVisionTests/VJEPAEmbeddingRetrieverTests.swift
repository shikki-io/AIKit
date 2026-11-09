// MARK: - VJEPAEmbeddingRetrieverTests
// TDDP step 19 (RED) + step 20 (GREEN): cosine-similarity top-K retrieval.
//
// Spec §5 Phase 2 deliverable 5 + §7 Scenario 2.
// Gate: retrieval returns correct nearest neighbor on test corpus.

#if canImport(SQLite3)
import Foundation
import Testing
@testable import AIKitVision

@Suite("VJEPAEmbeddingRetriever")
struct VJEPAEmbeddingRetrieverTests {

    // MARK: - Helpers

    /// Build a mean-pooled feature vector from a frame-level embedding so
    /// retrieval can score videos without per-frame alignment.
    private static func meanPool(_ e: VJEPAEmbedding) -> [Float] {
        var out = [Float](repeating: 0, count: e.featureDim)
        for t in 0..<e.frameCount {
            for d in 0..<e.featureDim {
                out[d] += e.data[t * e.featureDim + d]
            }
        }
        let scale = 1.0 / Float(max(1, e.frameCount))
        for d in 0..<e.featureDim { out[d] *= scale }
        return out
    }

    private static func embedding(frames: Int, dim: Int, base: [Float]) -> VJEPAEmbedding {
        var flat = [Float](repeating: 0, count: frames * dim)
        for t in 0..<frames {
            for d in 0..<dim {
                flat[t * dim + d] = base[d]
            }
        }
        return VJEPAEmbedding(frameCount: frames, featureDim: dim, data: flat)
    }

    private static func seededStore() throws -> EmbeddingStore {
        let store = try EmbeddingStore(path: ":memory:")
        // Three embeddings with distinct principal direction:
        // A lies along e0, B along e1, C along e2.
        let dim = 4
        let clipA = embedding(frames: 2, dim: dim, base: [1.0, 0.01, 0.01, 0.01])
        let clipB = embedding(frames: 2, dim: dim, base: [0.01, 1.0, 0.01, 0.01])
        let clipC = embedding(frames: 2, dim: dim, base: [0.01, 0.01, 1.0, 0.01])
        _ = try store.insert(videoId: "A", sourcePath: "/tmp/a.mov", embedding: clipA, modelVersion: "v2.1")
        _ = try store.insert(videoId: "B", sourcePath: "/tmp/b.mov", embedding: clipB, modelVersion: "v2.1")
        _ = try store.insert(videoId: "C", sourcePath: "/tmp/c.mov", embedding: clipC, modelVersion: "v2.1")
        return store
    }

    // MARK: - Cosine helper unit tests

    @Test("cosine similarity of identical vectors is 1.0")
    func cosineIdentity() {
        let v: [Float] = [1, 2, 3, 4]
        let s = EmbeddingRetriever.cosineSimilarity(v, v)
        #expect(abs(s - 1.0) < 1e-6)
    }

    @Test("cosine similarity of orthogonal vectors is ~0.0")
    func cosineOrthogonal() {
        let a: [Float] = [1, 0, 0, 0]
        let b: [Float] = [0, 1, 0, 0]
        let s = EmbeddingRetriever.cosineSimilarity(a, b)
        #expect(abs(s) < 1e-6)
    }

    @Test("cosine similarity of opposite vectors is -1.0")
    func cosineOpposite() {
        let a: [Float] = [1, 2, 3, 4]
        let b = a.map { -$0 }
        let s = EmbeddingRetriever.cosineSimilarity(a, b)
        #expect(abs(s - -1.0) < 1e-6)
    }

    // MARK: - Top-K retrieval

    @Test("EmbeddingRetriever returns clip A as nearest for an A-like query (BR: top-K ranking)")
    func topKNearestA() throws {
        let store = try Self.seededStore()
        let retriever = EmbeddingRetriever(store: store)
        // Query close to clip A's direction.
        let query = Self.embedding(frames: 1, dim: 4, base: [0.9, 0.02, 0.02, 0.0])
        let results = try retriever.topK(query: query, k: 3)
        #expect(results.count == 3)
        #expect(results[0].record.videoId == "A")
        #expect(results[0].similarity > results[1].similarity)
        #expect(results[1].similarity > results[2].similarity - 1e-6)
    }

    @Test("EmbeddingRetriever nearest neighbor picks B when query lies along e1")
    func nearestB() throws {
        let store = try Self.seededStore()
        let retriever = EmbeddingRetriever(store: store)
        let query = Self.embedding(frames: 1, dim: 4, base: [0.01, 1.0, 0.05, 0.0])
        let nearest = try #require(try retriever.nearest(query: query))
        #expect(nearest.record.videoId == "B")
    }

    @Test("EmbeddingRetriever nearest returns nil on empty store")
    func emptyStore() throws {
        let store = try EmbeddingStore(path: ":memory:")
        let retriever = EmbeddingRetriever(store: store)
        let query = Self.embedding(frames: 1, dim: 4, base: [1, 0, 0, 0])
        #expect(try retriever.nearest(query: query) == nil)
    }

    @Test("EmbeddingRetriever.topK truncates to requested k")
    func topKTruncates() throws {
        let store = try Self.seededStore()
        let retriever = EmbeddingRetriever(store: store)
        let query = Self.embedding(frames: 1, dim: 4, base: [1, 0, 0, 0])
        let results = try retriever.topK(query: query, k: 2)
        #expect(results.count == 2)
    }

    @Test("EmbeddingRetriever mean-pools frame-level embeddings before comparison")
    func meanPoolContract() throws {
        let store = try Self.seededStore()
        let retriever = EmbeddingRetriever(store: store)
        let queryFlat = Self.meanPool(Self.embedding(frames: 8, dim: 4, base: [1.0, 0.01, 0.01, 0.01]))
        #expect(queryFlat.count == 4)
        let nearest = try #require(try retriever.nearestByVector(query: queryFlat))
        #expect(nearest.record.videoId == "A")
    }
}

#endif
