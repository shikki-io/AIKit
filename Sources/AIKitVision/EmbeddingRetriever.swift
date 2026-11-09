// MARK: - EmbeddingRetriever
// Top-K cosine similarity search across stored V-JEPA embeddings
// (spec §5 Phase 2 deliverable 5).
//
// V1: brute-force scan of the local SQLite enclave — O(N*D) per query.
// For the expected scale (hundreds to a few thousand recordings per
// dev machine), this is cheap and avoids a vector-index dependency.
//
// Scoring: mean-pool each stored `[T, D]` embedding to `[D]`, then cosine
// similarity against the (mean-pooled) query. Mean pooling matches the
// semantic contract from §7 Scenario 2 — "find the recording that looks
// like this clip" — without committing to a specific frame alignment.

#if canImport(SQLite3)
import Foundation

// MARK: - Result

public struct EmbeddingRetrievalResult: Sendable, Equatable {
    public let record: VideoEmbeddingRecord
    public let similarity: Float

    public init(record: VideoEmbeddingRecord, similarity: Float) {
        self.record = record
        self.similarity = similarity
    }
}

// MARK: - Retriever

public struct EmbeddingRetriever: Sendable {

    public let store: EmbeddingStore

    public init(store: EmbeddingStore) {
        self.store = store
    }

    // MARK: Query API

    /// Return the top-K stored embeddings ranked by cosine similarity
    /// against the (mean-pooled) query.
    public func topK(query: VJEPAEmbedding, k: Int) throws -> [EmbeddingRetrievalResult] {
        let queryVector = Self.meanPool(query)
        return try topKByVector(query: queryVector, k: k)
    }

    /// Return the single nearest stored embedding, or nil if the store is
    /// empty.
    public func nearest(query: VJEPAEmbedding) throws -> EmbeddingRetrievalResult? {
        try topK(query: query, k: 1).first
    }

    /// Retrieval by a pre-pooled vector. Useful when the caller has already
    /// pooled frames (e.g. the agent tool) or wants to query with a
    /// hand-authored vector for testing.
    public func topKByVector(query: [Float], k: Int) throws -> [EmbeddingRetrievalResult] {
        guard k > 0 else { return [] }
        let rows = try store.all()
        var scored: [EmbeddingRetrievalResult] = []
        scored.reserveCapacity(rows.count)
        for row in rows {
            let pooled = Self.meanPool(row.embedding)
            // Guard against mismatched dims — skip rather than throw so a
            // mixed-model store degrades gracefully.
            guard pooled.count == query.count else { continue }
            let sim = Self.cosineSimilarity(query, pooled)
            scored.append(EmbeddingRetrievalResult(record: row, similarity: sim))
        }
        scored.sort { $0.similarity > $1.similarity }
        return Array(scored.prefix(k))
    }

    public func nearestByVector(query: [Float]) throws -> EmbeddingRetrievalResult? {
        try topKByVector(query: query, k: 1).first
    }

    // MARK: - Math

    /// Cosine similarity between two equal-length float vectors.
    /// Returns 0 if either norm is zero (defensive — avoids NaN).
    public static func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        precondition(a.count == b.count, "cosineSimilarity: length mismatch")
        var dot: Float = 0
        var normA: Float = 0
        var normB: Float = 0
        for i in 0..<a.count {
            dot += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }
        let denom = (normA.squareRoot()) * (normB.squareRoot())
        guard denom > 0 else { return 0 }
        return dot / denom
    }

    /// Mean-pool a `[T, D]` embedding into `[D]`.
    public static func meanPool(_ e: VJEPAEmbedding) -> [Float] {
        guard e.frameCount > 0 else { return [] }
        var out = [Float](repeating: 0, count: e.featureDim)
        for t in 0..<e.frameCount {
            for d in 0..<e.featureDim {
                out[d] += e.data[t * e.featureDim + d]
            }
        }
        let scale = 1.0 / Float(e.frameCount)
        for d in 0..<e.featureDim { out[d] *= scale }
        return out
    }
}

#endif
