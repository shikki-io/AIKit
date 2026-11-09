// MARK: - ScreenRecordingExtractor
// Spec §5 Phase 2 deliverable 4.
//
// AVFoundation → uniform frame sample → VJEPAInference → EmbeddingStore.
// Entry point for the end-to-end pipeline: pass a video URL, get back the
// stored record (including the generated row id) so callers can thread it
// straight into `EmbeddingRetriever`.
//
// The preprocessor + backend choice is inherited from the wrapped
// `VJEPANode`, so the extractor doesn't duplicate configuration.

#if canImport(AVFoundation) && canImport(SQLite3)
import AVFoundation
import Foundation

public struct ScreenRecordingExtractor: Sendable {

    public let node: VJEPANode
    public let store: EmbeddingStore

    public init(node: VJEPANode, store: EmbeddingStore) {
        self.node = node
        self.store = store
    }

    /// Extract frames from the video at `url`, embed them with the wrapped
    /// node, and persist the result. Returns the full persisted record.
    ///
    /// - Parameter videoId: stable logical id for the clip. If nil, derived
    ///   from the file's last path component so the same recording is not
    ///   duplicated on repeated extraction.
    @discardableResult
    public func extractAndStore(
        videoAt url: URL,
        videoId: String? = nil
    ) async throws -> VideoEmbeddingRecord {
        let embedding = try await node.embed(videoAt: url)
        let vid = videoId ?? Self.defaultVideoId(for: url)
        let id = try store.insert(
            videoId: vid,
            sourcePath: url.path,
            embedding: embedding,
            modelVersion: node.modelVersion
        )
        guard let record = try store.fetch(id: id) else {
            throw EmbeddingStoreError.corrupt("insert succeeded but fetch returned nil for id=\(id)")
        }
        return record
    }

    /// Derive a stable video id from the file path — last component minus
    /// extension, lowercased, spaces→dashes. Good enough for v1.
    public static func defaultVideoId(for url: URL) -> String {
        let base = url.deletingPathExtension().lastPathComponent
        let lowered = base.lowercased()
        let spaced = lowered.replacingOccurrences(of: " ", with: "-")
        return spaced.isEmpty ? "unknown" : spaced
    }
}

#endif
