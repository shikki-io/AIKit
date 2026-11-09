// MARK: - VJEPAEmbeddingExtractorTests
// Spec §5 Phase 2 deliverable 4 — ScreenRecordingExtractor end-to-end.

#if canImport(AVFoundation) && canImport(SQLite3)
import AVFoundation
import Foundation
import Testing
@testable import AIKitVision

@Suite("VJEPAEmbeddingExtractor")
struct VJEPAEmbeddingExtractorTests {

    private static func makeNode() -> VJEPANode {
        let encoder = VJEPAEncoder(config: .vitB16)
        let preprocessor = VideoFramePreprocessor(targetFrameCount: 8, targetSize: 224)
        let inference = VJEPAInference(
            encoder: encoder,
            preprocessor: preprocessor,
            backend: ReferenceVisionBackend()
        )
        return VJEPANode(
            nodeId: "vjepa-test",
            hostname: "test.local",
            inference: inference,
            modelVersion: "vjepa2-vitb16-v2.1"
        )
    }

    @Test("ScreenRecordingExtractor embeds a video and persists the record")
    func extractAndStore() async throws {
        let store = try EmbeddingStore(path: ":memory:")
        let node = Self.makeNode()
        let extractor = ScreenRecordingExtractor(node: node, store: store)

        let url = try await VideoFramePreprocessorTests.renderFixtureVideo(frameCount: 16)
        let record = try await extractor.extractAndStore(videoAt: url, videoId: "fixture-01")

        #expect(record.videoId == "fixture-01")
        #expect(record.modelVersion == "vjepa2-vitb16-v2.1")
        #expect(record.embedding.frameCount == 8)
        #expect(record.embedding.featureDim == 768)

        // The row is actually in the store.
        let all = try store.all()
        #expect(all.count == 1)
        #expect(all.first?.id == record.id)
    }

    @Test("ScreenRecordingExtractor derives stable videoId from filename")
    func defaultVideoIdDerivation() {
        let url = URL(fileURLWithPath: "/tmp/Debug Session.mov")
        let id = ScreenRecordingExtractor.defaultVideoId(for: url)
        #expect(id == "debug-session")
    }
}

#endif
