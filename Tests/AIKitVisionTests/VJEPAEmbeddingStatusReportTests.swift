// MARK: - VJEPAEmbeddingStatusReportTests
// Spec §5 Phase 2 deliverable 7 — `shi vision status` backing type.

#if canImport(SQLite3) && canImport(AVFoundation)
import Foundation
import Testing
@testable import AIKitVision

@Suite("VJEPAEmbeddingStatusReport")
struct VJEPAEmbeddingStatusReportTests {

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

    @Test("VisionStatusReport.make composes a report from a live node and store")
    func make() async throws {
        let node = Self.makeNode()
        let store = try EmbeddingStore(path: ":memory:")
        let emb = VJEPAEmbedding(frameCount: 1, featureDim: 2, data: [0.5, 0.5])
        _ = try store.insert(videoId: "a", sourcePath: "/tmp/a.mov", embedding: emb, modelVersion: "v2.1")

        let report = try await VisionStatusReport.make(node: node, store: store)
        #expect(report.modelLoaded == true)
        #expect(report.modelName == "vjepa-vitb16")
        #expect(report.modelVersion == "vjepa2-vitb16-v2.1")
        #expect(report.inputFrames == 8)
        #expect(report.embeddingDim == 768)
        #expect(report.embeddingsCount == 1)
    }

    @Test("VisionStatusReport renders human-readable text")
    func renderText() async throws {
        let report = VisionStatusReport(
            modelLoaded: true,
            modelName: "vjepa-vitb16",
            modelVersion: "v2.1",
            inputFrames: 8,
            embeddingDim: 768,
            latencyP50Ms: 120,
            memoryPeakMB: 42,
            embeddingsStorePath: "/tmp/embeddings.sqlite",
            embeddingsCount: 7
        )
        let text = report.renderText()
        #expect(text.contains("vjepa-vitb16"))
        #expect(text.contains("yes"))
        #expect(text.contains("768"))
        #expect(text.contains("120"))
        #expect(text.contains("7"))
    }
}

#endif
