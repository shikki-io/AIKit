// MARK: - VJEPAEmbeddingVideoAnalyzeTests
// Spec §5 Phase 2 deliverable 6 + §7 Scenario 2.

#if canImport(AVFoundation) && canImport(SQLite3)
import AVFoundation
import Foundation
import Testing
@testable import AIKitVision

@Suite("VJEPAEmbeddingVideoAnalyze")
struct VJEPAEmbeddingVideoAnalyzeTests {

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

    @Test("VideoAnalyze tool identity")
    func identity() {
        #expect(VideoAnalyzeTool.name == "VideoAnalyze")
    }

    @Test("VideoAnalyze returns stored record metadata + neighbours from local history")
    func endToEnd() async throws {
        let store = try EmbeddingStore(path: ":memory:")
        let node = Self.makeNode()
        let tool = VideoAnalyzeTool(node: node, store: store)

        // Prime history with two fixture recordings.
        let fixtureA = try await VideoFramePreprocessorTests.renderFixtureVideo(frameCount: 16)
        let fixtureB = try await VideoFramePreprocessorTests.renderFixtureVideo(frameCount: 12)
        let extractor = ScreenRecordingExtractor(node: node, store: store)
        _ = try await extractor.extractAndStore(videoAt: fixtureA, videoId: "hist-a")
        _ = try await extractor.extractAndStore(videoAt: fixtureB, videoId: "hist-b")

        // Ask the tool to analyze a fresh recording.
        let freshURL = try await VideoFramePreprocessorTests.renderFixtureVideo(frameCount: 16)
        let input = VideoAnalyzeInput(videoPath: freshURL.path, videoId: "fresh", topK: 2)
        let output = try await tool.execute(input: input)

        #expect(output.videoId == "fresh")
        #expect(output.frameCount == 8)
        #expect(output.embeddingDim == 768)
        #expect(output.modelVersion == "vjepa2-vitb16-v2.1")
        // Up to topK neighbours, excluding the newly-inserted record itself.
        #expect(output.neighbors.count <= 2)
        #expect(output.neighbors.allSatisfy { $0.videoId != "fresh" })
        // Similarity values are finite and in [-1, 1].
        for n in output.neighbors {
            #expect(n.similarity.isFinite)
            #expect(n.similarity >= -1.001 && n.similarity <= 1.001)
        }
    }

    @Test("VideoAnalyze input encodes/decodes via JSONEncoder")
    func inputCodable() throws {
        let input = VideoAnalyzeInput(videoPath: "/tmp/x.mov", videoId: "x", topK: 5)
        let data = try JSONEncoder().encode(input)
        let round = try JSONDecoder().decode(VideoAnalyzeInput.self, from: data)
        #expect(round == input)
    }
}

#endif
