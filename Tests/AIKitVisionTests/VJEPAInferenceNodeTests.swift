// MARK: - VJEPAInferenceNodeTests
// TDDP step 13 (RED) + step 14 (GREEN): VJEPANode conforms to AgentProvider
// semantics and returns embeddings.
// TDDP step 15 (RED) + step 16 (GREEN): heartbeat payload with model metadata.
//
// Spec: features/shikki-vjepa-mlx-node.md §5 Phase 2 + BR-06 + BR-16.

#if canImport(AVFoundation)
import AVFoundation
import Foundation
import Testing
@testable import AIKitVision

@Suite("VJEPAInferenceNode")
struct VJEPAInferenceNodeTests {

    // MARK: - Helpers

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

    // MARK: - TDDP 13/14: Node conforms to AgentProvider-ish contract + returns embeddings

    @Test("VJEPANode.name identifies the provider")
    func name() {
        let node = Self.makeNode()
        #expect(node.name == "vjepa-vitb16")
    }

    @Test("VJEPANode.embed returns [T, 768] embedding for preprocessed frames")
    func embedFramesReturnsShape() throws {
        let node = Self.makeNode()
        let frames = VideoFrameTensor(
            frameCount: 4,
            channels: 3,
            height: 224,
            width: 224,
            data: [Float](repeating: 0.1, count: 4 * 3 * 224 * 224)
        )
        let embedding = try node.embed(frames: frames)
        #expect(embedding.frameCount == 4)
        #expect(embedding.featureDim == 768)
        #expect(embedding.data.count == 4 * 768)
    }

    @Test("VJEPANode.embed(videoAt:) returns non-empty embedding for fixture video")
    func embedVideoReturnsEmbedding() async throws {
        let node = Self.makeNode()
        let url = try await VideoFramePreprocessorTests.renderFixtureVideo(frameCount: 16)
        let embedding = try await node.embed(videoAt: url)
        #expect(embedding.frameCount == 8)   // targetFrameCount on preprocessor
        #expect(embedding.featureDim == 768)
    }

    // MARK: - TDDP 15/16: Heartbeat with BR-16 metadata

    @Test("VJEPANode heartbeat includes BR-16 model metadata")
    func heartbeatMetadata() async {
        let node = Self.makeNode()
        let hb = await node.makeHeartbeat()
        #expect(hb.metadata.modelName == "vjepa-vitb16")
        #expect(hb.metadata.modelVersion == "vjepa2-vitb16-v2.1")
        #expect(hb.metadata.inputFrames == 8)
        #expect(hb.metadata.embeddingDim == 768)
        #expect(hb.metadata.latencyP50Ms >= 0)
        #expect(hb.metadata.memoryPeakMB >= 0)
    }

    @Test("VJEPANode heartbeat registers as provider NodeKind on correct subject")
    func heartbeatSubject() {
        #expect(VJEPANode.heartbeatSubject == "shikki.arena.nodes.heartbeat")
        #expect(VJEPANode.nodeKind == "provider")
    }

    @Test("VJEPANode publishes heartbeat to NATS client")
    func publishesHeartbeat() async throws {
        let recorder = RecordingVisionNATSClient()
        let node = Self.makeNode()
        try await node.publishHeartbeat(via: recorder)
        let published = await recorder.captured()
        #expect(published.count == 1)
        #expect(published.first?.subject == "shikki.arena.nodes.heartbeat")
        #expect(published.first?.payload.isEmpty == false)
    }
}

// MARK: - Test double: recording NATS client

final actor RecordingVisionNATSClient: VisionNATSPublisher {
    struct Published: Sendable, Equatable {
        let subject: String
        let payload: Data
    }

    private var messages: [Published] = []

    func publish(_ data: Data, subject: String) async throws {
        messages.append(Published(subject: subject, payload: data))
    }

    var isConnected: Bool { true }

    func captured() -> [Published] { messages }
}

#endif
