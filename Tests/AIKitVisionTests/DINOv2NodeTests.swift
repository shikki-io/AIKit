// MARK: - DINOv2NodeTests
// TDDP step 25 (RED) + step 26 (GREEN): DINOv2Node produces a [1, 384]
// embedding from a test image using a deterministic backend.
//
// Spec: features/shikki-vjepa-mlx-node.md §5 Phase 4 + §14.3.

#if canImport(CoreGraphics)
import CoreGraphics
import Foundation
import Testing
@testable import AIKitVision

@Suite("DINOv2Node")
struct DINOv2NodeTests {

    // MARK: - Helpers

    private static func makeSolidImage(width: Int = 64, height: Int = 64, gray: UInt8 = 128) -> CGImage {
        let bytesPerRow = width * 4
        var buffer = [UInt8](repeating: 0, count: width * height * 4)
        for i in stride(from: 0, to: buffer.count, by: 4) {
            buffer[i + 0] = gray
            buffer[i + 1] = gray
            buffer[i + 2] = gray
            buffer[i + 3] = 255
        }
        let space = CGColorSpaceCreateDeviceRGB()
        let info: UInt32 = CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
        let provider = CGDataProvider(data: Data(buffer) as CFData)!
        return CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: bytesPerRow,
            space: space,
            bitmapInfo: CGBitmapInfo(rawValue: info),
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        )!
    }

    private static func makeNode(backend: DINOv2Backend = MockDINOv2Backend()) -> DINOv2Node {
        DINOv2Node(
            nodeId: "dinov2-test",
            hostname: "test.local",
            backend: backend,
            modelVersion: "dinov2-vits14-v1.0"
        )
    }

    // MARK: - TDDP 25/26 — shape contract

    @Test("DINOv2Node.embed returns [1, 384] embedding")
    func embedReturnsShape() throws {
        let node = Self.makeNode()
        let image = Self.makeSolidImage()
        let embedding = try node.embed(image: image)
        #expect(embedding.featureDim == 384)
        #expect(embedding.data.count == 384)
    }

    @Test("DINOv2Node.name identifies the provider")
    func name() {
        let node = Self.makeNode()
        #expect(node.name == "dinov2-vits14")
    }

    @Test("DINOv2Node embedding is deterministic for identical inputs")
    func deterministic() throws {
        let node = Self.makeNode()
        let image = Self.makeSolidImage(gray: 200)
        let a = try node.embed(image: image)
        let b = try node.embed(image: image)
        #expect(a.data == b.data)
    }

    @Test("DINOv2Node embeddings differ for different images")
    func differentImagesDifferentEmbeddings() throws {
        let node = Self.makeNode()
        let darker = Self.makeSolidImage(gray: 30)
        let lighter = Self.makeSolidImage(gray: 220)
        let a = try node.embed(image: darker)
        let b = try node.embed(image: lighter)
        #expect(a.data != b.data)
    }

    // MARK: - Heartbeat contract (mirrors VJEPANode)

    @Test("DINOv2Node heartbeat includes model metadata")
    func heartbeatMetadata() async {
        let node = Self.makeNode()
        let hb = await node.makeHeartbeat()
        #expect(hb.metadata.modelName == "dinov2-vits14")
        #expect(hb.metadata.modelVersion == "dinov2-vits14-v1.0")
        #expect(hb.metadata.embeddingDim == 384)
        #expect(hb.metadata.inputFrames == 1)  // single-image model
    }

    @Test("DINOv2Node heartbeat subject matches arena convention")
    func heartbeatSubject() {
        #expect(DINOv2Node.heartbeatSubject == "shikki.arena.nodes.heartbeat")
        #expect(DINOv2Node.nodeKind == "provider")
    }

    @Test("DINOv2Node publishes heartbeat to NATS recorder")
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

#endif
