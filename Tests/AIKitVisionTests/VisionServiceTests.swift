// MARK: - VisionServiceTests
// TDDP step 27 (RED) + step 28 (GREEN): VisionService routes images to
// DINOv2 and videos to V-JEPA based on input type.
//
// Spec: features/shikki-vjepa-mlx-node.md §5 Phase 4 deliverable 7 +
// Scenario 7 (§ acceptance criteria — file extension routing).

#if canImport(AVFoundation) && canImport(CoreGraphics)
import AVFoundation
import CoreGraphics
import Foundation
import ImageIO
import Testing
@testable import AIKitVision

@Suite("VisionService")
struct VisionServiceTests {

    // MARK: - Helpers

    private static func makeService() -> VisionService {
        let videoNode = DINOv2NodeTests_Companion.makeVJEPANode()
        let imageNode = DINOv2NodeTests_Companion.makeDINOv2Node()
        return VisionService(videoNode: videoNode, imageNode: imageNode)
    }

    // MARK: - Type classification

    @Test("classify .png → .image")
    func classifyPng() {
        #expect(VisionService.classify(url: URL(fileURLWithPath: "/tmp/a.png")) == .image)
    }

    @Test("classify .jpg / .jpeg / .heic / .gif → .image")
    func classifyImageExtensions() {
        #expect(VisionService.classify(url: URL(fileURLWithPath: "/tmp/a.jpg")) == .image)
        #expect(VisionService.classify(url: URL(fileURLWithPath: "/tmp/a.jpeg")) == .image)
        #expect(VisionService.classify(url: URL(fileURLWithPath: "/tmp/a.HEIC")) == .image)
        #expect(VisionService.classify(url: URL(fileURLWithPath: "/tmp/a.gif")) == .image)
    }

    @Test("classify .mov / .mp4 / .m4v → .video")
    func classifyVideoExtensions() {
        #expect(VisionService.classify(url: URL(fileURLWithPath: "/tmp/a.mov")) == .video)
        #expect(VisionService.classify(url: URL(fileURLWithPath: "/tmp/a.mp4")) == .video)
        #expect(VisionService.classify(url: URL(fileURLWithPath: "/tmp/a.M4V")) == .video)
    }

    @Test("classify unknown extension → .unknown")
    func classifyUnknown() {
        #expect(VisionService.classify(url: URL(fileURLWithPath: "/tmp/a.txt")) == .unknown)
    }

    // MARK: - Routing

    @Test("VisionService routes PNG URL to DINOv2")
    func routesImageToDINOv2() async throws {
        let service = Self.makeService()
        let pngURL = try DINOv2NodeTests_Companion.writeSolidPNG(gray: 128)
        defer { try? FileManager.default.removeItem(at: pngURL) }
        let result = try await service.embed(fileAt: pngURL)
        switch result {
        case .image(let embedding):
            #expect(embedding.featureDim == 384)
        case .video:
            Issue.record("expected .image routing for .png")
        }
    }

    @Test("VisionService routes MOV URL to V-JEPA")
    func routesVideoToVJEPA() async throws {
        let service = Self.makeService()
        let videoURL = try await VideoFramePreprocessorTests.renderFixtureVideo(frameCount: 16)
        let result = try await service.embed(fileAt: videoURL)
        switch result {
        case .video(let embedding):
            #expect(embedding.featureDim == 768)
            #expect(embedding.frameCount > 0)
        case .image:
            Issue.record("expected .video routing for .mov")
        }
    }

    @Test("VisionService rejects unknown extension")
    func rejectsUnknown() async {
        let service = Self.makeService()
        let txtURL = URL(fileURLWithPath: "/tmp/not-a-valid-input.txt")
        await #expect(throws: VisionServiceError.self) {
            _ = try await service.embed(fileAt: txtURL)
        }
    }
}

// MARK: - Test companion

/// Shared setup used by VisionServiceTests + DINOv2VisualQCTests.
/// Kept outside the `DINOv2NodeTests` suite struct so both files can use it
/// without re-declaring the fixture helpers.
enum DINOv2NodeTests_Companion {

    static func makeDINOv2Node() -> DINOv2Node {
        DINOv2Node(
            nodeId: "dinov2-test",
            hostname: "test.local",
            backend: MockDINOv2Backend(),
            modelVersion: "dinov2-vits14-v1.0"
        )
    }

    static func makeVJEPANode() -> VJEPANode {
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

    /// Write a 64×64 solid-grey PNG to a tmp file and return its URL.
    static func writeSolidPNG(gray: UInt8) throws -> URL {
        let size = 64
        let bytesPerRow = size * 4
        var buffer = [UInt8](repeating: 0, count: size * bytesPerRow)
        for i in stride(from: 0, to: buffer.count, by: 4) {
            buffer[i + 0] = gray
            buffer[i + 1] = gray
            buffer[i + 2] = gray
            buffer[i + 3] = 255
        }
        let space = CGColorSpaceCreateDeviceRGB()
        let info: UInt32 = CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
        let provider = CGDataProvider(data: Data(buffer) as CFData)!
        guard let image = CGImage(
            width: size,
            height: size,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: bytesPerRow,
            space: space,
            bitmapInfo: CGBitmapInfo(rawValue: info),
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        ) else {
            throw VisionServiceError.unsupportedInput("could not create CGImage")
        }

        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("dinov2-test-\(UUID().uuidString).png")
        guard let dest = CGImageDestinationCreateWithURL(tmp as CFURL, "public.png" as CFString, 1, nil) else {
            throw VisionServiceError.unsupportedInput("could not create image destination")
        }
        CGImageDestinationAddImage(dest, image, nil)
        guard CGImageDestinationFinalize(dest) else {
            throw VisionServiceError.unsupportedInput("could not finalize image")
        }
        return tmp
    }
}

#endif
