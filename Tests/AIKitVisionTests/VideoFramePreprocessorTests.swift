// MARK: - VideoFramePreprocessorTests
// TDDP step 5 (RED) + step 6 (GREEN): uniform temporal sampling + resize + normalize.
//
// See features/shikki-vjepa-mlx-node.md §5 Phase 1, BR-04 + BR-14.

#if canImport(AVFoundation)
import AVFoundation
import CoreGraphics
import CoreVideo
import Foundation
import Testing
@testable import AIKitVision

@Suite("VideoFramePreprocessor")
struct VideoFramePreprocessorTests {

    // MARK: Helpers

    /// Render a synthetic 2-second, 30 fps, 320x240 MP4 into a tmp file.
    /// Each frame is a deterministic gradient — luminance varies with frame index.
    static func renderFixtureVideo(frameCount: Int = 60, size: Int = 240) async throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("vjepa-fixture-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let url = dir.appendingPathComponent("fixture.mov")

        let width = size
        let height = size
        let writer = try AVAssetWriter(outputURL: url, fileType: .mov)
        let settings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
        ]
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        input.expectsMediaDataInRealTime = false
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
        ]
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input,
            sourcePixelBufferAttributes: attrs
        )
        writer.add(input)
        writer.startWriting()
        writer.startSession(atSourceTime: .zero)

        for i in 0..<frameCount {
            // Poll the input until it is ready — simpler and deterministic than a callback queue.
            while !input.isReadyForMoreMediaData {
                try await Task.sleep(nanoseconds: 2_000_000)
            }
            var pixelBuffer: CVPixelBuffer?
            CVPixelBufferPoolCreatePixelBuffer(nil, adaptor.pixelBufferPool!, &pixelBuffer)
            guard let buf = pixelBuffer else { continue }
            CVPixelBufferLockBaseAddress(buf, [])
            let ptr = CVPixelBufferGetBaseAddress(buf)!
            let bpr = CVPixelBufferGetBytesPerRow(buf)
            let luminance = UInt8((i * 4) % 255)
            for y in 0..<height {
                let rowPtr = ptr.advanced(by: y * bpr).assumingMemoryBound(to: UInt8.self)
                for x in 0..<width {
                    rowPtr[4 * x + 0] = UInt8((x &* 3) & 0xFF)   // B
                    rowPtr[4 * x + 1] = luminance                 // G
                    rowPtr[4 * x + 2] = UInt8((y &* 5) & 0xFF)   // R
                    rowPtr[4 * x + 3] = 255                       // A
                }
            }
            CVPixelBufferUnlockBaseAddress(buf, [])
            let pts = CMTime(value: CMTimeValue(i), timescale: 30)
            adaptor.append(buf, withPresentationTime: pts)
        }
        input.markAsFinished()
        // Awaited, not blocked on: a DispatchSemaphore here parks a cooperative
        // thread, and with the suite's parallel tests the pool runs out of
        // threads and the run hangs at 0% CPU.
        await writer.finishWriting()
        return url
    }

    // MARK: Frame count

    @Test("preprocessor extracts exactly 16 frames from a 60-frame fixture")
    func test_uniformSamplingCountIs16() async throws {
        let url = try await Self.renderFixtureVideo(frameCount: 60)
        let preproc = VideoFramePreprocessor(targetFrameCount: 16, targetSize: 224)
        let tensor = try await preproc.process(videoAt: url)
        #expect(tensor.frameCount == 16)
    }

    @Test("preprocessor extracts exactly 8 frames when configured for 8")
    func test_uniformSamplingCountIs8() async throws {
        let url = try await Self.renderFixtureVideo(frameCount: 30)
        let preproc = VideoFramePreprocessor(targetFrameCount: 8, targetSize: 224)
        let tensor = try await preproc.process(videoAt: url)
        #expect(tensor.frameCount == 8)
    }

    // MARK: Output shape

    @Test("preprocessed tensor has shape [T, 3, 224, 224]")
    func test_outputShapeIs224() async throws {
        let url = try await Self.renderFixtureVideo(frameCount: 16)
        let preproc = VideoFramePreprocessor(targetFrameCount: 8, targetSize: 224)
        let tensor = try await preproc.process(videoAt: url)
        #expect(tensor.frameCount == 8)
        #expect(tensor.channels == 3)
        #expect(tensor.height == 224)
        #expect(tensor.width == 224)
        let expectedElements = 8 * 3 * 224 * 224
        #expect(tensor.data.count == expectedElements)
    }

    // MARK: ImageNet normalisation

    @Test("values are ImageNet-normalised — within plausible range [-3, 3]")
    func test_imageNetNormalization() async throws {
        let url = try await Self.renderFixtureVideo(frameCount: 16)
        let preproc = VideoFramePreprocessor(targetFrameCount: 8, targetSize: 224)
        let tensor = try await preproc.process(videoAt: url)
        let values = tensor.data
        let minVal = values.min() ?? 0
        let maxVal = values.max() ?? 0
        // ImageNet-normalised uint8 inputs land in [-2.12, 2.64]. Add slack.
        #expect(minVal >= -3.0)
        #expect(maxVal <= 3.0)
    }
}
#endif
