// MARK: - VideoFramePreprocessor
// AVFoundation → uniform temporal sampling → 224×224 resize → ImageNet normalize.
//
// Pipeline (spec §5 Phase 1, BR-04 + BR-14):
//   1. Open the video with AVURLAsset.
//   2. Uniformly sample `targetFrameCount` timestamps across the full duration.
//   3. Use AVAssetImageGenerator (with tight tolerances) to pull CGImages.
//   4. Resize each frame to `targetSize × targetSize` on a CoreGraphics context.
//   5. Convert BGRA uint8 → RGB float32, divide by 255.
//   6. Apply ImageNet mean/std per channel.
//   7. Return a `VideoFrameTensor` with shape [T, 3, H, W] (CHW order).
//
// TDDP steps 5 (RED) + 6 (GREEN).

#if canImport(AVFoundation)
import AVFoundation
import CoreGraphics
import Foundation
import ImageIO

// MARK: - Tensor value type

/// Preprocessed video frames in CHW order, ready to feed the V-JEPA encoder.
public struct VideoFrameTensor: Sendable, Equatable {
    /// Number of sampled frames (`T`).
    public let frameCount: Int
    /// Channel count. Always 3 (RGB) post-preprocessing.
    public let channels: Int
    /// Spatial height in pixels.
    public let height: Int
    /// Spatial width in pixels.
    public let width: Int
    /// Row-major float32 tensor, layout `[T, C, H, W]`.
    public let data: [Float]

    public init(frameCount: Int, channels: Int, height: Int, width: Int, data: [Float]) {
        self.frameCount = frameCount
        self.channels = channels
        self.height = height
        self.width = width
        self.data = data
    }

    /// Element offset for `tensor[t, c, y, x]`.
    public func offset(t: Int, c: Int, y: Int, x: Int) -> Int {
        ((t * channels + c) * height + y) * width + x
    }
}

// MARK: - Errors

public enum VideoFramePreprocessorError: Error, Sendable, Equatable {
    case missingVideoTrack
    case frameGenerationFailed(timestamp: Double)
    case unsupportedFrameCount(Int)
}

// MARK: - Preprocessor

/// Uniform-sampling video preprocessor for V-JEPA inputs.
/// One per-channel RGB constant.
///
/// A named type rather than a `(r:g:b:)` tuple: three members is past
/// SwiftLint's `large_tuple` cap, and these are read positionally in the
/// normalisation loop where a transposed channel is silent.
public struct RGBTriple: Sendable, Equatable {
    public let r: Float
    public let g: Float
    public let b: Float

    public init(r: Float, g: Float, b: Float) {
        self.r = r
        self.g = g
        self.b = b
    }
}

public struct VideoFramePreprocessor: Sendable {

    /// Number of frames the encoder consumes. Spec BR-04: 8-16.
    public let targetFrameCount: Int
    /// Square output resolution. ViT-B/16 → 224.
    public let targetSize: Int

    // ImageNet RGB mean/std.
    public static let imageNetMean = RGBTriple(r: 0.485, g: 0.456, b: 0.406)
    public static let imageNetStd = RGBTriple(r: 0.229, g: 0.224, b: 0.225)

    public init(targetFrameCount: Int, targetSize: Int = 224) {
        self.targetFrameCount = targetFrameCount
        self.targetSize = targetSize
    }

    // MARK: Public API

    /// Decode, sample, resize and normalise a video file.
    ///
    /// - Parameter url: local file URL (e.g. from a screen recording).
    /// - Returns: `VideoFrameTensor` with shape `[targetFrameCount, 3, targetSize, targetSize]`.
    public func process(videoAt url: URL) async throws -> VideoFrameTensor {
        guard targetFrameCount > 0, targetFrameCount <= 64 else {
            throw VideoFramePreprocessorError.unsupportedFrameCount(targetFrameCount)
        }
        let asset = AVURLAsset(url: url)
        let duration = try await asset.load(.duration)
        let totalSeconds = CMTimeGetSeconds(duration)

        // Uniform sampling across (0, totalSeconds). Midpoints of `T` equal bins.
        let timestamps: [CMTime] = (0..<targetFrameCount).map { i in
            let ratio = (Double(i) + 0.5) / Double(targetFrameCount)
            return CMTime(seconds: max(0.0, totalSeconds * ratio), preferredTimescale: 600)
        }

        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero

        var data = [Float](repeating: 0, count: targetFrameCount * 3 * targetSize * targetSize)
        for (index, ts) in timestamps.enumerated() {
            let cgImage: CGImage
            do {
                cgImage = try await generator.image(at: ts).image
            } catch {
                throw VideoFramePreprocessorError.frameGenerationFailed(timestamp: CMTimeGetSeconds(ts))
            }
            try fillFrame(cgImage: cgImage, into: &data, frameIndex: index)
        }
        return VideoFrameTensor(
            frameCount: targetFrameCount,
            channels: 3,
            height: targetSize,
            width: targetSize,
            data: data
        )
    }

    // MARK: Internal

    private func fillFrame(cgImage: CGImage, into data: inout [Float], frameIndex: Int) throws {
        let bytesPerRow = targetSize * 4
        let byteCount = targetSize * bytesPerRow
        var buffer = [UInt8](repeating: 0, count: byteCount)

        try buffer.withUnsafeMutableBytes { raw in
            guard let base = raw.baseAddress else { return }
            let space = CGColorSpaceCreateDeviceRGB()
            // premultipliedLast → RGBA byte order, which we decode as R, G, B, A.
            let info: UInt32 = CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
            guard let ctx = CGContext(
                data: base,
                width: targetSize,
                height: targetSize,
                bitsPerComponent: 8,
                bytesPerRow: bytesPerRow,
                space: space,
                bitmapInfo: info
            ) else {
                throw VideoFramePreprocessorError.frameGenerationFailed(timestamp: -1)
            }
            ctx.interpolationQuality = .high
            let rect = CGRect(x: 0, y: 0, width: targetSize, height: targetSize)
            ctx.draw(cgImage, in: rect)
        }

        let mean = Self.imageNetMean
        let std = Self.imageNetStd
        let size = targetSize

        for y in 0..<size {
            for x in 0..<size {
                let pixel = (y * size + x) * 4
                let r = Float(buffer[pixel + 0]) / 255.0
                let g = Float(buffer[pixel + 1]) / 255.0
                let b = Float(buffer[pixel + 2]) / 255.0
                let rIdx = ((frameIndex * 3 + 0) * size + y) * size + x
                let gIdx = ((frameIndex * 3 + 1) * size + y) * size + x
                let bIdx = ((frameIndex * 3 + 2) * size + y) * size + x
                data[rIdx] = (r - mean.r) / std.r
                data[gIdx] = (g - mean.g) / std.g
                data[bIdx] = (b - mean.b) / std.b
            }
        }
    }
}
#endif
