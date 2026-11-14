// MARK: - DINOv2LayoutAnalyzer
// Spec: features/shikki-vjepa-mlx-node.md §5 Phase 4 deliverable 6.
//
// Scaffold for document-layout analysis via DINOv2 attention maps. The
// real implementation will extract per-patch attention weights from the
// CoreML model, cluster them, and label regions (header / body /
// sidebar / code block / footer). That requires the real CoreML model
// (attention-head outputs exposed by the export) — not yet available.
//
// For now the analyzer ships a deterministic heuristic over vertical
// luminance bands so the API surface is stable, tests cover the
// contract, and downstream layout consumers can wire in without
// waiting on the CoreML gap.

#if canImport(CoreGraphics)
import CoreGraphics
import Foundation

// MARK: - Role taxonomy

public enum DINOv2LayoutRole: String, Codable, Sendable, CaseIterable, Equatable {
    case header
    case body
    case sidebar
    case codeBlock
    case footer
    case unknown
}

// MARK: - Geometry

/// Normalized axis-aligned rect in `[0, 1]² `, origin top-left.
public struct DINOv2Rect: Sendable, Equatable {
    public let minX: Float
    public let minY: Float
    public let maxX: Float
    public let maxY: Float

    public init(minX: Float, minY: Float, maxX: Float, maxY: Float) {
        self.minX = minX
        self.minY = minY
        self.maxX = maxX
        self.maxY = maxY
    }
}

// MARK: - Region

public struct DINOv2LayoutRegion: Sendable, Equatable {
    public let role: DINOv2LayoutRole
    public let rect: DINOv2Rect
    /// Attention / confidence score in `[0, 1]`. Real models produce this
    /// from the CLS attention map; the heuristic returns a band-wise
    /// luminance delta.
    public let confidence: Float

    public init(role: DINOv2LayoutRole, rect: DINOv2Rect, confidence: Float) {
        self.role = role
        self.rect = rect
        self.confidence = confidence
    }
}

// MARK: - Layout

public struct DINOv2Layout: Sendable, Equatable {
    public let regions: [DINOv2LayoutRegion]

    public init(regions: [DINOv2LayoutRegion]) {
        self.regions = regions
    }
}

// MARK: - Analyzer

/// Deterministic document-layout analyzer — heuristic scaffold for the
/// real CoreML attention-map path.
///
/// Algorithm (heuristic, v1):
///   1. downsample image to `targetSize × targetSize`
///   2. compute per-row mean luminance
///   3. group contiguous rows with similar luminance into bands
///   4. label first band `.header`, last band `.footer`, interior bands
///      alternating `.body` / `.sidebar`, darker bands → `.codeBlock`
///
/// When the CoreML .mlpackage ships attention heads, this implementation
/// is swapped for a real attention-map decoder behind the same
/// `analyze(image:)` signature — callers do not change.
public struct DINOv2LayoutAnalyzer: Sendable {

    public let targetSize: Int
    public let minBandHeight: Int

    public init(targetSize: Int = 64, minBandHeight: Int = 4) {
        self.targetSize = targetSize
        self.minBandHeight = minBandHeight
    }

    public func analyze(image: CGImage) throws -> DINOv2Layout {
        let rows = try Self.rowLuminance(image: image, size: targetSize)
        guard !rows.isEmpty else { return DINOv2Layout(regions: []) }

        // Group contiguous rows within 0.08 luminance of each other.
        var bands: [LuminanceBand] = []
        var bandStart = 0
        var accum: Float = rows[0]
        var count = 1
        for i in 1..<rows.count {
            let runningMean = accum / Float(count)
            if abs(rows[i] - runningMean) < 0.08 {
                accum += rows[i]
                count += 1
            } else {
                bands.append(LuminanceBand(start: bandStart, end: i - 1, mean: accum / Float(count)))
                bandStart = i
                accum = rows[i]
                count = 1
            }
        }
        bands.append(LuminanceBand(start: bandStart, end: rows.count - 1, mean: accum / Float(count)))

        // Drop bands thinner than `minBandHeight`.
        let filtered = bands.filter { ($0.end - $0.start + 1) >= minBandHeight }
        let usable = filtered.isEmpty ? bands : filtered

        // Global luminance range for "darker ⇒ .codeBlock" hint.
        let means = usable.map(\.mean)
        let minL = means.min() ?? 0
        let maxL = means.max() ?? 1
        let range = max(maxL - minL, 1e-3)

        var regions: [DINOv2LayoutRegion] = []
        for (idx, band) in usable.enumerated() {
            let role: DINOv2LayoutRole
            if idx == 0 {
                role = .header
            } else if idx == usable.count - 1 && usable.count > 2 {
                role = .footer
            } else if (band.mean - minL) / range < 0.25 {
                role = .codeBlock
            } else if idx % 2 == 0 {
                role = .sidebar
            } else {
                role = .body
            }
            let rect = DINOv2Rect(
                minX: 0,
                minY: Float(band.start) / Float(rows.count),
                maxX: 1.0,
                maxY: Float(band.end + 1) / Float(rows.count)
            )
            let confidence = min(1.0, Float(band.end - band.start + 1) / Float(rows.count) + 0.1)
            regions.append(DINOv2LayoutRegion(role: role, rect: rect, confidence: confidence))
        }
        return DINOv2Layout(regions: regions)
    }

    // MARK: Luminance extraction

    static func rowLuminance(image: CGImage, size: Int) throws -> [Float] {
        let bytesPerRow = size * 4
        var buffer = [UInt8](repeating: 0, count: size * bytesPerRow)
        let space = CGColorSpaceCreateDeviceRGB()
        let info: UInt32 = CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
        try buffer.withUnsafeMutableBytes { raw in
            guard let base = raw.baseAddress, let ctx = CGContext(
                data: base,
                width: size,
                height: size,
                bitsPerComponent: 8,
                bytesPerRow: bytesPerRow,
                space: space,
                bitmapInfo: info
            ) else {
                throw DINOv2Error.preprocessingFailed
            }
            ctx.interpolationQuality = .high
            ctx.draw(image, in: CGRect(x: 0, y: 0, width: size, height: size))
        }
        var rows = [Float](repeating: 0, count: size)
        for y in 0..<size {
            var acc: Float = 0
            for x in 0..<size {
                let p = (y * size + x) * 4
                let r = Float(buffer[p + 0]) / 255.0
                let g = Float(buffer[p + 1]) / 255.0
                let b = Float(buffer[p + 2]) / 255.0
                // Rec. 709 luminance.
                acc += 0.2126 * r + 0.7152 * g + 0.0722 * b
            }
            rows[y] = acc / Float(size)
        }
        return rows
    }
}

#endif

/// A contiguous run of rows within one luminance threshold of each other.
///
/// A named type rather than the `(start:end:mean:)` tuple: three members is
/// past SwiftLint's `large_tuple` cap, and `start`/`end` are both `Int`.
private struct LuminanceBand {
    let start: Int
    let end: Int
    let mean: Float
}
