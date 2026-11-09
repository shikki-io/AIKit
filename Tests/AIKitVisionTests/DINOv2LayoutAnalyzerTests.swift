// MARK: - DINOv2LayoutAnalyzerTests
// Spec: features/shikki-vjepa-mlx-node.md §5 Phase 4 deliverable 6.
// Scaffolds the document-layout region analyzer — real attention-map
// decoding lands when the CoreML model ships. For now the analyzer uses
// a deterministic heuristic over the normalized image tensor so the
// downstream API surface is stable.

#if canImport(CoreGraphics)
import CoreGraphics
import Foundation
import Testing
@testable import AIKitVision

@Suite("DINOv2LayoutAnalyzer")
struct DINOv2LayoutAnalyzerTests {

    // MARK: - Helpers

    private static func makeStripedImage(regions: Int = 4) -> CGImage {
        let size = 64
        let bytesPerRow = size * 4
        var buffer = [UInt8](repeating: 0, count: size * bytesPerRow)
        let stripeHeight = max(1, size / regions)
        for y in 0..<size {
            let band = y / stripeHeight
            let gray = UInt8(40 + (band * 60) % 200)
            for x in 0..<size {
                let p = (y * size + x) * 4
                buffer[p + 0] = gray
                buffer[p + 1] = gray
                buffer[p + 2] = gray
                buffer[p + 3] = 255
            }
        }
        let space = CGColorSpaceCreateDeviceRGB()
        let info: UInt32 = CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
        let provider = CGDataProvider(data: Data(buffer) as CFData)!
        return CGImage(
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
        )!
    }

    // MARK: - Region detection contract

    @Test("layout analyzer returns non-empty regions for a multi-band image")
    func detectsRegions() throws {
        let analyzer = DINOv2LayoutAnalyzer()
        let image = Self.makeStripedImage(regions: 4)
        let layout = try analyzer.analyze(image: image)
        #expect(!layout.regions.isEmpty)
    }

    @Test("all region rects are inside normalized [0, 1] × [0, 1] space")
    func regionRectsNormalized() throws {
        let analyzer = DINOv2LayoutAnalyzer()
        let image = Self.makeStripedImage(regions: 4)
        let layout = try analyzer.analyze(image: image)
        for region in layout.regions {
            #expect(region.rect.minX >= 0)
            #expect(region.rect.minY >= 0)
            #expect(region.rect.maxX <= 1.0 + 1e-5)
            #expect(region.rect.maxY <= 1.0 + 1e-5)
        }
    }

    @Test("layout analyzer is deterministic for the same image")
    func deterministic() throws {
        let analyzer = DINOv2LayoutAnalyzer()
        let image = Self.makeStripedImage(regions: 4)
        let a = try analyzer.analyze(image: image)
        let b = try analyzer.analyze(image: image)
        #expect(a.regions == b.regions)
    }

    @Test("each region carries a known role label")
    func regionRoles() throws {
        let analyzer = DINOv2LayoutAnalyzer()
        let image = Self.makeStripedImage(regions: 4)
        let layout = try analyzer.analyze(image: image)
        let roles = Set(layout.regions.map(\.role))
        #expect(!roles.isEmpty)
        for role in roles {
            #expect(DINOv2LayoutRole.allCases.contains(role))
        }
    }
}

#endif
