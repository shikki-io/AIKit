// MARK: - DINOv2VisualQCTests
// Spec: features/shikki-vjepa-mlx-node.md §5 Phase 4 deliverable 4 +
// Scenario 6 (Kagami visual QC with DINOv2).
//
// Gate: "Kagami visual QC detects intentional UI regression (moved button)
// with > 0.9 confidence" (§5 Phase 4).

#if canImport(CoreGraphics)
import CoreGraphics
import Foundation
import Testing
@testable import AIKitVision

@Suite("DINOv2VisualQC")
struct DINOv2VisualQCTests {

    // MARK: - Fixtures

    /// Render a 64×64 "UI mock" with a solid background and a filled square
    /// "button" at a given top-left pixel. Two mocks with the same button
    /// position are visually identical; mocks with the button at different
    /// positions simulate the "moved button" regression from spec Scenario 6.
    private static func renderMockUI(
        buttonOrigin: (x: Int, y: Int),
        buttonSize: Int = 16,
        bgGray: UInt8 = 220,
        buttonGray: UInt8 = 40
    ) -> CGImage {
        let size = 64
        let bytesPerRow = size * 4
        var buffer = [UInt8](repeating: 0, count: size * bytesPerRow)
        for y in 0..<size {
            for x in 0..<size {
                let pixel = (y * size + x) * 4
                let inButton = x >= buttonOrigin.x
                    && x < buttonOrigin.x + buttonSize
                    && y >= buttonOrigin.y
                    && y < buttonOrigin.y + buttonSize
                let g = inButton ? buttonGray : bgGray
                buffer[pixel + 0] = g
                buffer[pixel + 1] = g
                buffer[pixel + 2] = g
                buffer[pixel + 3] = 255
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

    private static func makeQC() -> DINOv2VisualQC {
        DINOv2VisualQC(node: DINOv2NodeTests_Companion.makeDINOv2Node())
    }

    // MARK: - Cosine sim math

    @Test("cosineSimilarity of identical vectors is 1.0")
    func cosineIdentity() {
        let v: [Float] = [0.1, 0.2, 0.3, 0.4]
        #expect(abs(DINOv2VisualQC.cosineSimilarity(v, v) - 1.0) < 1e-6)
    }

    @Test("cosineSimilarity of orthogonal vectors is 0.0")
    func cosineOrthogonal() {
        let a: [Float] = [1, 0, 0, 0]
        let b: [Float] = [0, 1, 0, 0]
        #expect(abs(DINOv2VisualQC.cosineSimilarity(a, b)) < 1e-6)
    }

    // MARK: - QC contract

    @Test("identical screenshots pass visual QC (similarity ≥ threshold)")
    func identicalPass() async throws {
        let qc = Self.makeQC()
        let a = Self.renderMockUI(buttonOrigin: (x: 10, y: 10))
        let b = Self.renderMockUI(buttonOrigin: (x: 10, y: 10))
        let result = try await qc.compare(current: a, reference: b)
        #expect(result.similarity >= DINOv2VisualQC.defaultThreshold)
        #expect(result.passed == true)
        #expect(result.regressionConfidence < 0.5)
    }

    // MARK: - Gate: "moved button" regression > 0.9 confidence

    @Test("moved-button regression is detected with > 0.9 confidence (gate)")
    func movedButtonDetected() async throws {
        let qc = Self.makeQC()
        let reference = Self.renderMockUI(buttonOrigin: (x: 10, y: 10))
        let moved = Self.renderMockUI(buttonOrigin: (x: 40, y: 40))
        let result = try await qc.compare(current: moved, reference: reference)
        #expect(result.passed == false)
        #expect(result.regressionConfidence > 0.9, "gate: > 0.9 confidence, got \(result.regressionConfidence)")
    }

    @Test("compare result carries the computed similarity + threshold used")
    func resultCarriesContext() async throws {
        let qc = Self.makeQC()
        let reference = Self.renderMockUI(buttonOrigin: (x: 10, y: 10))
        let moved = Self.renderMockUI(buttonOrigin: (x: 40, y: 40))
        let result = try await qc.compare(current: moved, reference: reference, threshold: 0.95)
        #expect(result.threshold == 0.95)
        #expect(result.similarity >= -1.0 && result.similarity <= 1.0)
    }
}

#endif
