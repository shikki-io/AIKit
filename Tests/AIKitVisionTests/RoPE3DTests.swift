// MARK: - RoPE3DTests
// TDDP step 3 (RED) + step 4 (GREEN): 3D Rotary Position Embedding.
//
// See features/shikki-vjepa-mlx-node.md §4 (RoPE 3D extension).

import Foundation
import Testing
@testable import AIKitVision

@Suite("RoPE3D")
struct RoPE3DTests {

    // MARK: Base case (x=0, y=0, t=0) — rotation identity

    @Test("rotation at (x=0, y=0, t=0) is the identity transform")
    func test_zeroPositionIsIdentity() {
        let rope = RoPE3D(headDim: 64, maxX: 14, maxY: 14, maxT: 8)
        let (cos, sin) = rope.angles(x: 0, y: 0, t: 0)
        // cos(0) = 1, sin(0) = 0 across all frequency bands.
        for value in cos {
            #expect(abs(value - 1.0) < 1e-6)
        }
        for value in sin {
            #expect(abs(value) < 1e-6)
        }
    }

    // MARK: Shape + split across axes

    @Test("angles length equals headDim / 2 per axis concatenated (x + y + t)")
    func test_angleVectorShape() {
        let rope = RoPE3D(headDim: 64, maxX: 14, maxY: 14, maxT: 8)
        let (cos, sin) = rope.angles(x: 3, y: 4, t: 2)
        // Head dim is split into 3 axes, each contributing headDim/6 frequency
        // bands with a cos/sin pair. For headDim=64 → 64/6 = 10 bands per axis
        // with 4 leftover dims. The axis bands cover 60 of the 64; the trailing
        // 4 remain unrotated (baked into the build). The returned cos/sin cover
        // the full headDim/2 rotation pairs: 32.
        #expect(cos.count == 32)
        #expect(sin.count == 32)
    }

    // MARK: Reference value — sanity of frequency base

    @Test("x=1 on lowest frequency equals cos(1) / sin(1)")
    func test_xAxisLowestFrequency() {
        let rope = RoPE3D(headDim: 64, maxX: 14, maxY: 14, maxT: 8)
        let (cos, sin) = rope.angles(x: 1, y: 0, t: 0)
        // Lowest frequency for the x axis is 1.0 (theta = pos * 1.0^0 = 1).
        // i.e. the FIRST x-axis pair must be exactly cos(1), sin(1).
        let expectedCos = Foundation.cos(Float(1.0))
        let expectedSin = Foundation.sin(Float(1.0))
        #expect(abs(cos[0] - expectedCos) < 1e-5)
        #expect(abs(sin[0] - expectedSin) < 1e-5)
    }

    // MARK: Rotation applied to a 2D vector round-trips

    @Test("apply + apply-inverse on a patch vector is the identity (numerically)")
    func test_applyAndInverseRoundTrip() {
        let rope = RoPE3D(headDim: 64, maxX: 14, maxY: 14, maxT: 8)
        // Random-ish deterministic patch vector.
        var vector: [Float] = (0..<64).map { Float(sin(Double($0) * 0.137)) }
        let original = vector
        rope.apply(inPlace: &vector, x: 5, y: 7, t: 3)
        rope.applyInverse(inPlace: &vector, x: 5, y: 7, t: 3)
        for i in 0..<64 {
            #expect(abs(vector[i] - original[i]) < 1e-4)
        }
    }
}
