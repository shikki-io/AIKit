// MARK: - RoPE3D
// 3D Rotary Position Embedding for V-JEPA 2 ViT-B/16.
//
// V-JEPA 2 uses an axially-factorised 3D RoPE. Each attention head's
// `headDim` is split into three axis groups (x spatial, y spatial,
// t temporal). Inside each group, frequency bands follow the standard
// RoPE inverse-frequency schedule:
//
//     inv_freq[k] = 1 / base^(2k / axisDim)
//     theta(pos, k) = pos * inv_freq[k]
//     rotate((a, b)) = (a·cos θ − b·sin θ, a·sin θ + b·cos θ)
//
// For ViT-B headDim=64, splitting 3-ways we use axisDim=20 (10 rotation
// pairs) per axis, leaving 4 trailing dims unrotated (parity with the
// PyTorch reference impl and what the safetensors checkpoint expects).
//
// See features/shikki-vjepa-mlx-node.md §4, risk register §12 (RoPE must
// match PyTorch reference). TDDP steps 3 (RED) + 4 (GREEN).

import Foundation

/// Axially-factorised 3D Rotary Position Embedding.
///
/// Builds cosine/sine tables for every (x, y, t) position inside a bounded
/// grid. Exposes an `angles(...)` query returning the concatenated cos/sin
/// pair vectors expected by the attention kernel, and in-place
/// `apply(...)` helpers that rotate per-head query/key vectors.
public struct RoPE3D: Sendable {

    // MARK: Stored properties

    /// Full attention head dimension (e.g. 64 for ViT-B).
    public let headDim: Int
    /// Per-axis rotation-pair count. `axisPairs * 2 * 3` must be ≤ headDim.
    public let axisPairs: Int
    /// Frequency base used in `1 / base^(2k/axisDim)`. Standard RoPE = 10000.
    public let base: Float

    /// Precomputed cos/sin tables per axis, addressed by position then pair index.
    /// `xCos[pos][pair]` ← cos(pos * inv_freq[pair]) for the x axis, etc.
    private let xCos: [[Float]]
    private let xSin: [[Float]]
    private let yCos: [[Float]]
    private let ySin: [[Float]]
    private let tCos: [[Float]]
    private let tSin: [[Float]]

    // MARK: Init

    public init(headDim: Int, maxX: Int, maxY: Int, maxT: Int, base: Float = 10_000.0) {
        precondition(headDim % 2 == 0, "headDim must be even for RoPE")
        self.headDim = headDim
        self.base = base
        // Split headDim/2 rotation pairs across 3 axes as evenly as possible.
        // The remainder (pairs not assigned to any axis) is left unrotated.
        let totalPairs = headDim / 2
        self.axisPairs = totalPairs / 3

        let xTables = RoPE3D.buildTable(axisPairs: axisPairs, maxPos: maxX, base: base)
        let yTables = RoPE3D.buildTable(axisPairs: axisPairs, maxPos: maxY, base: base)
        let tTables = RoPE3D.buildTable(axisPairs: axisPairs, maxPos: maxT, base: base)
        self.xCos = xTables.cos
        self.xSin = xTables.sin
        self.yCos = yTables.cos
        self.ySin = yTables.sin
        self.tCos = tTables.cos
        self.tSin = tTables.sin
    }

    // MARK: Table builder

    private static func buildTable(
        axisPairs: Int,
        maxPos: Int,
        base: Float
    ) -> (cos: [[Float]], sin: [[Float]]) {
        // inv_freq[k] = 1 / base ^ (2k / axisDim) where axisDim = axisPairs * 2.
        let axisDim = max(1, axisPairs * 2)
        var invFreq = [Float](repeating: 0, count: axisPairs)
        for k in 0..<axisPairs {
            let exponent = Float(2 * k) / Float(axisDim)
            invFreq[k] = 1.0 / Foundation.pow(base, exponent)
        }
        var cosTable = [[Float]](repeating: [Float](repeating: 1, count: axisPairs), count: max(1, maxPos))
        var sinTable = [[Float]](repeating: [Float](repeating: 0, count: axisPairs), count: max(1, maxPos))
        for pos in 0..<max(1, maxPos) {
            for k in 0..<axisPairs {
                let theta = Float(pos) * invFreq[k]
                cosTable[pos][k] = Foundation.cos(theta)
                sinTable[pos][k] = Foundation.sin(theta)
            }
        }
        return (cosTable, sinTable)
    }

    // MARK: Public query

    /// Return the cos/sin vectors used to rotate a head's (Q or K) vector at
    /// the given (x, y, t) patch position.
    ///
    /// The returned arrays have length `headDim / 2`: the first `axisPairs`
    /// entries correspond to the x axis, next `axisPairs` to y, next to t,
    /// and any remainder is filled with (cos=1, sin=0) → no rotation.
    public func angles(x: Int, y: Int, t: Int) -> (cos: [Float], sin: [Float]) {
        let total = headDim / 2
        var cos = [Float](repeating: 1.0, count: total)
        var sin = [Float](repeating: 0.0, count: total)
        let safeX = min(max(x, 0), xCos.count - 1)
        let safeY = min(max(y, 0), yCos.count - 1)
        let safeT = min(max(t, 0), tCos.count - 1)

        for k in 0..<axisPairs {
            cos[k] = xCos[safeX][k]
            sin[k] = xSin[safeX][k]
        }
        for k in 0..<axisPairs {
            cos[axisPairs + k] = yCos[safeY][k]
            sin[axisPairs + k] = ySin[safeY][k]
        }
        for k in 0..<axisPairs {
            cos[2 * axisPairs + k] = tCos[safeT][k]
            sin[2 * axisPairs + k] = tSin[safeT][k]
        }
        // Trailing entries beyond 3*axisPairs remain (cos=1, sin=0).
        return (cos, sin)
    }

    // MARK: Rotation

    /// Apply the RoPE rotation in place to a per-head vector of length `headDim`.
    ///
    /// Pairs `(vector[2k], vector[2k+1])` are rotated by the angle encoded in
    /// `angles(x, y, t)`.
    public func apply(inPlace vector: inout [Float], x: Int, y: Int, t: Int) {
        let (cos, sin) = angles(x: x, y: y, t: t)
        rotate(&vector, cos: cos, sin: sin, invert: false)
    }

    /// Inverse rotation — used for the round-trip sanity check and, eventually,
    /// to recover positions from rotated states (debugging aid).
    public func applyInverse(inPlace vector: inout [Float], x: Int, y: Int, t: Int) {
        let (cos, sin) = angles(x: x, y: y, t: t)
        rotate(&vector, cos: cos, sin: sin, invert: true)
    }

    private func rotate(_ vector: inout [Float], cos: [Float], sin: [Float], invert: Bool) {
        precondition(vector.count == headDim, "vector length must match headDim")
        let sign: Float = invert ? -1.0 : 1.0
        for k in 0..<(headDim / 2) {
            let a = vector[2 * k]
            let b = vector[2 * k + 1]
            let c = cos[k]
            let s = sin[k] * sign
            vector[2 * k] = a * c - b * s
            vector[2 * k + 1] = a * s + b * c
        }
    }
}
