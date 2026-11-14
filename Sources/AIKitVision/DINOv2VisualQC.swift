// MARK: - DINOv2VisualQC
// Spec: features/shikki-vjepa-mlx-node.md §5 Phase 4 deliverable 4 +
// Scenario 6 (Kagami visual QC with DINOv2).
//
// Compares two screenshots via DINOv2 cosine similarity and exposes a
// pass/fail result + regression confidence for @Kagami's CI gate.
//
// Gate (spec §5 Phase 4):
//   "Kagami visual QC detects intentional UI regression (moved button)
//    with > 0.9 confidence"
//
// The default similarity threshold is 0.95 (spec §9 open question 5 and
// the stated scenario). `regressionConfidence` is a monotone mapping of
// `threshold - similarity` into `[0, 1]` so Kagami can expose a single
// knob to agents.

#if canImport(CoreGraphics)
import CoreGraphics
import Foundation

// MARK: - Result

public struct DINOv2VisualQCResult: Sendable, Equatable {
    /// Cosine similarity of the two embeddings, in `[-1, 1]` (typically `[0, 1]`
    /// for L2-normalized outputs).
    public let similarity: Float
    /// Threshold applied for the pass/fail decision.
    public let threshold: Float
    /// `true` iff `similarity >= threshold`.
    public let passed: Bool
    /// Confidence that a regression occurred, in `[0, 1]`. Zero when identical,
    /// saturates to ~1 as similarity drops well below threshold.
    public let regressionConfidence: Float

    public init(similarity: Float, threshold: Float, passed: Bool, regressionConfidence: Float) {
        self.similarity = similarity
        self.threshold = threshold
        self.passed = passed
        self.regressionConfidence = regressionConfidence
    }
}

// MARK: - Service

/// Visual QC via DINOv2 embeddings — compares two images, reports a
/// pass/fail + confidence.
public struct DINOv2VisualQC: Sendable {

    public let node: DINOv2Node
    public static let defaultThreshold: Float = 0.95

    public init(node: DINOv2Node) {
        self.node = node
    }

    /// Embed both images, compute cosine similarity, apply threshold.
    public func compare(
        current: CGImage,
        reference: CGImage,
        threshold: Float = DINOv2VisualQC.defaultThreshold
    ) async throws -> DINOv2VisualQCResult {
        let currentEmbedding = try node.embed(image: current)
        let referenceEmbedding = try node.embed(image: reference)
        let sim = Self.cosineSimilarity(currentEmbedding.data, referenceEmbedding.data)
        let passed = sim >= threshold
        // Regression confidence: linearly maps `(threshold - similarity)` into [0, 1],
        // scaled so a drop of `1 - threshold` below threshold saturates near 1.
        let delta = threshold - sim
        let denom = max(1 - threshold, 1e-3)
        let raw = delta / Float(denom)
        let confidence = max(0, min(1, raw))
        return DINOv2VisualQCResult(
            similarity: sim,
            threshold: threshold,
            passed: passed,
            regressionConfidence: confidence
        )
    }

    // MARK: Math

    /// Dot-product cosine similarity. Inputs do not have to be unit-norm —
    /// the divisor normalizes. Returns 0 when either vector is zero.
    public static func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count, !a.isEmpty else { return 0 }
        var dot: Float = 0
        var na: Float = 0
        var nb: Float = 0
        for i in 0..<a.count {
            dot += a[i] * b[i]
            na += a[i] * a[i]
            nb += b[i] * b[i]
        }
        let denom = sqrtf(na) * sqrtf(nb)
        guard denom > 1e-12 else { return 0 }
        return dot / denom
    }
}

#endif
