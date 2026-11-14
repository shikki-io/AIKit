// MARK: - VJEPAEncoder
// V-JEPA 2 ViT-B/16 encoder — structural model + forward pipeline.
//
// This file defines the structural representation of the V-JEPA 2 ViT-B
// encoder (12 transformer blocks, 768 hidden, 12 heads, 3D patch embed).
// The tensor compute itself is routed through a `VisionBackend` protocol
// so that the production MLX backend (`mlx-swift`) and a pure-Swift
// reference backend (used in CI / tests where MLX is unavailable) can
// both slot in without changing the surface.
//
// See: features/shikki-vjepa-mlx-node.md §4 + §5 Phase 1
// TDDP steps: 1 (RED) + 2 (GREEN), 7 (RED) + 8 (GREEN), 9 (RED) + 10 (GREEN).

import Foundation

// MARK: - Config

/// Architectural configuration for a V-JEPA encoder variant.
///
/// Only `vitB16` is in scope for Phase 1 (BR-02). Larger variants
/// (ViT-L, ViT-H) are reserved for the future distributed inference
/// path (`shikki-vision-cluster-inference.md`) and MUST NOT be
/// instantiated here.
public struct VJEPAEncoderConfig: Sendable, Equatable {
    /// Transformer depth. ViT-B = 12.
    public let numLayers: Int
    /// Hidden dimension (embedding width). ViT-B = 768.
    public let hiddenDim: Int
    /// Attention head count. ViT-B = 12 (head dim = 64).
    public let numHeads: Int
    /// MLP expansion ratio (standard ViT = 4).
    public let mlpRatio: Int
    /// Spatial patch size in pixels (ViT-B/16 = 16).
    public let patchSize: Int
    /// Temporal patch size in frames (V-JEPA 2 default = 2).
    public let temporalPatchSize: Int
    /// Square input resolution per frame.
    public let imageSize: Int
    /// Post-norm layer norm epsilon.
    public let layerNormEps: Float

    public init(
        numLayers: Int,
        hiddenDim: Int,
        numHeads: Int,
        mlpRatio: Int,
        patchSize: Int,
        temporalPatchSize: Int,
        imageSize: Int,
        layerNormEps: Float
    ) {
        self.numLayers = numLayers
        self.hiddenDim = hiddenDim
        self.numHeads = numHeads
        self.mlpRatio = mlpRatio
        self.patchSize = patchSize
        self.temporalPatchSize = temporalPatchSize
        self.imageSize = imageSize
        self.layerNormEps = layerNormEps
    }

    /// ViT-B/16 — the target V-JEPA 2 variant for Phase 1 (BR-02).
    public static let vitB16 = VJEPAEncoderConfig(
        numLayers: 12,
        hiddenDim: 768,
        numHeads: 12,
        mlpRatio: 4,
        patchSize: 16,
        temporalPatchSize: 2,
        imageSize: 224,
        layerNormEps: 1e-6
    )

    /// Head dimension — hidden / heads.
    public var headDim: Int { hiddenDim / numHeads }

    /// Patches per frame along each spatial axis.
    public var spatialPatchGrid: Int { imageSize / patchSize }

    /// Total spatial patches per frame.
    public var patchesPerFrame: Int { spatialPatchGrid * spatialPatchGrid }
}

// MARK: - Transformer block

/// A single transformer block (pre-norm, attn + MLP) inside V-JEPA.
///
/// The block is an immutable description of layer parameters; the
/// forward pass is delegated to the injected `VisionBackend`.
public struct TransformerBlock: Sendable, Equatable {
    public let hiddenDim: Int
    public let numHeads: Int
    public let mlpHiddenDim: Int
    public let layerIndex: Int

    public init(hiddenDim: Int, numHeads: Int, mlpRatio: Int, layerIndex: Int) {
        self.hiddenDim = hiddenDim
        self.numHeads = numHeads
        self.mlpHiddenDim = hiddenDim * mlpRatio
        self.layerIndex = layerIndex
    }

    /// Head dimension (used for RoPE frequency tables + attention reshape).
    public var headDim: Int { hiddenDim / numHeads }
}

// MARK: - Encoder

/// Structural V-JEPA encoder — 3D patch embedding + N transformer blocks + final layer norm.
public struct VJEPAEncoder: Sendable {

    public typealias Config = VJEPAEncoderConfig

    public let config: Config
    public let blocks: [TransformerBlock]

    public init(config: Config) {
        self.config = config
        self.blocks = (0..<config.numLayers).map { i in
            TransformerBlock(
                hiddenDim: config.hiddenDim,
                numHeads: config.numHeads,
                mlpRatio: config.mlpRatio,
                layerIndex: i
            )
        }
    }
}
