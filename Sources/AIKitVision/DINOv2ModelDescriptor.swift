// MARK: - DINOv2ModelDescriptor
// Pinned model descriptors for the DINOv2 family — same pattern as the
// VJEPA descriptors used by `ModelWeightLoader`.
//
// Spec: features/shikki-vjepa-mlx-node.md §14.3 — DINOv2 ViT-S/14,
// 21M params, Apache-2.0, `~/.shikki/models/dinov2-vits14/` cache.
//
// `remoteURL`, `expectedSHA256`, and `expectedBytes` are placeholder
// sentinels until the real mirror lands. Tests inject their own
// descriptor (see `DINOv2ModelCacheTests`).

import Foundation

public struct DINOv2ModelDescriptor: Sendable {
    public let descriptor: ModelWeightLoader.ModelDescriptor

    public init(descriptor: ModelWeightLoader.ModelDescriptor) {
        self.descriptor = descriptor
    }

    /// Canonical ViT-S/14 (21M) descriptor. Placeholder SHA + URL until
    /// the .mlpackage mirror is published — update when the Phase 4
    /// conversion script produces a final artefact.
    public static let vits14 = DINOv2ModelDescriptor(
        descriptor: ModelWeightLoader.ModelDescriptor(
            modelId: "dinov2-vits14",
            fileName: "dinov2-vits14.mlpackage.zip",
            remoteURL: URL(string: "https://huggingface.co/shikki/dinov2-vits14-coreml/resolve/main/dinov2-vits14.mlpackage.zip")!,
            expectedSHA256: String(repeating: "0", count: 64),
            expectedBytes: 0
        )
    )
}
