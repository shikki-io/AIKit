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

    /// The repository and revision the ViT-S/14 blob is published from.
    ///
    /// Named here rather than spelled into a URL, so the route itself stays in
    /// `Endpoints/DINOv2HubEndPoint` — BR-AIL-16, enforced by
    /// `EndPointRouteTests`.
    private static let vits14RepositoryID = "shikki/dinov2-vits14-coreml"
    private static let vits14Revision = "main"
    private static let vits14FileName = "dinov2-vits14.mlpackage.zip"

    /// Canonical ViT-S/14 (21M) descriptor. Placeholder SHA + bytes until
    /// the .mlpackage mirror is published — update when the Phase 4
    /// conversion script produces a final artefact.
    ///
    /// OPTIONAL, and this is a deliberate narrowing of what used to be a
    /// non-optional `static let`. The URL is built by
    /// `DINOv2HubEndPoint.resolveFile`, whose `url` is optional because a URL
    /// assembled from a repository id can fail to form. The previous spelling
    /// hid that behind `URL(string:)!` — the exact force-unwrap the operator
    /// cited in AIKit#18 review r4500469694 against `HuggingFaceClient`. A
    /// caller that cannot proceed without the descriptor now has to say so,
    /// rather than the process trapping.
    ///
    /// For these three constant inputs the URL does form; the optionality is
    /// the TYPE being honest about the operation, not a hedge about this value.
    public static let vits14: DINOv2ModelDescriptor? = {
        guard
            let remoteURL = DINOv2HubEndPoint.resolveFile(
                id: vits14RepositoryID,
                revision: vits14Revision,
                filename: vits14FileName
            ).url
        else { return nil }
        return DINOv2ModelDescriptor(
            descriptor: ModelWeightLoader.ModelDescriptor(
                modelId: "dinov2-vits14",
                fileName: vits14FileName,
                remoteURL: remoteURL,
                expectedSHA256: String(repeating: "0", count: 64),
                expectedBytes: 0
            )
        )
    }()
}
