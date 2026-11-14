// DINOv2HubEndPoint — the one place AIKitVision spells a HuggingFace Hub route.
//
// WHY THIS FILE EXISTS, since a second copy of a route table is normally the
// defect and not the fix.
//
// `DINOv2ModelDescriptor` used to build its download URL by interpolation and
// force-unwrap it:
//
//     URL(string: "https://huggingface.co/shikki/dinov2-vits14-coreml/resolve/main/…")!
//
// That is the exact pattern the operator already ruled on once, in AIKit#18
// review r4500469694 ("Why this is not part of the hugginface endpoint?"),
// against `HuggingFaceClient.downloadURL`. The answer there was BR-AIL-16:
// route strings live in ONE place, inside an `Endpoints/` directory, and
// `EndPointRouteTests` enforces it by scanning `Sources/` for `/resolve/`,
// `"/v1` and `"/api/v1` outside such a directory. The rebase of this branch
// onto the epic is what first ran that ratchet over this module, and it caught
// this line.
//
// The obvious fix — reuse `AIKit`'s `HuggingFaceEndPoint.resolveFile` — is NOT
// available, and the reason is a boundary worth stating rather than
// rediscovering: that enum is `internal` to `AIKit`. Reaching it from here
// would mean making it `public`, which puts a provider-side route table into
// AIKit's exported surface — the precise thing BR-AIL-21's allowlist exists to
// refuse ("no provider API may be in it"), and `AIKitVision` deliberately does
// not depend on `AIKit` at all (Package.swift: it keeps `AIKit` MLX-free and
// lets a vision-only app link neither mlx-swift-lm nor swift-transformers).
//
// So the route gets ONE owner on this side of the boundary. The duplication is
// a module boundary, not a copy-paste: two modules that may not share a type
// each name the route once, and each is scanned by the same ratchet.
//
// If `AIKitVision` ever does depend on `AIKit`, this file is what to delete.

import Foundation

/// The HuggingFace Hub routes AIKitVision reaches.
///
/// | Route                                   | Method | Host path    |
/// |-----------------------------------------|--------|--------------|
/// | `/{id}/resolve/{revision}/{filename}`   | GET    | at the root  |
///
/// Served from the host ROOT and not from `/api`, which is why there is no
/// shared API prefix here — the same per-case reasoning AIKit's own table
/// carries.
public enum DINOv2HubEndPoint: Sendable {
    /// `GET /{id}/resolve/{revision}/{filename}` — the bytes of one file.
    case resolveFile(id: String, revision: String, filename: String)

    private static let scheme = "https"
    private static let host = "huggingface.co"

    /// The path components of this route, already split — never a string with
    /// separators in it, so no call site constructs a path by interpolation.
    private var pathComponents: [String] {
        switch self {
        case .resolveFile(let id, let revision, let filename):
            // `id` is a repo id and carries its own separator (`owner/name`),
            // which is why it is split rather than appended whole.
            id.split(separator: "/").map(String.init) + ["resolve", revision, filename]
        }
    }

    /// The absolute URL for this route.
    ///
    /// Optional and never force-unwrapped: a URL built from a caller's
    /// repository id can fail to form, and the force-unwrap that used to live
    /// at the call site is what the review cited. A caller that cannot proceed
    /// without the URL says so itself.
    public var url: URL? {
        var components = URLComponents()
        components.scheme = Self.scheme
        components.host = Self.host
        // `URLComponents.path` wants a single leading-slash string; building it
        // from an array is what keeps the separator out of the call sites.
        components.path = "/" + pathComponents.joined(separator: "/")
        return components.url
    }
}
