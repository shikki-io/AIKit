// MARK: - VisionService
// Unified vision entry point — routes still images to DINOv2Node and
// videos to VJEPANode based on input type.
//
// Spec: features/shikki-vjepa-mlx-node.md §5 Phase 4 deliverable 7,
// Scenario 7 (extension-based routing) + TDDP steps 27-28.

#if canImport(AVFoundation) && canImport(CoreGraphics)
import AVFoundation
import CoreGraphics
import Foundation
import ImageIO

// MARK: - Errors

public enum VisionServiceError: Error, Sendable, Equatable {
    /// File extension did not classify as image or video.
    case unsupportedInput(String)
    /// Input URL could not be read / decoded.
    case inputUnreadable(URL)
}

// MARK: - Input classification

public enum VisionInputKind: Sendable, Equatable {
    case image
    case video
    case unknown
}

// MARK: - Result

/// Tagged result — caller inspects `.image` vs `.video` to know which
/// backend responded. Both carry `featureDim` for a uniform
/// similarity-search story.
public enum VisionEmbedding: Sendable, Equatable {
    case image(DINOv2Embedding)
    case video(VJEPAEmbedding)
}

// MARK: - Service

/// Single entry point that routes vision inputs to the right provider node.
///
/// Routing rules (deterministic, extension-based — see Scenario 7):
///   - `.png`, `.jpg`, `.jpeg`, `.heic`, `.gif`, `.bmp`, `.tiff` → DINOv2Node
///   - `.mov`, `.mp4`, `.m4v`, `.webm`, `.mkv`            → VJEPANode
///   - anything else                                       → `VisionServiceError.unsupportedInput`
///
/// `VisionService` is `Sendable` and safe to share across actors; it holds
/// two actor-isolated nodes and awaits their async embedding APIs.
public struct VisionService: Sendable {

    public let videoNode: VJEPANode
    public let imageNode: DINOv2Node

    public init(videoNode: VJEPANode, imageNode: DINOv2Node) {
        self.videoNode = videoNode
        self.imageNode = imageNode
    }

    // MARK: Classification

    public static let imageExtensions: Set<String> = [
        "png", "jpg", "jpeg", "heic", "heif", "gif", "bmp", "tif", "tiff", "webp",
    ]
    public static let videoExtensions: Set<String> = [
        "mov", "mp4", "m4v", "webm", "mkv", "avi",
    ]

    public static func classify(url: URL) -> VisionInputKind {
        let ext = url.pathExtension.lowercased()
        if imageExtensions.contains(ext) { return .image }
        if videoExtensions.contains(ext) { return .video }
        return .unknown
    }

    // MARK: Unified embed API

    /// Embed a file — routes to DINOv2 or V-JEPA based on extension.
    public func embed(fileAt url: URL) async throws -> VisionEmbedding {
        switch Self.classify(url: url) {
        case .image:
            let image = try Self.loadCGImage(at: url)
            let embedding = try imageNode.embed(image: image)
            return .image(embedding)
        case .video:
            let embedding = try await videoNode.embed(videoAt: url)
            return .video(embedding)
        case .unknown:
            throw VisionServiceError.unsupportedInput(url.pathExtension)
        }
    }

    /// Embed a raw `CGImage` — always routes to DINOv2.
    public func embed(image: CGImage) async throws -> DINOv2Embedding {
        try imageNode.embed(image: image)
    }

    // MARK: CGImage loading

    static func loadCGImage(at url: URL) throws -> CGImage {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else {
            throw VisionServiceError.inputUnreadable(url)
        }
        guard let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw VisionServiceError.inputUnreadable(url)
        }
        return image
    }
}

#endif
