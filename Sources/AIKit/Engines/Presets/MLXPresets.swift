#if canImport(Darwin)
import Foundation

public extension MLXEngine {

    /// Wan2.1 Text-to-Video 1.3B — fits in 8GB unified memory, 480p, fast.
    static let wan21_t2v_1_3b = ModelDescriptor(
        id: ModelIdentifier(provider: "mlx", modelId: "wan2.1-t2v-1.3b"),
        name: "Wan2.1 T2V 1.3B",
        author: "Alibaba",
        description: "Wan2.1 Text-to-Video 1.3B — lightweight video generation for 8GB+ Macs",
        capabilities: .videoGeneration,
        format: .mlx,
        parameters: "1.3B",
        quantization: "Q4",
        sizeBytes: 800_000_000,
        architecture: "wan2.1",
        domain: .video,
        huggingFaceId: "mlx-community/wan2.1-t2v-1.3b-4bit",
        tags: ["video", "t2v", "mlx", "lightweight"]
    )

    /// Wan2.1 Text-to-Video 14B — needs 40GB+ unified memory, 720p, cinema quality.
    static let wan21_t2v_14b = ModelDescriptor(
        id: ModelIdentifier(provider: "mlx", modelId: "wan2.1-t2v-14b"),
        name: "Wan2.1 T2V 14B",
        author: "Alibaba",
        description: "Wan2.1 Text-to-Video 14B — high quality video generation for 40GB+ Macs",
        capabilities: .videoGeneration,
        format: .mlx,
        parameters: "14B",
        quantization: "Q4",
        sizeBytes: 8_000_000_000,
        architecture: "wan2.1",
        domain: .video,
        huggingFaceId: "mlx-community/wan2.1-t2v-14b-4bit",
        tags: ["video", "t2v", "mlx", "cinema"]
    )

    /// LTX-2 — 19B (Q4 distilled ~10B), 720p-1080p.
    static let ltx2 = ModelDescriptor(
        id: ModelIdentifier(provider: "mlx", modelId: "ltx-2"),
        name: "LTX-2",
        author: "Lightricks",
        description: "LTX-2 video generation — fast inference, up to 1080p",
        capabilities: .videoGeneration,
        format: .mlx,
        parameters: "19B",
        quantization: "Q4",
        sizeBytes: 10_000_000_000,
        architecture: "ltx",
        domain: .video,
        huggingFaceId: "mlx-community/ltx-2-4bit",
        tags: ["video", "t2v", "mlx", "fast"]
    )

    /// Wan2.1 Image-to-Video 14B — animate still images.
    static let wan21_i2v_14b = ModelDescriptor(
        id: ModelIdentifier(provider: "mlx", modelId: "wan2.1-i2v-14b"),
        name: "Wan2.1 I2V 14B",
        author: "Alibaba",
        description: "Wan2.1 Image-to-Video 14B — animate still images into video",
        capabilities: [.videoGeneration, .vision],
        format: .mlx,
        parameters: "14B",
        quantization: "Q4",
        sizeBytes: 8_000_000_000,
        architecture: "wan2.1",
        domain: .video,
        huggingFaceId: "mlx-community/wan2.1-i2v-14b-4bit",
        tags: ["video", "i2v", "mlx", "animate"]
    )

    /// Video model catalog for the model library UI.
    static var videoModelCatalog: [ModelDescriptor] {
        [wan21_t2v_1_3b, wan21_t2v_14b, ltx2, wan21_i2v_14b]
    }
}

#endif
