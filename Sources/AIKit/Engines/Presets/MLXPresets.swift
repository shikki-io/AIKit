import Foundation

// MARK: - The video preset grid
//
// W25 of spec `ai-layer-aikit-core-2026-10-31`. Before this, a video request's
// geometry was six loose integers on `VideoGenerationOptions` (width, height,
// numFrames, fps, …) with two hand-written presets beside them, so "480x320 at
// 24 fps" and "1280x720 at 24 fps" were the only shapes anybody could ask for
// without spelling four numbers, and nothing could refuse a shape the hardware
// cannot serve — a 4K request on an iPhone was accepted and then simply never
// finished.
//
// These types are DELIBERATELY outside any platform gate. They are data: a
// resolution, a frame rate, and the product of the two. `DeviceCapability`
// refuses a combination on every host, including the ones with no MLX at all
// (BR-AIL-02), so the grid has to exist where ShellKit and MLX do not.

// MARK: - VideoResolution

/// The frame geometries AIKit generates video at.
///
/// Three, not an open `(width, height)` pair: a resolution is a choice from a
/// closed set the models are trained for, and an arbitrary pair invites the
/// 481x321 that no model serves and no ceiling describes.
public enum VideoResolution: String, Sendable, Codable, Hashable, CaseIterable {

    /// 1280 × 720.
    case hd720

    /// 1920 × 1080.
    case fullHD1080

    /// 3840 × 2160.
    case uhd4K

    /// Frame width in pixels.
    public var width: Int {
        switch self {
        case .hd720: 1280
        case .fullHD1080: 1920
        case .uhd4K: 3840
        }
    }

    /// Frame height in pixels.
    public var height: Int {
        switch self {
        case .hd720: 720
        case .fullHD1080: 1080
        case .uhd4K: 2160
        }
    }

    /// Pixels in one frame.
    public var pixelsPerFrame: Int { width * height }

    /// How this resolution is written for a person: `720p`, `1080p`, `4K`.
    public var displayName: String {
        switch self {
        case .hd720: "720p"
        case .fullHD1080: "1080p"
        case .uhd4K: "4K"
        }
    }
}

// MARK: - VideoFrameRate

/// The frame rates AIKit generates video at.
///
/// The raw value IS the rate in frames per second, so a caller never converts
/// and the CLI argument is the raw value unchanged.
public enum VideoFrameRate: Int, Sendable, Codable, Hashable, CaseIterable {

    /// 24 fps — cinema.
    case fps24 = 24

    /// 30 fps — the broadcast rate, and the default.
    case fps30 = 30

    /// 60 fps — smooth motion.
    case fps60 = 60

    /// 120 fps — slow motion at half speed.
    case fps120 = 120

    /// 240 fps — slow motion at an eighth speed. No shipped Apple silicon is
    /// budgeted for this above 720p; it is in the grid because the enum
    /// describes what can be ASKED for, and ``DeviceCapability`` is what
    /// answers.
    case fps240 = 240

    /// Frames per second.
    public var framesPerSecond: Int { rawValue }
}

// MARK: - MLXVideoPreset

/// One video shape: a resolution and a frame rate.
///
/// The unit `DeviceCapability` refuses on, and the unit a caller names instead
/// of four integers. Its ``pixelsPerSecond`` is the one number that orders the
/// grid, because generation cost scales with pixels emitted per second of
/// output rather than with either dimension alone — which is why 720p at 240
/// fps is more expensive than 1080p at 60.
public struct MLXVideoPreset: Sendable, Codable, Hashable {

    /// Frame geometry.
    public let resolution: VideoResolution

    /// Frames per second.
    public let frameRate: VideoFrameRate

    public init(resolution: VideoResolution, frameRate: VideoFrameRate) {
        self.resolution = resolution
        self.frameRate = frameRate
    }

    /// Pixels of output per second of video — the cost this preset carries.
    public var pixelsPerSecond: Int {
        resolution.pixelsPerFrame * frameRate.framesPerSecond
    }

    /// Frames needed for `seconds` of video at this rate.
    ///
    /// - Parameter seconds: duration in seconds; values below one frame's worth
    ///   still produce one frame, because a zero-frame generation is not a
    ///   video and the CLI would reject it.
    public func numFrames(seconds: Double) -> Int {
        max(1, Int((Double(frameRate.framesPerSecond) * seconds).rounded()))
    }

    /// `720p30`, `4K120` — short enough for a log line or a picker row.
    public var displayName: String {
        "\(resolution.displayName)\(frameRate.framesPerSecond)"
    }

    // MARK: Named points on the grid

    /// 1080p at 30 fps — the default shape, and the ceiling of the current
    /// iPhone line.
    public static let standard = MLXVideoPreset(resolution: .fullHD1080, frameRate: .fps30)

    /// 720p at 30 fps — the floor, and what an A16 phone is budgeted.
    public static let compact = MLXVideoPreset(resolution: .hd720, frameRate: .fps30)

    /// 4K at 60 fps — the most the grid offers at 4K, and an M5/M6 ceiling.
    public static let cinema = MLXVideoPreset(resolution: .uhd4K, frameRate: .fps60)

    /// Every resolution × frame-rate combination, resolution-major.
    ///
    /// The full cross product, generated rather than listed: a listed grid
    /// drifts from the enums the moment either gains a case, and
    /// ``MLXVideoPresetTests`` pins the count to `resolutions × rates`.
    public static let grid: [MLXVideoPreset] = VideoResolution.allCases.flatMap { resolution in
        VideoFrameRate.allCases.map { MLXVideoPreset(resolution: resolution, frameRate: $0) }
    }
}

// MARK: - Video model catalog

#if canImport(Darwin)

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
