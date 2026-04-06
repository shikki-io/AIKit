#if os(macOS)
import Foundation

/// Generates video from text or image prompts via mlx-video CLI.
/// The request's first user message is used as the generation prompt.
/// Response content = output video file path.
public struct MLXVideoProvider: AIProvider, Sendable {
    public let id: String
    public let displayName: String
    public let capabilities: AICapabilities = [.videoGeneration, .vision]

    private let pythonPath: String
    private let modelName: String
    private let defaultOptions: VideoGenerationOptions

    public init(
        id: String,
        displayName: String,
        pythonPath: String,
        modelName: String,
        defaultOptions: VideoGenerationOptions = .default
    ) {
        self.id = id
        self.displayName = displayName
        self.pythonPath = pythonPath
        self.modelName = modelName
        self.defaultOptions = defaultOptions
    }

    public var status: AIProviderStatus {
        get async {
            ShellRunner.commandExists(pythonPath) ? .ready : .unavailable
        }
    }

    /// Generate video from prompt. Returns file path in response content.
    public func complete(request: AIRequest) async throws -> AIResponse {
        let startTime = ContinuousClock.now

        guard let prompt = request.messages.first(where: { $0.role == .user })?.content else {
            throw AIKitError.requestFailed("No user message found for video generation prompt")
        }

        let options = defaultOptions
        let outputPath = options.outputPath ?? NSTemporaryDirectory() + "aikit_video_\(UUID().uuidString).mp4"

        var arguments = [
            "-m", "mlx_video",
            "--model", modelName,
            "--prompt", prompt,
            "--width", String(options.width),
            "--height", String(options.height),
            "--num-frames", String(options.numFrames),
            "--fps", String(options.fps),
            "--num-inference-steps", String(options.numInferenceSteps),
            "--guidance-scale", String(options.guidanceScale),
            "--output", outputPath,
        ]

        if let seed = options.seed {
            arguments += ["--seed", String(seed)]
        }

        if let inputImage = options.inputImagePath {
            arguments += ["--input-image", inputImage]
        }

        let result = try await ShellRunner.run(pythonPath, arguments: arguments)

        guard result.exitCode == 0 else {
            throw AIKitError.requestFailed("mlx-video failed (exit \(result.exitCode)): \(result.stderr)")
        }

        let elapsed = startTime.duration(to: .now)
        let latencyMs = Int(elapsed.components.seconds * 1000 + elapsed.components.attoseconds / 1_000_000_000_000_000)

        return AIResponse(
            content: outputPath,
            model: modelName,
            tokensUsed: TokenUsage(prompt: 0, completion: 0),
            latencyMs: latencyMs
        )
    }

    /// Stream progress during video generation.
    public func stream(request: AIRequest) async throws -> AsyncThrowingStream<AIChunk, Error> {
        guard let prompt = request.messages.first(where: { $0.role == .user })?.content else {
            throw AIKitError.requestFailed("No user message found for video generation prompt")
        }

        let options = defaultOptions
        let outputPath = options.outputPath ?? NSTemporaryDirectory() + "aikit_video_\(UUID().uuidString).mp4"

        var arguments = [
            "-m", "mlx_video",
            "--model", modelName,
            "--prompt", prompt,
            "--width", String(options.width),
            "--height", String(options.height),
            "--num-frames", String(options.numFrames),
            "--fps", String(options.fps),
            "--num-inference-steps", String(options.numInferenceSteps),
            "--guidance-scale", String(options.guidanceScale),
            "--output", outputPath,
        ]

        if let seed = options.seed {
            arguments += ["--seed", String(seed)]
        }

        if let inputImage = options.inputImagePath {
            arguments += ["--input-image", inputImage]
        }

        let pythonCmd = pythonPath
        let finalArguments = arguments
        return AsyncThrowingStream { continuation in
            Task {
                do {
                    let result = try await ShellRunner.stream(
                        pythonCmd,
                        arguments: finalArguments
                    ) { line in
                        // Forward progress lines as chunks.
                        continuation.yield(AIChunk(delta: line + "\n"))
                    }

                    if result.exitCode == 0 {
                        continuation.yield(AIChunk(delta: outputPath, isComplete: true))
                        continuation.finish()
                    } else {
                        continuation.finish(
                            throwing: AIKitError.requestFailed("mlx-video failed (exit \(result.exitCode)): \(result.stderr)")
                        )
                    }
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
}

/// Video generation specific request options.
public struct VideoGenerationOptions: Sendable, Codable, Equatable {
    public var width: Int
    public var height: Int
    public var numFrames: Int
    public var fps: Int
    public var numInferenceSteps: Int
    public var guidanceScale: Double
    public var seed: Int?
    public var outputPath: String?
    /// For I2V (image-to-video): input image path.
    public var inputImagePath: String?

    public init(
        width: Int = 480,
        height: Int = 320,
        numFrames: Int = 49,
        fps: Int = 24,
        numInferenceSteps: Int = 30,
        guidanceScale: Double = 5.0,
        seed: Int? = nil,
        outputPath: String? = nil,
        inputImagePath: String? = nil
    ) {
        self.width = width
        self.height = height
        self.numFrames = numFrames
        self.fps = fps
        self.numInferenceSteps = numInferenceSteps
        self.guidanceScale = guidanceScale
        self.seed = seed
        self.outputPath = outputPath
        self.inputImagePath = inputImagePath
    }

    /// Default: 480x320, 49 frames (~2s at 24fps), 30 inference steps.
    public static let `default` = VideoGenerationOptions()

    /// HD 720p: 1280x720, 97 frames (~4s at 24fps), 50 inference steps.
    public static let hd720p = VideoGenerationOptions(
        width: 1280,
        height: 720,
        numFrames: 97,
        fps: 24,
        numInferenceSteps: 50,
        guidanceScale: 5.0
    )
}

#endif
