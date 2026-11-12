// STATUS: unproven and not wired. Built and unit-tested at the helper level only;
// NO caller in AIKit or the fleet constructs it, and it has NOT been run on real
// weights / real inference (needs a mlx-video install the tests never exercise).
// Whether to keep it is the operator's call (W6 deliverable of ai-layer spec 8badc87f).

#if os(macOS)
    import Foundation

    // MARK: - Errors

    /// Why a video generation did not start, or did not finish.
    ///
    /// W25 of spec `ai-layer-aikit-core-2026-10-31`. ``presetRefused`` is the
    /// case this wave exists for: a preset the hardware cannot serve is refused
    /// before a subprocess is launched, carrying
    /// ``VideoPresetRefusal`` so the caller reads the LIMIT (budgeted pixels per
    /// second, or the host with no MLX at all) rather than a sentence.
    ///
    /// The other cases replace `AIKitError.requestFailed` strings that this
    /// provider used to throw for its own failures, so a caller can tell "the
    /// device said no" from "mlx-video crashed" without parsing text.
    enum MLXVideoError: Error, LocalizedError, Equatable {

        /// ``DeviceCapability`` refused the preset.
        case presetRefused(VideoPresetRefusal, preset: MLXVideoPreset)

        /// The request carried no user message to generate from.
        case noPrompt

        /// `mlx-video` exited non-zero.
        case nonZeroExit(exitCode: Int32, stderr: String)

        var errorDescription: String? {
            switch self {
            case .presetRefused(let refusal, let preset):
                switch refusal {
                case .unsupportedPlatform(let platform):
                    "\(preset.displayName) cannot be generated on \(platform.rawValue): "
                        + "there is no MLX video runtime on this host"
                case .throughput(let required, let ceiling, let deviceClass):
                    "\(preset.displayName) needs \(required) pixels/s; "
                        + "\(deviceClass.rawValue) is budgeted \(ceiling) pixels/s"
                }
            case .noPrompt:
                "No user message found for video generation prompt"
            case .nonZeroExit(let exitCode, let stderr):
                "mlx-video failed (exit \(exitCode)): \(stderr)"
            }
        }
    }

    // MARK: - MLXVideoProvider

    /// Generates video from text or image prompts via mlx-video CLI.
    /// The request's first user message is used as the generation prompt.
    /// Response content = output video file path.
    struct MLXVideoProvider: AIProvider, Sendable {
        let id: String
        let displayName: String
        let capabilities: AICapabilities = [.videoGeneration, .vision]

        private let pythonPath: String
        private let modelName: String
        private let defaultOptions: VideoGenerationOptions

        /// The hardware this provider refuses against.
        ///
        /// Injected rather than read at the call site so a test can describe an
        /// iPhone from a Mac — the refusal is the behaviour under test, and it
        /// is untestable if the verdict comes from whatever machine runs the
        /// suite.
        private let device: DeviceCapability

        init(
            id: String,
            displayName: String,
            pythonPath: String,
            modelName: String,
            defaultOptions: VideoGenerationOptions = .default,
            device: DeviceCapability = .current()
        ) {
            self.id = id
            self.displayName = displayName
            self.pythonPath = pythonPath
            self.modelName = modelName
            self.defaultOptions = defaultOptions
            self.device = device
        }

        var status: AIProviderStatus {
            get async {
                ShellRunner.commandExists(pythonPath) ? .ready : .unavailable
            }
        }

        // MARK: - One preparation for both paths

        /// What both `complete` and `stream` need before launching anything:
        /// a prompt, a device verdict, and the argument list.
        ///
        /// Built once because it was written twice. The two paths had identical
        /// fourteen-line argument arrays, and a flag added to one would have
        /// silently not reached the other.
        private func prepare() throws -> (arguments: [String], outputPath: String) {
            let options = defaultOptions

            // The device verdict comes FIRST: refusing before the prompt check
            // would hide a missing prompt, and refusing after launching would
            // waste a subprocess on work that cannot finish.
            if let preset = options.preset, let refusal = device.refusal(for: preset) {
                throw MLXVideoError.presetRefused(refusal, preset: preset)
            }

            let outputPath = options.outputPath ?? NSTemporaryDirectory() + "aikit_video_\(UUID().uuidString).mp4"

            var arguments = [
                "-m", "mlx_video",
                "--model", modelName,
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

            return (arguments, outputPath)
        }

        /// The prompt, and the argument list carrying it.
        private func prepare(for request: AIRequest) throws -> (arguments: [String], outputPath: String) {
            guard let prompt = request.messages.first(where: { $0.role == .user })?.content else {
                throw MLXVideoError.noPrompt
            }
            var prepared = try prepare()
            prepared.arguments += ["--prompt", prompt]
            return prepared
        }

        /// Generate video from prompt. Returns file path in response content.
        func complete(request: AIRequest) async throws -> AIResponse {
            let startTime = ContinuousClock.now
            let (arguments, outputPath) = try prepare(for: request)

            let result = try await ShellRunner.run(pythonPath, arguments: arguments)

            guard result.exitCode == 0 else {
                throw MLXVideoError.nonZeroExit(exitCode: result.exitCode, stderr: result.stderr)
            }

            let elapsed = startTime.duration(to: .now)
            let latencyMs = Int(
                elapsed.components.seconds * 1000 + elapsed.components.attoseconds / 1_000_000_000_000_000)

            return AIResponse(
                content: outputPath,
                model: modelName,
                tokensUsed: TokenUsage(prompt: nil, completion: nil),
                latencyMs: latencyMs,
                providerID: id,
                outputs: [AIOutput(kind: .video, url: URL(fileURLWithPath: outputPath))]
            )
        }

        /// Stream progress during video generation.
        func stream(request: AIRequest) async throws -> AsyncThrowingStream<AIChunk, Error> {
            let (arguments, outputPath) = try prepare(for: request)
            let pythonCmd = pythonPath

            return AsyncThrowingStream { continuation in
                Task {
                    do {
                        let result = try await ShellRunner.stream(
                            pythonCmd,
                            arguments: arguments
                        ) { line in
                            // Forward progress lines as chunks.
                            continuation.yield(AIChunk(delta: line + "\n"))
                        }

                        if result.exitCode == 0 {
                            continuation.yield(AIChunk(delta: outputPath, isComplete: true))
                            continuation.finish()
                        } else {
                            continuation.finish(
                                throwing: MLXVideoError.nonZeroExit(
                                    exitCode: result.exitCode,
                                    stderr: result.stderr
                                )
                            )
                        }
                    } catch {
                        continuation.finish(throwing: error)
                    }
                }
            }
        }
    }

    // MARK: - VideoGenerationOptions

    /// Video generation specific request options.
    struct VideoGenerationOptions: Sendable, Codable, Equatable {
        var width: Int
        var height: Int
        var numFrames: Int
        var fps: Int
        var numInferenceSteps: Int
        var guidanceScale: Double
        var seed: Int?
        var outputPath: String?
        /// For I2V (image-to-video): input image path.
        var inputImagePath: String?

        /// The grid preset these options came from, when they came from one.
        ///
        /// `nil` for options built from loose integers, and that is not a
        /// shortcoming to paper over: geometry does NOT round-trip to a preset.
        /// `480x320` — this type's own historical default — is on no grid point,
        /// so inferring a preset from width and height would have to pick a
        /// nearest neighbour and would then refuse, or allow, a shape nobody
        /// asked for. Options with no preset are passed to the CLI as given and
        /// carry no device verdict; options built through
        /// ``init(preset:seconds:numInferenceSteps:guidanceScale:seed:outputPath:inputImagePath:)``
        /// are refused by ``DeviceCapability`` before a subprocess starts.
        ///
        /// Optional and synthesized-`Codable`, so a stored payload written
        /// before W25 still decodes: the key is simply absent.
        var preset: MLXVideoPreset?

        init(
            width: Int = 480,
            height: Int = 320,
            numFrames: Int = 49,
            fps: Int = 24,
            numInferenceSteps: Int = 30,
            guidanceScale: Double = 5.0,
            seed: Int? = nil,
            outputPath: String? = nil,
            inputImagePath: String? = nil,
            preset: MLXVideoPreset? = nil
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
            self.preset = preset
        }

        /// Options for a grid preset, for `seconds` of video.
        ///
        /// Geometry and frame count are DERIVED, so width, height, fps and
        /// numFrames cannot disagree with each other or with the preset the
        /// device was asked about.
        ///
        /// - Parameters:
        ///   - preset: the resolution × frame-rate point to generate at.
        ///   - seconds: output duration. The default is two seconds, which at
        ///     24 fps is 48 frames — the neighbourhood of this type's legacy
        ///     49-frame default, kept so switching to a preset does not quietly
        ///     change how long a generation takes.
        init(
            preset: MLXVideoPreset,
            seconds: Double = 2.0,
            numInferenceSteps: Int = 30,
            guidanceScale: Double = 5.0,
            seed: Int? = nil,
            outputPath: String? = nil,
            inputImagePath: String? = nil
        ) {
            self.init(
                width: preset.resolution.width,
                height: preset.resolution.height,
                numFrames: preset.numFrames(seconds: seconds),
                fps: preset.frameRate.framesPerSecond,
                numInferenceSteps: numInferenceSteps,
                guidanceScale: guidanceScale,
                seed: seed,
                outputPath: outputPath,
                inputImagePath: inputImagePath,
                preset: preset
            )
        }

        /// Default: 480x320, 49 frames (~2s at 24fps), 30 inference steps.
        ///
        /// Carries no preset: 480x320 is on no grid point. Kept unchanged
        /// because it is what every existing caller and stored payload means.
        static let `default` = VideoGenerationOptions()

        /// HD 720p: 1280x720, 97 frames (~4s at 24fps), 50 inference steps.
        static let hd720p = VideoGenerationOptions(
            width: 1280,
            height: 720,
            numFrames: 97,
            fps: 24,
            numInferenceSteps: 50,
            guidanceScale: 5.0,
            preset: MLXVideoPreset(resolution: .hd720, frameRate: .fps24)
        )
    }

#endif
