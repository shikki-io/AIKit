import Foundation

// MARK: - The voice contract on the existing request
//
// W6 of spec `ai-layer-aikit-core-2026-10-31` (S3.2, BR-AIL-01).
//
// A transcription is NOT a new noun. There is no `Transcribing` protocol and no
// `TranscriptionRequest`: a transcription is an `AIRequest` carrying an audio
// input, served by an `AIProvider` that advertises `.voiceToText`. That is why
// the contract lives here as an extension on the request AIKit already has.
//
// This part of the file is deliberately OUTSIDE the ShellKit gate below: the
// Core ML voice path in `CoreMLEngine` is the iOS half of S3.5 and needs the
// same rule, and ShellKit is macOS-only (BR-AIL-02).

extension AIRequest {

    /// The audio this request carries, resolved against what a provider can do.
    ///
    /// This is the one place that decides whether a given provider may serve a
    /// given request's audio, so every voice provider — the whisper.cpp CLI on
    /// macOS, Core ML on iOS — refuses the same way and for the same reason.
    ///
    /// - Parameter capabilities: the serving provider's `AICapabilities`.
    /// - Returns: the first audio input the request carries.
    /// - Throws: ``AIKitError/noProviderAvailable(_:)`` with `.voiceToText`
    ///   when the provider cannot hear at all, and
    ///   ``AIKitError/requestFailed(_:)`` when the provider can hear but the
    ///   request carries nothing for it to listen to.
    public func audioInput(servedBy capabilities: AICapabilities) throws -> AIInput {
        guard capabilities.contains(.voiceToText) else {
            throw AIKitError.noProviderAvailable(.voiceToText)
        }
        guard let audio = inputs.first(where: { $0.kind == .audio }) else {
            let carried = inputs.map(\.kind.rawValue).joined(separator: ", ")
            throw AIKitError.requestFailed(
                "A transcription needs an audio input; this request carries "
                    + (inputs.isEmpty ? "none" : "[\(carried)]")
            )
        }
        return audio
    }

    /// Refuses when this request carries audio the provider cannot serve.
    ///
    /// The mirror of ``audioInput(servedBy:)`` for a provider that is *not* a
    /// voice provider. A text or vision model handed a `.wav` would otherwise
    /// read `messages`, ignore the audio and answer about nothing — a wrong
    /// answer that looks like a right one. Calling this first turns that into
    /// a typed refusal.
    ///
    /// A request with no audio is served normally, so a text provider pays
    /// nothing for this guard.
    ///
    /// - Parameter capabilities: the serving provider's `AICapabilities`.
    /// - Throws: ``AIKitError/noProviderAvailable(_:)`` with `.voiceToText`.
    public func rejectUnservableAudio(by capabilities: AICapabilities) throws {
        guard inputs.contains(where: { $0.kind == .audio }) else { return }
        // Derived from the same rule rather than restating it: one capability
        // decision, two entry points.
        _ = try audioInput(servedBy: capabilities)
    }
}

// MARK: - Platform gate
//
// Mirrors `CLISubprocessProvider` (W1, BR-AIL-02): ShellKit builds on
// `Foundation.Process`, which the iOS SDK does not have, so the whole provider
// compiles only where ShellKit can be imported. On iOS the voice capability is
// served by `CoreMLEngine` instead — which engine serves a task is config at
// the point of use, never a code branch (BR-AIL-08).
#if canImport(ShellKit)

    import Logging
    import ShellKit

    // MARK: - WhisperCLIProvider

    /// Transcribes the audio an `AIRequest` carries by running whisper.cpp's
    /// `whisper-cli`, and answers as any other `AIProvider` does.
    ///
    /// This is the macOS half of S3.5 and it takes over from shikki's
    /// `WhisperTranscriber`, which coupled transcription to shikki's own
    /// vocabulary (`ProviderProfileStore`, `ShikkiDI`) and to fetching the
    /// audio itself with `yt-dlp`. Here the audio arrives in the request and
    /// the binary and weights arrive as config, so the provider knows nothing
    /// about where either came from (BR-AIL-03).
    ///
    /// Both paths are config, never a hardcoded location: pass `binaryPath`
    /// and `weightsPath` explicitly, or fill them from
    /// ``resolveBinary(searchPaths:)`` and ``resolveWeights(in:preferring:)``.
    ///
    /// ## Measured behaviour of the binary
    ///
    /// Verified against `whisper-cli` (homebrew `whisper-cpp`) on 2026-10-31,
    /// because all three facts shape the code below:
    ///
    /// - `--output-txt` writes `<--output-file>.txt`, and with no
    ///   `--output-file` it writes next to the input instead: `a/b.wav` becomes
    ///   `a/b.wav.txt`, extension **appended**. This provider always passes
    ///   `--output-file` into a directory the call owns, so the transcript
    ///   never lands in the caller's folder and a `.txt` left by an earlier
    ///   run can never be read back as a fresh transcript.
    /// - Even with `--no-prints`, stdout carries timestamped segments
    ///   (`[00:00:00.000 --> 00:00:04.380] …`) while the `.txt` carries the
    ///   clean transcript. The `.txt` is therefore read first and stdout is
    ///   only a fallback.
    /// - A file with no speech in it (a 440 Hz tone) exits **0** and writes an
    ///   **empty** `.txt`. Exit 0 is not proof of a transcript, which is why
    ///   an empty result throws ``TranscriptionError/emptyTranscript(_:)``
    ///   instead of being returned as a successful empty answer.
    public struct WhisperCLIProvider: AIProvider, Sendable {

        // MARK: - Errors

        /// Why a transcription could not be produced.
        ///
        /// Every case names what was missing or what the binary said. None of
        /// them is reachable with a transcript in hand, so a caller that gets
        /// a value back got real text.
        public enum TranscriptionError: Error, LocalizedError, Equatable {
            /// `binaryPath` is not an executable file.
            case binaryNotFound(String)
            /// `weightsPath` does not exist.
            case weightsNotFound(String)
            /// The request's audio is not a readable local file.
            case audioNotReadable(String)
            /// The working files this call needs could not be created on disk.
            case audioNotStageable(String)
            /// `whisper-cli` exited non-zero.
            case nonZeroExit(exitCode: Int32, stderr: String)
            /// `whisper-cli` outlived `timeout` and was killed.
            case timedOut(limit: TimeInterval)
            /// `whisper-cli` could not be launched at all.
            case launchFailed(String)
            /// The binary succeeded and produced no words.
            case emptyTranscript(String)

            public var errorDescription: String? {
                switch self {
                case .binaryNotFound(let path):
                    "whisper-cli not found at \(path). Install: brew install whisper-cpp"
                case .weightsNotFound(let path):
                    "Whisper weights not found at \(path). Download: whisper-cpp --download-model large-v3-turbo"
                case .audioNotReadable(let detail):
                    "The request's audio is not a readable local file: \(detail)"
                case .audioNotStageable(let detail):
                    "Could not create this transcription's working files: \(detail)"
                case .nonZeroExit(let exitCode, let stderr):
                    "whisper-cli exited \(exitCode): \(stderr)"
                case .timedOut(let limit):
                    "whisper-cli exceeded its \(Int(limit))s limit and was killed"
                case .launchFailed(let reason):
                    "whisper-cli could not be launched: \(reason)"
                case .emptyTranscript(let path):
                    "whisper-cli exited 0 but produced no transcript for \(path) — "
                        + "the audio may carry no speech, or the weights may be wrong for its language"
                }
            }
        }

        // MARK: - AIProvider

        public let id: String
        public let displayName: String

        /// Voice, and only voice. A request with no audio is refused by
        /// ``AIRequest/audioInput(servedBy:)``, not served as text.
        public let capabilities: AICapabilities = [.voiceToText]

        // MARK: - Configuration

        /// Absolute path to `whisper-cli` (or `whisper-cpp`).
        public let binaryPath: String

        /// Absolute path to the GGML weights, e.g. `ggml-large-v3-turbo.bin`.
        public let weightsPath: String

        /// BCP-47 language code, or `"auto"` to let the model detect it.
        public let language: String

        /// Hard wall-clock limit for one transcription.
        public let timeout: TimeInterval

        // MARK: - Collaborators

        private let executor: any ShellExecutorProtocol
        private let logger: Logger

        // MARK: - Init

        /// Creates a provider over an installed whisper.cpp.
        ///
        /// - Parameters:
        ///   - id: `AIProvider.id`, stamped onto every response.
        ///   - displayName: human-readable name.
        ///   - binaryPath: absolute path to `whisper-cli`. Required rather than
        ///     resolved here so the model a caller runs is always visible in
        ///     config instead of decided inside the provider.
        ///   - weightsPath: absolute path to the GGML weights.
        ///   - language: BCP-47 code, or `"auto"` for detection.
        ///   - timeout: wall-clock limit for one run. The default matches
        ///     shikki's shipped budget: local inference on Apple Silicon, so
        ///     proportional to audio length rather than to the network.
        ///   - executor: shell executor; `TimedShellExecutor` drains both pipes
        ///     and kills on timeout, which a bare `Process` does not.
        ///   - logger: diagnostic logger.
        public init(
            id: String = "whisper-cli",
            displayName: String = "Whisper (whisper.cpp)",
            binaryPath: String,
            weightsPath: String,
            language: String = "auto",
            timeout: TimeInterval = 600,
            executor: any ShellExecutorProtocol = TimedShellExecutor(),
            logger: Logger = Logger(label: "aikit.whisper-cli-provider")
        ) {
            self.id = id
            self.displayName = displayName
            self.binaryPath = binaryPath
            self.weightsPath = weightsPath
            self.language = language
            self.timeout = timeout
            self.executor = executor
            self.logger = logger
        }

        // MARK: - Status

        /// `.ready` only when both the binary and the weights are on disk.
        ///
        /// Reported as `.unavailable` rather than `.error` because nothing is
        /// broken: whisper.cpp simply is not installed here, and a router
        /// should move on to the next provider.
        public var status: AIProviderStatus {
            get async {
                let fm = FileManager.default
                guard fm.isExecutableFile(atPath: binaryPath), fm.fileExists(atPath: Self.expandingTilde(weightsPath))
                else {
                    return .unavailable
                }
                return .ready
            }
        }

        // MARK: - Transcription

        /// Transcribes the request's audio input.
        ///
        /// - Returns: a response whose `content` is the transcript and whose
        ///   `providerID` is this provider's `id`.
        /// - Throws: ``TranscriptionError``, or
        ///   ``AIKitError/requestFailed(_:)`` when the request carries no
        ///   audio.
        public func complete(request: AIRequest) async throws -> AIResponse {
            let audio = try request.audioInput(servedBy: capabilities)
            let start = ContinuousClock.now

            let fm = FileManager.default
            guard fm.isExecutableFile(atPath: binaryPath) else {
                throw TranscriptionError.binaryNotFound(binaryPath)
            }
            let weights = Self.expandingTilde(weightsPath)
            guard fm.fileExists(atPath: weights) else {
                throw TranscriptionError.weightsNotFound(weights)
            }

            let staged = try Self.stage(audio)
            defer { staged.cleanUp() }

            // A directory this call owns, for the transcript to land in.
            // Without it the binary writes next to the input, which both
            // litters the caller's folder and leaves a stale `.txt` that a
            // later run could read back as if it were fresh.
            let scratch = URL(fileURLWithPath: NSTemporaryDirectory())
                .appendingPathComponent("aikit-whisper-\(UUID().uuidString)", isDirectory: true)
            do {
                try fm.createDirectory(at: scratch, withIntermediateDirectories: true)
            } catch {
                throw TranscriptionError.audioNotStageable(error.localizedDescription)
            }
            defer { try? fm.removeItem(at: scratch) }

            let outputStem = scratch.appendingPathComponent("transcript").path
            let arguments = Self.arguments(
                binaryPath: binaryPath,
                weights: weights,
                language: language,
                audioPath: staged.path,
                outputStem: outputStem
            )

            let result: ShellCommandResult
            do {
                result = try await executor.run(
                    arguments, cwd: nil, env: nil, timeout: timeout, stdin: nil
                )
            } catch ShellError.timeout(_, let limit) {
                throw TranscriptionError.timedOut(limit: limit)
            } catch ShellError.launchFailed(_, let underlying) {
                throw TranscriptionError.launchFailed(underlying)
            } catch {
                throw TranscriptionError.launchFailed(String(describing: error))
            }

            guard result.exitCode == 0 else {
                let stderr = result.stderrString
                logger.error(
                    "whisper-cli non-zero exit",
                    metadata: [
                        "provider": "\(id)",
                        "exit_code": "\(result.exitCode)",
                        "stderr_preview": "\(String(stderr.prefix(200)))",
                    ]
                )
                throw TranscriptionError.nonZeroExit(exitCode: result.exitCode, stderr: stderr)
            }

            guard let transcript = Self.transcript(at: outputStem + ".txt", orStdout: result.stdoutString) else {
                // Exit 0 with nothing to show for it. Measured on a tone-only
                // file: the trap this case exists to close.
                logger.warning(
                    "whisper-cli produced no transcript",
                    metadata: ["provider": "\(id)", "audio": "\(staged.path)"]
                )
                throw TranscriptionError.emptyTranscript(staged.path)
            }

            let elapsed = start.duration(to: .now)
            let latencyMs = Int(
                elapsed.components.seconds * 1000
                    + elapsed.components.attoseconds / 1_000_000_000_000_000
            )

            return AIResponse(
                content: transcript,
                model: URL(fileURLWithPath: weights).lastPathComponent,
                tokensUsed: TokenUsage(prompt: 0, completion: 0),
                latencyMs: latencyMs,
                providerID: id
            )
        }

        // MARK: - Argument list

        /// The argument list for one transcription.
        ///
        /// Exposed so a test asserts the real list the provider runs rather
        /// than a second copy of it written from the same reasoning.
        ///
        /// `--no-prints` silences the progress banner; it does *not* silence
        /// the timestamped segments on stdout, which is why `--output-txt` is
        /// the transcript's real home. GPU acceleration is left on: whisper.cpp
        /// uses Metal by default on Apple Silicon and `--no-gpu` would give
        /// that up.
        ///
        /// - Parameter outputStem: the `--output-file` path, *without* an
        ///   extension — the binary appends `.txt` itself.
        public static func arguments(
            binaryPath: String,
            weights: String,
            language: String,
            audioPath: String,
            outputStem: String
        ) -> [String] {
            [
                binaryPath,
                "--model", weights,
                "--language", language,
                "--output-txt",
                "--output-file", outputStem,
                "--no-prints",
                "--file", audioPath,
            ]
        }

        // MARK: - Reading the result

        /// The transcript, preferring the `.txt` the binary wrote and falling
        /// back to stdout. `nil` when neither carries a word, which the caller
        /// turns into ``TranscriptionError/emptyTranscript(_:)``.
        static func transcript(at path: String, orStdout stdout: String) -> String? {
            if let text = try? String(contentsOfFile: path, encoding: .utf8) {
                let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty { return trimmed }
            }
            let trimmedStdout = stdout.trimmingCharacters(in: .whitespacesAndNewlines)
            return trimmedStdout.isEmpty ? nil : trimmedStdout
        }

        // MARK: - Staging the audio

        /// An audio file on disk for the binary to read, plus whatever cleanup
        /// putting it there earned.
        struct StagedAudio {
            let path: String
            private let temporary: Bool

            init(path: String, temporary: Bool) {
                self.path = path
                self.temporary = temporary
            }

            /// Removes the file only when this provider is the one that wrote it.
            func cleanUp() {
                guard temporary else { return }
                try? FileManager.default.removeItem(atPath: path)
            }
        }

        /// Resolves an `AIInput` to a readable local path.
        ///
        /// A URL input is used where it already lives — a recording can be
        /// hundreds of megabytes and copying it would double that for nothing.
        /// A data input has no file behind it, so it is written to a temporary
        /// one, which `StagedAudio.cleanUp()` removes.
        static func stage(_ audio: AIInput) throws -> StagedAudio {
            switch audio.source {
            case .url(let url):
                guard url.isFileURL else {
                    throw TranscriptionError.audioNotReadable(
                        "\(url.absoluteString) is not a file URL; fetch it before transcribing"
                    )
                }
                guard FileManager.default.fileExists(atPath: url.path) else {
                    throw TranscriptionError.audioNotReadable("no file at \(url.path)")
                }
                return StagedAudio(path: url.path, temporary: false)

            case .data(let data):
                guard !data.isEmpty else {
                    throw TranscriptionError.audioNotReadable("the request carries 0 bytes of audio")
                }
                let name = "aikit-whisper-\(UUID().uuidString).\(Self.fileExtension(for: audio.mimeType))"
                let url = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent(name)
                do {
                    try data.write(to: url)
                } catch {
                    throw TranscriptionError.audioNotStageable(error.localizedDescription)
                }
                return StagedAudio(path: url.path, temporary: true)
            }
        }

        /// Container extension for a data-carried input, from its media type.
        /// whisper.cpp decodes via miniaudio and reads the container, so this
        /// only has to be honest, not exhaustive.
        static func fileExtension(for mimeType: String?) -> String {
            switch mimeType?.lowercased() {
            case "audio/wav", "audio/x-wav", "audio/wave": "wav"
            case "audio/mpeg", "audio/mp3": "mp3"
            case "audio/mp4", "audio/m4a", "audio/x-m4a": "m4a"
            case "audio/flac", "audio/x-flac": "flac"
            case "audio/ogg", "audio/opus": "ogg"
            default: "wav"
            }
        }

        // MARK: - Config resolution

        /// Binary names whisper.cpp has shipped under, newest first.
        public static let binaryNames = ["whisper-cli", "whisper-cpp"]

        /// Weights in descending preference.
        ///
        /// `large-v3-turbo` leads because it is the accuracy/speed point the
        /// fleet settled on and the one flsh runs; the smaller models are the
        /// fallbacks for a machine that cannot hold it. The Flsh voice spec
        /// (`features/flsh-whisper-e4b-voice-stack.md`, BR-01) names
        /// whisper-small as the on-device floor and large as the Mac upgrade,
        /// and this is that order read from the large end.
        public static let weightsPreference = [
            "ggml-large-v3-turbo.bin",
            "ggml-large-v3.bin",
            "ggml-large.bin",
            "ggml-medium.bin",
            "ggml-small.bin",
            "ggml-base.bin",
            "ggml-base.en.bin",
        ]

        /// Where whisper.cpp keeps its weights once installed.
        public static var defaultWeightsDirectories: [URL] {
            let home = FileManager.default.homeDirectoryForCurrentUser
            return [
                home.appendingPathComponent(".local/share/whisper-cpp/models", isDirectory: true),
                home.appendingPathComponent(".aikit/whisper-models", isDirectory: true),
            ]
        }

        /// The first `whisper-cli` on `searchPaths`, or `nil` when whisper.cpp
        /// is not installed.
        ///
        /// Returns an optional on purpose: a bare binary name handed to the
        /// init would fail later as a launch error, where "not installed" is
        /// the thing the caller actually needs to know.
        public static func resolveBinary(
            searchPaths: [String] = Self.pathEntries()
        ) -> String? {
            for directory in searchPaths {
                for name in binaryNames {
                    let candidate = URL(fileURLWithPath: directory).appendingPathComponent(name).path
                    if FileManager.default.isExecutableFile(atPath: candidate) { return candidate }
                }
            }
            return nil
        }

        /// The most preferred weights present in `directories`, or `nil`.
        public static func resolveWeights(
            in directories: [URL] = Self.defaultWeightsDirectories,
            preferring preference: [String] = Self.weightsPreference
        ) -> String? {
            for directory in directories {
                for name in preference {
                    let candidate = directory.appendingPathComponent(name).path
                    if FileManager.default.fileExists(atPath: candidate) { return candidate }
                }
            }
            return nil
        }

        /// `PATH`, split into directories.
        public static func pathEntries(
            _ path: String? = ProcessInfo.processInfo.environment["PATH"]
        ) -> [String] {
            (path ?? "").split(separator: ":").map(String.init).filter { !$0.isEmpty }
        }

        private static func expandingTilde(_ path: String) -> String {
            (path as NSString).expandingTildeInPath
        }
    }

#endif  // canImport(ShellKit)
