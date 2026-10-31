import Foundation
import Testing

@testable import AIKit

#if canImport(CoreML)
    import CoreML
#endif
#if canImport(ShellKit)
    import ShellKit
#endif

// MARK: - TranscriptionProviderTests
//
// W6 of spec `ai-layer-aikit-core-2026-10-31`. Covers T-AIL-10
// (`refusesWithoutVoice`) and T-AIL-11 (`fixtureThroughMock`), plus the
// behaviour of the two providers W6 ships.
//
// No test reaches a real provider (BR-AIL-12): the CLI provider runs against
// `MockShellExecutor`, and the routing test runs against `MockAIProvider`.
// `whisper-cli` itself is never launched here.

@Suite("Voice through the same request")
struct TranscriptionProviderTests {

    // MARK: - Fixtures

    /// A directory that lives for one test and takes its files with it.
    final class TempDirectory {
        let url: URL

        init() throws {
            url = URL(fileURLWithPath: NSTemporaryDirectory())
                .appendingPathComponent("aikit-w6-\(UUID().uuidString)", isDirectory: true)
            try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        }

        deinit {
            try? FileManager.default.removeItem(at: url)
        }

        func file(_ name: String, contents: String = "") throws -> URL {
            let path = url.appendingPathComponent(name)
            try contents.write(to: path, atomically: true, encoding: .utf8)
            return path
        }

        /// An executable stand-in for `whisper-cli`. Never run: it exists so
        /// `isExecutableFile` is satisfied by a real file rather than mocked.
        func executable(_ name: String) throws -> URL {
            let path = url.appendingPathComponent(name)
            try "#!/bin/sh\nexit 0\n".write(to: path, atomically: true, encoding: .utf8)
            try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: path.path)
            return path
        }

        /// A real 16 kHz mono 16-bit PCM WAV of `seconds` of silence.
        ///
        /// Written by hand rather than through AVFoundation so the fixture is
        /// a known-good file by construction — `AVAudioFile` reading it is
        /// then a result, not an assumption.
        func wav(_ name: String, seconds: Double = 1.0, sampleRate: Int = 16_000) throws -> URL {
            let frames = Int(Double(sampleRate) * seconds)
            let dataBytes = frames * 2
            var bytes = Data()

            func append(_ string: String) { bytes.append(contentsOf: Array(string.utf8)) }
            func append32(_ value: Int) {
                bytes.append(contentsOf: withUnsafeBytes(of: UInt32(value).littleEndian, Array.init))
            }
            func append16(_ value: Int) {
                bytes.append(contentsOf: withUnsafeBytes(of: UInt16(value).littleEndian, Array.init))
            }

            append("RIFF")
            append32(36 + dataBytes)
            append("WAVE")
            append("fmt ")
            append32(16)  // PCM header size
            append16(1)  // PCM
            append16(1)  // mono
            append32(sampleRate)
            append32(sampleRate * 2)  // byte rate
            append16(2)  // block align
            append16(16)  // bits per sample
            append("data")
            append32(dataBytes)
            bytes.append(Data(count: dataBytes))

            let path = url.appendingPathComponent(name)
            try bytes.write(to: path)
            return path
        }
    }

    static func makeAudioRequest(
        url: URL,
        mimeType: String? = "audio/wav",
        prompt: String = "Transcribe this."
    ) -> AIRequest {
        AIRequest(
            messages: [AIMessage(role: .user, content: prompt)],
            inputs: [AIInput(kind: .audio, url: url, mimeType: mimeType)]
        )
    }

    static func makeTextRequest() -> AIRequest {
        AIRequest(messages: [AIMessage(role: .user, content: "No audio here.")])
    }

    // MARK: - T-AIL-10 — typed refusal

    @Test("T-AIL-10 refusesWithoutVoice: a provider without voiceToText refuses an audio request")
    func refusesWithoutVoice() throws {
        let request = Self.makeAudioRequest(url: URL(fileURLWithPath: "/tmp/does-not-need-to-exist.wav"))

        // Asking for the audio as a non-voice provider is refused outright.
        #expect(throws: AIKitError.noProviderAvailable(.voiceToText)) {
            _ = try request.audioInput(servedBy: .textGeneration)
        }

        // And the guard a text or vision provider actually calls — the same
        // one `CoreMLProvider.complete` runs — refuses the same way.
        #expect(throws: AIKitError.noProviderAvailable(.voiceToText)) {
            try request.rejectUnservableAudio(by: [.vision, .ocr])
        }
    }

    @Test("A capability set that merely includes voice is enough to be served")
    func voiceAmongOtherCapabilitiesIsServed() throws {
        let audio = URL(fileURLWithPath: "/tmp/mixed.wav")
        let request = Self.makeAudioRequest(url: audio)
        let resolved = try request.audioInput(servedBy: [.textGeneration, .voiceToText])
        #expect(resolved.url == audio)
    }

    @Test("A voice provider receives the audio the request carries")
    func voiceProviderReceivesTheAudio() throws {
        let audio = URL(fileURLWithPath: "/tmp/meeting.wav")
        let request = Self.makeAudioRequest(url: audio)

        let resolved = try request.audioInput(servedBy: .voiceToText)
        #expect(resolved.kind == .audio)
        #expect(resolved.url == audio)
        #expect(resolved.mimeType == "audio/wav")
    }

    @Test("A voice provider handed no audio refuses, naming what the request carried")
    func voiceProviderWithoutAudioRefuses() {
        let imageOnly = AIRequest(
            messages: [AIMessage(role: .user, content: "Transcribe this.")],
            inputs: [AIInput(kind: .image, url: URL(fileURLWithPath: "/tmp/frame.png"))]
        )

        #expect(throws: AIKitError.self) {
            _ = try imageOnly.audioInput(servedBy: .voiceToText)
        }

        // The refusal says which inputs did arrive, so a caller that sent the
        // wrong kind is not left guessing.
        do {
            _ = try imageOnly.audioInput(servedBy: .voiceToText)
            Issue.record("A request with no audio must not resolve an audio input")
        } catch let error as AIKitError {
            guard case .requestFailed(let reason) = error else {
                Issue.record("Expected .requestFailed, got \(error)")
                return
            }
            #expect(reason.contains("image"))
        } catch {
            Issue.record("Expected AIKitError, got \(error)")
        }
    }

    @Test("A request with no media is served normally by a text provider")
    func textRequestPassesTheGuard() throws {
        // The guard must cost a text-only request nothing: no audio, no throw.
        try Self.makeTextRequest().rejectUnservableAudio(by: .textGeneration)
        try Self.makeTextRequest().rejectUnservableAudio(by: [.vision, .ocr])
    }

    @Test("Media a provider cannot serve is only refused when it is audio")
    func nonAudioMediaIsNotRefusedByTheVoiceGuard() throws {
        let imageRequest = AIRequest(
            messages: [AIMessage(role: .user, content: "Describe this.")],
            inputs: [AIInput(kind: .image, url: URL(fileURLWithPath: "/tmp/frame.png"))]
        )
        // A vision provider handed an image is exactly the right pairing; the
        // voice guard must stay out of its way.
        try imageRequest.rejectUnservableAudio(by: [.vision])
    }

    @Test("The first audio input is the one served")
    func firstAudioInputWins() throws {
        let first = URL(fileURLWithPath: "/tmp/first.wav")
        let second = URL(fileURLWithPath: "/tmp/second.wav")
        let request = AIRequest(
            messages: [AIMessage(role: .user, content: "Transcribe.")],
            inputs: [
                AIInput(kind: .image, url: URL(fileURLWithPath: "/tmp/cover.png")),
                AIInput(kind: .audio, url: first),
                AIInput(kind: .audio, url: second),
            ]
        )
        #expect(try request.audioInput(servedBy: .voiceToText).url == first)
    }

    // MARK: - T-AIL-11 — the audio URL reaches the provider

    @Test("T-AIL-11 fixtureThroughMock: an audio URL reaches the provider through the router")
    func fixtureThroughMock() async throws {
        let temp = try TempDirectory()
        let fixture = try temp.wav("fixture.wav")

        let text = MockAIProvider(id: "text", capabilities: .textGeneration)
        let voice = MockAIProvider(id: "voice", capabilities: [.voiceToText])
        let router = ProviderRouter(providers: [text, voice])

        // Routing by capability is all it takes — a transcription needs no
        // special dispatch, which is the whole point of BR-AIL-01.
        let chosen = await router.route(capabilities: .voiceToText)
        #expect(chosen?.id == "voice")

        let request = Self.makeAudioRequest(url: fixture)
        let response = try await #require(chosen).complete(request: request)

        // The URL arrived intact at the provider that was asked to serve it.
        #expect(voice.completeCallCount == 1)
        #expect(voice.lastRequest?.inputs.count == 1)
        #expect(voice.lastRequest?.inputs.first?.kind == .audio)
        #expect(voice.lastRequest?.inputs.first?.url == fixture)
        #expect(voice.lastRequest?.inputs.first?.mimeType == "audio/wav")

        // And the text provider was never asked.
        #expect(text.completeCallCount == 0)

        // The answer names who produced it (BR-AIL-06).
        #expect(response.providerID == "voice")
    }

    @Test("Audio carried as data reaches the provider by value")
    func dataCarriedAudioReachesTheProvider() async throws {
        let bytes = Data([0x52, 0x49, 0x46, 0x46, 0x00, 0x01, 0x02, 0x03])
        let voice = MockAIProvider(id: "voice", capabilities: [.voiceToText])
        let request = AIRequest(
            messages: [AIMessage(role: .user, content: "Transcribe.")],
            inputs: [AIInput(kind: .audio, data: bytes, mimeType: "audio/wav")]
        )

        _ = try await voice.complete(request: request)

        #expect(voice.lastRequest?.inputs.first?.data == bytes)
        #expect(voice.lastRequest?.inputs.first?.url == nil)
    }

    @Test("A request carrying audio survives a Codable round trip")
    func audioRequestRoundTrips() throws {
        let request = Self.makeAudioRequest(url: URL(fileURLWithPath: "/tmp/round.wav"))
        let decoded = try JSONDecoder().decode(
            AIRequest.self, from: try JSONEncoder().encode(request)
        )
        #expect(decoded.inputs == request.inputs)
        #expect(try decoded.audioInput(servedBy: .voiceToText).url == URL(fileURLWithPath: "/tmp/round.wav"))
    }
}

// MARK: - WhisperCLIProvider
//
// Gated like the provider: ShellKit is macOS-only (BR-AIL-02).
#if canImport(ShellKit)

    /// Stands in for the `whisper-cli` binary.
    ///
    /// Records the invocation and reproduces the one side effect the real
    /// binary has: writing the transcript to `<--output-file>.txt`. Modelling
    /// that rather than pre-planting a file means the provider is tested
    /// against the path it actually chose, not one the test guessed.
    ///
    /// Deliberately not `MockShellExecutor` (from `CLISubprocessProviderInitTests`):
    /// that mock returns a canned result and cannot write a file, which is the
    /// half of whisper's contract that matters here.
    final class StubWhisperCLI: ShellExecutorProtocol, @unchecked Sendable {
        /// Every argument list the provider ran, in order.
        var calls: [[String]] = []
        /// Written to `<--output-file>.txt` when the stub "succeeds". `nil`
        /// writes nothing — the binary that produced no transcript.
        var transcript: String?
        var stdout = ""
        var stderr = ""
        var exitCode: Int32 = 0
        var toThrow: (any Error)?
        /// The timeout the provider asked for.
        var observedTimeout: TimeInterval?
        /// Whether the audio file existed at the moment of the call.
        var audioExistedAtCallTime: Bool?

        init(transcript: String? = nil) {
            self.transcript = transcript
        }

        /// The `--output-file` value from an argument list.
        static func outputStem(in args: [String]) -> String? {
            guard let index = args.firstIndex(of: "--output-file"), index + 1 < args.count else {
                return nil
            }
            return args[index + 1]
        }

        func run(
            _ args: [String],
            cwd: String?,
            env: [String: String]?,
            timeout: TimeInterval,
            stdin: Data?
        ) async throws -> ShellCommandResult {
            calls.append(args)
            observedTimeout = timeout
            if let audio = args.last {
                audioExistedAtCallTime = FileManager.default.fileExists(atPath: audio)
            }
            if let toThrow { throw toThrow }
            if let transcript, let stem = Self.outputStem(in: args) {
                try? transcript.write(
                    to: URL(fileURLWithPath: stem + ".txt"), atomically: true, encoding: .utf8
                )
            }
            return ShellCommandResult(
                exitCode: exitCode,
                stdout: Data(stdout.utf8),
                stderr: Data(stderr.utf8),
                duration: 0.1
            )
        }
    }

    @Suite("WhisperCLIProvider")
    struct WhisperCLIProviderTests {

        typealias Temp = TranscriptionProviderTests.TempDirectory

        /// A provider and the two things a test needs to steer and inspect it.
        struct SUT {
            let provider: WhisperCLIProvider
            let binary: StubWhisperCLI
            let weights: URL
        }

        /// Builds a provider over an executable stand-in and real weights,
        /// with ``StubWhisperCLI`` standing in for the binary.
        static func makeSUT(
            in temp: Temp,
            transcript: String? = nil,
            stdout: String = "",
            stderr: String = "",
            exitCode: Int32 = 0,
            language: String = "auto",
            timeout: TimeInterval = 600
        ) throws -> SUT {
            let binaryFile = try temp.executable("whisper-cli")
            let weights = try temp.file("ggml-large-v3-turbo.bin", contents: "not-really-weights")
            let binary = StubWhisperCLI(transcript: transcript)
            binary.stdout = stdout
            binary.stderr = stderr
            binary.exitCode = exitCode
            let provider = WhisperCLIProvider(
                binaryPath: binaryFile.path,
                weightsPath: weights.path,
                language: language,
                timeout: timeout,
                executor: binary
            )
            return SUT(provider: provider, binary: binary, weights: weights)
        }

        // MARK: - Surface

        @Test("The provider advertises voice, and only voice")
        func capabilities() throws {
            let temp = try Temp()
            let sut = try Self.makeSUT(in: temp)
            #expect(sut.provider.capabilities == [.voiceToText])
            #expect(sut.provider.capabilities.contains(.voiceToText))
            #expect(!sut.provider.capabilities.contains(.textGeneration))
            #expect(sut.provider.id == "whisper-cli")
        }

        @Test("The provider is routable for voice like any other provider")
        func routableAsAnyProvider() async throws {
            let temp = try Temp()
            let sut = try Self.makeSUT(in: temp)
            let router = ProviderRouter(providers: [
                MockAIProvider(id: "text", capabilities: .textGeneration),
                sut.provider,
            ])
            let chosen = await router.route(capabilities: .voiceToText)
            #expect(chosen?.id == "whisper-cli")
        }

        // MARK: - The argument list

        @Test("The argument list names the weights, the language, the audio and the output")
        func argumentList() {
            let args = WhisperCLIProvider.arguments(
                binaryPath: "/opt/homebrew/bin/whisper-cli",
                weights: "/models/ggml-large-v3-turbo.bin",
                language: "fr",
                audioPath: "/tmp/talk.wav",
                outputStem: "/tmp/scratch/transcript"
            )
            #expect(args.first == "/opt/homebrew/bin/whisper-cli")
            #expect(args.contains("--output-txt"))
            #expect(args.contains("--no-prints"))
            // Measured against the real binary: --no-gpu is absent on purpose,
            // so Metal stays on.
            #expect(!args.contains("--no-gpu"))

            func value(after flag: String) -> String? {
                guard let index = args.firstIndex(of: flag), index + 1 < args.count else { return nil }
                return args[index + 1]
            }
            #expect(value(after: "--model") == "/models/ggml-large-v3-turbo.bin")
            #expect(value(after: "--language") == "fr")
            #expect(value(after: "--file") == "/tmp/talk.wav")
            // Measured: --output-file takes the stem and the binary appends
            // ".txt" itself, so passing an extension here would produce
            // "transcript.txt.txt".
            #expect(value(after: "--output-file") == "/tmp/scratch/transcript")
            #expect(value(after: "--output-file")?.hasSuffix(".txt") == false)
        }

        @Test("complete runs the argument list the provider publishes")
        func completeRunsThePublishedArguments() async throws {
            let temp = try Temp()
            let sut = try Self.makeSUT(in: temp, transcript: "Some words.", language: "fr")
            let audio = try temp.wav("talk.wav")
            _ = try await sut.provider.complete(
                request: TranscriptionProviderTests.makeAudioRequest(url: audio)
            )

            #expect(sut.binary.calls.count == 1)
            let ran = try #require(sut.binary.calls.first)

            // The shipped list and the asserted list are the same function —
            // not two copies written from the same reasoning (BR-EC-02). The
            // output stem is read back from the real call because the provider
            // picks a fresh one per run.
            let stem = try #require(StubWhisperCLI.outputStem(in: ran))
            let expected = WhisperCLIProvider.arguments(
                binaryPath: sut.provider.binaryPath,
                weights: sut.weights.path,
                language: "fr",
                audioPath: audio.path,
                outputStem: stem
            )
            #expect(ran == expected)
            #expect(sut.binary.observedTimeout == 600)
            #expect(sut.binary.audioExistedAtCallTime == true)
        }

        @Test("The transcript is written outside the caller's directory")
        func transcriptStaysOutOfTheCallersDirectory() async throws {
            let temp = try Temp()
            let sut = try Self.makeSUT(in: temp, transcript: "Some words.")
            let audio = try temp.wav("talk.wav")
            _ = try await sut.provider.complete(
                request: TranscriptionProviderTests.makeAudioRequest(url: audio)
            )

            let ran = try #require(sut.binary.calls.first)
            let stem = try #require(StubWhisperCLI.outputStem(in: ran))
            // Not next to the audio: that is the default the provider overrides.
            #expect(!stem.hasPrefix(temp.url.path))
            #expect(!FileManager.default.fileExists(atPath: audio.path + ".txt"))
            // And the scratch directory did not outlive the call.
            #expect(!FileManager.default.fileExists(atPath: stem + ".txt"))
        }

        @Test("A stale transcript from an earlier run cannot be read back")
        func staleTranscriptIsNotReadBack() async throws {
            let temp = try Temp()
            // The binary writes nothing this time...
            let sut = try Self.makeSUT(in: temp, transcript: nil)
            let audio = try temp.wav("talk.wav")
            // ...and a transcript from some earlier run sits beside the audio.
            try "A transcript from last week."
                .write(to: URL(fileURLWithPath: audio.path + ".txt"), atomically: true, encoding: .utf8)

            // Because the provider directs output into its own scratch
            // directory, the stale file is invisible to it: this is an empty
            // transcript, not last week's words.
            await #expect(throws: WhisperCLIProvider.TranscriptionError.self) {
                _ = try await sut.provider.complete(
                    request: TranscriptionProviderTests.makeAudioRequest(url: audio)
                )
            }
        }

        // MARK: - The happy path

        @Test("The transcript comes from the .txt the binary writes")
        func transcriptFromTheTextFile() async throws {
            let temp = try Temp()
            let sut = try Self.makeSUT(in: temp, transcript: "Voice goes through the same request.")
            let audio = try temp.wav("speech.wav")

            let response = try await sut.provider.complete(
                request: TranscriptionProviderTests.makeAudioRequest(url: audio)
            )

            #expect(response.content == "Voice goes through the same request.")
            #expect(response.providerID == "whisper-cli")
            #expect(response.model == "ggml-large-v3-turbo.bin")
            #expect(response.outputs.isEmpty)
        }

        @Test("The clean .txt wins over the timestamped stdout")
        func textFileWinsOverStdout() async throws {
            let temp = try Temp()
            // Measured: with --no-prints, stdout still carries segments.
            let sut = try Self.makeSUT(
                in: temp,
                transcript: "Voice goes through the same request.",
                stdout: "[00:00:00.000 --> 00:00:04.380]   Voice goes through the same request."
            )
            let audio = try temp.wav("speech.wav")

            let response = try await sut.provider.complete(
                request: TranscriptionProviderTests.makeAudioRequest(url: audio)
            )
            #expect(response.content == "Voice goes through the same request.")
            #expect(!response.content.contains("-->"))
        }

        @Test("Stdout is the fallback when no .txt was written")
        func stdoutFallback() async throws {
            let temp = try Temp()
            let sut = try Self.makeSUT(in: temp, transcript: nil, stdout: "A transcript from stdout.")
            let audio = try temp.wav("speech.wav")

            let response = try await sut.provider.complete(
                request: TranscriptionProviderTests.makeAudioRequest(url: audio)
            )
            #expect(response.content == "A transcript from stdout.")
        }

        @Test("A transcript is trimmed of the binary's trailing newline")
        func transcriptIsTrimmed() async throws {
            let temp = try Temp()
            let sut = try Self.makeSUT(in: temp, transcript: "\nVoice, with whitespace around it.\n\n")
            let audio = try temp.wav("speech.wav")

            let response = try await sut.provider.complete(
                request: TranscriptionProviderTests.makeAudioRequest(url: audio)
            )
            #expect(response.content == "Voice, with whitespace around it.")
        }

        // MARK: - Silent-success trap

        @Test("Exit 0 with no transcript is an error, never an empty answer")
        func exitZeroWithNoTranscriptThrows() async throws {
            // Measured against the real binary: a 440 Hz tone exits 0 and
            // writes an empty .txt. Returning "" here would report success
            // for a transcription that never happened.
            let temp = try Temp()
            let sut = try Self.makeSUT(in: temp, transcript: "", stdout: "", exitCode: 0)
            let audio = try temp.wav("tone.wav")

            await #expect(throws: WhisperCLIProvider.TranscriptionError.self) {
                _ = try await sut.provider.complete(
                    request: TranscriptionProviderTests.makeAudioRequest(url: audio)
                )
            }
        }

        @Test("Whitespace is not a transcript")
        func whitespaceIsNotATranscript() async throws {
            let temp = try Temp()
            let sut = try Self.makeSUT(in: temp, transcript: "  \n ", stdout: "   \n\t  ")
            let audio = try temp.wav("tone.wav")

            await #expect(throws: WhisperCLIProvider.TranscriptionError.self) {
                _ = try await sut.provider.complete(
                    request: TranscriptionProviderTests.makeAudioRequest(url: audio)
                )
            }
        }

        // MARK: - Failure modes

        @Test("A non-zero exit surfaces the binary's stderr")
        func nonZeroExitSurfacesStderr() async throws {
            let temp = try Temp()
            let sut = try Self.makeSUT(
                in: temp, stderr: "error: failed to initialize whisper context", exitCode: 1
            )
            let audio = try temp.wav("speech.wav")

            await #expect(
                throws: WhisperCLIProvider.TranscriptionError.nonZeroExit(
                    exitCode: 1, stderr: "error: failed to initialize whisper context"
                )
            ) {
                _ = try await sut.provider.complete(
                    request: TranscriptionProviderTests.makeAudioRequest(url: audio)
                )
            }
        }

        @Test("A non-zero exit is an error even when a transcript was written")
        func nonZeroExitBeatsAPartialTranscript() async throws {
            // A crash halfway through can leave a partial .txt behind. Exit
            // code decides, not the presence of a file.
            let temp = try Temp()
            let sut = try Self.makeSUT(
                in: temp, transcript: "Half a sen", stderr: "error: out of memory", exitCode: 3
            )
            let audio = try temp.wav("speech.wav")

            await #expect(throws: WhisperCLIProvider.TranscriptionError.self) {
                _ = try await sut.provider.complete(
                    request: TranscriptionProviderTests.makeAudioRequest(url: audio)
                )
            }
        }

        @Test("A timeout is a timeout, not a launch failure")
        func timeoutIsTyped() async throws {
            let temp = try Temp()
            let sut = try Self.makeSUT(in: temp)
            sut.binary.toThrow = ShellError.timeout(args: ["whisper-cli"], limit: 600)
            let audio = try temp.wav("long.wav")

            await #expect(throws: WhisperCLIProvider.TranscriptionError.timedOut(limit: 600)) {
                _ = try await sut.provider.complete(
                    request: TranscriptionProviderTests.makeAudioRequest(url: audio)
                )
            }
        }

        @Test("A launch failure keeps the reason")
        func launchFailureKeepsTheReason() async throws {
            let temp = try Temp()
            let sut = try Self.makeSUT(in: temp)
            sut.binary.toThrow = ShellError.launchFailed(args: ["whisper-cli"], underlying: "ENOENT")
            let audio = try temp.wav("speech.wav")

            await #expect(throws: WhisperCLIProvider.TranscriptionError.launchFailed("ENOENT")) {
                _ = try await sut.provider.complete(
                    request: TranscriptionProviderTests.makeAudioRequest(url: audio)
                )
            }
        }

        @Test("An unexpected executor error is still a launch failure, not a crash")
        func unexpectedErrorIsTyped() async throws {
            struct Surprise: Error {}
            let temp = try Temp()
            let sut = try Self.makeSUT(in: temp)
            sut.binary.toThrow = Surprise()
            let audio = try temp.wav("speech.wav")

            await #expect(throws: WhisperCLIProvider.TranscriptionError.self) {
                _ = try await sut.provider.complete(
                    request: TranscriptionProviderTests.makeAudioRequest(url: audio)
                )
            }
        }

        @Test("A missing binary is named, and nothing is run")
        func missingBinaryIsNamed() async throws {
            let temp = try Temp()
            let weights = try temp.file("ggml-small.bin", contents: "w")
            let binary = StubWhisperCLI()
            let absent = temp.url.appendingPathComponent("absent-whisper").path
            let provider = WhisperCLIProvider(
                binaryPath: absent, weightsPath: weights.path, executor: binary
            )
            let audio = try temp.wav("speech.wav")

            await #expect(throws: WhisperCLIProvider.TranscriptionError.binaryNotFound(absent)) {
                _ = try await provider.complete(
                    request: TranscriptionProviderTests.makeAudioRequest(url: audio)
                )
            }
            #expect(binary.calls.isEmpty)
        }

        @Test("Missing weights are named, and nothing is run")
        func missingWeightsAreNamed() async throws {
            let temp = try Temp()
            let binaryFile = try temp.executable("whisper-cli")
            let absent = temp.url.appendingPathComponent("ggml-absent.bin")
            let binary = StubWhisperCLI()
            let provider = WhisperCLIProvider(
                binaryPath: binaryFile.path, weightsPath: absent.path, executor: binary
            )
            let audio = try temp.wav("speech.wav")

            await #expect(
                throws: WhisperCLIProvider.TranscriptionError.weightsNotFound(absent.path)
            ) {
                _ = try await provider.complete(
                    request: TranscriptionProviderTests.makeAudioRequest(url: audio)
                )
            }
            #expect(binary.calls.isEmpty)
        }

        @Test("A remote audio URL is refused rather than silently skipped")
        func remoteURLIsRefused() async throws {
            let temp = try Temp()
            let sut = try Self.makeSUT(in: temp)
            let request = TranscriptionProviderTests.makeAudioRequest(
                url: URL(string: "https://example.com/talk.wav")!
            )

            await #expect(throws: WhisperCLIProvider.TranscriptionError.self) {
                _ = try await sut.provider.complete(request: request)
            }
            #expect(sut.binary.calls.isEmpty)
        }

        @Test("An audio URL with no file behind it is refused")
        func absentAudioFileIsRefused() async throws {
            let temp = try Temp()
            let sut = try Self.makeSUT(in: temp)
            let absent = temp.url.appendingPathComponent("never-written.wav")

            await #expect(
                throws: WhisperCLIProvider.TranscriptionError.audioNotReadable("no file at \(absent.path)")
            ) {
                _ = try await sut.provider.complete(
                    request: TranscriptionProviderTests.makeAudioRequest(url: absent)
                )
            }
            #expect(sut.binary.calls.isEmpty)
        }

        @Test("A request with no audio is refused before anything is launched")
        func requestWithoutAudioIsRefused() async throws {
            let temp = try Temp()
            let sut = try Self.makeSUT(in: temp)

            await #expect(throws: AIKitError.self) {
                _ = try await sut.provider.complete(request: TranscriptionProviderTests.makeTextRequest())
            }
            #expect(sut.binary.calls.isEmpty)
        }

        // MARK: - Staging audio carried by value

        @Test("Audio carried as data is staged on disk and removed afterwards")
        func dataCarriedAudioIsStagedAndCleanedUp() async throws {
            let temp = try Temp()
            let sut = try Self.makeSUT(in: temp, transcript: "Staged and transcribed.")
            let wavBytes = try Data(contentsOf: try temp.wav("source.wav"))

            let request = AIRequest(
                messages: [AIMessage(role: .user, content: "Transcribe.")],
                inputs: [AIInput(kind: .audio, data: wavBytes, mimeType: "audio/wav")]
            )
            let response = try await sut.provider.complete(request: request)
            #expect(response.content == "Staged and transcribed.")

            // The binary was handed a real path that existed when it ran...
            let staged = try #require(sut.binary.calls.first?.last)
            #expect(staged.hasSuffix(".wav"))
            #expect(sut.binary.audioExistedAtCallTime == true)
            // ...and the temporary file did not outlive the call.
            #expect(!FileManager.default.fileExists(atPath: staged))
        }

        @Test("Audio carried by URL is left where the caller put it")
        func urlCarriedAudioIsNotDeleted() async throws {
            let temp = try Temp()
            let sut = try Self.makeSUT(in: temp, transcript: "Words.")
            let audio = try temp.wav("keep-me.wav")

            _ = try await sut.provider.complete(
                request: TranscriptionProviderTests.makeAudioRequest(url: audio)
            )
            // A recording is the caller's file; transcribing it must not
            // consume it.
            #expect(FileManager.default.fileExists(atPath: audio.path))
        }

        @Test("Empty audio data is refused")
        func emptyAudioDataIsRefused() async throws {
            let temp = try Temp()
            let sut = try Self.makeSUT(in: temp)
            let request = AIRequest(
                messages: [AIMessage(role: .user, content: "Transcribe.")],
                inputs: [AIInput(kind: .audio, data: Data(), mimeType: "audio/wav")]
            )

            await #expect(throws: WhisperCLIProvider.TranscriptionError.self) {
                _ = try await sut.provider.complete(request: request)
            }
            #expect(sut.binary.calls.isEmpty)
        }

        @Test(
            "A staged file takes its extension from the media type",
            arguments: [
                ("audio/wav", "wav"),
                ("audio/x-wav", "wav"),
                ("audio/mpeg", "mp3"),
                ("audio/mp3", "mp3"),
                ("audio/m4a", "m4a"),
                ("audio/x-m4a", "m4a"),
                ("audio/flac", "flac"),
                ("audio/ogg", "ogg"),
                ("audio/opus", "ogg"),
            ]
        )
        func stagedExtensionFollowsTheMediaType(pair: (mime: String, expected: String)) {
            #expect(WhisperCLIProvider.fileExtension(for: pair.mime) == pair.expected)
        }

        @Test("An unknown or absent media type stages as wav")
        func unknownMediaTypeStagesAsWav() {
            #expect(WhisperCLIProvider.fileExtension(for: nil) == "wav")
            #expect(WhisperCLIProvider.fileExtension(for: "application/octet-stream") == "wav")
            // The lookup is case-insensitive: callers copy media types verbatim.
            #expect(WhisperCLIProvider.fileExtension(for: "AUDIO/MPEG") == "mp3")
        }

        // MARK: - Status

        @Test("Status is ready only when both the binary and the weights are present")
        func statusReflectsWhatIsInstalled() async throws {
            let temp = try Temp()
            let sut = try Self.makeSUT(in: temp)
            #expect(await sut.provider.status == .ready)

            let noBinary = WhisperCLIProvider(
                binaryPath: temp.url.appendingPathComponent("absent").path,
                weightsPath: sut.weights.path
            )
            #expect(await noBinary.status == .unavailable)

            let noWeights = WhisperCLIProvider(
                binaryPath: sut.provider.binaryPath,
                weightsPath: temp.url.appendingPathComponent("absent.bin").path
            )
            #expect(await noWeights.status == .unavailable)
        }

        @Test("An unavailable provider is not routed to")
        func unavailableProviderIsNotRouted() async throws {
            let temp = try Temp()
            let provider = WhisperCLIProvider(
                binaryPath: temp.url.appendingPathComponent("absent").path,
                weightsPath: temp.url.appendingPathComponent("absent.bin").path
            )
            let router = ProviderRouter(providers: [provider])
            #expect(await router.route(capabilities: .voiceToText) == nil)
        }

        // MARK: - Config resolution ("model reachable from config")

        @Test("The binary is resolved from the search path, newest name first")
        func binaryResolvedFromSearchPath() throws {
            let temp = try Temp()
            #expect(WhisperCLIProvider.resolveBinary(searchPaths: [temp.url.path]) == nil)

            // The legacy name is found when it is the only one installed...
            let legacy = try temp.executable("whisper-cpp")
            #expect(WhisperCLIProvider.resolveBinary(searchPaths: [temp.url.path]) == legacy.path)

            // ...and loses to the current name once that exists.
            let current = try temp.executable("whisper-cli")
            #expect(WhisperCLIProvider.resolveBinary(searchPaths: [temp.url.path]) == current.path)
        }

        @Test("A non-executable file of the right name is not the binary")
        func nonExecutableIsNotTheBinary() throws {
            let temp = try Temp()
            _ = try temp.file("whisper-cli", contents: "just a text file")
            #expect(WhisperCLIProvider.resolveBinary(searchPaths: [temp.url.path]) == nil)
        }

        @Test("Weights are resolved in descending preference")
        func weightsResolvedInPreferenceOrder() throws {
            let temp = try Temp()
            #expect(WhisperCLIProvider.resolveWeights(in: [temp.url]) == nil)

            let base = try temp.file("ggml-base.bin", contents: "w")
            #expect(WhisperCLIProvider.resolveWeights(in: [temp.url]) == base.path)

            // large-v3-turbo is the fleet's choice and outranks base.
            let turbo = try temp.file("ggml-large-v3-turbo.bin", contents: "w")
            #expect(WhisperCLIProvider.resolveWeights(in: [temp.url]) == turbo.path)

            // A caller that wants something else passes its own order rather
            // than AIKit guessing from the host (BR-AIL-08).
            #expect(
                WhisperCLIProvider.resolveWeights(in: [temp.url], preferring: ["ggml-base.bin"]) == base.path
            )
        }

        @Test("Earlier directories win over later ones")
        func earlierDirectoryWins() throws {
            let first = try Temp()
            let second = try Temp()
            let preferred = try first.file("ggml-base.bin", contents: "w")
            _ = try second.file("ggml-large-v3-turbo.bin", contents: "w")
            #expect(
                WhisperCLIProvider.resolveWeights(in: [first.url, second.url]) == preferred.path
            )
        }

        @Test("The weights preference is ordered large to small")
        func weightsPreferenceIsOrdered() throws {
            let preference = WhisperCLIProvider.weightsPreference
            #expect(preference.first == "ggml-large-v3-turbo.bin")
            let large = try #require(preference.firstIndex(of: "ggml-large-v3.bin"))
            let small = try #require(preference.firstIndex(of: "ggml-small.bin"))
            #expect(large < small)
            #expect(WhisperCLIProvider.binaryNames == ["whisper-cli", "whisper-cpp"])
        }

        @Test("An empty or absent PATH yields no search directories")
        func pathEntriesHandlesAnEmptyPath() {
            #expect(WhisperCLIProvider.pathEntries(nil).isEmpty)
            #expect(WhisperCLIProvider.pathEntries("").isEmpty)
            #expect(WhisperCLIProvider.pathEntries("/usr/bin::/opt/homebrew/bin") == ["/usr/bin", "/opt/homebrew/bin"])
        }
    }

#endif  // canImport(ShellKit)

// MARK: - CoreML voice path
//
// The iOS half of S3.5. End-to-end Core ML transcription needs real Whisper
// weights, which these tests do not ship and must not download, so what is
// asserted here is everything up to the model call: which descriptor takes the
// voice path, how weights are found, how an audio file is decoded, and which
// model inputs the provider refuses by name instead of filling with nonsense.
#if canImport(CoreML)

    @Suite("CoreMLEngine voice path")
    struct CoreMLVoicePathTests {

        typealias Temp = TranscriptionProviderTests.TempDirectory

        static func makeDescriptor(
            modelId: String = "whisper-small",
            capabilities: AICapabilities = .voiceToText,
            domain: ModelDomain = .voice,
            localPath: URL? = nil
        ) -> ModelDescriptor {
            ModelDescriptor(
                id: ModelIdentifier(provider: "coreml", modelId: modelId),
                name: "Whisper Small",
                author: "openai",
                description: "Speech to text",
                capabilities: capabilities,
                format: .coreml,
                parameters: "244M",
                sizeBytes: 460_000_000,
                architecture: "whisper",
                domain: domain,
                localPath: localPath
            )
        }

        // MARK: - Which descriptor takes the voice path

        @Test("A voice descriptor is recognised by either signal")
        func voiceDescriptorRecognised() {
            // Domain alone is enough.
            #expect(CoreMLEngine.isVoice(Self.makeDescriptor(capabilities: [], domain: .voice)))
            // Capability alone is enough.
            #expect(CoreMLEngine.isVoice(Self.makeDescriptor(capabilities: .voiceToText, domain: .llm)))
            // Both together, naturally.
            #expect(CoreMLEngine.isVoice(Self.makeDescriptor()))
        }

        @Test("A vision descriptor does not take the voice path")
        func visionDescriptorIsNotVoice() {
            #expect(!CoreMLEngine.isVoice(Self.makeDescriptor(capabilities: .vision, domain: .vision)))
            #expect(!CoreMLEngine.isVoice(Self.makeDescriptor(capabilities: [.ocr], domain: .vision)))
        }

        @Test("Loading voice weights that are not there names the model, not the domain")
        func loadingAbsentVoiceWeightsThrows() async throws {
            let temp = try Temp()
            let engine = CoreMLEngine(modelsDirectory: temp.url)
            let descriptor = Self.makeDescriptor()

            await #expect(throws: AIKitError.modelNotFound(descriptor.id)) {
                _ = try await engine.loadModel(descriptor)
            }
            #expect(engine.loadedModels().isEmpty)
        }

        // MARK: - Weights from config

        @Test("Voice weights are resolved from the models directory in preference order")
        func voiceWeightsResolvedInPreferenceOrder() throws {
            let temp = try Temp()
            let engine = CoreMLEngine(modelsDirectory: temp.url)
            #expect(engine.resolveVoiceWeights() == nil)

            let base = try temp.file("whisper-base.mlmodelc", contents: "m")
            #expect(engine.resolveVoiceWeights() == base)

            // The Flsh default (small) outranks base.
            let small = try temp.file("whisper-small.mlmodelc", contents: "m")
            #expect(engine.resolveVoiceWeights() == small)

            // A Mac caller with the headroom passes its own order.
            #expect(
                engine.resolveVoiceWeights(preferring: ["whisper-base.mlmodelc"]) == base
            )
        }

        @Test("The Core ML preference starts at the Flsh on-device default")
        func preferenceStartsAtSmall() {
            // features/flsh-whisper-e4b-voice-stack.md BR-01: Whisper-small is
            // the on-device default, large is the Mac upgrade.
            #expect(CoreMLEngine.voiceWeightsPreference.first == "whisper-small.mlmodelc")
            #expect(CoreMLEngine.voiceWeightsPreference.contains("whisper-base.mlmodelc"))
        }

        // MARK: - Resolving the request's audio

        @Test("A file URL is used where it already lives")
        func fileURLIsUsedInPlace() throws {
            let temp = try Temp()
            let audio = try temp.wav("speech.wav")
            #expect(try CoreMLWhisperProvider.fileURL(for: AIInput(kind: .audio, url: audio)) == audio)
        }

        @Test("A remote URL and an absent file are both refused")
        func unreadableAudioIsRefused() throws {
            let temp = try Temp()
            #expect(throws: AIKitError.self) {
                _ = try CoreMLWhisperProvider.fileURL(
                    for: AIInput(kind: .audio, url: URL(string: "https://example.com/a.wav")!)
                )
            }
            #expect(throws: AIKitError.self) {
                _ = try CoreMLWhisperProvider.fileURL(
                    for: AIInput(kind: .audio, url: temp.url.appendingPathComponent("absent.wav"))
                )
            }
            #expect(throws: AIKitError.self) {
                _ = try CoreMLWhisperProvider.fileURL(for: AIInput(kind: .audio, data: Data()))
            }
        }

        @Test("Audio carried as data is staged as a readable file")
        func dataIsStagedAsAFile() throws {
            let temp = try Temp()
            let bytes = try Data(contentsOf: try temp.wav("source.wav"))
            let staged = try CoreMLWhisperProvider.fileURL(for: AIInput(kind: .audio, data: bytes))
            defer { try? FileManager.default.removeItem(at: staged) }
            #expect(FileManager.default.fileExists(atPath: staged.path))
            #expect(try Data(contentsOf: staged) == bytes)
        }

        // MARK: - Decoding to what Whisper is trained on

        @Test("A 16 kHz mono file decodes to one sample per frame")
        func decodesSixteenKiloHertzMono() throws {
            let temp = try Temp()
            let audio = try temp.wav("one-second.wav", seconds: 1.0, sampleRate: 16_000)
            let samples = try CoreMLWhisperProvider.samples(at: audio)
            #expect(samples.count == 16_000)
            #expect(CoreMLWhisperProvider.sampleRate == 16_000)
        }

        @Test("A 44.1 kHz file is resampled to 16 kHz rather than reinterpreted")
        func resamplesToSixteenKiloHertz() throws {
            let temp = try Temp()
            let audio = try temp.wav("cd-rate.wav", seconds: 1.0, sampleRate: 44_100)
            let samples = try CoreMLWhisperProvider.samples(at: audio)
            // One second at 16 kHz, within a frame of conversion slack.
            #expect(abs(samples.count - 16_000) <= 256)
        }

        @Test("A file that is not audio is refused, not decoded")
        func nonAudioFileIsRefused() throws {
            let temp = try Temp()
            let notAudio = try temp.file("notes.txt", contents: "this is not a waveform")
            #expect(throws: AIKitError.self) {
                _ = try CoreMLWhisperProvider.samples(at: notAudio)
            }
        }

        // MARK: - Which model inputs can be filled from an audio file

        @Test("A one-dimensional window is a window of samples")
        func sampleWindowSqueezesTheBatchAxis() {
            #expect(CoreMLWhisperProvider.sampleWindow(in: [480_000]) == 480_000)
            // A leading batch axis is not a second real axis.
            #expect(CoreMLWhisperProvider.sampleWindow(in: [1, 480_000]) == 480_000)
            #expect(CoreMLWhisperProvider.sampleWindow(in: [1, 1, 480_000]) == 480_000)
        }

        @Test("A spectrogram shape is not a window of samples")
        func melShapeIsNotASampleWindow() {
            // 80 mel bins × 3000 frames: the encoder's input, which only the
            // mel stage can produce.
            #expect(CoreMLWhisperProvider.sampleWindow(in: [80, 3_000]) == nil)
            #expect(CoreMLWhisperProvider.sampleWindow(in: [1, 80, 3_000]) == nil)
            #expect(CoreMLWhisperProvider.sampleWindow(in: [1]) == nil)
            #expect(CoreMLWhisperProvider.sampleWindow(in: []) == nil)
        }

        @Test("A short clip is zero-padded to the declared window")
        func shortClipIsPadded() throws {
            let array = try CoreMLWhisperProvider.multiArray(
                from: [0.5, -0.25],
                window: 4,
                shape: [4],
                dataType: .float32,
                feature: "audio"
            )
            #expect(array.count == 4)
            #expect(abs(array[0].floatValue - 0.5) < 0.0001)
            #expect(abs(array[1].floatValue + 0.25) < 0.0001)
            // The tail is silence, not uninitialised memory.
            #expect(array[2].floatValue == 0)
            #expect(array[3].floatValue == 0)
        }

        @Test("A long clip is truncated to the declared window")
        func longClipIsTruncated() throws {
            let array = try CoreMLWhisperProvider.multiArray(
                from: [1, 2, 3, 4, 5],
                window: 2,
                shape: [2],
                dataType: .float32,
                feature: "audio"
            )
            #expect(array.count == 2)
            #expect(array[0].floatValue == 1)
            #expect(array[1].floatValue == 2)
        }

        @Test("A non-float window is refused by name")
        func nonFloatWindowIsRefused() {
            #expect(throws: AIKitError.self) {
                _ = try CoreMLWhisperProvider.multiArray(
                    from: [0.5], window: 1, shape: [1], dataType: .int32, feature: "audio"
                )
            }
        }

        @Test("A model declaring no inputs is refused")
        func modelWithNoInputsIsRefused() throws {
            let temp = try Temp()
            let audio = try temp.wav("speech.wav")
            #expect(throws: AIKitError.self) {
                _ = try CoreMLWhisperProvider.featureProvider(for: [:], audioURL: audio)
            }
        }

        // MARK: - Never an empty transcript

        @Test("No string output means no transcript, and no transcript is an error")
        func emptyOutputIsNotATranscript() throws {
            // An empty string is what the stub this provider replaces returned
            // forever; it must read as "nothing", not as a successful answer.
            let blank = try MLDictionaryFeatureProvider(dictionary: ["text": "   " as NSString])
            #expect(CoreMLWhisperProvider.transcript(in: blank) == nil)

            let numeric = try MLDictionaryFeatureProvider(dictionary: ["logits": NSNumber(value: 1.0)])
            #expect(CoreMLWhisperProvider.transcript(in: numeric) == nil)
        }

        @Test("A string output is the transcript, trimmed")
        func stringOutputIsTheTranscript() throws {
            let output = try MLDictionaryFeatureProvider(
                dictionary: ["text": "  Voice goes through the same request.\n" as NSString]
            )
            #expect(CoreMLWhisperProvider.transcript(in: output) == "Voice goes through the same request.")
        }
    }

#endif  // canImport(CoreML)
