import Foundation
import Testing

@testable import AIKit

// Spoken audio captured from a real microphone, transcribed through the same
// AIRequest every other provider takes. `scripts/voice-proof.sh` records the wav.
#if canImport(ShellKit)
    @Suite("Voice loopback — live")
    struct VoiceLoopbackLiveTests {
        private static let environment = ProcessInfo.processInfo.environment

        @Test("A spoken phrase comes back through WhisperCLIProvider")
        func spokenPhraseIsTranscribed() async throws {
            let wav = try #require(Self.environment["AIKIT_VOICE_WAV"])
            let binary = try #require(Self.environment["AIKIT_WHISPER_CLI"])
            let weights = try #require(Self.environment["AIKIT_WHISPER_MODEL"])
            let expected = try #require(Self.environment["AIKIT_VOICE_EXPECTED"])
                .lowercased().split(separator: " ").map(String.init)

            let provider = WhisperCLIProvider(binaryPath: binary, weightsPath: weights)
            let request = AIRequest(
                messages: [],
                inputs: [AIInput(kind: .audio, source: .url(URL(fileURLWithPath: wav)))]
            )
            let response = try await provider.complete(request: request)

            let heard = response.content.lowercased()
            let matched = expected.filter(heard.contains)
            #expect(response.providerID == provider.id)
            #expect(matched.count * 2 >= expected.count, "heard: \(heard)")
        }
    }
#endif
