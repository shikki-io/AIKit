import Foundation
import Testing

@testable import AIKit

/// W2 of spec ai-layer-aikit-core-2026-10-31: a request carries media inputs
/// (BR-AIL-05) and a response names who answered (BR-AIL-06), without
/// invalidating a payload a caller stored before either field existed.
@Suite("AIRequest inputs and AIResponse provenance")
struct AIRequestInputTests {

    // MARK: - Fixtures

    /// Exactly what the pre-W2 encoder wrote for a fully populated request:
    /// no `inputs` key, because the field did not exist.
    private let legacyRequestJSON = """
        {
          "messages": [
            { "role": "system", "content": "You are helpful." },
            { "role": "user", "content": "Hello" }
          ],
          "systemPrompt": "Be concise",
          "temperature": 0.5,
          "maxTokens": 256,
          "model": { "provider": "openai", "modelId": "gpt-5" },
          "tools": [
            { "name": "search", "description": "Search the web", "parametersSchema": "{}" }
          ],
          "responseFormat": "json"
        }
        """

    /// The pre-W2 encoder's output for a response: no `providerID`, no `outputs`.
    private let legacyResponseJSON = """
        {
          "content": "Hello!",
          "model": "gpt-5",
          "tokensUsed": { "prompt": 10, "completion": 5, "total": 15 },
          "latencyMs": 42
        }
        """

    private func roundTrip<T: Codable>(_ value: T) throws -> T {
        let data = try JSONEncoder().encode(value)
        return try JSONDecoder().decode(T.self, from: data)
    }

    // MARK: - T-AIL-03 — a stored payload stays readable

    @Test("A request stored before inputs existed still decodes, with no inputs")
    func decodesOldPayload() throws {
        let data = Data(legacyRequestJSON.utf8)

        let decoded = try JSONDecoder().decode(AIRequest.self, from: data)

        #expect(decoded.inputs.isEmpty)
        // Every pre-existing field survives the custom decoder.
        #expect(decoded.messages.count == 2)
        #expect(decoded.messages[0].role == .system)
        #expect(decoded.messages[1].content == "Hello")
        #expect(decoded.systemPrompt == "Be concise")
        #expect(decoded.temperature == 0.5)
        #expect(decoded.maxTokens == 256)
        #expect(decoded.model?.provider == "openai")
        #expect(decoded.model?.modelId == "gpt-5")
        #expect(decoded.tools?.first?.name == "search")
        #expect(decoded.responseFormat == .json)
    }

    @Test("A minimal old request — only the keys the old encoder always wrote")
    func decodesMinimalOldPayload() throws {
        let json = """
            { "messages": [ { "role": "user", "content": "hi" } ], "temperature": 0.7, "maxTokens": 1024 }
            """

        let decoded = try JSONDecoder().decode(AIRequest.self, from: Data(json.utf8))

        #expect(decoded.inputs.isEmpty)
        #expect(decoded.messages.count == 1)
        #expect(decoded.systemPrompt == nil)
        #expect(decoded.tools == nil)
        #expect(decoded.responseFormat == nil)
    }

    @Test("A response stored before providerID existed still decodes, unnamed")
    func decodesOldResponsePayload() throws {
        let data = Data(legacyResponseJSON.utf8)

        let decoded = try JSONDecoder().decode(AIResponse.self, from: data)

        // The silent-fail this guards: a cache read throwing on an old row.
        #expect(decoded.providerID.isEmpty)
        #expect(decoded.outputs.isEmpty)
        #expect(decoded.content == "Hello!")
        #expect(decoded.model == "gpt-5")
        #expect(decoded.tokensUsed.total == 15)
        #expect(decoded.latencyMs == 42)
    }

    // MARK: - T-AIL-04 — both input forms round trip

    @Test("An input carried by URL round trips")
    func carriesURL() throws {
        let audioURL = URL(fileURLWithPath: "/tmp/interview.wav")
        let request = AIRequest(
            messages: [AIMessage(role: .user, content: "Transcribe this")],
            inputs: [AIInput(kind: .audio, url: audioURL, mimeType: "audio/wav")],
        )

        let decoded = try roundTrip(request)

        #expect(decoded.inputs.count == 1)
        #expect(decoded.inputs[0].kind == .audio)
        #expect(decoded.inputs[0].source == .url(audioURL))
        #expect(decoded.inputs[0].url == audioURL)
        #expect(decoded.inputs[0].data == nil)
        #expect(decoded.inputs[0].mimeType == "audio/wav")
    }

    @Test("An input carried by data round trips")
    func carriesData() throws {
        let bytes = Data([0x52, 0x49, 0x46, 0x46, 0x00, 0xFF])
        let request = AIRequest(
            messages: [AIMessage(role: .user, content: "Transcribe this")],
            inputs: [AIInput(kind: .audio, data: bytes, mimeType: "audio/wav")],
        )

        let decoded = try roundTrip(request)

        #expect(decoded.inputs.count == 1)
        #expect(decoded.inputs[0].source == .data(bytes))
        #expect(decoded.inputs[0].data == bytes)
        #expect(decoded.inputs[0].url == nil)
    }

    @Test("Both input forms round trip together, in order")
    func carriesURLAndData() throws {
        let imageURL = URL(string: "https://example.invalid/cat.png")!
        let bytes = Data([0x89, 0x50, 0x4E, 0x47])
        let request = AIRequest(
            messages: [AIMessage(role: .user, content: "Describe both")],
            inputs: [
                AIInput(kind: .image, url: imageURL, mimeType: "image/png"),
                AIInput(kind: .image, data: bytes, mimeType: "image/png"),
                AIInput(kind: .document, url: URL(fileURLWithPath: "/tmp/brief.pdf")),
            ],
        )

        let decoded = try roundTrip(request)

        #expect(decoded.inputs == request.inputs)
        #expect(decoded.inputs.count == 3)
        #expect(decoded.inputs[0].url == imageURL)
        #expect(decoded.inputs[1].data == bytes)
        #expect(decoded.inputs[2].kind == .document)
        #expect(decoded.inputs[2].mimeType == nil)
    }

    @Test("Every media kind round trips")
    func everyKindRoundTrips() throws {
        let kinds: [AIMediaKind] = [.audio, .image, .video, .document, .text]

        for kind in kinds {
            let input = AIInput(kind: kind, url: URL(fileURLWithPath: "/tmp/f"))
            #expect(try roundTrip(input).kind == kind)
        }
    }

    @Test("A media source carrying neither a url nor data is rejected, not guessed")
    func rejectsEmptyMediaSource() throws {
        let json = """
            { "kind": "audio", "source": { } }
            """

        #expect(throws: DecodingError.self) {
            try JSONDecoder().decode(AIInput.self, from: Data(json.utf8))
        }
    }

    @Test("An input encodes to a flat, stable wire shape")
    func inputWireShape() throws {
        let input = AIInput(kind: .audio, url: URL(fileURLWithPath: "/tmp/a.wav"), mimeType: "audio/wav")

        let data = try JSONEncoder().encode(input)
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]

        #expect(json?["kind"] as? String == "audio")
        #expect(json?["mimeType"] as? String == "audio/wav")
        let source = json?["source"] as? [String: Any]
        #expect(source?["url"] as? String == "file:///tmp/a.wav")
        #expect(source?.keys.contains("data") == false)
    }

    // MARK: - Outputs mirror inputs (BR-AIL-05)

    @Test("Outputs round trip in both forms")
    func carriesOutputs() throws {
        let rendered = URL(fileURLWithPath: "/tmp/out.mp4")
        let response = AIResponse(
            content: "",
            model: "wan-2.2",
            tokensUsed: TokenUsage(prompt: 0, completion: 0),
            latencyMs: 9,
            providerID: "mlx-video",
            outputs: [
                AIOutput(kind: .video, url: rendered, mimeType: "video/mp4"),
                AIOutput(kind: .image, data: Data([0x01, 0x02])),
            ],
        )

        let decoded = try roundTrip(response)

        #expect(decoded == response)
        #expect(decoded.outputs.count == 2)
        #expect(decoded.outputs[0].url == rendered)
        #expect(decoded.outputs[0].mimeType == "video/mp4")
        #expect(decoded.outputs[1].data == Data([0x01, 0x02]))
    }

    // MARK: - BR-AIL-06 — a response names who answered

    @Test("A response round trips its provenance")
    func responseCarriesProvenance() throws {
        let response = AIResponse(
            content: "Hello!",
            model: "devstral-small-2507",
            tokensUsed: TokenUsage(prompt: 10, completion: 5),
            latencyMs: 42,
            providerID: "lmstudio-local",
        )

        let decoded = try roundTrip(response)

        #expect(decoded.providerID == "lmstudio-local")
        #expect(decoded.model == "devstral-small-2507")
    }

    @Test("A provider names itself in the response it returns")
    func providerNamesItself() async throws {
        let provider = MockAIProvider(id: "lmstudio-local", capabilities: .textGeneration)

        let response = try await provider.complete(
            request: AIRequest(messages: [AIMessage(role: .user, content: "hi")])
        )

        #expect(response.providerID == "lmstudio-local")
    }

    @Test("A canned response that already names its author keeps that name")
    func cannedProvenanceIsNotOverwritten() async throws {
        let canned = AIResponse(
            content: "from elsewhere",
            model: "m",
            tokensUsed: TokenUsage(prompt: 1, completion: 1),
            latencyMs: 1,
            providerID: "some-other-provider",
        )
        let provider = MockAIProvider(id: "mock", responses: [canned])

        let response = try await provider.complete(
            request: AIRequest(messages: [AIMessage(role: .user, content: "hi")])
        )

        #expect(response.providerID == "some-other-provider")
    }

    // MARK: - An input survives the trip to a provider

    @Test("The inputs a caller attached reach the provider intact")
    func inputsReachTheProvider() async throws {
        let audioURL = URL(fileURLWithPath: "/tmp/interview.wav")
        let provider = MockAIProvider(id: "whisper", capabilities: .voiceToText)
        let request = AIRequest(
            messages: [AIMessage(role: .user, content: "Transcribe")],
            inputs: [AIInput(kind: .audio, url: audioURL, mimeType: "audio/wav")],
        )

        _ = try await provider.complete(request: request)

        #expect(provider.receivedRequests.count == 1)
        #expect(provider.lastRequest?.inputs.first?.url == audioURL)
        #expect(provider.lastRequest?.inputs.first?.kind == .audio)
    }

    @Test("A text-only request reaches the provider carrying no inputs")
    func textOnlyRequestCarriesNoInputs() async throws {
        let provider = MockAIProvider()

        _ = try await provider.complete(
            request: AIRequest(messages: [AIMessage(role: .user, content: "hi")])
        )

        #expect(provider.lastRequest?.inputs.isEmpty == true)
    }
}
