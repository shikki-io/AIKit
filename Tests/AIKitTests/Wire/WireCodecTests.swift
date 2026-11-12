import Foundation
import NetKit
import Testing

@testable import AIKit

// MARK: - WireCodecTests
//
// T-AIL-40. BR-AIL-17: a codec declares what it supports, and a request field
// it cannot carry throws `AIKitError.unsupported(field:family:)` BEFORE
// anything is sent. Nothing is dropped silently.
//
// What this replaces, measured at 81aa16c: `AnthropicEndPoint` built its body
// without reading `request.tools` or `request.responseFormat` at all;
// `LMStudioEndPoint` did the same and flattened roles into `"role: text"`;
// `OpenAIEndPoint` ignored `request.inputs`. In every case `AICapabilities`
// went on advertising the feature, so a caller's tool definitions simply never
// reached the model and nothing said so.

@Suite("WireCodecTests — a codec refuses what it cannot carry")
struct WireCodecRefusalTests {

    // MARK: - The matrix

    @Test("tools are refused by every family that does not carry them", arguments: wireFamilyFixtures)
    func refusesTools(fixture: WireFamilyFixture) async throws {
        var request = makeFixtureRequest()
        request.tools = [AITool(name: "grep", description: "search", parametersSchema: "{\"type\":\"object\"}")]

        if fixture.codec.capabilities.tools {
            // `openai-chat` does carry them, and sends them.
            let body = try await sentBody(fixture, request: request)
            let tools = try #require(body["tools"] as? [[String: Any]])
            #expect(tools.count == 1)
        } else {
            try await expectRefusal(fixture, request: request, field: "tools")
        }
    }

    @Test("a response format is refused by every family that does not send one", arguments: wireFamilyFixtures)
    func refusesResponseFormat(fixture: WireFamilyFixture) async throws {
        var request = makeFixtureRequest()
        request.responseFormat = .json

        if fixture.codec.capabilities.responseFormats.contains(.json) {
            let body = try await sentBody(fixture, request: request)
            let format = try #require(body["response_format"] as? [String: Any])
            #expect(format["type"] as? String == "json_object")
        } else {
            try await expectRefusal(fixture, request: request, field: "responseFormat=json")
        }
    }

    @Test(
        "every media input kind is refused, by every family",
        arguments: wireFamilyFixtures,
        [AIMediaKind.audio, .image, .video, .document, .text]
    )
    func refusesEveryInputKind(fixture: WireFamilyFixture, kind: AIMediaKind) async throws {
        // No HTTP family carries media today. Declaring that and refusing is
        // the whole change: `request.inputs` used to be ignored outright while
        // `AICapabilities.vision` advertised it.
        #expect(fixture.codec.capabilities.inputKinds.isEmpty)

        var request = makeFixtureRequest()
        request.inputs = [AIInput(kind: kind, url: URL(fileURLWithPath: "/tmp/sample"))]
        try await expectRefusal(fixture, request: request, field: "inputs[].kind=\(kind.rawValue)")
    }

    @Test("an image attached to a message is refused, not dropped", arguments: wireFamilyFixtures)
    func refusesMessageImages(fixture: WireFamilyFixture) async throws {
        let request = AIRequest(
            messages: [AIMessage(role: .user, content: "what is this", images: [Data([0x89, 0x50])])],
            maxTokens: 16
        )
        try await expectRefusal(fixture, request: request, field: "messages[].images")
    }

    @Test("a role a family cannot keep is refused, never relabelled", arguments: wireFamilyFixtures)
    func refusesRolesItCannotKeep(fixture: WireFamilyFixture) async throws {
        for role in [AIMessage.Role.system, .user, .assistant, .tool] {
            let message =
                role == .tool
                ? AIMessage(role: .tool, content: "42", toolCallId: "call_1")
                : AIMessage(role: role, content: "hi")
            let request = AIRequest(messages: [AIMessage(role: .user, content: "hi"), message], maxTokens: 16)

            if fixture.codec.capabilities.roles.contains(role) {
                _ = try await sentBody(fixture, request: request)
            } else {
                try await expectRefusal(fixture, request: request, field: "messages[].role=\(role.rawValue)")
            }
        }
    }

    // MARK: - Tool-call ids are never minted

    @Test("a tool call with no id is refused instead of being given a fresh one")
    func refusesAMissingToolCallID() async throws {
        // `OpenAIEndPoint` used to write `id: tc.id ?? UUID().uuidString`,
        // handing the server an id it never issued, so the tool result that
        // followed could not be matched to the call.
        let fixture = try #require(wireFamilyFixtures.first { $0.family == .openAIChat })
        let request = AIRequest(
            messages: [
                AIMessage(role: .user, content: "search"),
                AIMessage(
                    role: .assistant,
                    content: "",
                    toolCalls: [AIToolCall(id: nil, name: "grep", arguments: "{}")]
                ),
            ],
            maxTokens: 16
        )
        try await expectRefusal(fixture, request: request, field: "messages[].toolCalls[].id is missing")
    }

    @Test("a tool call with an empty id is refused too")
    func refusesAnEmptyToolCallID() async throws {
        let fixture = try #require(wireFamilyFixtures.first { $0.family == .openAIChat })
        let request = AIRequest(
            messages: [
                AIMessage(role: .user, content: "search"),
                AIMessage(
                    role: .assistant,
                    content: "",
                    toolCalls: [AIToolCall(id: "", name: "grep", arguments: "{}")]
                ),
            ],
            maxTokens: 16
        )
        try await expectRefusal(fixture, request: request, field: "messages[].toolCalls[].id is missing")
    }

    @Test("the server's tool-call id is the one that goes back out")
    func keepsTheServersToolCallID() async throws {
        let fixture = try #require(wireFamilyFixtures.first { $0.family == .openAIChat })
        let request = AIRequest(
            messages: [
                AIMessage(role: .user, content: "search"),
                AIMessage(
                    role: .assistant,
                    content: "",
                    toolCalls: [AIToolCall(id: "call_abc123", name: "grep", arguments: "{\"q\":\"x\"}")]
                ),
                AIMessage(role: .tool, content: "found", toolCallId: "call_abc123"),
            ],
            maxTokens: 16
        )
        let body = try await sentBody(fixture, request: request)
        let messages = try #require(body["messages"] as? [[String: Any]])
        let assistant = try #require(messages.first { $0["role"] as? String == "assistant" })
        let calls = try #require(assistant["tool_calls"] as? [[String: Any]])
        #expect(calls.first?["id"] as? String == "call_abc123")
        let tool = try #require(messages.first { $0["role"] as? String == "tool" })
        #expect(tool["tool_call_id"] as? String == "call_abc123")
    }

    @Test("a tool result with no id to echo is refused")
    func refusesAToolResultWithNoID() async throws {
        let fixture = try #require(wireFamilyFixtures.first { $0.family == .openAIChat })
        let request = AIRequest(
            messages: [AIMessage(role: .user, content: "x"), AIMessage(role: .tool, content: "42")],
            maxTokens: 16
        )
        try await expectRefusal(fixture, request: request, field: "messages[].toolCallId is missing")
    }

    // MARK: - LM Studio native, the narrowest family

    @Test("the native family refuses a multi-role conversation rather than flattening it")
    func nativeRefusesAConversation() async throws {
        let fixture = try #require(wireFamilyFixtures.first { $0.family == .lmStudioNative })
        let request = AIRequest(
            messages: [
                AIMessage(role: .user, content: "Name a colour."),
                AIMessage(role: .assistant, content: "Blue."),
                AIMessage(role: .user, content: "Another."),
            ],
            maxTokens: 16
        )
        try await expectRefusal(fixture, request: request, field: "messages[].role=assistant")
    }

    @Test("the native family keeps the system prompt in its own documented field")
    func nativeKeepsTheSystemPromptStructurally() async throws {
        let fixture = try #require(wireFamilyFixtures.first { $0.family == .lmStudioNative })
        let body = try await sentBody(
            fixture,
            request: makeFixtureRequest("Name a colour.", systemPrompt: "Reply with one word only.")
        )
        // `system_prompt`, measured honoured by the live server (it raised
        // `input_tokens` to 19 and changed the answer). NOT a "system: …"
        // prefix glued onto the prompt.
        #expect(body["system_prompt"] as? String == "Reply with one word only.")
        let input = try #require(body["input"] as? [[String: Any]])
        #expect(input.count == 1)
        #expect(input.first?["content"] as? String == "Name a colour.")
        #expect(input.first?["type"] as? String == "text")
    }

    @Test("the native family sends every user turn as its own typed part, in order")
    func nativeKeepsUserTurnsInOrder() async throws {
        let fixture = try #require(wireFamilyFixtures.first { $0.family == .lmStudioNative })
        let request = AIRequest(
            messages: [
                AIMessage(role: .user, content: "Name a colour."),
                AIMessage(role: .user, content: "Make it blue."),
            ],
            maxTokens: 16
        )
        let body = try await sentBody(fixture, request: request)
        let input = try #require(body["input"] as? [[String: Any]])
        #expect(input.map { $0["content"] as? String } == ["Name a colour.", "Make it blue."])
        // Measured: `{"type":"text","text":…}` is refused by the server with
        // `'input.0.content' is required`.
        #expect(input.allSatisfy { $0["content"] != nil && $0["text"] == nil })
    }

    @Test("the native family refuses a request with no user message to send")
    func nativeRefusesAnEmptyInput() async throws {
        let fixture = try #require(wireFamilyFixtures.first { $0.family == .lmStudioNative })
        let request = AIRequest(messages: [AIMessage(role: .system, content: "be brief")], maxTokens: 16)
        try await expectRefusal(fixture, request: request, field: "messages: no user message to send as 'input'")
    }

    @Test("the native family uses max_output_tokens, never max_tokens")
    func nativeUsesTheBudgetFieldItAccepts() async throws {
        let fixture = try #require(wireFamilyFixtures.first { $0.family == .lmStudioNative })
        let body = try await sentBody(fixture, request: makeFixtureRequest(maxTokens: 64))
        #expect(body["max_output_tokens"] as? Int == 64)
        // Measured: `max_tokens` is `Unrecognized key(s) in object` — a 400.
        #expect(body["max_tokens"] == nil)
    }

    @Test("the native family omits a non-positive budget rather than sending -1")
    func nativeOmitsANonPositiveBudget() async throws {
        let fixture = try #require(wireFamilyFixtures.first { $0.family == .lmStudioNative })
        let body = try await sentBody(fixture, request: makeFixtureRequest(maxTokens: -1))
        #expect(body["max_output_tokens"] == nil)
    }

    // MARK: - Where each family puts a system prompt

    @Test("anthropic sends the system prompt top-level, never as a message")
    func anthropicLiftsTheSystemPrompt() async throws {
        let fixture = try #require(wireFamilyFixtures.first { $0.family == .anthropicMessages })
        let request = AIRequest(
            messages: [AIMessage(role: .system, content: "from the array"), AIMessage(role: .user, content: "hi")],
            systemPrompt: "from the field",
            maxTokens: 16
        )
        let body = try await sentBody(fixture, request: request)
        #expect(body["system"] as? String == "from the field\n\nfrom the array")
        let messages = try #require(body["messages"] as? [[String: Any]])
        #expect(messages.allSatisfy { $0["role"] as? String != "system" })
    }

    @Test("responses sends the system prompt as instructions")
    func responsesUsesInstructions() async throws {
        let fixture = try #require(wireFamilyFixtures.first { $0.family == .openAIResponses })
        let body = try await sentBody(fixture, request: makeFixtureRequest(systemPrompt: "be brief"))
        #expect(body["instructions"] as? String == "be brief")
    }

    @Test("chat completions sends the system prompt as the first message")
    func chatUsesAFirstMessage() async throws {
        let fixture = try #require(wireFamilyFixtures.first { $0.family == .openAIChat })
        let body = try await sentBody(fixture, request: makeFixtureRequest(systemPrompt: "be brief"))
        let messages = try #require(body["messages"] as? [[String: Any]])
        #expect(messages.first?["role"] as? String == "system")
        #expect(messages.first?["content"] as? String == "be brief")
    }

    // MARK: - Helpers

    /// Sends a request through the family's real provider on a stub transport,
    /// and returns the JSON body that went out.
    private func sentBody(_ fixture: WireFamilyFixture, request: AIRequest) async throws -> [String: Any] {
        let network = MockHTTPTransport()
        network.answer(json: fixture.reply)
        let provider = HTTPAIProvider(
            entry: try makeFixtureEntry(family: fixture.family),
            codec: fixture.codec,
            network: network
        )
        _ = try await provider.complete(request: request)
        return try fixtureBodyObject(try #require(network.sentRequest))
    }

    /// Asserts the family refuses this request by name, AND that nothing was
    /// sent — the mock network sees zero requests.
    private func expectRefusal(
        _ fixture: WireFamilyFixture,
        request: AIRequest,
        field: String
    ) async throws {
        let network = MockHTTPTransport()
        network.answer(json: fixture.reply)
        let provider = HTTPAIProvider(
            entry: try makeFixtureEntry(family: fixture.family),
            codec: fixture.codec,
            network: network
        )

        await #expect(throws: AIKitError.unsupported(field: field, family: fixture.family)) {
            _ = try await provider.complete(request: request)
        }
        #expect(network.sentRequests.isEmpty, "\(fixture.family.rawValue) sent a request it should have refused")
    }
}

// MARK: - Round trips

@Suite("WireCodecTests — every family round-trips its captured reply")
struct WireCodecRoundTripTests {

    @Test("the fixture table covers every HTTP family, and nothing else")
    func theTableCoversEveryHTTPFamily() {
        #expect(Set(wireFamilyFixtures.map(\.family)) == Set(AIProviderFamily.httpFamilies))
        #expect(wireFamilyFixtures.count == AIProviderFamily.httpFamilies.count)
    }

    @Test("a declared codec speaks the family it was asked for", arguments: AIProviderFamily.httpFamilies)
    func aCodecSpeaksItsFamily(family: AIProviderFamily) throws {
        #expect(try family.codec().family == family)
    }

    @Test("a local family has no codec, and says so", arguments: [
        AIProviderFamily.cliSubprocess, .mlxInProcess, .coreML,
    ])
    func aLocalFamilyHasNoCodec(family: AIProviderFamily) {
        #expect(!family.isHTTP)
        #expect(throws: AIKitError.self) { _ = try family.codec() }
    }

    @Test("a captured reply becomes an AIResponse", arguments: wireFamilyFixtures)
    func aCapturedReplyBecomesAResponse(fixture: WireFamilyFixture) throws {
        let context = try makeFixtureContext(family: fixture.family, route: fixture.route)
        let response = try fixture.codec.response(from: Data(fixture.reply.utf8), context: context)

        #expect(response.content == "pong")
        // The reply names the model that answered; the requested name is only
        // a fallback (BR-AIL-06).
        #expect(response.model == capturedModel)
        #expect(response.providerID == "lmstudio")
        #expect(response.tokensUsed.prompt == fixture.promptTokens)
        #expect(response.tokensUsed.completion == fixture.completionTokens)
        #expect(response.latencyMs == 7)
    }

    @Test("a captured stream becomes the same answer, chunk by chunk", arguments: wireFamilyFixtures)
    func aCapturedStreamBecomesTheSameAnswer(fixture: WireFamilyFixture) throws {
        let chunks = try fixtureChunks(fixture)
        #expect(chunks.map(\.delta).joined() == "pong")
        #expect(chunks.last?.isComplete == true)

        // The terminators are all at the tail, and no chunk both carries text
        // and ends the stream — so a consumer that stops at the first
        // `isComplete` loses nothing.
        //
        // NOT "exactly one terminator": `/v1/chat/completions` is measured to
        // send BOTH `finish_reason: "stop"` and a following `data: [DONE]`,
        // and the codec honours either because a server may send one alone.
        let firstComplete = try #require(chunks.firstIndex { $0.isComplete })
        let tailIsAllTerminators = chunks[firstComplete...].allSatisfy { $0.isComplete }
        #expect(tailIsAllTerminators)
        #expect(chunks.allSatisfy { $0.delta.isEmpty || !$0.isComplete })
    }

    @Test("a lifecycle or label line yields no chunk", arguments: wireFamilyFixtures)
    func lifecycleLinesYieldNothing(fixture: WireFamilyFixture) throws {
        let context = try makeFixtureContext(family: fixture.family, route: fixture.route)
        for line in ["", "event: whatever", ": keep-alive", "id: 7"] {
            #expect(try fixture.codec.chunk(fromLine: line, context: context) == nil, "line: \(line)")
        }
    }

    @Test("a malformed data payload is reported, not skipped", arguments: wireFamilyFixtures)
    func aMalformedChunkIsReported(fixture: WireFamilyFixture) throws {
        let context = try makeFixtureContext(family: fixture.family, route: fixture.route)
        #expect(throws: AIKitError.self) {
            _ = try fixture.codec.chunk(fromLine: "data: {not json", context: context)
        }
    }

    @Test("a 2xx carrying an error body surfaces the server's words", arguments: wireFamilyFixtures)
    func anErrorBodyOnSuccessSurfaces(fixture: WireFamilyFixture) throws {
        let context = try makeFixtureContext(family: fixture.family, route: fixture.route)
        let body = #"{"error": {"message": "Model unloaded, try again"}}"#
        let error = #expect(throws: AIKitError.self) {
            _ = try fixture.codec.response(from: Data(body.utf8), context: context)
        }
        // BR-AIL-10: the text, and where it came from.
        let described = try #require(error?.errorDescription)
        #expect(described.contains("Model unloaded, try again"))
        #expect(described.contains("127.0.0.1:1234"))
        #expect(described.contains(fixture.route))
    }

    @Test("a bare string error body surfaces too", arguments: wireFamilyFixtures)
    func aBareStringErrorBodySurfaces(fixture: WireFamilyFixture) throws {
        let context = try makeFixtureContext(family: fixture.family, route: fixture.route)
        let error = #expect(throws: AIKitError.self) {
            _ = try fixture.codec.response(from: Data(#"{"error": "no model"}"#.utf8), context: context)
        }
        #expect(error?.errorDescription?.contains("no model") == true)
    }

    @Test("a reply with no content is a typed empty reply, naming the target", arguments: wireFamilyFixtures)
    func anEmptyReplyIsTyped(fixture: WireFamilyFixture) throws {
        let context = try makeFixtureContext(family: fixture.family, route: fixture.route)
        let empty: String =
            switch fixture.family {
            case .openAIChat: #"{"id":"x","model":"m","choices":[]}"#
            case .openAIResponses: #"{"id":"x","model":"m","output":[],"error":null}"#
            case .anthropicMessages: #"{"id":"x","role":"assistant","model":"m","content":[]}"#
            default: #"{"response_id":"x","model_instance_id":"m","output":[]}"#
            }

        #expect(throws: AIKitError.emptyReply(target: context.target)) {
            _ = try fixture.codec.response(from: Data(empty.utf8), context: context)
        }
    }

    @Test("a null error on a successful responses reply is not an error")
    func aNullErrorIsNotAnError() throws {
        // The trap `/v1/responses` sets on EVERY successful reply.
        let fixture = try #require(wireFamilyFixtures.first { $0.family == .openAIResponses })
        let context = try makeFixtureContext(family: fixture.family, route: fixture.route)
        #expect(try fixture.codec.response(from: Data(fixture.reply.utf8), context: context).content == "pong")
    }

    @Test("an error value that is present but malformed throws, rather than reading as a success")
    func aMalformedErrorBodyThrows() throws {
        let fixture = try #require(wireFamilyFixtures.first { $0.family == .openAIChat })
        let context = try makeFixtureContext(family: fixture.family, route: fixture.route)
        #expect(throws: AIKitError.self) {
            _ = try fixture.codec.response(from: Data(#"{"error": 42}"#.utf8), context: context)
        }
    }

    @Test("a reply that does not decode at all is reported against its target", arguments: wireFamilyFixtures)
    func anUndecodableReplyIsReported(fixture: WireFamilyFixture) throws {
        let context = try makeFixtureContext(family: fixture.family, route: fixture.route)
        let error = #expect(throws: AIKitError.self) {
            _ = try fixture.codec.response(from: Data(#"{"unexpected":true}"#.utf8), context: context)
        }
        guard case .replyUndecodable(let target, _) = try #require(error) else {
            Issue.record("expected .replyUndecodable, got \(String(describing: error))")
            return
        }
        #expect(target == context.target)
    }
}

// MARK: - Capability declarations

@Suite("WireCodecTests — a codec's declaration matches what it sends")
struct WireCodecCapabilityTests {

    @Test("every family can stream, and every family reads its own stream", arguments: wireFamilyFixtures)
    func everyFamilyStreams(fixture: WireFamilyFixture) throws {
        #expect(fixture.codec.capabilities.streaming)
        #expect(try !fixtureChunks(fixture).isEmpty)
    }

    @Test("every family can carry a system prompt somewhere", arguments: wireFamilyFixtures)
    func everyFamilyCarriesASystemPrompt(fixture: WireFamilyFixture) {
        #expect(fixture.codec.capabilities.systemPrompt)
    }

    @Test("a user role is the one role every family keeps", arguments: wireFamilyFixtures)
    func everyFamilyKeepsAUserRole(fixture: WireFamilyFixture) {
        #expect(fixture.codec.capabilities.roles.contains(.user))
    }

    @Test("only chat completions claims tools, and it is the only one that sends them")
    func onlyChatClaimsTools() {
        let claiming = wireFamilyFixtures.filter { $0.codec.capabilities.tools }.map(\.family)
        #expect(claiming == [.openAIChat])
    }

    @Test("a provider never advertises a capability its codec drops", arguments: wireFamilyFixtures)
    func aProviderAdvertisesOnlyWhatItsCodecCarries(fixture: WireFamilyFixture) throws {
        let provider = HTTPAIProvider(
            entry: try makeFixtureEntry(family: fixture.family),
            codec: fixture.codec,
            network: MockHTTPTransport()
        )
        // This is the inversion the wave removes: capabilities are DERIVED
        // from the codec, so `.toolUse` cannot be claimed by a family that
        // refuses tools.
        #expect(provider.capabilities.contains(.toolUse) == fixture.codec.capabilities.tools)
        #expect(provider.capabilities.contains(.vision) == fixture.codec.capabilities.inputKinds.contains(.image))
    }

    @Test("a credential goes in the header its family reads it from", arguments: wireFamilyFixtures)
    func aCredentialGoesInTheRightHeader(fixture: WireFamilyFixture) async throws {
        let network = MockHTTPTransport()
        network.answer(json: fixture.reply)
        let provider = HTTPAIProvider(
            entry: try makeFixtureEntry(family: fixture.family),
            codec: fixture.codec,
            network: network,
            credential: "sk-secret"
        )
        _ = try await provider.complete(request: makeFixtureRequest())
        let headers = try #require(network.sentRequest?.allHTTPHeaderFields)

        if fixture.family == .anthropicMessages {
            #expect(headers["x-api-key"] == "sk-secret")
            #expect(headers["anthropic-version"] == "2023-06-01")
        } else {
            #expect(headers["Authorization"] == "Bearer sk-secret")
        }
    }

    @Test("no credential means no auth header", arguments: wireFamilyFixtures)
    func noCredentialMeansNoHeader(fixture: WireFamilyFixture) async throws {
        let network = MockHTTPTransport()
        network.answer(json: fixture.reply)
        let provider = HTTPAIProvider(
            entry: try makeFixtureEntry(family: fixture.family),
            codec: fixture.codec,
            network: network
        )
        _ = try await provider.complete(request: makeFixtureRequest())
        let headers = try #require(network.sentRequest?.allHTTPHeaderFields)
        #expect(headers["Authorization"] == nil)
        #expect(headers["x-api-key"] == nil)
    }
}
