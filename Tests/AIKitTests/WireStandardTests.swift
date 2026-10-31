import Foundation
import NetKit
import Testing

@testable import AIKit

// MARK: - WireStandardTests
//
// T-AIL-06, T-AIL-17, T-AIL-18. Every wire standard is a declared EndPoint
// (BR-AIL-09): `kind` names the standard, the standard owns the whole path, and
// nothing parses or strips a configured URL.
//
// PROVENANCE OF THE FIXTURES. Every reply below is a real one, captured from
// the operator's LM Studio at 127.0.0.1:1234 on 2026-10-31 with
// `mistralai_devstral-small-2-24b-instruct-2512-mlx` loaded, each from a
// prompt of "Reply with exactly: pong". They are pinned here verbatim,
// including the fields AIKit does not read (`logprobs`, `stats`,
// `system_fingerprint`, `reasoning_content`), because a fixture trimmed to what
// the decoder wants stops being evidence that the decoder survives the real
// thing.
//
// They are Swift literals rather than files under `Tests/AIKitTests/Fixtures/`
// because this repo has no on-disk fixture convention and loading them from
// disk would need a `resources:` declaration in `Package.swift` — a file this
// wave does not own. The pin is the literal.
//
// No test here reaches a server (BR-AIL-12): the transport is
// `MockNetworkService`, and everything above it — endpoint, DTO, mapping — is
// the real code path a dispatch takes.

// MARK: - Captured replies

/// `/v1/chat/completions` — HTTP 200.
private let chatCompletionsReply = """
    {
      "id": "chatcmpl-khd9f53ms98pkmkg2wo91",
      "object": "chat.completion",
      "created": 1793457204,
      "model": "mistralai_devstral-small-2-24b-instruct-2512-mlx",
      "choices": [
        {
          "index": 0,
          "message": {
            "role": "assistant",
            "content": "pong",
            "reasoning_content": "",
            "tool_calls": []
          },
          "logprobs": null,
          "finish_reason": "stop"
        }
      ],
      "usage": {
        "prompt_tokens": 9,
        "completion_tokens": 3,
        "total_tokens": 12,
        "completion_tokens_details": { "reasoning_tokens": 0 }
      },
      "stats": {},
      "system_fingerprint": "mistralai_devstral-small-2-24b-instruct-2512-mlx"
    }
    """

/// `/v1/responses` — HTTP 200. Note `"error": null` on a SUCCESSFUL reply.
private let responsesReply = """
    {
      "id": "resp_b13bdc5d60556b007624d79ebde5ff0358439b1d87179228",
      "object": "response",
      "created_at": 1793457212,
      "completed_at": 1793457212,
      "status": "completed",
      "incomplete_details": null,
      "model": "mistralai_devstral-small-2-24b-instruct-2512-mlx",
      "previous_response_id": null,
      "instructions": null,
      "output": [
        {
          "id": "msg_afef27hbd5u5kzzic98z5o",
          "type": "message",
          "role": "assistant",
          "status": "completed",
          "content": [
            { "type": "output_text", "text": "pong", "annotations": [], "logprobs": [] }
          ]
        }
      ],
      "error": null,
      "tools": [],
      "tool_choice": "auto",
      "usage": {
        "input_tokens": 9,
        "output_tokens": 2,
        "total_tokens": 11,
        "input_tokens_details": { "cached_tokens": 8 },
        "output_tokens_details": { "reasoning_tokens": 0 }
      },
      "max_output_tokens": 16,
      "store": true,
      "service_tier": "default",
      "metadata": {}
    }
    """

/// `/v1/messages` — HTTP 200.
private let anthropicMessagesReply = """
    {
      "id": "msg_0302tng5wkeadg5qtbnjrzn",
      "type": "message",
      "role": "assistant",
      "content": [
        { "type": "text", "text": "pong" }
      ],
      "model": "mistralai_devstral-small-2-24b-instruct-2512-mlx",
      "stop_reason": "end_turn",
      "stop_sequence": null,
      "usage": {
        "input_tokens": 9,
        "output_tokens": 3,
        "cache_read_input_tokens": 8
      }
    }
    """

/// `/api/v1/chat` — HTTP 200.
private let lmStudioNativeReply = """
    {
      "model_instance_id": "mistralai_devstral-small-2-24b-instruct-2512-mlx",
      "output": [
        { "type": "message", "content": "pong" }
      ],
      "stats": {
        "input_tokens": 9,
        "total_output_tokens": 3,
        "reasoning_output_tokens": 0,
        "tokens_per_second": 35.10915764798229,
        "time_to_first_token_seconds": 0.296
      },
      "response_id": "resp_ce67a81f5f12825ad85e85d8fbad7793f8ecf864afa4a16d"
    }
    """

/// The model every capture above was answered by.
private let capturedModel = "mistralai_devstral-small-2-24b-instruct-2512-mlx"

// MARK: - The fixture table

/// One captured reply per standard, with what it must become.
///
/// Not `private`: a `@Test(arguments:)` case takes it as a parameter, and a
/// parameter type has to be at least as visible as the method.
struct WireFixture: Sendable, CustomTestStringConvertible {
    let standard: AIWireStandard
    let reply: String
    let path: String
    let content: String
    let promptTokens: Int
    let completionTokens: Int

    /// Names the case in test output by its standard, so a failure reads
    /// `anthropic-messages` rather than a struct dump.
    var testDescription: String { standard.rawValue }
}

private let wireFixtures: [WireFixture] = [
    WireFixture(
        standard: .openAIChat,
        reply: chatCompletionsReply,
        path: "/v1/chat/completions",
        content: "pong",
        promptTokens: 9,
        completionTokens: 3
    ),
    WireFixture(
        standard: .openAIResponses,
        reply: responsesReply,
        path: "/v1/responses",
        content: "pong",
        promptTokens: 9,
        completionTokens: 2
    ),
    WireFixture(
        standard: .anthropicMessages,
        reply: anthropicMessagesReply,
        path: "/v1/messages",
        content: "pong",
        promptTokens: 9,
        completionTokens: 3
    ),
    WireFixture(
        standard: .lmStudioNative,
        reply: lmStudioNativeReply,
        path: "/api/v1/chat",
        content: "pong",
        promptTokens: 9,
        completionTokens: 3
    ),
]

// MARK: - Helpers

private let serverRoot = "http://127.0.0.1:1234"

private func makeServer(_ root: String = serverRoot) throws -> AIWireServer {
    try AIWireServer(serverRoot: root)
}

private func makeRequest(
    _ prompt: String = "Reply with exactly: pong",
    systemPrompt: String? = nil,
    maxTokens: Int = 16,
    temperature: Double = 0
) -> AIRequest {
    AIRequest(
        messages: [AIMessage(role: .user, content: prompt)],
        systemPrompt: systemPrompt,
        temperature: temperature,
        maxTokens: maxTokens
    )
}

/// Sends one request on `standard` against a mocked transport answering
/// `reply`, and hands back both the response and what went on the wire.
private func send(
    _ standard: AIWireStandard,
    answering reply: String,
    request: AIRequest = makeRequest(),
    root: String = serverRoot,
    apiKey: String? = nil,
    model: String = "requested-model",
    providerID: String = "lmstudio"
) async throws -> (response: AIResponse, sent: URLRequest) {
    let network = MockNetworkService()
    network.resultData = Data(reply.utf8)
    let response = try await standard.complete(
        request: request,
        target: try AIWireTarget(serverRoot: root, apiKey: apiKey, model: model, providerID: providerID),
        networkService: network
    )
    let sent = try #require(network.capturedRequests.first)
    return (response, sent)
}

/// The JSON body of a request, as a dictionary.
private func bodyObject(_ request: URLRequest) throws -> [String: Any] {
    let data = try #require(request.httpBody)
    return try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
}

// MARK: - T-AIL-17 — every standard round-trips

@Suite("AIWireStandard — every standard round-trips a captured reply")
struct WireStandardRoundTripTests {

    /// The guard that makes "one case per standard" true rather than hopeful:
    /// adding a fifth case to `AIWireStandard` without capturing a reply for it
    /// fails here instead of silently shipping an untested standard.
    @Test("the fixture table covers every declared standard, and nothing else")
    func fixtureTableCoversEveryStandard() {
        let covered = Set(wireFixtures.map(\.standard))
        #expect(covered == Set(AIWireStandard.allCases))
        #expect(wireFixtures.count == AIWireStandard.allCases.count, "one fixture per standard, no duplicates")
    }

    /// T-AIL-17. Fixture to typed: each standard's own reply shape becomes the
    /// same `AIResponse`, through the real endpoint and the real mapping.
    @Test("a captured reply becomes an AIResponse", arguments: wireFixtures)
    func capturedReplyBecomesAResponse(fixture: WireFixture) async throws {
        let (response, _) = try await send(fixture.standard, answering: fixture.reply)

        #expect(response.content == fixture.content, "\(fixture.standard.rawValue) lost the answer")
        #expect(response.tokensUsed.prompt == fixture.promptTokens)
        #expect(response.tokensUsed.completion == fixture.completionTokens)
        #expect(response.providerID == "lmstudio", "BR-AIL-06: the answer names who served it")
        // BR-AIL-06: the model is the one the REPLY names, not the one asked
        // for. Measured 2026-10-31: three of these four surfaces ignore an
        // unknown model and answer with whatever is loaded, so echoing the
        // request back here would be a response that lies about its author.
        #expect(response.model == capturedModel, "\(fixture.standard.rawValue) must name the model that answered")
        #expect(response.model != "requested-model")
    }

    /// T-AIL-17, the other direction: typed to fixture. Each standard puts the
    /// prompt where its own server demands it, under the key names measured
    /// against the live surface.
    @Test("a request is rendered into each standard's own shape", arguments: wireFixtures)
    func requestIsRenderedIntoEachShape(fixture: WireFixture) async throws {
        let request = makeRequest("Reply with exactly: pong", maxTokens: 16, temperature: 0)
        let (_, sent) = try await send(fixture.standard, answering: fixture.reply, request: request)
        let body = try bodyObject(sent)

        #expect(sent.httpMethod == "POST")
        #expect(body["model"] as? String == "requested-model")

        switch fixture.standard {
        case .openAIChat:
            let messages = try #require(body["messages"] as? [[String: Any]])
            #expect(messages.count == 1)
            #expect(messages[0]["role"] as? String == "user")
            #expect(messages[0]["content"] as? String == "Reply with exactly: pong")
            #expect(body["max_tokens"] as? Int == 16)

        case .openAIResponses:
            // `input` is the role array, `max_output_tokens` is the budget.
            let input = try #require(body["input"] as? [[String: Any]])
            #expect(input.count == 1)
            #expect(input[0]["role"] as? String == "user")
            #expect(input[0]["content"] as? String == "Reply with exactly: pong")
            #expect(body["max_output_tokens"] as? Int == 16)
            #expect(body["max_tokens"] == nil, "this standard's budget field is max_output_tokens")

        case .anthropicMessages:
            let messages = try #require(body["messages"] as? [[String: Any]])
            #expect(messages.count == 1)
            #expect(messages[0]["content"] as? String == "Reply with exactly: pong")
            #expect(body["max_tokens"] as? Int == 16)
            #expect(body["system"] == nil, "no system prompt was given, so none is sent")

        case .lmStudioNative:
            // Measured: a role array is refused here, and `max_tokens` is a
            // 400. A lone user message travels as exactly what the caller
            // wrote, with no invented labels around it.
            #expect(body["input"] as? String == "Reply with exactly: pong")
            #expect(body["max_output_tokens"] as? Int == 16)
            #expect(body["max_tokens"] == nil, "/api/v1/chat answers 400 to max_tokens")
            #expect(body["messages"] == nil, "/api/v1/chat refuses a messages array")
        }
    }

    /// The trap this envelope sets: `/v1/responses` sends `"error": null` on a
    /// successful reply, so an envelope that tested for the KEY would turn
    /// every good answer into a failure. The captured reply above carries that
    /// null, which is why this asserts on the real one.
    @Test("a null error on a successful responses reply is not an error")
    func nullErrorIsNotAnError() async throws {
        #expect(responsesReply.contains("\"error\": null"), "the fixture must still carry the trap it guards")
        let (response, _) = try await send(.openAIResponses, answering: responsesReply)
        #expect(response.content == "pong")
    }

    /// BR-AIL-10, on every standard: a 200 that carries an error body surfaces
    /// the server's own words, and the URL they came from.
    @Test("a 200 with an error body surfaces the server's text", arguments: AIWireStandard.allCases)
    func errorBodyOn200(standard: AIWireStandard) async throws {
        let errorBody = """
            {"error": {"message": "Invalid model identifier \\"no-such-model\\".", "type": "invalid_request"}}
            """

        await #expect(throws: AIKitError.self) {
            _ = try await send(standard, answering: errorBody)
        }

        do {
            _ = try await send(standard, answering: errorBody)
        } catch let error as AIKitError {
            guard case .requestFailed(let message) = error else {
                Issue.record("Expected requestFailed, got \(error)")
                return
            }
            #expect(message.contains("Invalid model identifier"), "the server's words travel")
            #expect(
                message.contains("\(serverRoot)\(standard.path)"),
                "the error names the URL it came from — got: \(message)"
            )
        }
    }

    /// A reply with no text at all is a typed refusal naming the standard, not
    /// an empty success a caller would store as an answer.
    @Test("a reply with no content is a typed refusal", arguments: AIWireStandard.allCases)
    func emptyReplyIsARefusal(standard: AIWireStandard) async throws {
        let empty: String
        switch standard {
        case .openAIChat: empty = #"{"id":"c","choices":[]}"#
        case .openAIResponses: empty = #"{"id":"r","output":[]}"#
        case .anthropicMessages: empty = #"{"id":"m","content":[]}"#
        case .lmStudioNative: empty = #"{"response_id":"r","output":[]}"#
        }

        do {
            _ = try await send(standard, answering: empty)
            Issue.record("\(standard.rawValue): expected a refusal")
        } catch let error as AIKitError {
            guard case .requestFailed(let message) = error else {
                Issue.record("Expected requestFailed, got \(error)")
                return
            }
            #expect(message.contains(standard.displayName), "the refusal names which standard came back empty")
        }
    }
}

// MARK: - T-AIL-06 / T-AIL-18 — kind selects the endpoint

@Suite("AIWireStandard — kind picks the path, nothing edits a URL")
struct WireStandardSelectionTests {

    /// T-AIL-06. A server ROOT plus a standard yields the full URL, version
    /// prefix included. No caller appended anything and no string was edited.
    @Test("a server root and a standard yield the full URL")
    func endpointOwnsThePath() async throws {
        let (_, sent) = try await send(.openAIChat, answering: chatCompletionsReply, root: "http://h:1")
        #expect(sent.url?.absoluteString == "http://h:1/v1/chat/completions")
    }

    /// T-AIL-06, per standard: each one owns its whole path.
    @Test("each standard owns its whole path", arguments: wireFixtures)
    func eachStandardOwnsItsPath(fixture: WireFixture) async throws {
        #expect(fixture.standard.path == fixture.path)
        let (_, sent) = try await send(fixture.standard, answering: fixture.reply, root: "http://h:1")
        #expect(sent.url?.absoluteString == "http://h:1\(fixture.path)")
    }

    /// T-AIL-18. One server, two providers, two standards: the `kind` each
    /// provider declares is what decides where its request goes. This is the
    /// measured shape of the operator's machine — a single LM Studio answering
    /// four standards on one port.
    @Test("one server backs two providers, and kind decides which path each takes")
    func kindSelectsTheEndPoint() async throws {
        let chat = try #require(AIWireStandard(kind: "openai-chat"))
        let anthropic = try #require(AIWireStandard(kind: "anthropic-messages"))

        let chatCall = try await send(chat, answering: chatCompletionsReply, providerID: "lmstudio-chat")
        let anthropicCall = try await send(anthropic, answering: anthropicMessagesReply, providerID: "lmstudio-claude")

        // Same host and port...
        #expect(chatCall.sent.url?.host == "127.0.0.1")
        #expect(anthropicCall.sent.url?.host == "127.0.0.1")
        #expect(chatCall.sent.url?.port == 1234)
        #expect(anthropicCall.sent.url?.port == 1234)
        // ...different paths, each chosen by the declared kind.
        #expect(chatCall.sent.url?.path == "/v1/chat/completions")
        #expect(anthropicCall.sent.url?.path == "/v1/messages")
        // ...and each answer names the provider that served it.
        #expect(chatCall.response.providerID == "lmstudio-chat")
        #expect(anthropicCall.response.providerID == "lmstudio-claude")
        // ...from one and the same loaded model.
        #expect(chatCall.response.model == capturedModel)
        #expect(anthropicCall.response.model == capturedModel)
    }

    /// The 7 provider profiles that exist today all carry
    /// `kind = "openai-compatible"`, written when nothing read `kind` at all.
    /// That value has to keep meaning what those profiles have always talked
    /// to, or W8's migration breaks a configuration that was valid yesterday.
    @Test("a kind string selects a standard, legacy spellings included")
    func kindStringsMapToStandards() {
        #expect(AIWireStandard(kind: "openai-compatible") == .openAIChat)
        #expect(AIWireStandard(kind: "OpenAI-Compatible") == .openAIChat, "case and spacing are not a config error")
        #expect(AIWireStandard(kind: "  openai-compatible  ") == .openAIChat)
        #expect(AIWireStandard(kind: "lmstudio") == .openAIChat)
        #expect(AIWireStandard(kind: "ollama") == .openAIChat)
        #expect(AIWireStandard(kind: "anthropic") == .anthropicMessages)

        // Every declared standard is reachable by its own raw value.
        for standard in AIWireStandard.allCases {
            #expect(AIWireStandard(kind: standard.rawValue) == standard)
        }

        #expect(AIWireStandard(kind: "not-a-standard") == nil, "an unknown kind is a config error, not a default")
        #expect(AIWireStandard(kind: "") == nil)
    }

    /// BR-EC-02: the generator and the enforcer share one code path, and the
    /// agreement is asserted rather than assumed. `OpenAIChatCompletionEndPoint`
    /// is what `OpenAIProvider` still sends through, and the chat family
    /// declares the same standard — if the path were written twice, this is
    /// where the drift would show.
    @Test("the chat path is written once, and both readers read it")
    func chatPathHasOneSourceOfTruth() throws {
        let legacy = OpenAIChatCompletionEndPoint(
            host: "h",
            port: 1,
            scheme: "http",
            apiKey: nil,
            requestBody: ChatCompletionRequest(
                model: "m",
                messages: [],
                temperature: 0,
                maxTokens: 1,
                stream: false,
                tools: nil,
                responseFormat: nil
            )
        )
        #expect(legacy.path == AIWireStandard.openAIChat.path)
        #expect(OpenAIChatWireFamily.standard == .openAIChat)
    }

    /// Each family declares the standard it is registered under — the switch in
    /// `AIWireStandard.endPoint` cannot be wired to the wrong family without
    /// this failing.
    @Test("the endpoint a standard builds declares that same standard", arguments: AIWireStandard.allCases)
    func familyMatchesItsStandard(standard: AIWireStandard) async throws {
        let request = makeRequest()
        let endpoint = standard.endPoint(
            target: try AIWireTarget(server: try makeServer(), model: "m", providerID: "p"),
            request: request,
            stream: false
        )
        #expect(endpoint.path == standard.path)
        #expect(endpoint.method == .POST)
        #expect(endpoint.header?["Content-Type"] == "application/json")
    }
}

// MARK: - BR-AIL-09 — base_url is a server root

@Suite("AIWireServer — base_url is a root, and a leftover path is named")
struct WireServerRootTests {

    @Test("a server root parses into scheme, host and port")
    func rootParses() throws {
        let server = try AIWireServer(serverRoot: "http://127.0.0.1:1234")
        #expect(server.scheme == "http")
        #expect(server.host == "127.0.0.1")
        #expect(server.port == 1234)
        #expect(server.description == "http://127.0.0.1:1234")
    }

    @Test("a trailing slash is a root written with a slash, not a path")
    func trailingSlashIsStillARoot() throws {
        let server = try AIWireServer(serverRoot: "http://127.0.0.1:1234/")
        #expect(server.host == "127.0.0.1")
        #expect(server.port == 1234)
    }

    @Test("a root with no port keeps none, so the scheme's default applies")
    func rootWithoutPort() throws {
        let server = try AIWireServer(serverRoot: "https://api.openai.com")
        #expect(server.scheme == "https")
        #expect(server.host == "api.openai.com")
        #expect(server.port == nil)
        #expect(server.description == "https://api.openai.com")
    }

    /// The silent failure this replaces: a `/v1` left in `base_url` after W8's
    /// migration produced `/v1/v1/chat/completions`, which LM Studio answers
    /// with a 200 and an error body — "looks like a bad reply". BR-AIL-09 says
    /// the standard owns the path, so a root carrying one is refused at parse
    /// time, by name, and NOT silently stripped.
    @Test(
        "a root carrying a path is refused, naming the leftover",
        arguments: [
            "http://127.0.0.1:1234/v1",
            "http://127.0.0.1:1234/v1/",
            "http://127.0.0.1:1234/api/v0",
        ])
    func rootWithAPathIsRefused(root: String) throws {
        do {
            _ = try AIWireServer(serverRoot: root)
            Issue.record("\(root): expected a refusal")
        } catch let error as AIKitError {
            guard case .requestFailed(let message) = error else {
                Issue.record("Expected requestFailed, got \(error)")
                return
            }
            #expect(message.contains(root.trimmingCharacters(in: CharacterSet(charactersIn: "/"))))
            #expect(message.contains("BR-AIL-09"), "the refusal says which rule it enforces")
            #expect(message.contains("server root"))
        }
    }

    /// A target is one provider profile row: the same root can be given to two
    /// standards, and the leftover-path refusal travels through its convenience
    /// initialiser rather than being re-implemented there.
    @Test("a target carries one provider's configuration, and refuses a root with a path")
    func targetCarriesOneProfile() throws {
        let target = try AIWireTarget(
            serverRoot: "http://127.0.0.1:1234",
            apiKey: "sk-test",
            model: "devstral",
            providerID: "lmstudio"
        )
        #expect(target.server.host == "127.0.0.1")
        #expect(target.server.port == 1234)
        #expect(target.apiKey == "sk-test")
        #expect(target.model == "devstral")
        #expect(target.providerID == "lmstudio")

        #expect(throws: AIKitError.self) {
            _ = try AIWireTarget(serverRoot: "http://127.0.0.1:1234/v1", model: "m", providerID: "p")
        }
    }

    @Test("something that is not a URL at all is refused by name", arguments: ["", "   ", "not a url"])
    func nonURLIsRefused(root: String) throws {
        #expect(throws: AIKitError.self) {
            _ = try AIWireServer(serverRoot: root)
        }
    }
}

// MARK: - Per-standard shape details

@Suite("Wire standards — the details each server actually demands")
struct WireStandardShapeTests {

    /// Anthropic's system prompt is a top-level field, not a message. Measured:
    /// sending it there changed the answer and raised `input_tokens` from 4 to
    /// 18, so the field is read.
    @Test("anthropic sends the system prompt top-level, never as a message")
    func anthropicLiftsTheSystemPrompt() async throws {
        let request = AIRequest(
            messages: [
                AIMessage(role: .system, content: "You are terse."),
                AIMessage(role: .user, content: "Name the capital of France."),
            ],
            systemPrompt: "Always answer in exactly one word."
        )
        let (_, sent) = try await send(.anthropicMessages, answering: anthropicMessagesReply, request: request)
        let body = try bodyObject(sent)

        let system = try #require(body["system"] as? String)
        #expect(system.contains("Always answer in exactly one word."))
        #expect(system.contains("You are terse."), "a system message in the array is lifted, not dropped")

        let messages = try #require(body["messages"] as? [[String: Any]])
        #expect(messages.count == 1, "the system message left the array")
        #expect(messages.allSatisfy { $0["role"] as? String != "system" })
    }

    /// The responses standard has its own name for a system prompt.
    @Test("responses sends the system prompt as instructions")
    func responsesSendsInstructions() async throws {
        let request = makeRequest(systemPrompt: "Always answer in exactly one word.")
        let (_, sent) = try await send(.openAIResponses, answering: responsesReply, request: request)
        let body = try bodyObject(sent)

        #expect(body["instructions"] as? String == "Always answer in exactly one word.")
        #expect(body["system"] == nil, "that is the Anthropic field, not this one")
    }

    /// The chat standard keeps sending a system message, which is where that
    /// standard puts it.
    @Test("chat completions sends the system prompt as the first message")
    func chatSendsASystemMessage() async throws {
        let request = makeRequest(systemPrompt: "You are terse.")
        let (_, sent) = try await send(.openAIChat, answering: chatCompletionsReply, request: request)
        let body = try bodyObject(sent)

        let messages = try #require(body["messages"] as? [[String: Any]])
        #expect(messages.count == 2)
        #expect(messages[0]["role"] as? String == "system")
        #expect(messages[0]["content"] as? String == "You are terse.")
        #expect(messages[1]["role"] as? String == "user")
    }

    /// `/api/v1/chat` has no conversation shape at this version, so a
    /// conversation is flattened — and a lone user message is NOT, because
    /// inventing labels around the one thing the caller wrote would change the
    /// prompt the model sees.
    @Test("the native standard flattens a conversation but not a lone message")
    func nativeFlattensOnlyWhenItMust() {
        let lone = AIRequest(messages: [AIMessage(role: .user, content: "Reply with exactly: pong")])
        #expect(AIWirePromptFlattening.flatten(lone) == "Reply with exactly: pong")

        let withSystem = AIRequest(
            messages: [AIMessage(role: .user, content: "hi")],
            systemPrompt: "Be terse."
        )
        #expect(AIWirePromptFlattening.flatten(withSystem) == "system: Be terse.\n\nuser: hi")

        let conversation = AIRequest(messages: [
            AIMessage(role: .user, content: "hi"),
            AIMessage(role: .assistant, content: "hello"),
            AIMessage(role: .user, content: "bye"),
        ])
        #expect(AIWirePromptFlattening.flatten(conversation) == "user: hi\n\nassistant: hello\n\nuser: bye")

        // An empty system prompt is no system prompt.
        let emptySystem = AIRequest(messages: [AIMessage(role: .user, content: "hi")], systemPrompt: "")
        #expect(AIWirePromptFlattening.flatten(emptySystem) == "hi")
    }

    /// Anthropic reads its key from `x-api-key`; the OpenAI surfaces read a
    /// bearer token. The standard says which, so a key configured once reaches
    /// the right header without the caller knowing the difference.
    @Test("the key goes in the header the standard reads it from")
    func keyGoesInTheRightHeader() async throws {
        let anthropic = try await send(.anthropicMessages, answering: anthropicMessagesReply, apiKey: "sk-test")
        #expect(anthropic.sent.value(forHTTPHeaderField: "x-api-key") == "sk-test")
        #expect(anthropic.sent.value(forHTTPHeaderField: "Authorization") == nil)
        #expect(anthropic.sent.value(forHTTPHeaderField: "anthropic-version") == "2023-06-01")

        for standard in [AIWireStandard.openAIChat, .openAIResponses, .lmStudioNative] {
            let fixture = try #require(wireFixtures.first { $0.standard == standard })
            let call = try await send(standard, answering: fixture.reply, apiKey: "sk-test")
            #expect(call.sent.value(forHTTPHeaderField: "Authorization") == "Bearer sk-test")
            #expect(call.sent.value(forHTTPHeaderField: "x-api-key") == nil)
        }
    }

    /// A local server takes no token, and an empty key must not become
    /// `Authorization: Bearer `.
    @Test("no key means no authorization header", arguments: AIWireStandard.allCases)
    func noKeyMeansNoHeader(standard: AIWireStandard) async throws {
        let fixture = try #require(wireFixtures.first { $0.standard == standard })
        for key in [nil, ""] as [String?] {
            let call = try await send(standard, answering: fixture.reply, apiKey: key)
            #expect(call.sent.value(forHTTPHeaderField: "Authorization") == nil)
            #expect(call.sent.value(forHTTPHeaderField: "x-api-key") == nil)
        }
    }

    /// A transport failure must travel as the type it was thrown as, because
    /// W5's fallback rule needs to tell a refused connection from a server that
    /// answered (BR-AIL-07). Flattening it into a string here would erase that.
    @Test("a transport failure is not flattened into a string", arguments: AIWireStandard.allCases)
    func transportFailureKeepsItsType(standard: AIWireStandard) async throws {
        let network = MockNetworkService()
        network.resultError = .unknown(URLError(.cannotConnectToHost))

        do {
            _ = try await standard.complete(
                request: makeRequest(),
                target: try AIWireTarget(server: try makeServer(), model: "m", providerID: "p"),
                networkService: network
            )
            Issue.record("\(standard.rawValue): expected the transport error")
        } catch let error as NetworkError {
            guard case .unknown(let underlying) = error else {
                Issue.record("Expected .unknown, got \(error)")
                return
            }
            #expect((underlying as? URLError)?.code == .cannotConnectToHost)
        }
    }
}
