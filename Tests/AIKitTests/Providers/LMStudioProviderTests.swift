import Foundation
import NetKit
import Testing

@testable import AIKit

// MARK: - LMStudioProviderTests
//
// LM Studio owns no wire format: it renders a prompt into an `AIRequest`,
// `HTTPAIProvider` with the `openai-chat` codec sends it, and what comes back
// is translated into `LMStudioError` so a chain's fallback rule keeps reading
// the cases it always read. What is LM Studio-specific and stays here is the
// `AgentProviding` face and the model-MANAGEMENT calls, which live on LM
// Studio's native API and have no OpenAI equivalent.
//
// These tests exercise the real path — stub transport, real endpoint, real
// codec, real provider (BR-AIL-12: no test reaches a server).
//
// The error mapping is now read from CASES, not from message prefixes. That
// was a real defect: `mapped(aiKitError:)` decided `.emptyContent` with
// `message.hasPrefix(AIWireStandard.emptyReplyPrefix(…))`, so one wire event
// got two verdicts depending on how its sentence had been composed
// (AIKit#18, panel finding on `LMStudioProvider.swift:150`).

// MARK: - Fixtures

/// A 200 chat completion, the shape LM Studio answers with.
private func makeChatCompletionJSON(content: String) -> Data {
    let json: [String: Any] = [
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": "test-model",
        "choices": [
            [
                "index": 0,
                "message": [
                    "role": "assistant",
                    "content": content,
                ],
                "finish_reason": "stop",
            ],
        ],
        "usage": [
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        ],
    ]
    return (try? JSONSerialization.data(withJSONObject: json)) ?? Data()
}

/// An error body, the nested form.
private func makeErrorJSON(message: String) -> Data {
    let json: [String: Any] = [
        "error": [
            "message": message,
            "type": "server_error",
        ],
    ]
    return (try? JSONSerialization.data(withJSONObject: json)) ?? Data()
}

private func makeSUT(
    answering data: Data? = nil,
    statusCode: Int = 200,
    failingWith error: (any Error)? = nil,
    baseURL: String = "http://127.0.0.1:1234",
    model: String = "qwen-2.5-coder",
    requestTimeout: TimeInterval = 300
) throws -> (provider: LMStudioProvider, network: MockHTTPTransport) {
    let network = MockHTTPTransport()
    network.body = data ?? Data()
    network.status = statusCode
    network.error = error
    let provider = try LMStudioProvider(
        baseURL: baseURL,
        model: model,
        requestTimeout: requestTimeout,
        transport: network
    )
    return (provider, network)
}

// MARK: - LMStudioProvider

@Suite("LMStudioProvider — one wire path, its own error vocabulary")
struct LMStudioProviderTests {

    // MARK: - The request on the wire

    @Test("sends an OpenAI-compatible chat completion to /v1/chat/completions")
    func sendsAnOpenAICompatibleRequest() async throws {
        let (provider, network) = try makeSUT(answering: makeChatCompletionJSON(content: "ok"))

        _ = try await provider.run(prompt: "Hello, world!", timeout: 30)

        #expect(network.sentRequests.count == 1)
        let request = try #require(network.sentRequest)
        #expect(request.url?.absoluteString == "http://127.0.0.1:1234/v1/chat/completions")
        #expect(request.httpMethod == "POST")
        #expect(request.allHTTPHeaderFields?["Content-Type"] == "application/json")
        #expect(request.allHTTPHeaderFields?["Authorization"] == nil, "a local server takes no bearer token")

        let body = try #require(try network.sentBodyObject())
        #expect(body["model"] as? String == "qwen-2.5-coder")
        #expect(body["stream"] as? Bool == false)
        #expect(body["temperature"] as? Double == 0.7)
        // -1 is LM Studio's "answer until the context is full", which is what
        // this provider has always sent.
        #expect(body["max_tokens"] as? Int == -1)

        let messages = body["messages"] as? [[String: String]]
        #expect(messages?.count == 1)
        #expect(messages?.first?["role"] == "user")
        #expect(messages?.first?["content"] == "Hello, world!")
    }

    @Test("the caller's timeout rides on the request, capped by the provider's own")
    func theRequestCarriesTheCallersTimeout() throws {
        // NetKit's request builder leaves `timeoutInterval` at URLRequest's
        // 60-second default, which would make a cold local model read as a dead
        // one. The transport the provider builds for a call carries the timeout
        // instead — this asserts the override is the one that runs, not that it
        // merely exists.
        let provider = try LMStudioProvider(baseURL: "http://127.0.0.1:1234", model: "m", requestTimeout: 300)

        let within = try #require(provider.provider(timeout: 120).network as? URLSessionHTTPTransport)
        #expect(within.timeout == 120)

        let capped = try #require(provider.provider(timeout: 900).network as? URLSessionHTTPTransport)
        #expect(capped.timeout == 300, "requestTimeout is the ceiling")

        let endpoint = OpenAIEndPoint.chatCompletions(
            address: try AIServerAddress(validating: "http://127.0.0.1:1234"),
            credential: nil,
            body: ChatCompletionRequest(
                model: "m",
                messages: [],
                temperature: 0.7,
                maxTokens: -1,
                stream: false,
                tools: nil,
                responseFormat: nil
            )
        )
        #expect(try within.createRequest(endPoint: endpoint).timeoutInterval == 120)
        #expect(try NetworkService().createRequest(endPoint: endpoint).timeoutInterval == 60)
    }

    @Test("an injected transport is used as it was given")
    func injectedTransportIsNotWrapped() throws {
        let (provider, network) = try makeSUT()
        #expect(provider.provider(timeout: 30).network as? MockHTTPTransport === network)
    }

    // MARK: - The answer

    @Test("parses an OpenAI-format chat completion")
    func parsesTheAnswer() async throws {
        let (provider, _) = try makeSUT(
            answering: makeChatCompletionJSON(content: "  This is the assistant response.\n")
        )
        let result = try await provider.run(prompt: "p", timeout: 30)
        #expect(result == "This is the assistant response.")
    }

    @Test("a reply with no choices is emptyContent")
    func noChoicesIsEmptyContent() async throws {
        let data = try JSONSerialization.data(withJSONObject: [
            "id": "chatcmpl-empty",
            "choices": [] as [[String: Any]],
        ])
        let (provider, _) = try makeSUT(answering: data)

        await #expect(throws: LMStudioProvider.LMStudioError.emptyContent) {
            _ = try await provider.run(prompt: "p", timeout: 30)
        }
    }

    @Test("an empty reply is recognised by its CASE, not by a message prefix")
    func emptyReplyIsRecognisedByCase() throws {
        let target = AIRequestTarget(
            entryID: "lmstudio",
            address: try AIServerAddress(validating: "http://127.0.0.1:1234"),
            route: "/v1/chat/completions"
        )
        let provider = try LMStudioProvider(baseURL: "http://127.0.0.1:1234", model: "qwen")
        #expect(provider.mapped(AIKitError.emptyReply(target: target)) == .emptyContent)
        // And the verdict does not depend on what the sentence happens to say.
        #expect(!FallbackProviderChain.isFallbackEligible(LMStudioProvider.LMStudioError.emptyContent))
    }

    @Test("a whitespace-only answer is emptyContent")
    func whitespaceAnswerIsEmptyContent() async throws {
        let (provider, _) = try makeSUT(answering: makeChatCompletionJSON(content: "   \n  "))

        await #expect(throws: LMStudioProvider.LMStudioError.emptyContent) {
            _ = try await provider.run(prompt: "p", timeout: 30)
        }
    }

    @Test("malformed JSON is invalidResponse, not a crash")
    func malformedJSONIsInvalidResponse() async throws {
        let (provider, _) = try makeSUT(answering: Data("not json at all".utf8))

        do {
            _ = try await provider.run(prompt: "p", timeout: 30)
            Issue.record("Expected invalidResponse")
        } catch let error as LMStudioProvider.LMStudioError {
            guard case .invalidResponse(let message) = error else {
                Issue.record("Expected invalidResponse, got \(error)")
                return
            }
            #expect(message.contains("Failed to decode response"))
            #expect(!FallbackProviderChain.isFallbackEligible(error))
        }
    }

    // MARK: - BR-AIL-10 — a 200 can carry an error

    @Test("a 200 with an error body surfaces the server's text")
    func errorBodyOn200SurfacesServerText() async throws {
        let (provider, _) = try makeSUT(
            answering: makeErrorJSON(message: "Model 'qwen-2.5-coder' is not loaded")
        )

        do {
            _ = try await provider.run(prompt: "p", timeout: 30)
            Issue.record("Expected the server's error to be thrown")
        } catch let error as LMStudioProvider.LMStudioError {
            guard case .invalidResponse(let message) = error else {
                Issue.record("Expected invalidResponse, got \(error)")
                return
            }
            #expect(message.contains("Model 'qwen-2.5-coder' is not loaded"))
            #expect(
                message.contains("http://127.0.0.1:1234/v1/chat/completions"),
                "the error names the target it came from"
            )
            #expect(
                !FallbackProviderChain.isFallbackEligible(error),
                "the server answered — this is not a reason to try the next provider"
            )
        }
    }

    @Test("a bare string error body surfaces too")
    func bareStringErrorBodyOn200() async throws {
        let data = try JSONSerialization.data(withJSONObject: ["error": "unexpected endpoint or method"])
        let (provider, _) = try makeSUT(answering: data)

        do {
            _ = try await provider.run(prompt: "p", timeout: 30)
            Issue.record("Expected the server's error to be thrown")
        } catch let error as LMStudioProvider.LMStudioError {
            guard case .invalidResponse(let message) = error else {
                Issue.record("Expected invalidResponse, got \(error)")
                return
            }
            #expect(message.contains("unexpected endpoint or method"))
        }
    }

    // MARK: - Status codes and transport

    @Test("429 is rateLimited, and rate limiting falls back")
    func rateLimitedOn429() async throws {
        let (provider, _) = try makeSUT(answering: Data("{}".utf8), statusCode: 429)

        await #expect(throws: LMStudioProvider.LMStudioError.rateLimited) {
            _ = try await provider.run(prompt: "p", timeout: 30)
        }
        #expect(FallbackProviderChain.isFallbackEligible(LMStudioProvider.LMStudioError.rateLimited))
    }

    @Test("500 keeps the server's text, and the chain falls back on it")
    func httpErrorOn500() async throws {
        let (provider, _) = try makeSUT(
            answering: makeErrorJSON(message: "model crashed"),
            statusCode: 500
        )

        do {
            _ = try await provider.run(prompt: "p", timeout: 30)
            Issue.record("Expected httpError")
        } catch let error as LMStudioProvider.LMStudioError {
            guard case .httpError(let code, let message) = error else {
                Issue.record("Expected httpError, got \(error)")
                return
            }
            #expect(code == 500)
            // BR-AIL-19: a non-2xx keeps the server's own text now. NetKit's
            // `sendRequest` discarded it, so this used to be the status alone.
            #expect(message.contains("model crashed"))
            // A 5xx IS fallback-eligible at the AIKit level: the server broke,
            // which is a fact about that server. `LMStudioError.httpError` is
            // the chain's own older vocabulary and keeps its own verdict.
            #expect(!FallbackProviderChain.isFallbackEligible(error))
        }
    }

    @Test("a 401 is not eligible and a 5xx is, read from the status")
    func statusDecidesEligibility() throws {
        let target = AIRequestTarget(
            entryID: "lmstudio",
            address: try AIServerAddress(validating: "http://127.0.0.1:1234"),
            route: "/v1/chat/completions"
        )
        #expect(!FallbackEligibility.allows(AIKitError.http(status: 401, body: "no key", target: target)))
        #expect(FallbackEligibility.allows(AIKitError.http(status: 429, body: "slow down", target: target)))
        #expect(FallbackEligibility.allows(AIKitError.http(status: 503, body: "busy", target: target)))
        #expect(!FallbackEligibility.allows(AIKitError.http(status: 404, body: "no route", target: target)))
    }

    @Test("a refused connection is connectionRefused and falls back")
    func refusedConnectionFallsBack() async throws {
        // NetKit wraps a transport failure in `.unknown`; the mapping unwraps
        // it, because the `URLError` inside is what decides the fallback.
        let (provider, _) = try makeSUT(failingWith: NetworkError.unknown(URLError(.cannotConnectToHost)))

        do {
            _ = try await provider.run(prompt: "p", timeout: 30)
            Issue.record("Expected connectionRefused")
        } catch let error as LMStudioProvider.LMStudioError {
            guard case .connectionRefused(let message) = error else {
                Issue.record("Expected connectionRefused, got \(error)")
                return
            }
            #expect(message.contains("127.0.0.1:1234"))
            #expect(FallbackProviderChain.isFallbackEligible(error))
        }
    }

    @Test("a bare URLError is read the same way as a wrapped one")
    func aBareURLErrorIsReadTheSameWay() async throws {
        // The production transport throws `URLError` directly — it does not go
        // through NetKit's `sendRequest` — so this is the shape that actually
        // arrives now.
        let (provider, _) = try makeSUT(failingWith: URLError(.cannotConnectToHost))

        do {
            _ = try await provider.run(prompt: "p", timeout: 30)
            Issue.record("Expected connectionRefused")
        } catch let error as LMStudioProvider.LMStudioError {
            guard case .connectionRefused = error else {
                Issue.record("Expected connectionRefused, got \(error)")
                return
            }
            #expect(FallbackProviderChain.isFallbackEligible(error))
        }
    }

    @Test("a timed-out request is connectionRefused, so a slow server falls back")
    func timedOutFallsBack() async throws {
        let (provider, _) = try makeSUT(failingWith: NetworkError.unknown(URLError(.timedOut)))

        do {
            _ = try await provider.run(prompt: "p", timeout: 30)
            Issue.record("Expected connectionRefused")
        } catch let error as LMStudioProvider.LMStudioError {
            #expect(FallbackProviderChain.isFallbackEligible(error))
        }
    }

    @Test("an answer that is not an HTTP answer is invalidResponse")
    func nonHTTPAnswerIsInvalidResponse() async throws {
        let (provider, _) = try makeSUT(failingWith: AIHTTPTransportError.notAnHTTPResponse)

        do {
            _ = try await provider.run(prompt: "p", timeout: 30)
            Issue.record("Expected invalidResponse")
        } catch let error as LMStudioProvider.LMStudioError {
            guard case .invalidResponse(let message) = error else {
                Issue.record("Expected invalidResponse, got \(error)")
                return
            }
            #expect(message.contains("not an HTTP answer"))
        }
    }

    @Test("a misconfigured address is a settings error, never a connection failure")
    func aBadAddressIsNotAConnectionFailure() throws {
        // Falling back here would hide a misconfigured entry behind a working
        // provider, and the operator would keep an address that never works.
        //
        // W22 moved WHERE this is caught: the init is throwing and
        // non-optional, so a `base_url` carrying `/v1` is refused when the
        // provider is BUILT. Before, a provider was handed out and the refusal
        // waited for the first `run`. Either way nothing is sent — and the
        // mapping below proves the verdict did not change with the timing.
        #expect(throws: AIKitError.self) {
            try makeSUT(baseURL: "http://127.0.0.1:1234/v1")
        }

        let working = try makeSUT().provider
        let error = working.mapped(
            AIKitError.invalidServerAddress(value: "http://127.0.0.1:1234/v1", reason: "it carries the path '/v1'")
        )
        guard case .invalidResponse(let message) = error else {
            Issue.record("Expected invalidResponse, got \(error)")
            return
        }
        #expect(message.contains("/v1"))
        #expect(!FallbackProviderChain.isFallbackEligible(error))
    }

    // MARK: - Configuration

    @Test("the configuration it was given is the configuration it reports")
    func explicitConfigurationWins() throws {
        // Once "an explicit configuration wins over the environment"; since
        // W22 there is no environment to win over — `LMSTUDIO_URL` and
        // `LMSTUDIO_MODEL` are gone from Sources (BR-AIL-20) and the init is
        // non-optional, so what settings said is all there is.
        let provider = try LMStudioProvider(baseURL: "http://custom:9999", model: "custom-model")
        #expect(provider.baseURL == "http://custom:9999")
        #expect(provider.model == "custom-model")
        #expect(provider.requestTimeout == 300)
    }

    @Test("the provider it builds names itself and the configured model")
    func theBuiltProviderNamesItself() throws {
        let (provider, _) = try makeSUT()
        let http = provider.provider(timeout: 30)
        #expect(http.id == "lmstudio")
        #expect(http.displayName == "LM Studio")
        #expect(http.entry.model == "qwen-2.5-coder")
        #expect(http.entry.address?.description == "http://127.0.0.1:1234")
        #expect(http.entry.family == .openAIChat)
        #expect(http.credential == nil)
        // Derived from the codec: chat completions carries tools, so the
        // provider may claim `.toolUse`. It cannot claim what the codec drops.
        #expect(http.capabilities.contains(.textGeneration))
    }

    @Test("the entry it serves carries the provider's own timeout")
    func theEntryCarriesTheTimeout() throws {
        let provider = try LMStudioProvider(baseURL: "http://127.0.0.1:1234", model: "m", requestTimeout: 90)
        #expect(provider.entry.requestTimeout == 90)
    }

    @Test("a bad address is refused when the provider is built, not at the first request")
    func aBadAddressIsRefusedAtConstruction() {
        // W22, BR-AIL-18: the throwing init IS the validation point. The
        // previous shape took `baseURL: String?`, fell back to
        // `LMSTUDIO_URL` and then to `http://127.0.0.1:1234`, and only noticed
        // a bad value when `run` built its entry — by which time a caller had
        // already been handed a provider.
        #expect(throws: AIKitError.self) {
            try LMStudioProvider(baseURL: "https://proxy.corp/openai/v1", model: "m")
        }
    }
}

// MARK: - Model management

@Suite("LMStudioProvider — the model calls that are LM Studio's own")
struct LMStudioModelManagementTests {

    /// `GET /v1/models` lists only what is LOADED; LM Studio's native
    /// `/api/v1/models` lists everything on the host. That is the half of this
    /// server with no OpenAI equivalent, and the reason this type still exists
    /// after the wire moved out (AIKit#18 r4493310420).
    @Test("hostedModels reads LM Studio's native models route")
    func hostedModelsReadsTheNativeRoute() async throws {
        let network = MockHTTPTransport()
        network.body = Data(
            #"{"data":[{"id":"devstral","type":"llm","is_loaded":true},{"id":"nomic","type":"embeddings"}]}"#.utf8
        )
        let provider = try LMStudioProvider(baseURL: "http://127.0.0.1:1234", model: "m", transport: network)

        let models = try await provider.hostedModels()
        #expect(models.map(\.id) == ["devstral", "nomic"])
        #expect(models.first?.isLoaded == true)
        #expect(models.last?.isLoaded == nil)
    }

    @Test("load posts the model to the documented load route")
    func loadPostsToTheLoadRoute() async throws {
        let network = MockHTTPTransport()
        network.body = Data("{}".utf8)
        let provider = try LMStudioProvider(baseURL: "http://127.0.0.1:1234", model: "m", transport: network)

        try await provider.load(model: "devstral", contextLength: 8192, ttl: 600)

        let request = try #require(network.sentRequest)
        #expect(request.url?.absoluteString == "http://127.0.0.1:1234/api/v1/models/load")
        #expect(request.httpMethod == "POST")
        let body = try #require(try network.sentBodyObject())
        #expect(body["model"] as? String == "devstral")
        #expect(body["context_length"] as? Int == 8192)
        #expect(body["ttl"] as? Int == 600)
    }

    @Test("load omits the options the caller did not give")
    func loadOmitsAbsentOptions() async throws {
        let network = MockHTTPTransport()
        network.body = Data("{}".utf8)
        let provider = try LMStudioProvider(baseURL: "http://127.0.0.1:1234", model: "m", transport: network)

        try await provider.load(model: "devstral")

        let body = try #require(try network.sentBodyObject())
        #expect(body["context_length"] == nil)
        #expect(body["ttl"] == nil)
    }

    @Test("unload posts the model to the documented unload route")
    func unloadPostsToTheUnloadRoute() async throws {
        let network = MockHTTPTransport()
        network.body = Data("{}".utf8)
        let provider = try LMStudioProvider(baseURL: "http://127.0.0.1:1234", model: "m", transport: network)

        try await provider.unload(model: "devstral")

        #expect(network.sentRequest?.url?.absoluteString == "http://127.0.0.1:1234/api/v1/models/unload")
        #expect(try network.sentBodyObject()?["model"] as? String == "devstral")
    }

    @Test("a management call that is refused throws, naming the status and the route")
    func aRefusedManagementCallThrows() async throws {
        let network = MockHTTPTransport()
        network.status = 404
        network.body = Data(#"{"error":{"message":"model not found"}}"#.utf8)
        let provider = try LMStudioProvider(baseURL: "http://127.0.0.1:1234", model: "m", transport: network)

        let error = await #expect(throws: AIKitError.self) {
            try await provider.load(model: "no-such-model")
        }
        guard case .http(let status, let body, let target) = try #require(error) else {
            Issue.record("expected .http, got \(String(describing: error))")
            return
        }
        #expect(status == 404)
        #expect(body.contains("model not found"))
        #expect(target.route == "/api/v1/models/load")
    }
}

// MARK: - A 200 can still be a refusal

@Suite("HTTPAIProvider — a 200 reply can still be a refusal")
struct HTTPAIProviderErrorSurfacingTests {

    /// T-AIL-07. A 2xx carrying an error body is not a decode failure, it is
    /// the server talking, and its words must reach the thrown error
    /// (BR-AIL-10).
    @Test("errorBodyOn200 puts the server's text in the thrown error")
    func errorBodyOn200() async throws {
        let network = MockHTTPTransport()
        network.body = try JSONSerialization.data(withJSONObject: [
            "error": ["message": "Unexpected endpoint or method. (POST /v1/v1/chat/completions)"]
        ])
        let provider = try HTTPAIProvider(
            entry: try makeFixtureEntry(family: .openAIChat, id: "openai-compatible"),
            network: network
        )

        do {
            _ = try await provider.complete(
                request: AIRequest(messages: [AIMessage(role: .user, content: "hi")])
            )
            Issue.record("Expected a thrown error")
        } catch let error as AIKitError {
            guard case .requestFailed(let message) = error else {
                Issue.record("Expected requestFailed, got \(error)")
                return
            }
            #expect(message.contains("Unexpected endpoint or method. (POST /v1/v1/chat/completions)"))
            #expect(message.contains("http://127.0.0.1:1234/v1/chat/completions"))
        }
    }

    @Test("a completion still decodes, and names the provider that answered")
    func completionStillDecodes() async throws {
        let network = MockHTTPTransport()
        network.body = Data(
            """
            {"id":"chatcmpl-1","choices":[{"index":0,"message":{"role":"assistant","content":"pong"},
            "finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}
            """.utf8)
        let provider = try HTTPAIProvider(
            entry: try makeFixtureEntry(family: .openAIChat, model: "qwen"),
            network: network
        )

        let response = try await provider.complete(
            request: AIRequest(messages: [AIMessage(role: .user, content: "ping")])
        )
        #expect(response.content == "pong")
        #expect(response.model == "qwen")
        #expect(response.providerID == "lmstudio")
        #expect(response.tokensUsed.total == 3)
    }

    @Test("an empty reply is a case LMStudioProvider maps without reading a sentence")
    func emptyReplyIsSharedByCase() async throws {
        // Generator and matcher agree on a CASE. This used to be two copies of
        // a sentence kept equal by a shared constant, which worked only as long
        // as nothing else composed a message that happened to match it.
        let network = MockHTTPTransport()
        network.body = try JSONSerialization.data(withJSONObject: [
            "id": "chatcmpl-empty",
            "choices": [] as [[String: Any]],
        ])
        let provider = try HTTPAIProvider(
            entry: try makeFixtureEntry(family: .openAIChat, model: "qwen"),
            network: network
        )

        do {
            _ = try await provider.complete(
                request: AIRequest(messages: [AIMessage(role: .user, content: "ping")])
            )
            Issue.record("Expected a thrown error")
        } catch {
            guard case .emptyReply(let target) = error as? AIKitError else {
                Issue.record("Expected AIKitError.emptyReply, got \(error)")
                return
            }
            #expect(target.route == "/v1/chat/completions")
            #expect(target.address.description == "http://127.0.0.1:1234")
            #expect(try LMStudioProvider(baseURL: fixtureServerRoot, model: "qwen").mapped(error) == .emptyContent)
        }
    }
}
