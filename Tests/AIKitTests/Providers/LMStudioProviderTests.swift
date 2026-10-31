import Foundation
import NetKit
import Testing

@testable import AIKit

// MARK: - LMStudioProviderTests
//
// LM Studio no longer owns a second copy of the OpenAI wire format: it renders
// a prompt into an `AIRequest`, `OpenAIProvider` sends it, and what comes back
// is translated into `LMStudioError` so a chain's fallback rule keeps reading
// the cases it always read. These tests therefore exercise the real path —
// mock transport, real endpoint, real provider — rather than two helpers that
// built a dictionary nothing sent (BR-AIL-12: no test reaches a server).
//
// `OpenAIProviderTests` (T-AIL-07) lives in this file because the wave declares
// no `OpenAIProviderTests.swift`, and the 200-with-an-error-body path it covers
// is the same path LM Studio reaches below.

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
    return try! JSONSerialization.data(withJSONObject: json)
}

/// An error body, the nested form.
private func makeErrorJSON(message: String) -> Data {
    let json: [String: Any] = [
        "error": [
            "message": message,
            "type": "server_error",
        ],
    ]
    return try! JSONSerialization.data(withJSONObject: json)
}

private func makeSUT(
    answering data: Data? = nil,
    statusCode: Int = 200,
    failingWith error: NetworkError? = nil,
    baseURL: String = "http://127.0.0.1:1234",
    model: String = "qwen-2.5-coder",
    requestTimeout: TimeInterval = 300
) -> (provider: LMStudioProvider, network: MockNetworkService) {
    let network = MockNetworkService()
    network.resultData = data
    network.statusCode = statusCode
    network.resultError = error
    let provider = LMStudioProvider(
        baseURL: baseURL,
        model: model,
        requestTimeout: requestTimeout,
        networkService: network
    )
    return (provider, network)
}

// MARK: - LMStudioProvider

@Suite("LMStudioProvider — one wire path, its own error vocabulary")
struct LMStudioProviderTests {

    // MARK: - The request on the wire

    @Test("sends an OpenAI-compatible chat completion to /v1/chat/completions")
    func sendsAnOpenAICompatibleRequest() async throws {
        let (provider, network) = makeSUT(answering: makeChatCompletionJSON(content: "ok"))

        _ = try await provider.run(prompt: "Hello, world!", timeout: 30)

        #expect(network.capturedRequests.count == 1)
        let request = try #require(network.capturedRequests.first)
        #expect(request.url?.absoluteString == "http://127.0.0.1:1234/v1/chat/completions")
        #expect(request.httpMethod == "POST")
        #expect(request.allHTTPHeaderFields?["Content-Type"] == "application/json")
        #expect(request.allHTTPHeaderFields?["Authorization"] == nil, "a local server takes no bearer token")

        let httpBody = try #require(request.httpBody)
        let body = try #require(JSONSerialization.jsonObject(with: httpBody) as? [String: Any])
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
        let provider = LMStudioProvider(baseURL: "http://127.0.0.1:1234", model: "m", requestTimeout: 300)

        let within = try #require(provider.provider(timeout: 120).networkService as? TimeoutNetworkService)
        #expect(within.timeoutInterval == 120)

        let capped = try #require(provider.provider(timeout: 900).networkService as? TimeoutNetworkService)
        #expect(capped.timeoutInterval == 300, "requestTimeout is the ceiling")

        let endpoint = OpenAIChatCompletionEndPoint(
            host: "127.0.0.1",
            port: 1234,
            scheme: "http",
            apiKey: nil,
            requestBody: ChatCompletionRequest(
                model: "m",
                messages: [],
                temperature: 0.7,
                maxTokens: -1,
                stream: false,
                tools: nil,
                responseFormat: nil
            )
        )
        #expect(within.createRequest(endPoint: endpoint).timeoutInterval == 120)
        #expect(NetworkService().createRequest(endPoint: endpoint).timeoutInterval == 60)
    }

    @Test("an injected transport is used as it was given")
    func injectedTransportIsNotWrapped() {
        let (provider, network) = makeSUT()
        #expect(provider.provider(timeout: 30).networkService as? MockNetworkService === network)
    }

    // MARK: - The answer

    @Test("parses an OpenAI-format chat completion")
    func parsesTheAnswer() async throws {
        let (provider, _) = makeSUT(
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
        let (provider, _) = makeSUT(answering: data)

        await #expect(throws: LMStudioProvider.LMStudioError.emptyContent) {
            _ = try await provider.run(prompt: "p", timeout: 30)
        }
    }

    @Test("a whitespace-only answer is emptyContent")
    func whitespaceAnswerIsEmptyContent() async throws {
        let (provider, _) = makeSUT(answering: makeChatCompletionJSON(content: "   \n  "))

        await #expect(throws: LMStudioProvider.LMStudioError.emptyContent) {
            _ = try await provider.run(prompt: "p", timeout: 30)
        }
    }

    @Test("malformed JSON is invalidResponse, not a crash")
    func malformedJSONIsInvalidResponse() async throws {
        let (provider, _) = makeSUT(answering: Data("not json at all".utf8))

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
        let (provider, _) = makeSUT(
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
                "the error names the URL it came from"
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
        let (provider, _) = makeSUT(answering: data)

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
        let (provider, _) = makeSUT(answering: Data("{}".utf8), statusCode: 429)

        await #expect(throws: LMStudioProvider.LMStudioError.rateLimited) {
            _ = try await provider.run(prompt: "p", timeout: 30)
        }
        #expect(FallbackProviderChain.isFallbackEligible(LMStudioProvider.LMStudioError.rateLimited))
    }

    @Test("500 is httpError and does not fall back")
    func httpErrorOn500() async throws {
        let (provider, _) = makeSUT(answering: Data("{}".utf8), statusCode: 500)

        do {
            _ = try await provider.run(prompt: "p", timeout: 30)
            Issue.record("Expected httpError")
        } catch let error as LMStudioProvider.LMStudioError {
            guard case .httpError(let code, let message) = error else {
                Issue.record("Expected httpError, got \(error)")
                return
            }
            #expect(code == 500)
            #expect(message.contains("http://127.0.0.1:1234"))
            #expect(!FallbackProviderChain.isFallbackEligible(error))
        }
    }

    @Test("a refused connection is connectionRefused and falls back")
    func refusedConnectionFallsBack() async throws {
        // NetKit wraps a transport failure in `.unknown`; the mapping unwraps
        // it, because the `URLError` inside is what decides the fallback.
        let (provider, _) = makeSUT(failingWith: .unknown(URLError(.cannotConnectToHost)))

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

    @Test("a timed-out request is connectionRefused, so a slow server falls back")
    func timedOutFallsBack() async throws {
        let (provider, _) = makeSUT(failingWith: .unknown(URLError(.timedOut)))

        do {
            _ = try await provider.run(prompt: "p", timeout: 30)
            Issue.record("Expected connectionRefused")
        } catch let error as LMStudioProvider.LMStudioError {
            #expect(FallbackProviderChain.isFallbackEligible(error))
        }
    }

    @Test("a non-HTTP answer is invalidResponse")
    func nonHTTPAnswerIsInvalidResponse() async throws {
        let (provider, _) = makeSUT(failingWith: .requestFailed(description: "Unexpected HTTP answer"))

        do {
            _ = try await provider.run(prompt: "p", timeout: 30)
            Issue.record("Expected invalidResponse")
        } catch let error as LMStudioProvider.LMStudioError {
            guard case .invalidResponse(let message) = error else {
                Issue.record("Expected invalidResponse, got \(error)")
                return
            }
            #expect(message.contains("Unexpected HTTP answer"))
        }
    }

    // MARK: - Configuration

    @Test("an explicit configuration wins over the environment")
    func explicitConfigurationWins() {
        let provider = LMStudioProvider(baseURL: "http://custom:9999", model: "custom-model")
        #expect(provider.baseURL == "http://custom:9999")
        #expect(provider.model == "custom-model")
        #expect(provider.requestTimeout == 300)
    }

    @Test("the provider it builds names itself and the configured model")
    func theBuiltProviderNamesItself() {
        let (provider, _) = makeSUT()
        let openAI = provider.provider(timeout: 30)
        #expect(openAI.id == "lmstudio")
        #expect(openAI.displayName == "LM Studio")
        #expect(openAI.modelName == "qwen-2.5-coder")
        #expect(openAI.baseURL == "http://127.0.0.1:1234")
        #expect(openAI.apiKey == nil)
        #expect(openAI.capabilities == .textGeneration)
    }
}

// MARK: - OpenAIProvider

@Suite("OpenAIProvider — a 200 reply can still be a refusal")
struct OpenAIProviderTests {

    /// T-AIL-07. The typed provider, straight: a 200 carrying an error body is
    /// not a decode failure, it is the server talking, and its words must reach
    /// the thrown error (BR-AIL-10).
    @Test("errorBodyOn200 puts the server's text in the thrown error")
    func errorBodyOn200() async throws {
        let network = MockNetworkService()
        network.resultData = try JSONSerialization.data(withJSONObject: [
            "error": ["message": "Unexpected endpoint or method. (POST /v1/v1/chat/completions)"]
        ])
        let provider = OpenAIProvider(
            id: "openai-compatible",
            displayName: "OpenAI Compatible",
            capabilities: .textGeneration,
            baseURL: "http://127.0.0.1:1234",
            modelName: "any",
            networkService: network
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
        let network = MockNetworkService()
        network.resultData = Data(
            """
            {"id":"chatcmpl-1","choices":[{"index":0,"message":{"role":"assistant","content":"pong"},
            "finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}
            """.utf8)
        let provider = OpenAIProvider(
            id: "lmstudio",
            displayName: "LM Studio",
            capabilities: .textGeneration,
            baseURL: "http://127.0.0.1:1234",
            modelName: "qwen",
            networkService: network
        )

        let response = try await provider.complete(
            request: AIRequest(messages: [AIMessage(role: .user, content: "ping")])
        )
        #expect(response.content == "pong")
        #expect(response.model == "qwen")
        #expect(response.providerID == "lmstudio")
        #expect(response.tokensUsed.total == 3)
    }

    @Test("the empty-reply message is the one LMStudioProvider matches on")
    func emptyReplyMessageIsShared() async throws {
        // Generator and matcher read one constant: this asserts the agreement
        // rather than assuming two copies of a sentence stay equal.
        let network = MockNetworkService()
        network.resultData = try JSONSerialization.data(withJSONObject: [
            "id": "chatcmpl-empty",
            "choices": [] as [[String: Any]],
        ])
        let provider = OpenAIProvider(
            id: "lmstudio",
            displayName: "LM Studio",
            capabilities: .textGeneration,
            baseURL: "http://127.0.0.1:1234",
            modelName: "qwen",
            networkService: network
        )

        do {
            _ = try await provider.complete(
                request: AIRequest(messages: [AIMessage(role: .user, content: "ping")])
            )
            Issue.record("Expected a thrown error")
        } catch {
            #expect(error as? AIKitError == .requestFailed(OpenAIProvider.emptyReplyMessage))
            #expect(LMStudioProvider(model: "qwen").mapped(error) == .emptyContent)
        }
    }
}
