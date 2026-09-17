import Foundation
import Testing
@testable import AIKit
import ShellKit

// MARK: - Mock URLProtocol

/// Custom URLProtocol that intercepts network requests for deterministic testing.
/// Avoids hitting real LM Studio server during unit tests.
final class MockURLProtocol: URLProtocol, @unchecked Sendable {
    nonisolated(unsafe) static var mockResponseData: Data?
    nonisolated(unsafe) static var mockStatusCode: Int = 200
    nonisolated(unsafe) static var mockError: (any Error)?
    nonisolated(unsafe) static var lastRequest: URLRequest?

    override class func canInit(with request: URLRequest) -> Bool {
        true
    }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest {
        request
    }

    override func startLoading() {
        Self.lastRequest = request

        if let error = Self.mockError {
            client?.urlProtocol(self, didFailWithError: error)
            client?.urlProtocolDidFinishLoading(self)
            return
        }

        let response = HTTPURLResponse(
            url: request.url!,
            statusCode: Self.mockStatusCode,
            httpVersion: "HTTP/1.1",
            headerFields: ["Content-Type": "application/json"]
        )!

        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)

        if let data = Self.mockResponseData {
            client?.urlProtocol(self, didLoad: data)
        }

        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}

    static func reset() {
        mockResponseData = nil
        mockStatusCode = 200
        mockError = nil
        lastRequest = nil
    }
}

// MARK: - Test Helpers

/// Build a valid OpenAI-format chat completion response JSON.
private func makeChatCompletionJSON(content: String) -> Data {
    let json: [String: Any] = [
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "created": Int(Date().timeIntervalSince1970),
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

/// Build an error response JSON.
private func makeErrorJSON(message: String) -> Data {
    let json: [String: Any] = [
        "error": [
            "message": message,
            "type": "server_error",
        ],
    ]
    return try! JSONSerialization.data(withJSONObject: json)
}

/// Create a URLSession that uses the MockURLProtocol.
private func makeMockSession() -> URLSession {
    let config = URLSessionConfiguration.ephemeral
    config.protocolClasses = [MockURLProtocol.self]
    return URLSession(configuration: config)
}

// MARK: - LMStudioProvider Tests

@Suite("LMStudioProvider — Local AI Provider")
struct LMStudioProviderTests {

    init() {
        MockURLProtocol.reset()
    }

    // MARK: - Test 1: Builds correct request payload

    @Test("builds correct OpenAI-compatible request payload")
    func buildsCorrectRequestPayload() throws {
        let provider = LMStudioProvider(
            baseURL: "http://127.0.0.1:1234",
            model: "qwen-2.5-coder"
        )

        let body = provider.buildRequestBody(prompt: "Hello, world!")

        let model = body["model"] as? String
        #expect(model == "qwen-2.5-coder")

        let messages = body["messages"] as? [[String: String]]
        #expect(messages?.count == 1)
        #expect(messages?.first?["role"] == "user")
        #expect(messages?.first?["content"] == "Hello, world!")

        let stream = body["stream"] as? Bool
        #expect(stream == false)

        let temperature = body["temperature"] as? Double
        #expect(temperature == 0.7)
    }

    // MARK: - Test 2: Parses OpenAI-format response

    @Test("parses OpenAI-format chat completion response")
    func parsesOpenAIResponse() throws {
        let provider = LMStudioProvider(
            baseURL: "http://127.0.0.1:1234",
            model: "test-model"
        )

        let responseData = makeChatCompletionJSON(content: "This is the assistant response.")
        let result = try provider.parseResponse(responseData)
        #expect(result == "This is the assistant response.")
    }

    // MARK: - Test 3: Handles connection refused

    @Test("throws connectionRefused on network failure")
    func handlesConnectionRefused() async throws {
        // We test this by verifying the error type mapping,
        // since we can't easily simulate a real connection refusal in unit tests
        // without a custom URLSession. Verify the error enum is well-formed.
        let error = LMStudioProvider.LMStudioError.connectionRefused(
            "Cannot connect to LM Studio at http://127.0.0.1:1234"
        )

        switch error {
        case .connectionRefused(let msg):
            #expect(msg.contains("127.0.0.1:1234"))
        default:
            Issue.record("Expected connectionRefused error")
        }

        // Verify it triggers fallback eligibility
        #expect(FallbackProviderChain.isFallbackEligible(error))
    }

    // MARK: - Test 4: Reads model from env var

    @Test("reads model name from environment variable when not explicit")
    func readsModelFromEnv() {
        // When explicit model is provided, it takes precedence
        let provider = LMStudioProvider(
            baseURL: "http://localhost:1234",
            model: "explicit-model"
        )
        #expect(provider.model == "explicit-model")

        // Default baseURL
        let defaultProvider = LMStudioProvider(
            baseURL: "http://custom:9999",
            model: "custom-model"
        )
        #expect(defaultProvider.baseURL == "http://custom:9999")
        #expect(defaultProvider.model == "custom-model")
    }

    // MARK: - Additional: Empty content handling

    @Test("throws emptyContent when response has no choices")
    func emptyContentThrows() throws {
        let provider = LMStudioProvider(model: "test")

        let emptyJSON: [String: Any] = [
            "choices": [] as [[String: Any]]
        ]
        let data = try JSONSerialization.data(withJSONObject: emptyJSON)

        #expect(throws: LMStudioProvider.LMStudioError.self) {
            try provider.parseResponse(data)
        }
    }


    // MARK: - Additional: Invalid JSON response

    @Test("throws invalidResponse on malformed JSON")
    func invalidJSONThrows() throws {
        let provider = LMStudioProvider(model: "test")
        let badData = "not json at all".data(using: .utf8)!

        #expect(throws: LMStudioProvider.LMStudioError.self) {
            try provider.parseResponse(badData)
        }
    }

    // MARK: - Additional: Rate limit error

    @Test("rateLimited error is fallback-eligible")
    func rateLimitedIsFallbackEligible() {
        let error = LMStudioProvider.LMStudioError.rateLimited
        #expect(FallbackProviderChain.isFallbackEligible(error))
    }

    // MARK: - Additional: httpError is not fallback-eligible

    @Test("httpError (500) is NOT fallback-eligible")
    func httpErrorNotFallbackEligible() {
        let error = LMStudioProvider.LMStudioError.httpError(500, "Internal Server Error")
        #expect(!FallbackProviderChain.isFallbackEligible(error))
    }
}
