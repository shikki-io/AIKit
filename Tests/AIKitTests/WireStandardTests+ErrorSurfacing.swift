import Foundation
import NetKit
import Testing

@testable import AIKit

// MARK: - Review fixes for PR #18
//
// 1. An unencodable body must reach the caller as a thrown error. The
//    endpoints' `body` is `try? bodyObject()`, so the only thing between a bad
//    body and a bodyless POST is the throwing guard on the dispatch path.
//    These tests prove that guard on every standard and on OpenAIProvider.
// 2. The reply mapper must carry the server's tool-call id through.

/// `.nan` is not representable in JSON, so `JSONEncoder` throws on it.
private func unencodableRequest() -> AIRequest {
    AIRequest(
        messages: [AIMessage(role: .user, content: "hi")],
        temperature: .nan,
        maxTokens: 16
    )
}

@Suite("Wire dispatch — an unencodable body throws, never a bodyless request")
struct WireStandardTestsBodyEncodeFailure {

    @Test("complete throws and sends nothing", arguments: AIWireStandard.allCases)
    func completeThrows(standard: AIWireStandard) async throws {
        let network = MockNetworkService()
        network.resultData = Data("{}".utf8)
        let target = try AIWireTarget(serverRoot: "http://127.0.0.1:1234", apiKey: nil, model: "m", providerID: "p")

        await #expect(throws: (any Error).self) {
            _ = try await standard.complete(request: unencodableRequest(), target: target, networkService: network)
        }
        #expect(network.capturedRequests.isEmpty)
    }

    @Test("endPoint(target:request:stream:) throws", arguments: AIWireStandard.allCases)
    func endPointThrows(standard: AIWireStandard) throws {
        let target = try AIWireTarget(serverRoot: "http://127.0.0.1:1234", apiKey: nil, model: "m", providerID: "p")
        #expect(throws: (any Error).self) {
            _ = try standard.endPoint(target: target, request: unencodableRequest(), stream: false)
        }
        #expect(throws: (any Error).self) {
            _ = try standard.endPoint(target: target, request: unencodableRequest(), stream: true)
        }
    }

    @Test("a valid request still yields a body")
    func validBodyIsPresent() throws {
        let target = try AIWireTarget(serverRoot: "http://127.0.0.1:1234", apiKey: nil, model: "m", providerID: "p")
        for standard in AIWireStandard.allCases {
            let valid = AIRequest(messages: [AIMessage(role: .user, content: "hi")], temperature: 0, maxTokens: 16)
            let endpoint = try standard.endPoint(target: target, request: valid, stream: false)
            #expect(endpoint.body != nil, "\(standard.rawValue)")
        }
    }

    @Test("OpenAIProvider.complete throws and sends nothing")
    func providerCompleteThrows() async throws {
        let network = MockNetworkService()
        network.resultData = Data("{}".utf8)
        let provider = OpenAIProvider(
            id: "p",
            displayName: "P",
            capabilities: [],
            baseURL: "http://127.0.0.1:1234",
            modelName: "m",
            networkService: network
        )
        await #expect(throws: (any Error).self) {
            _ = try await provider.complete(request: unencodableRequest())
        }
        #expect(network.capturedRequests.isEmpty)
    }
}

@Suite("ChatCompletionReply — tool-call id")
struct WireStandardTestsReplyToolCallID {

    @Test("the server's tool-call id survives the reply mapper")
    func idIsPreserved() throws {
        let json = """
            {
              "id": "chatcmpl-tool",
              "model": "m",
              "choices": [{
                "index": 0,
                "message": {
                  "role": "assistant",
                  "content": null,
                  "tool_calls": [{
                    "id": "call_abc",
                    "type": "function",
                    "function": {"name": "search", "arguments": "{}"}
                  }]
                },
                "finish_reason": "tool_calls"
              }]
            }
            """
        let dto = try JSONDecoder().decode(ChatCompletionResponseDTO.self, from: Data(json.utf8))
        let response = try ChatCompletionReply.completion(dto).response(
            requestedModel: "m", providerID: "p", latencyMs: 1, requestURL: "http://x"
        )
        #expect(response.toolCalls?.first?.id == "call_abc")
        #expect(response.toolCalls?.first?.name == "search")
    }
}
