import Foundation
import NetKit
import Testing

@testable import AIKit

// A reply that carries no usage reports no usage. It must not read as zero
// tokens: a zero silently under-charges a budget and passes a limit check.
// Replies are minimal on purpose: only the fields the decoder requires.

private struct NoUsageCase: Sendable, CustomTestStringConvertible {
    let standard: AIWireStandard
    let reply: String
    var testDescription: String { standard.rawValue }
}

private let noUsageCases: [NoUsageCase] = [
    NoUsageCase(
        standard: .openAIChat,
        reply: #"{"id":"x","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}]}"#
    ),
    NoUsageCase(
        standard: .openAIResponses,
        reply: #"{"id":"r","output":[{"type":"message","content":[{"type":"output_text","text":"hi"}]}]}"#
    ),
    NoUsageCase(
        standard: .anthropicMessages,
        reply: #"{"id":"m","content":[{"type":"text","text":"hi"}]}"#
    ),
    NoUsageCase(
        standard: .lmStudioNative,
        reply: #"{"output":[{"type":"message","content":"hi"}]}"#
    ),
]

private func complete(_ standard: AIWireStandard, answering reply: String) async throws -> AIResponse {
    let network = MockNetworkService()
    network.resultData = Data(reply.utf8)
    return try await standard.complete(
        request: AIRequest(messages: [AIMessage(role: .user, content: "p")], temperature: 0, maxTokens: 16),
        target: try AIWireTarget(serverRoot: "http://127.0.0.1:1234", apiKey: nil, model: "m", providerID: "lmstudio"),
        networkService: network
    )
}

extension WireStandardShapeTests {
    @Test("missing usage: a reply without usage decodes to nil counts", arguments: noUsageCases)
    fileprivate func absentUsageIsNil(testCase: NoUsageCase) async throws {
        let response = try await complete(testCase.standard, answering: testCase.reply)

        #expect(response.content == "hi")
        #expect(response.tokensUsed.prompt == nil)
        #expect(response.tokensUsed.completion == nil)
        #expect(response.tokensUsed.total == nil)
    }

    @Test("missing usage: a total is only claimed when both sides are known")
    func totalNeedsBothSides() {
        #expect(TokenUsage(prompt: 3, completion: nil).total == nil)
        #expect(TokenUsage(prompt: nil, completion: 4).total == nil)
        #expect(TokenUsage(prompt: 3, completion: 4).total == 7)
    }

    @Test("missing usage: unknown counts survive an encode and decode round trip as unknown")
    func unknownRoundTrips() throws {
        let data = try JSONEncoder().encode(TokenUsage(prompt: 5, completion: nil))
        let decoded = try JSONDecoder().decode(TokenUsage.self, from: data)
        #expect(decoded.prompt == 5)
        #expect(decoded.completion == nil)
        #expect(decoded.total == nil)
    }

    @Test("error body: an error body that is present but malformed throws instead of reading as a completion (NetKit wraps the DecodingError)")
    func malformedErrorBodyThrows() async {
        let reply = #"{"error":{"code":7},"id":"x","choices":[]}"#
        await #expect(throws: NetworkError.self) {
            _ = try await complete(.openAIChat, answering: reply)
        }
    }
}
