import Foundation
import NetKit
import Testing

@testable import AIKit

// A reply that carries no usage reports no usage. It must not read as zero
// tokens: a zero silently under-charges a budget and passes a limit check.
// Replies are minimal on purpose: only the fields the decoder requires.

private struct NoUsageCase: Sendable, CustomTestStringConvertible {
    let family: AIProviderFamily
    let reply: String
    var testDescription: String { family.rawValue }
}

private let noUsageCases: [NoUsageCase] = [
    NoUsageCase(
        family: .openAIChat,
        reply:
            #"{"id":"x","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}]}"#
    ),
    NoUsageCase(
        family: .openAIResponses,
        reply: #"{"id":"r","output":[{"type":"message","content":[{"type":"output_text","text":"hi"}]}]}"#
    ),
    NoUsageCase(
        family: .anthropicMessages,
        reply: #"{"id":"m","content":[{"type":"text","text":"hi"}]}"#
    ),
    NoUsageCase(
        family: .lmStudioNative,
        reply: #"{"output":[{"type":"message","content":"hi"}]}"#
    ),
]

@Suite("WireCodecTests — a reply that claims no usage claims none")
struct WireCodecUsageTests {

    @Test("a reply without usage decodes to nil counts", arguments: noUsageCases)
    fileprivate func absentUsageIsNil(testCase: NoUsageCase) throws {
        let fixture = try #require(wireFamilyFixtures.first { $0.family == testCase.family })
        let context = try makeFixtureContext(family: fixture.family, route: fixture.route)
        let response = try fixture.codec.response(from: Data(testCase.reply.utf8), context: context)

        #expect(response.content == "hi")
        #expect(response.tokensUsed.prompt == nil)
        #expect(response.tokensUsed.completion == nil)
        #expect(response.tokensUsed.total == nil)
    }

    @Test("a reply that names no model falls back to the model that was asked for", arguments: noUsageCases)
    fileprivate func absentModelFallsBackToTheRequested(testCase: NoUsageCase) throws {
        let fixture = try #require(wireFamilyFixtures.first { $0.family == testCase.family })
        let context = try makeFixtureContext(family: fixture.family, route: fixture.route)
        let response = try fixture.codec.response(from: Data(testCase.reply.utf8), context: context)
        // Three of these four replies carry no `model`; the chat one carries
        // none either in this minimal shape. The requested name is the only
        // thing left to report, and it is reported as such.
        #expect(response.model == context.requestedModel)
    }

    @Test("a total is only claimed when both sides are known")
    func totalNeedsBothSides() {
        #expect(TokenUsage(prompt: 3, completion: nil).total == nil)
        #expect(TokenUsage(prompt: nil, completion: 4).total == nil)
        #expect(TokenUsage(prompt: 3, completion: 4).total == 7)
    }

    @Test("unknown counts survive an encode and decode round trip as unknown")
    func unknownRoundTrips() throws {
        let data = try JSONEncoder().encode(TokenUsage(prompt: 5, completion: nil))
        let decoded = try JSONDecoder().decode(TokenUsage.self, from: data)
        #expect(decoded.prompt == 5)
        #expect(decoded.completion == nil)
        #expect(decoded.total == nil)
    }

    @Test("an error body that is present but malformed throws instead of reading as a completion")
    func malformedErrorBodyThrows() throws {
        let fixture = try #require(wireFamilyFixtures.first { $0.family == .openAIChat })
        let context = try makeFixtureContext(family: fixture.family, route: fixture.route)
        // `{"error":{"code":7}}` has an `error` that is present, non-null and
        // undecodable. It must NOT read as "no error, so a completion".
        #expect(throws: AIKitError.self) {
            _ = try fixture.codec.response(
                from: Data(#"{"error":{"code":7},"id":"x","choices":[]}"#.utf8),
                context: context
            )
        }
    }
}
