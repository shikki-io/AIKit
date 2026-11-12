import Foundation
import NetKit
import Testing

@testable import AIKit

// A reply that carries no usage reports no usage. It must not read as zero
// tokens: a zero silently under-charges a budget and passes a limit check.
//
// W24: the four replies these tests read used to be hand-written minimal
// payloads, one per family (AIKit#18 r4500580810: *"Is it normal all the reply
// data is not build from a typed model with those fake data?"*). They are now
// MINTED from each family's own reply DTO with `promptTokens: nil` and
// `model: nil`, which is what "a reply that claims no usage and names no
// model" IS — the absence is expressed by the type rather than by a payload
// that happens to omit two keys. A key that moves is now a compile error here
// instead of a field that quietly decodes to nil.

@Suite("WireCodecTests — a reply that claims no usage claims none")
struct WireCodecUsageTests {

    /// A reply with no `usage` and no `model`, for one family.
    private func mintUsagelessReply(_ family: AIProviderFamily) throws -> String {
        try mintReply(family: family, model: nil, promptTokens: nil, completionTokens: nil)
    }

    @Test("a reply without usage decodes to nil counts", arguments: AIProviderFamily.httpFamilies)
    func absentUsageIsNil(family: AIProviderFamily) throws {
        let fixture = try #require(wireFamilyFixtures.first { $0.family == family })
        let context = try makeFixtureContext(family: family, route: fixture.route)
        let response = try fixture.codec.response(
            from: Data(try mintUsagelessReply(family).utf8), context: context)

        #expect(response.content == fakeReplyText)
        #expect(response.tokensUsed.prompt == nil)
        #expect(response.tokensUsed.completion == nil)
        #expect(response.tokensUsed.total == nil)
    }

    @Test(
        "a reply that names no model falls back to the model that was asked for",
        arguments: AIProviderFamily.httpFamilies
    )
    func absentModelFallsBackToTheRequested(family: AIProviderFamily) throws {
        let fixture = try #require(wireFamilyFixtures.first { $0.family == family })
        let context = try makeFixtureContext(family: family, route: fixture.route)
        let response = try fixture.codec.response(
            from: Data(try mintUsagelessReply(family).utf8), context: context)
        // The minted reply omits `model` entirely, so the requested name is the
        // only thing left to report — and it is reported as such rather than
        // left blank (BR-AIL-06).
        #expect(response.model == context.requestedModel)
    }

    @Test("a minted reply omits the keys it was minted without", arguments: AIProviderFamily.httpFamilies)
    func theMintedReplyReallyOmitsThem(family: AIProviderFamily) throws {
        // The silent fail this closes: if `mintReply` wrote `usage: null` or a
        // zeroed usage object instead of omitting the key, both tests above
        // would still pass for the wrong reason — the codec would be reading a
        // present-but-empty usage rather than an absent one, and the nil they
        // assert would prove nothing about an absent key.
        let minted = try mintUsagelessReply(family)
        let object = try #require(
            try JSONSerialization.jsonObject(with: Data(minted.utf8)) as? [String: Any])

        for key in ["usage", "stats", "model", "model_instance_id"] {
            #expect(object[key] == nil, "\(family.rawValue) minted a '\(key)' it was asked to omit")
        }
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
        //
        // Hand-written, and declared as such: `FixtureWireError` writes a
        // well-formed envelope by construction, so the only way to state "an
        // envelope the decoder cannot read" is to write one by hand.
        #expect(throws: AIKitError.self) {
            _ = try fixture.codec.response(
                from: Data(#"{"error":{"code":7},"id":"x","choices":[]}"#.utf8),
                context: context
            )
        }
    }
}
