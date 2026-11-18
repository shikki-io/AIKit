// StreamingChunkToolCallTests.swift — a streamed tool call reaches the consumer.
//
// Operator, 2026-11-18: "yes we have to have one API open for making this
// interaction generic against all of our ai provider like this it's one usage
// and AIKit manage the complexity internally".
//
// The one API existed for TEXT. `ChatCompletionChunkDTO` has always parsed the
// role, the tool-call deltas and the finish reason, and `OpenAIChatCodec.chunk`
// threw all three away before returning `AIChunk(delta:isComplete:)`. A chunk
// carrying ONLY a tool-call step hit `guard !delta.isEmpty || isComplete` and
// was dropped, so a consumer streaming a tool call received nothing at all —
// which is why shikki's `LocalHTTPAgentRunner` carries its own
// `ChatCompletionReply` / `ToolCall` and decodes the wire a second time.
//
// `toolCallArrivesWhenItIsTheOnlyNews` is the falsifiable one: it returns nil
// at this branch's parent.

import Testing

@testable import AIKit

@Suite("Streaming chunks carry tool calls, roles and finish reasons")
struct StreamingChunkToolCallTests {

    private func chunk(_ line: String) throws -> AIChunk? {
        let context = try makeFixtureContext(family: .openAIChat, route: "/v1/chat/completions")
        return try OpenAIChatCodec().chunk(fromLine: line, context: context)
    }

    @Test("a chunk whose only news is a tool-call step is delivered, not dropped")
    func toolCallArrivesWhenItIsTheOnlyNews() throws {
        let line = try mintChatChunkLine(
            toolCalls: [FixtureToolCallDelta(index: 0, id: "call_abc", name: "get_weather", arguments: "")]
        )
        let got = try #require(try chunk(line), "a tool-call step is news; dropping it loses the call")
        #expect(got.delta.isEmpty)
        #expect(got.isComplete == false)
        let deltas = try #require(got.toolCallDeltas)
        #expect(deltas.count == 1)
        #expect(deltas[0].index == 0)
        #expect(deltas[0].id == "call_abc")
        #expect(deltas[0].name == "get_weather")
    }

    @Test("argument text arrives in fragments the consumer joins by index")
    func argumentFragmentsCarryTheirIndex() throws {
        let first = try #require(try chunk(try mintChatChunkLine(
            toolCalls: [FixtureToolCallDelta(index: 0, id: "call_abc", name: "get_weather", arguments: fixtureToolArgumentsHead)]
        )))
        let second = try #require(try chunk(try mintChatChunkLine(
            toolCalls: [FixtureToolCallDelta(index: 0, id: nil, name: nil, arguments: fixtureToolArgumentsTail)]
        )))

        // The second fragment carries neither id nor name — that is the wire's
        // shape, and it is why `index` is what joins them.
        #expect(try #require(second.toolCallDeltas)[0].id == nil)
        #expect(try #require(second.toolCallDeltas)[0].name == nil)

        let joined = [first, second]
            .compactMap(\.toolCallDeltas)
            .flatMap { $0 }
            .filter { $0.index == 0 }
            .compactMap(\.argumentsFragment)
            .joined()
        #expect(joined == fixtureToolArgumentsJoined)
    }

    @Test("two calls in one chunk keep their own indices")
    func parallelCallsKeepTheirIndices() throws {
        let got = try #require(try chunk(try mintChatChunkLine(toolCalls: [
            FixtureToolCallDelta(index: 0, id: "call_a", name: "get_weather", arguments: nil),
            FixtureToolCallDelta(index: 1, id: "call_b", name: "get_time", arguments: nil),
        ])))
        let deltas = try #require(got.toolCallDeltas)
        #expect(deltas.map(\.index) == [0, 1])
        #expect(deltas.map(\.name) == ["get_weather", "get_time"])
    }

    @Test("a provider that omits index falls back to array position")
    func omittedIndexFallsBackToPosition() throws {
        let got = try #require(try chunk(try mintChatChunkLine(toolCalls: [
            FixtureToolCallDelta(index: nil, id: "call_a", name: "one", arguments: nil),
            FixtureToolCallDelta(index: nil, id: "call_b", name: "two", arguments: nil),
        ])))
        #expect(try #require(got.toolCallDeltas).map(\.index) == [0, 1])
    }

    @Test("finish_reason tool_calls is why it stopped, and that it stopped")
    func finishReasonToolCalls() throws {
        let got = try #require(try chunk(try mintChatChunkLine(finishReason: "tool_calls")))
        #expect(got.isComplete)
        #expect(got.finishReason == .toolCalls)
    }

    @Test("an unknown finish reason round-trips instead of failing the stream")
    func unknownFinishReasonSurvives() throws {
        // The open-vocabulary reason AIFinishReason is a RawRepresentable struct
        // and not an enum: a provider we have not seen must not break a reply
        // that is otherwise good.
        let got = try #require(try chunk(try mintChatChunkLine(finishReason: "max_output_tokens")))
        #expect(got.isComplete)
        #expect(got.finishReason?.rawValue == "max_output_tokens")
        #expect(got.finishReason != .stop)
    }

    @Test("the role-only priming chunk is still dropped")
    func roleOnlyChunkStaysDropped() throws {
        // Regression guard: the captured stream opens with
        // `{"delta":{"role":"assistant"}}`, which is not news. Widening the
        // model must not start emitting it.
        #expect(try chunk(try mintChatChunkLine(role: "assistant")) == nil)
    }

    @Test("a text chunk still reports its role and choice index")
    func textChunkCarriesRoleAndIndex() throws {
        let got = try #require(try chunk(try mintChatChunkLine(content: "p", role: "assistant")))
        #expect(got.delta == "p")
        #expect(got.role == .assistant)
        #expect(got.choiceIndex == 0)
        #expect(got.toolCallDeltas == nil)
    }

    @Test("the captured text stream decodes to the same deltas as before")
    func capturedTextStreamIsUnchanged() throws {
        // The existing fixture is a real LM Studio capture. Widening AIChunk
        // must leave its text answer identical.
        let fixture = try #require(wireFamilyFixtures.first { $0.family == .openAIChat })
        let chunks = try fixtureChunks(fixture)

        // Measured, not assumed. The capture yields four chunks: two text
        // deltas, then TWO terminal ones — the `finish_reason: "stop"` chunk
        // and then the `[DONE]` sentinel, which carries no reason because
        // `[DONE]` is not a choice. The mid-run `{"content":""}` line is still
        // dropped as not-news. That shape is unchanged by this branch; only the
        // fields each chunk now carries are new.
        #expect(chunks.map(\.delta) == ["p", "ong", "", ""])
        #expect(chunks.map(\.isComplete) == [false, false, true, true])
        #expect(chunks[2].finishReason == .stop)
        #expect(chunks[3].finishReason == nil, "[DONE] states the end, never a reason")
        #expect(chunks.allSatisfy { $0.toolCallDeltas == nil }, "a text stream calls no tool")
        #expect(chunks[0].role == .assistant, "the capture declares the role on its first chunk")
    }
}
