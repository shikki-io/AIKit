import Foundation
@testable import AIKit
import Testing

@Suite("ClaudeCodeStreamJSONParser")
struct ClaudeCodeStreamJSONParserTests {

    // MARK: - system events

    @Test("recognizes system init and captures subtype + session id")
    func systemInit() {
        var parser = ClaudeCodeStreamJSONParser()
        let line = #"{"type":"system","subtype":"init","session_id":"sess-42","model":"claude-x"}"# + "\n"
        let events = parser.consume(line)
        #expect(events.count == 1)
        guard case .system(let subtype, let sessionID) = events[0].kind else {
            Issue.record("Expected .system, got \(events[0].kind)")
            return
        }
        #expect(subtype == "init")
        #expect(sessionID == "sess-42")
    }

    // MARK: - assistant events

    @Test("assistant text block emits assistantText")
    func assistantText() {
        var parser = ClaudeCodeStreamJSONParser()
        let line = #"{"type":"assistant","message":{"content":[{"type":"text","text":"hi there"}]}}"# + "\n"
        let events = parser.consume(line)
        // Expect exactly one assistantText event (no usage block in this input).
        #expect(events.count == 1)
        guard case .assistantText(let text) = events[0].kind else {
            Issue.record("Expected .assistantText, got \(events[0].kind)")
            return
        }
        #expect(text == "hi there")
    }

    @Test("assistant tool_use block emits toolUse with normalized input")
    func assistantToolUse() {
        var parser = ClaudeCodeStreamJSONParser()
        let line =
            #"{"type":"assistant","message":{"content":[{"type":"tool_use","id":"toolu_1","name":"Read","input":{"file_path":"a.swift"}}]}}"#
            + "\n"
        let events = parser.consume(line)
        #expect(events.count == 1)
        guard case .toolUse(let id, let name, let input) = events[0].kind else {
            Issue.record("Expected .toolUse, got \(events[0].kind)")
            return
        }
        #expect(id == "toolu_1")
        #expect(name == "Read")
        // Encoded with .sortedKeys so we can assert exact equality.
        #expect(input == #"{"file_path":"a.swift"}"#)
    }

    @Test("assistant message with usage emits both assistantText and usage")
    func assistantWithUsage() {
        var parser = ClaudeCodeStreamJSONParser()
        let line =
            #"{"type":"assistant","message":{"content":[{"type":"text","text":"ok"}],"usage":{"input_tokens":11,"output_tokens":22}}}"#
            + "\n"
        let events = parser.consume(line)
        #expect(events.count == 2)
        guard case .assistantText = events[0].kind,
            case .usage(let inTok, let outTok, let cost) = events[1].kind
        else {
            Issue.record("Unexpected kinds: \(events.map(\.kind))")
            return
        }
        #expect(inTok == 11)
        #expect(outTok == 22)
        #expect(cost == nil)
    }

    // MARK: - user / tool_result events

    @Test("user tool_result emits toolResult and preserves isError flag")
    func userToolResult() {
        var parser = ClaudeCodeStreamJSONParser()
        let line =
            #"{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"file body","is_error":true}]}}"#
            + "\n"
        let events = parser.consume(line)
        #expect(events.count == 1)
        guard case .toolResult(let callID, let output, let isError) = events[0].kind else {
            Issue.record("Expected .toolResult, got \(events[0].kind)")
            return
        }
        #expect(callID == "toolu_1")
        #expect(output == "file body")
        #expect(isError == true)
    }

    // MARK: - result events

    @Test("result event surfaces summary, cost, and preceding usage")
    func resultEvent() {
        var parser = ClaudeCodeStreamJSONParser()
        let line =
            #"{"type":"result","subtype":"success","is_error":false,"result":"done","total_cost_usd":0.0123,"usage":{"input_tokens":100,"output_tokens":50}}"#
            + "\n"
        let events = parser.consume(line)
        #expect(events.count == 2)
        guard case .usage(_, _, let usageCost) = events[0].kind,
            case .result(let isError, let summary, let resultCost) = events[1].kind
        else {
            Issue.record("Unexpected kinds: \(events.map(\.kind))")
            return
        }
        #expect(usageCost == 0.0123)
        #expect(isError == false)
        #expect(summary == "done")
        #expect(resultCost == 0.0123)
    }

    @Test("result event accepts legacy cost_usd key")
    func resultLegacyCostKey() {
        var parser = ClaudeCodeStreamJSONParser()
        let line = #"{"type":"result","subtype":"success","is_error":false,"result":"done","cost_usd":0.5}"# + "\n"
        let events = parser.consume(line)
        #expect(events.count == 1)
        guard case .result(_, _, let cost) = events[0].kind else {
            Issue.record("Expected .result, got \(events[0].kind)")
            return
        }
        #expect(cost == 0.5)
    }

    // MARK: - Buffering + framing

    @Test("chunk boundaries in mid-line are buffered until newline arrives")
    func chunkedInput() {
        var parser = ClaudeCodeStreamJSONParser()
        let a = parser.consume(#"{"type":"system","subtype":"in"#)
        let b = parser.consume(#"it","session_id":"s1"}"#)
        let c = parser.consume("\n")
        #expect(a.isEmpty)
        #expect(b.isEmpty)
        #expect(c.count == 1)
        guard case .system(let subtype, let sessionID) = c[0].kind else {
            Issue.record("Expected .system")
            return
        }
        #expect(subtype == "init")
        #expect(sessionID == "s1")
    }

    @Test("flush drains a trailing line missing its newline")
    func flushDrain() {
        var parser = ClaudeCodeStreamJSONParser()
        let mid = parser.consume(#"{"type":"system","subtype":"end"}"#)
        #expect(mid.isEmpty)
        let drained = parser.flush()
        #expect(drained.count == 1)
        guard case .system(let subtype, _) = drained[0].kind else {
            Issue.record("Expected .system on flush")
            return
        }
        #expect(subtype == "end")
    }

    // MARK: - Error surfacing

    @Test("malformed JSON becomes a parseError event, not silent loss")
    func malformedLine() {
        var parser = ClaudeCodeStreamJSONParser()
        let events = parser.consume("this is not json\n")
        #expect(events.count == 1)
        guard case .parseError = events[0].kind else {
            Issue.record("Expected .parseError, got \(events[0].kind)")
            return
        }
    }

    @Test("blank lines are skipped, not emitted")
    func blankLinesSkipped() {
        var parser = ClaudeCodeStreamJSONParser()
        let events = parser.consume("\n   \n\t\n")
        #expect(events.isEmpty)
    }
}
