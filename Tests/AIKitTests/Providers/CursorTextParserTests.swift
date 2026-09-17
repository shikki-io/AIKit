import Foundation
@testable import AIKit
import Testing

@Suite("CursorTextParser")
struct CursorTextParserTests {

    @Test("plain text lines become assistantText events")
    func plainProse() {
        var parser = CursorTextParser()
        let events = parser.consume("Hello world\nSecond line\n")
        #expect(events.count == 2)
        guard case .assistantText(let a) = events[0].kind,
            case .assistantText(let b) = events[1].kind
        else {
            Issue.record("Expected two .assistantText, got \(events.map(\.kind))")
            return
        }
        #expect(a == "Hello world")
        #expect(b == "Second line")
    }

    @Test("[system] marker emits a system event with payload as subtype")
    func systemMarker() {
        var parser = CursorTextParser()
        let events = parser.consume("[system] session started\n")
        #expect(events.count == 1)
        guard case .system(let subtype, let sessionID) = events[0].kind else {
            Issue.record("Expected .system, got \(events[0].kind)")
            return
        }
        #expect(subtype == "session started")
        #expect(sessionID == nil)
    }

    @Test("[tool] marker splits name from input")
    func toolMarker() {
        var parser = CursorTextParser()
        let events = parser.consume(#"[tool] Read {"path":"a.txt"}"# + "\n")
        #expect(events.count == 1)
        guard case .toolUse(let id, let name, let input) = events[0].kind else {
            Issue.record("Expected .toolUse, got \(events[0].kind)")
            return
        }
        #expect(id == nil)
        #expect(name == "Read")
        #expect(input == #"{"path":"a.txt"}"#)
    }

    @Test("[tool-result] marker emits toolResult without an error flag")
    func toolResultMarker() {
        var parser = CursorTextParser()
        let events = parser.consume("[tool-result] 42 lines read\n")
        #expect(events.count == 1)
        guard case .toolResult(let callID, let output, let isError) = events[0].kind else {
            Issue.record("Expected .toolResult, got \(events[0].kind)")
            return
        }
        #expect(callID == nil)
        #expect(output == "42 lines read")
        #expect(isError == false)
    }

    @Test("edit / create / delete markers map to the right FileEditKind")
    func fileEditMarkers() {
        var parser = CursorTextParser()
        let events = parser.consume(
            """
            [edit] src/a.swift
            [create] src/b.swift
            [delete] src/c.swift

            """)
        #expect(events.count == 3)
        let expected: [(String, StreamEvent.FileEditKind)] = [
            ("src/a.swift", .modified),
            ("src/b.swift", .created),
            ("src/c.swift", .deleted),
        ]
        for (event, (path, kind)) in zip(events, expected) {
            guard case .fileEdit(let p, let k) = event.kind else {
                Issue.record("Expected .fileEdit, got \(event.kind)")
                continue
            }
            #expect(p == path)
            #expect(k == kind)
        }
    }

    @Test("[usage] parses in/out/cost tokens regardless of ordering")
    func usageMarker() {
        var parser = CursorTextParser()
        let events = parser.consume("[usage] cost=0.42 out=200 in=100\n")
        #expect(events.count == 1)
        guard case .usage(let inTok, let outTok, let cost) = events[0].kind else {
            Issue.record("Expected .usage, got \(events[0].kind)")
            return
        }
        #expect(inTok == 100)
        #expect(outTok == 200)
        #expect(cost == 0.42)
    }

    @Test("[done] emits a success result with the trailing summary")
    func doneMarker() {
        var parser = CursorTextParser()
        let events = parser.consume("[done] all good\n")
        #expect(events.count == 1)
        guard case .result(let isError, let summary, _) = events[0].kind else {
            Issue.record("Expected .result, got \(events[0].kind)")
            return
        }
        #expect(isError == false)
        #expect(summary == "all good")
    }

    @Test("[error] emits a failed result")
    func errorMarker() {
        var parser = CursorTextParser()
        let events = parser.consume("[error] boom\n")
        #expect(events.count == 1)
        guard case .result(let isError, let summary, _) = events[0].kind else {
            Issue.record("Expected .result, got \(events[0].kind)")
            return
        }
        #expect(isError == true)
        #expect(summary == "boom")
    }

    @Test("unknown marker is preserved as raw, never dropped")
    func unknownMarker() {
        var parser = CursorTextParser()
        let events = parser.consume("[banana] mystery\n")
        #expect(events.count == 1)
        guard case .raw = events[0].kind else {
            Issue.record("Expected .raw, got \(events[0].kind)")
            return
        }
    }

    @Test("chunk boundaries are buffered until newline arrives")
    func chunkedInput() {
        var parser = CursorTextParser()
        let a = parser.consume("hel")
        let b = parser.consume("lo")
        let c = parser.consume(" world\n")
        #expect(a.isEmpty)
        #expect(b.isEmpty)
        #expect(c.count == 1)
        guard case .assistantText(let text) = c[0].kind else {
            Issue.record("Expected .assistantText")
            return
        }
        #expect(text == "hello world")
    }

    @Test("flush drains any trailing line missing its newline")
    func flushDrain() {
        var parser = CursorTextParser()
        _ = parser.consume("trailing text")
        let events = parser.flush()
        #expect(events.count == 1)
        guard case .assistantText(let t) = events[0].kind else {
            Issue.record("Expected .assistantText on flush")
            return
        }
        #expect(t == "trailing text")
    }
}
