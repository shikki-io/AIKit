import Foundation
@testable import AIKit
import Testing

@Suite("AiderMixedParser")
struct AiderMixedParserTests {

    @Test("prose lines become assistantText events")
    func plainProse() {
        var parser = AiderMixedParser()
        let events = parser.consume("Let me look at the code.\nHere is my plan.\n")
        #expect(events.count == 2)
        for event in events {
            guard case .assistantText = event.kind else {
                Issue.record("Expected .assistantText, got \(event.kind)")
                return
            }
        }
    }

    @Test("blank lines are skipped between prose")
    func blankLinesSkipped() {
        var parser = AiderMixedParser()
        let events = parser.consume("first\n\n  \nsecond\n")
        #expect(events.count == 2)
    }

    @Test("SEARCH/REPLACE block emits a single fileEdit with the preceding filename")
    func searchReplaceEmitsFileEdit() {
        var parser = AiderMixedParser()
        let stream = """
            path/to/thing.py
            <<<<<<< SEARCH
            old_line
            =======
            new_line
            >>>>>>> REPLACE

            """
        let events = parser.consume(stream)
        // The filename line is also emitted as prose (it hadn't yet
        // been claimed by a block when parsed). The fileEdit event
        // fires when the REPLACE closer arrives.
        let editEvents = events.filter {
            if case .fileEdit = $0.kind { return true } else { return false }
        }
        #expect(editEvents.count == 1)
        guard case .fileEdit(let path, let kind) = editEvents[0].kind else {
            Issue.record("Expected .fileEdit")
            return
        }
        #expect(path == "path/to/thing.py")
        #expect(kind == .modified)
    }

    @Test("SEARCH/REPLACE with no preceding filename emits a parseError")
    func searchReplaceMissingFilename() {
        var parser = AiderMixedParser()
        let stream = """
            <<<<<<< SEARCH
            old
            =======
            new
            >>>>>>> REPLACE

            """
        let events = parser.consume(stream)
        let errors = events.filter {
            if case .parseError = $0.kind { return true } else { return false }
        }
        #expect(errors.count == 1)
    }

    @Test("`Applied edit to <path>` marker emits a fileEdit event")
    func appliedEditMarker() {
        var parser = AiderMixedParser()
        let events = parser.consume("Applied edit to path/to/file.py\n")
        #expect(events.count == 1)
        guard case .fileEdit(let path, let kind) = events[0].kind else {
            Issue.record("Expected .fileEdit, got \(events[0].kind)")
            return
        }
        #expect(path == "path/to/file.py")
        #expect(kind == .modified)
    }

    @Test("`Commit ...` marker emits a result event")
    func commitMarker() {
        var parser = AiderMixedParser()
        let events = parser.consume("Commit abc123 apply refactor\n")
        #expect(events.count == 1)
        guard case .result(let isError, let summary, let cost) = events[0].kind else {
            Issue.record("Expected .result, got \(events[0].kind)")
            return
        }
        #expect(isError == false)
        #expect(summary == "Commit abc123 apply refactor")
        #expect(cost == nil)
    }

    @Test("chunk boundaries mid-block still emit the correct fileEdit")
    func chunkedSearchReplace() {
        var parser = AiderMixedParser()
        var events: [StreamEvent] = []
        events += parser.consume("file.py\n<<<<<<< SEARCH\n")
        events += parser.consume("old\n=======\n")
        events += parser.consume("new\n>>>>>>> REPLACE\n")
        let edits = events.filter {
            if case .fileEdit = $0.kind { return true } else { return false }
        }
        #expect(edits.count == 1)
        guard case .fileEdit(let path, _) = edits[0].kind else {
            Issue.record("Expected .fileEdit")
            return
        }
        #expect(path == "file.py")
    }

    @Test("stream ending mid-block surfaces a parseError on flush")
    func flushMidBlock() {
        var parser = AiderMixedParser()
        _ = parser.consume("file.py\n<<<<<<< SEARCH\nold\n")
        let events = parser.flush()
        let errors = events.filter {
            if case .parseError = $0.kind { return true } else { return false }
        }
        #expect(errors.count == 1)
    }

    @Test("flush on a clean stream produces no events")
    func flushCleanNoEvents() {
        var parser = AiderMixedParser()
        _ = parser.consume("prose line\n")
        let events = parser.flush()
        #expect(events.isEmpty)
    }
}
