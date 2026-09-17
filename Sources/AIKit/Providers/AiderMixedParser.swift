import Foundation

// MARK: - AiderMixedParser

/// Parses the mixed prose + SEARCH/REPLACE stream produced by the
/// [Aider](https://aider.chat) CLI.
///
/// Aider does not offer a JSON stream mode. Its stdout intersperses
/// three things:
///
/// 1. **Assistant prose** — free-form model output.
/// 2. **File edit blocks** — SEARCH/REPLACE hunks in a fenced form:
///
///        path/to/file.py
///        <<<<<<< SEARCH
///        old
///        =======
///        new
///        >>>>>>> REPLACE
///
///    The parser detects the `<<<<<<< SEARCH` opener, remembers the
///    `path/to/file.py` line immediately preceding it, and emits a
///    single ``StreamEvent/Kind/fileEdit(path:kind:)`` when the
///    matching `>>>>>>> REPLACE` closer arrives.
///
/// 3. **Apply / commit markers** — lines like `Applied edit to path`,
///    `Commit <sha> <msg>`. Recognized markers are surfaced as typed
///    events (file edit / result); everything else is prose.
///
/// A SEARCH/REPLACE block whose leading path is empty (`old → new`
/// with no preceding filename) emits a
/// ``StreamEvent/Kind/parseError(reason:)`` so silent no-ops are visible.
public struct AiderMixedParser: StreamParsing {

    // MARK: State machine

    private enum Mode: Equatable {
        case prose
        case insideSearch
        case insideReplace
    }

    private var buffer = LineBuffer()
    private var mode: Mode = .prose
    /// The last non-empty prose line — held as the candidate filename
    /// for a SEARCH/REPLACE opener that may follow on the next line.
    private var lastCandidatePath: String?
    /// The filename bound to the currently open SEARCH block.
    private var currentEditPath: String?
    /// Raw source of the current block, for the emitted event's `raw`.
    private var currentBlockRaw: [String] = []

    public init() {}

    // MARK: - StreamParsing

    public mutating func consume(_ chunk: String) -> [StreamEvent] {
        var events: [StreamEvent] = []
        for line in buffer.append(chunk) {
            events.append(contentsOf: handle(line))
        }
        return events
    }

    public mutating func flush() -> [StreamEvent] {
        var events: [StreamEvent] = []
        if let leftover = buffer.drain() {
            events.append(contentsOf: handle(leftover))
        }
        if mode != .prose {
            events.append(
                StreamEvent(
                    kind: .parseError(reason: "stream ended inside a SEARCH/REPLACE block"),
                    raw: currentBlockRaw.joined(separator: "\n")
                ))
            resetBlockState()
        }
        return events
    }

    // MARK: - Per-line handling

    private mutating func handle(_ line: String) -> [StreamEvent] {
        switch mode {
        case .prose: return handleProse(line)
        case .insideSearch: return handleSearch(line)
        case .insideReplace: return handleReplace(line)
        }
    }

    private mutating func handleProse(_ line: String) -> [StreamEvent] {
        let trimmed = line.trimmingCharacters(in: .whitespaces)

        if trimmed == "<<<<<<< SEARCH" {
            mode = .insideSearch
            currentEditPath = lastCandidatePath
            currentBlockRaw = [line]
            lastCandidatePath = nil
            return []
        }

        if trimmed.isEmpty {
            return []
        }

        if let event = recognizeMarker(trimmed, raw: line) {
            lastCandidatePath = nil
            return [event]
        }

        // Any non-marker, non-empty line is potentially the filename
        // that will precede a SEARCH opener on the next line — remember
        // it, but also emit it as prose so nothing is dropped if it
        // turned out to be plain text.
        lastCandidatePath = trimmed
        return [StreamEvent(kind: .assistantText(trimmed), raw: line)]
    }

    private mutating func handleSearch(_ line: String) -> [StreamEvent] {
        currentBlockRaw.append(line)
        if line.trimmingCharacters(in: .whitespaces) == "=======" {
            mode = .insideReplace
        }
        return []
    }

    private mutating func handleReplace(_ line: String) -> [StreamEvent] {
        currentBlockRaw.append(line)
        if line.trimmingCharacters(in: .whitespaces) == ">>>>>>> REPLACE" {
            let raw = currentBlockRaw.joined(separator: "\n")
            let event: StreamEvent
            if let path = currentEditPath, !path.isEmpty {
                event = StreamEvent(kind: .fileEdit(path: path, kind: .modified), raw: raw)
            } else {
                event = StreamEvent(
                    kind: .parseError(reason: "SEARCH/REPLACE block missing preceding filename"),
                    raw: raw
                )
            }
            resetBlockState()
            mode = .prose
            return [event]
        }
        return []
    }

    private mutating func resetBlockState() {
        currentEditPath = nil
        currentBlockRaw = []
    }

    // MARK: - Marker recognition

    /// Aider commit / apply markers we surface as typed events.
    private static let appliedEditPrefix = "Applied edit to "
    private static let commitPrefix = "Commit "

    private func recognizeMarker(_ trimmed: String, raw: String) -> StreamEvent? {
        if trimmed.hasPrefix(Self.appliedEditPrefix) {
            let path = String(trimmed.dropFirst(Self.appliedEditPrefix.count))
                .trimmingCharacters(in: .whitespaces)
            guard !path.isEmpty else { return nil }
            return StreamEvent(kind: .fileEdit(path: path, kind: .modified), raw: raw)
        }
        if trimmed.hasPrefix(Self.commitPrefix) {
            return StreamEvent(
                kind: .result(isError: false, summary: trimmed, costUSD: nil),
                raw: raw
            )
        }
        return nil
    }
}
