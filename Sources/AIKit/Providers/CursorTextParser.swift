import Foundation

// MARK: - CursorTextParser

/// Parses the plain-text stream produced by the Cursor Agent CLI.
///
/// Unlike Claude Code's structured stream-json format, Cursor emits
/// human-readable prose interleaved with a handful of bracketed
/// meta-markers. The parser recognizes the following patterns; every
/// other line is preserved as ``StreamEvent/Kind/assistantText(_:)``.
///
/// | Line prefix               | Emitted event                                     |
/// |---------------------------|---------------------------------------------------|
/// | `[system] <text>`         | ``StreamEvent/Kind/system(subtype:sessionID:)``   |
/// | `[tool] <name> <input>`   | ``StreamEvent/Kind/toolUse(id:name:input:)``      |
/// | `[tool-result] <text>`    | ``StreamEvent/Kind/toolResult(callID:output:isError:)`` |
/// | `[edit] <path>`           | ``StreamEvent/Kind/fileEdit(path:kind:)`` (modified) |
/// | `[create] <path>`         | ``StreamEvent/Kind/fileEdit(path:kind:)`` (created)  |
/// | `[delete] <path>`         | ``StreamEvent/Kind/fileEdit(path:kind:)`` (deleted)  |
/// | `[usage] in=N out=M cost=X` | ``StreamEvent/Kind/usage(inputTokens:outputTokens:costUSD:)`` |
/// | `[done] <summary>`        | ``StreamEvent/Kind/result(isError:summary:costUSD:)`` (success) |
/// | `[error] <summary>`       | ``StreamEvent/Kind/result(isError:summary:costUSD:)`` (failure) |
///
/// Empty lines are dropped. The set of markers is deliberately small —
/// the parser's job is to normalize, not to enrich.
public struct CursorTextParser: StreamParsing {

    private var buffer = LineBuffer()

    public init() {}

    // MARK: - StreamParsing

    public mutating func consume(_ chunk: String) -> [StreamEvent] {
        buffer.append(chunk).compactMap(Self.parseLine)
    }

    public mutating func flush() -> [StreamEvent] {
        guard let leftover = buffer.drain(),
            let event = Self.parseLine(leftover)
        else { return [] }
        return [event]
    }

    // MARK: - Line-level parsing

    /// Marker prefix → typed handler.
    ///
    /// Handlers receive the payload (everything after the marker) and
    /// the verbatim source line. Order matters only for readability —
    /// prefixes are matched exhaustively, not by insertion order.
    private static func parseLine(_ line: String) -> StreamEvent? {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty else { return nil }

        if let (marker, payload) = splitMarker(trimmed) {
            return handleMarker(marker: marker, payload: payload, raw: line)
        }

        return StreamEvent(kind: .assistantText(trimmed), raw: line)
    }

    private static func splitMarker(_ line: String) -> (String, String)? {
        guard line.hasPrefix("["), let closing = line.firstIndex(of: "]") else {
            return nil
        }
        let marker = String(line[line.index(after: line.startIndex)..<closing])
        let payload = String(line[line.index(after: closing)...])
            .trimmingCharacters(in: .whitespaces)
        return (marker.lowercased(), payload)
    }

    private static func handleMarker(marker: String, payload: String, raw: String) -> StreamEvent {
        switch marker {
        case "system":
            return StreamEvent(kind: .system(subtype: payload, sessionID: nil), raw: raw)
        case "tool":
            let (name, input) = splitFirstToken(payload)
            return StreamEvent(kind: .toolUse(id: nil, name: name, input: input), raw: raw)
        case "tool-result":
            return StreamEvent(kind: .toolResult(callID: nil, output: payload, isError: false), raw: raw)
        case "edit":
            return StreamEvent(kind: .fileEdit(path: payload, kind: .modified), raw: raw)
        case "create":
            return StreamEvent(kind: .fileEdit(path: payload, kind: .created), raw: raw)
        case "delete":
            return StreamEvent(kind: .fileEdit(path: payload, kind: .deleted), raw: raw)
        case "usage":
            return parseUsagePayload(payload, raw: raw)
        case "done":
            return StreamEvent(
                kind: .result(isError: false, summary: payload.isEmpty ? nil : payload, costUSD: nil),
                raw: raw
            )
        case "error":
            return StreamEvent(
                kind: .result(isError: true, summary: payload.isEmpty ? nil : payload, costUSD: nil),
                raw: raw
            )
        default:
            // Preserve unknown markers as raw so telemetry can spot
            // provider-version drift without silent loss.
            return StreamEvent(kind: .raw(raw), raw: raw)
        }
    }

    private static func splitFirstToken(_ s: String) -> (String, String) {
        guard let space = s.firstIndex(where: { $0.isWhitespace }) else {
            return (s, "")
        }
        let head = String(s[..<space])
        let tail = String(s[s.index(after: space)...]).trimmingCharacters(in: .whitespaces)
        return (head, tail)
    }

    /// Parse `in=N out=M cost=X` (any order, whitespace-separated).
    private static func parseUsagePayload(_ payload: String, raw: String) -> StreamEvent {
        var input = 0
        var output = 0
        var cost: Double?
        for token in payload.split(whereSeparator: { $0.isWhitespace }) {
            let parts = token.split(separator: "=", maxSplits: 1)
            guard parts.count == 2 else { continue }
            let key = String(parts[0]).lowercased()
            let value = String(parts[1])
            switch key {
            case "in", "input", "input_tokens":
                if let v = Int(value) { input = v }
            case "out", "output", "output_tokens":
                if let v = Int(value) { output = v }
            case "cost", "cost_usd", "total_cost_usd":
                if let v = Double(value) { cost = v }
            default:
                continue
            }
        }
        return StreamEvent(
            kind: .usage(inputTokens: input, outputTokens: output, costUSD: cost),
            raw: raw
        )
    }
}
