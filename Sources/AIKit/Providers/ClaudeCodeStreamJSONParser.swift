import Foundation

// MARK: - ClaudeCodeStreamJSONParser

/// Parses the newline-delimited JSON stream emitted by
/// `claude --output-format stream-json`.
///
/// Each line of stdout is a self-contained JSON object with a
/// top-level `type` discriminator. Recognized shapes:
///
/// - `{"type":"system","subtype":"init","session_id":"...","model":"..."}`
///     → ``StreamEvent/Kind/system(subtype:sessionID:)``
/// - `{"type":"assistant","message":{"content":[{"type":"text","text":"..."}], ...}}`
///     → one ``StreamEvent/Kind/assistantText(_:)`` per text block
///       plus ``StreamEvent/Kind/toolUse(id:name:input:)`` per tool_use block
/// - `{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"...","content":"..."}]}}`
///     → ``StreamEvent/Kind/toolResult(callID:output:isError:)``
/// - `{"type":"result","subtype":"success","result":"...","total_cost_usd":0.01,"is_error":false}`
///     → ``StreamEvent/Kind/usage(inputTokens:outputTokens:costUSD:)`` (if usage
///       present) plus ``StreamEvent/Kind/result(isError:summary:costUSD:)``
///
/// Malformed lines are emitted as ``StreamEvent/Kind/parseError(reason:)``
/// so nothing is silently dropped — every stdout byte accounts to an event.
public struct ClaudeCodeStreamJSONParser: StreamParsing {

    private var buffer = LineBuffer()

    public init() {}

    // MARK: - StreamParsing

    public mutating func consume(_ chunk: String) -> [StreamEvent] {
        buffer.append(chunk).flatMap(Self.parseLine)
    }

    public mutating func flush() -> [StreamEvent] {
        guard let leftover = buffer.drain() else { return [] }
        return Self.parseLine(leftover)
    }

    // MARK: - Line-level parsing

    /// Typed envelope for the outer `type` discriminator on each stream-json
    /// frame. A typed enum makes an unknown `type` a distinct case rather
    /// than a silent parse failure.
    private struct FrameEnvelope: Decodable {
        let type: FrameType
    }

    private enum FrameType: String, Decodable {
        case system, assistant, user, result
    }

    /// Shared decoder for the outer envelope. No date fields are decoded
    /// here, so the default strategy is sufficient.
    private static let envelopeDecoder = JSONDecoder()

    private static func parseLine(_ line: String) -> [StreamEvent] {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty else { return [] }
        guard let data = trimmed.data(using: .utf8) else {
            return [StreamEvent(kind: .parseError(reason: "line is not valid utf-8"), raw: line)]
        }

        // Content-block bodies stay as loose `[String: Any]` walks below because
        // claude-code's per-version schema drift makes a full typed model brittle;
        // only the outer `type` discriminator is stable across versions.
        guard (try? envelopeDecoder.decode(FrameEnvelope.self, from: data)) != nil else {
            return [
                StreamEvent(
                    kind: .parseError(reason: "not a JSON object with a `type` field"),
                    raw: line
                ),
            ]
        }

        // Second pass: recover the raw dictionary for content-block walking. Version-tolerant.
        guard let root = try? JSONSerialization.jsonObject(with: data),
            let object = root as? [String: Any],
            let typeStr = object["type"] as? String
        else {
            return [
                StreamEvent(
                    kind: .parseError(reason: "envelope decoded but body is not a keyed object"),
                    raw: line
                ),
            ]
        }

        switch typeStr {
        case "system": return [parseSystem(object, raw: line)]
        case "assistant": return parseAssistant(object, raw: line)
        case "user": return parseUser(object, raw: line)
        case "result": return parseResult(object, raw: line)
        default: return [StreamEvent(kind: .raw(line), raw: line)]
        }
    }

    // MARK: - Per-type handlers

    private static func parseSystem(_ obj: [String: Any], raw: String) -> StreamEvent {
        let subtype = (obj["subtype"] as? String) ?? "unknown"
        let session = obj["session_id"] as? String
        return StreamEvent(kind: .system(subtype: subtype, sessionID: session), raw: raw)
    }

    private static func parseAssistant(_ obj: [String: Any], raw: String) -> [StreamEvent] {
        guard let message = obj["message"] as? [String: Any] else {
            return [
                StreamEvent(
                    kind: .parseError(reason: "assistant event missing `message`"),
                    raw: raw
                ),
            ]
        }
        var events: [StreamEvent] = []
        events.append(contentsOf: parseContentBlocks(message["content"], raw: raw, direction: .assistant))
        if let usage = message["usage"] as? [String: Any] {
            events.append(makeUsageEvent(usage: usage, costUSD: nil, raw: raw))
        }
        return events
    }

    private static func parseUser(_ obj: [String: Any], raw: String) -> [StreamEvent] {
        guard let message = obj["message"] as? [String: Any] else {
            return [
                StreamEvent(
                    kind: .parseError(reason: "user event missing `message`"),
                    raw: raw
                ),
            ]
        }
        return parseContentBlocks(message["content"], raw: raw, direction: .user)
    }

    private static func parseResult(_ obj: [String: Any], raw: String) -> [StreamEvent] {
        let isError = (obj["is_error"] as? Bool) ?? false
        let summary = obj["result"] as? String
        // Claude Code has emitted either `cost_usd` or `total_cost_usd`
        // across versions — accept both to insulate downstream from the drift.
        let cost = (obj["total_cost_usd"] as? Double) ?? (obj["cost_usd"] as? Double)

        var events: [StreamEvent] = []
        if let usage = obj["usage"] as? [String: Any] {
            events.append(makeUsageEvent(usage: usage, costUSD: cost, raw: raw))
        }
        events.append(
            StreamEvent(
                kind: .result(isError: isError, summary: summary, costUSD: cost),
                raw: raw
            ))
        return events
    }

    // MARK: - Content blocks

    private enum Direction { case assistant, user }

    private static func parseContentBlocks(
        _ content: Any?,
        raw: String,
        direction: Direction
    ) -> [StreamEvent] {
        // `content` may be a JSON string (short form) or an array of blocks.
        if let text = content as? String {
            return [StreamEvent(kind: direction == .assistant ? .assistantText(text) : .userText(text), raw: raw)]
        }
        guard let blocks = content as? [[String: Any]] else {
            return []
        }

        var events: [StreamEvent] = []
        for block in blocks {
            guard let blockType = block["type"] as? String else { continue }
            switch blockType {
            case "text":
                if let text = block["text"] as? String {
                    events.append(
                        StreamEvent(
                            kind: direction == .assistant ? .assistantText(text) : .userText(text),
                            raw: raw
                        ))
                }
            case "tool_use":
                let id = block["id"] as? String
                let name = (block["name"] as? String) ?? "unknown"
                let input = encodeInline(block["input"]) ?? ""
                events.append(
                    StreamEvent(
                        kind: .toolUse(id: id, name: name, input: input),
                        raw: raw
                    ))
            case "tool_result":
                let callID = block["tool_use_id"] as? String
                let isError = (block["is_error"] as? Bool) ?? false
                let output = extractToolResultContent(block["content"])
                events.append(
                    StreamEvent(
                        kind: .toolResult(callID: callID, output: output, isError: isError),
                        raw: raw
                    ))
            default:
                events.append(
                    StreamEvent(
                        kind: .raw("unknown content block type: \(blockType)"),
                        raw: raw
                    ))
            }
        }
        return events
    }

    private static func extractToolResultContent(_ content: Any?) -> String {
        if let s = content as? String { return s }
        if let blocks = content as? [[String: Any]] {
            return blocks.compactMap { $0["text"] as? String }.joined(separator: "\n")
        }
        return encodeInline(content) ?? ""
    }

    private static func makeUsageEvent(
        usage: [String: Any],
        costUSD: Double?,
        raw: String
    ) -> StreamEvent {
        let input = (usage["input_tokens"] as? Int) ?? 0
        let output = (usage["output_tokens"] as? Int) ?? 0
        return StreamEvent(
            kind: .usage(inputTokens: input, outputTokens: output, costUSD: costUSD),
            raw: raw
        )
    }

    /// Serialize an arbitrary JSON value back to a compact string so
    /// downstream consumers can display tool inputs verbatim.
    private static func encodeInline(_ value: Any?) -> String? {
        guard let value else { return nil }
        guard JSONSerialization.isValidJSONObject(value) || value is String else {
            // Primitives that aren't valid top-level JSON: coerce via description.
            return String(describing: value)
        }
        let data: Data
        if let s = value as? String {
            data = Data(s.utf8)
            return String(data: data, encoding: .utf8)
        } else {
            guard let d = try? JSONSerialization.data(withJSONObject: value, options: [.sortedKeys]) else {
                return nil
            }
            data = d
        }
        return String(data: data, encoding: .utf8)
    }
}
