import Foundation

// MARK: - StreamEvent

/// A typed event emitted by a CLI-subprocess provider stream parser.
///
/// Concrete parsers (Claude Code JSON, Cursor text, Aider mixed) normalize
/// their heterogeneous stdout formats into this single event vocabulary so
/// downstream consumers (dispatch bridge, UI, telemetry) do not need
/// per-provider decoding paths.
public struct StreamEvent: Sendable, Equatable {

    /// The typed kind of event.
    public enum Kind: Sendable, Equatable {
        /// Session-level signal from the CLI (init, config, tool list).
        case system(subtype: String, sessionID: String?)
        /// A chunk of assistant-produced natural-language text.
        case assistantText(String)
        /// A tool invocation initiated by the model.
        case toolUse(id: String?, name: String, input: String)
        /// A tool's response returned back into the loop.
        case toolResult(callID: String?, output: String, isError: Bool)
        /// A user-authored message echoed back through the stream.
        case userText(String)
        /// A file-system edit reported by the provider.
        case fileEdit(path: String, kind: FileEditKind)
        /// Token / cost accounting reported at any point in the stream.
        case usage(inputTokens: Int, outputTokens: Int, costUSD: Double?)
        /// Terminal result event — the CLI has finished a turn or run.
        case result(isError: Bool, summary: String?, costUSD: Double?)
        /// Unrecognized-but-preserved output (opaque pass-through).
        case raw(String)
        /// Parser encountered malformed input on this line — surfaced
        /// rather than swallowed so callers can decide policy.
        case parseError(reason: String)
    }

    /// The direction of a file-edit event reported by a provider.
    public enum FileEditKind: String, Sendable, Equatable, Codable {
        case created
        case modified
        case deleted
    }

    /// The typed event.
    public let kind: Kind
    /// Verbatim source line(s) that produced this event, for tracing.
    public let raw: String

    public init(kind: Kind, raw: String) {
        self.kind = kind
        self.raw = raw
    }
}

// MARK: - StreamParser Protocol

/// A stateful parser that converts a provider's stdout stream into
/// typed ``StreamEvent`` values.
///
/// Implementations consume arbitrary UTF-8 chunks (which may or may not
/// be line-aligned) and emit zero or more events per call. ``flush()``
/// must be invoked once the underlying process closes stdout so any
/// buffered partial line or in-progress block is either emitted or
/// reported as a ``StreamEvent.Kind/parseError``.
///
/// Parsers are single-consumer / single-thread: they do not synchronize
/// their own state. Callers hold one parser per subprocess.
public protocol StreamParsing: Sendable {
    /// Feed a chunk of stdout. Chunks may span multiple lines and may
    /// end mid-line — implementations buffer the trailing partial.
    mutating func consume(_ chunk: String) -> [StreamEvent]

    /// Signal end-of-stream. Any buffered partial content is drained.
    mutating func flush() -> [StreamEvent]
}

// MARK: - LineBuffer

/// Small helper used by all three parsers to accumulate stdout chunks
/// into complete lines before dispatching to per-line handlers.
///
/// Kept internal on purpose: it is an implementation primitive, not
/// part of the parser vocabulary.
struct LineBuffer {
    private var pending: String = ""

    /// Append `chunk` and return every newline-terminated line that
    /// became complete. The trailing partial (if any) stays buffered.
    mutating func append(_ chunk: String) -> [String] {
        pending.append(chunk)
        var out: [String] = []
        while let newlineIndex = pending.firstIndex(of: "\n") {
            let line = pending[..<newlineIndex]
            out.append(String(line))
            pending = String(pending[pending.index(after: newlineIndex)...])
        }
        return out
    }

    /// Return whatever partial line remains and clear internal state.
    mutating func drain() -> String? {
        guard !pending.isEmpty else { return nil }
        let leftover = pending
        pending = ""
        return leftover
    }
}
