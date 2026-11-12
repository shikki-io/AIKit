import Foundation
import Testing

@testable import AIKit

// MARK: - The no-localhost-fallback ratchet
//
// T-AIL-44, the kagami scope `AIKitNoLocalhostFallback`. BR-AIL-18: "There is
// no `?? "http"`, no `?? "127.0.0.1"`, no localhost fallback anywhere in
// Sources."
//
// WHY A RATCHET AND NOT A CODE REVIEW. The pattern this forbids is not a typo,
// it is a habit, and it reads as defensive programming on the way in. Six of
// the panel's ten bad addresses were EXECUTED against the operator's own
// machine because `URLComponents(string:)?.scheme ?? "http"` and
// `?.host ?? "127.0.0.1"` answered "this string is not an address" with "then
// use the local one" — and one of those requests carried a bearer key
// (AIKit#18 r4500281951, r4500300887). Deleting the four occurrences fixes
// today; this file is what makes the fifth one fail a gate.
//
// WHAT IS SCANNED. `Sources/`, every target, with COMMENTS REMOVED first. The
// files that deleted these fallbacks document what they deleted — quoting
// `?? "http://localhost:8080"` in a doc comment is how the next reader learns
// why the expression ends where it does — and a ratchet that counted its own
// documentation would force that explanation out of the repo. Comment
// stripping is string-literal aware on purpose: `"http://127.0.0.1"` contains
// `//`, so a naive "cut at the first //" would truncate the very line it has
// to inspect and the ratchet would pass by missing it.
//
// WHAT IS NOT SCANNED, and the judgment call in it. A `??` whose right-hand
// side is a localhost literal is a fallback: it answers an absent or
// unparseable value with the local machine. A DECLARED address is not —
// `OpenAICompatibleEngine.lmStudio()` and `MCPEngine.koharu(port:)` are
// factories for a local server, named after it, and their default argument is
// that server's address rather than a rescue from a missing one. Those
// literals stay, and this comment is the record that they were looked at
// rather than overlooked. If a later wave wants them gone too, the rule to add
// is "no address literal in a default argument", which is a different rule
// from this one.

@Suite("No localhost fallback, and no route literal, anywhere in Sources")
struct NoLocalhostFallbackRatchetTests {

    // MARK: - The rules

    /// The string literals on the right-hand side of a `??` in `text`.
    ///
    /// A function rather than a stored `Regex`, because a `Regex` is not
    /// `Sendable` and a `static let` holding one is a data race under strict
    /// concurrency.
    ///
    /// It matches `?? "http"`, `?? "http://localhost:8080"`,
    /// `?? "http://127.0.0.1:1234"` and `?? "localhost"` alike, because what
    /// makes them one defect is the shape — an absent address answered with a
    /// literal one — not the spelling of the host.
    static func fallbackLiterals(in text: some StringProtocol) -> [String] {
        String(text).matches(of: #/\?\?\s*"(?<literal>[^"\n]*)"/#).map { String($0.output.literal) }
    }

    /// Whether a `??` literal names a host, which is what makes the fallback
    /// one of BR-AIL-18's.
    static func namesAHost(_ literal: String) -> Bool {
        let lowered = literal.lowercased()
        return forbiddenHosts.contains { lowered.contains($0) }
    }

    /// The host words that make such a fallback a localhost fallback.
    static let forbiddenHosts = ["http", "https", "localhost", "127.0.0.1", "0.0.0.0", "::1"]

    /// Route prefixes that belong to an endpoint enum and nowhere else
    /// (BR-AIL-09, BR-AIL-16).
    static let routeLiterals = ["\"/v1", "\"/api/v1"]

    // MARK: - T-AIL-44

    @Test("no ?? falls back to a host literal in Sources")
    func noLocalhostFallback() throws {
        var offenders: [String] = []

        for file in try Self.swiftFiles(under: Self.sourcesRoot) {
            let code = Self.codeOnly(try String(contentsOf: file, encoding: .utf8))
            for (offset, line) in code.split(separator: "\n", omittingEmptySubsequences: false).enumerated() {
                for literal in Self.fallbackLiterals(in: line) where Self.namesAHost(literal) {
                    offenders.append("\(Self.relative(file)):\(offset + 1) — ?? \"\(literal)\"")
                }
            }
        }

        #expect(
            offenders.isEmpty,
            """
            BR-AIL-18: an absent or unparseable address is an error naming the \
            value, never a request to the local machine. Offending fallbacks: \
            \(offenders.joined(separator: " | "))
            """
        )
    }

    @Test("no route literal is written outside Sources/AIKit/Engines/Endpoints")
    func noRouteLiteralOutsideEndpoints() throws {
        // The second half of T-AIL-44. A route belongs to its `EndPoint` enum:
        // the concatenation this forbids is the one that produced
        // `http://host:1234/v1/v1/models` in `ProviderHealthCheck` and
        // reported a healthy server as down.
        //
        // `Registry/Endpoints/` is allowed for the same reason
        // `Engines/Endpoints/` is: `HuggingFaceEndPoint` lives there and is an
        // endpoint enum, so the rule is "an Endpoints directory", which is
        // also how the W20 enforcer spells it.
        var offenders: [String] = []

        for file in try Self.swiftFiles(under: Self.sourcesRoot) {
            guard !file.path.contains("/Endpoints/") else { continue }
            let code = Self.codeOnly(try String(contentsOf: file, encoding: .utf8))
            let hits = Self.routeLiterals.filter { code.contains($0) }
            guard !hits.isEmpty else { continue }
            offenders.append("\(Self.relative(file)) — \(hits.joined(separator: ", "))")
        }

        #expect(offenders.isEmpty, "route literals outside an Endpoints directory: \(offenders)")
    }

    @Test("the forbidden shapes are still recognised, so a pass means a clean tree")
    func theRatchetStillCatchesWhatItIsFor() throws {
        // A ratchet that has stopped matching passes forever. These are the
        // four expressions W21 deleted, fed back through the same two rules:
        // if the detector has rotted, this test goes red instead of the gate
        // going quietly green (BR-EC-02 — the enforcer is verified, not
        // assumed).
        let deleted = [
            #"self.baseURL = baseURL ?? env["LMSTUDIO_URL"] ?? "http://127.0.0.1:1234""#,
            #"ProcessInfo.processInfo.environment["MLX_LM_BASE_URL"] ?? "http://localhost:8080""#,
            #"let scheme = URLComponents(string: raw)?.scheme ?? "http""#,
            #"let host = URLComponents(string: raw)?.host ?? "127.0.0.1""#,
        ]

        for line in deleted {
            #expect(
                Self.fallbackLiterals(in: line).contains(where: Self.namesAHost),
                "the fallback rule no longer recognises: \(line)"
            )
        }

        // And the shapes it must NOT flag, so the rule stays usable: a
        // non-address fallback, and a declared address that is not a fallback.
        let allowed = [
            #"self.model = model ?? "local-model""#,
            #"public static func lmStudio(baseURL: String = "http://127.0.0.1:1234")"#,
        ]
        for line in allowed {
            #expect(
                !Self.fallbackLiterals(in: line).contains(where: Self.namesAHost),
                "the fallback rule flags something it should not: \(line)"
            )
        }

        #expect(Self.codeOnly(#"let a = x // "http://127.0.0.1" in a comment"#).contains("//") == false)
        #expect(Self.codeOnly(#"let a = y ?? "http://127.0.0.1""#).contains("127.0.0.1"))
    }

    @Test("the scan actually read the Sources tree")
    func theScanFoundFiles() throws {
        // The silent-fail this closes: a wrong root makes both rules pass on
        // an empty file list, and the scope reports green having inspected
        // nothing. The floor is well under the real count (76 at W21) so a
        // deleted file never turns this red for the wrong reason.
        let files = try Self.swiftFiles(under: Self.sourcesRoot)
        #expect(files.count >= 50, "only \(files.count) Swift files found under \(Self.sourcesRoot.path)")
        #expect(files.contains { $0.lastPathComponent == "AIServerAddress.swift" })
    }

    // MARK: - Scanning

    /// `Sources/`, found from this file rather than from the process's working
    /// directory, which a test runner does not promise.
    static var sourcesRoot: URL {
        URL(fileURLWithPath: #filePath)  // Tests/AIKitTests/Wire/<self>.swift
            .deletingLastPathComponent()  // Wire
            .deletingLastPathComponent()  // AIKitTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // <root>
            .appendingPathComponent("Sources")
    }

    static func relative(_ file: URL) -> String {
        let root = sourcesRoot.deletingLastPathComponent().path
        return file.path.hasPrefix(root)
            ? String(file.path.dropFirst(root.count + 1))
            : file.path
    }

    static func swiftFiles(under directory: URL) throws -> [URL] {
        guard
            let walker = FileManager.default.enumerator(
                at: directory,
                includingPropertiesForKeys: nil,
                options: [.skipsHiddenFiles]
            )
        else { return [] }
        return walker.compactMap { $0 as? URL }.filter { $0.pathExtension == "swift" }
    }

    /// `source` with every comment replaced by spaces and every newline kept,
    /// so a match's line number is still the file's line number.
    ///
    /// String-literal aware, which is the whole difficulty: `"http://host"`
    /// carries a `//`, and a stripper that cut there would hide exactly the
    /// literals this file exists to find. Handles `//`, `/* */` (nested, as
    /// Swift allows), `"…"` with escapes, and `"""…"""`.
    static func codeOnly(_ source: String) -> String {
        var scanner = CommentStripper(source: source)
        return scanner.run()
    }
}

// MARK: - CommentStripper

/// The little state machine `codeOnly` runs.
///
/// A type rather than one function because the five states each need the
/// cursor and the lookahead, and threading those through a single `while`
/// put it over the complexity the linter allows — which was fair: the
/// states are what a reader wants to see named.
private struct CommentStripper {
    enum State { case code, lineComment, blockComment, string, multilineString }

    let characters: [Character]
    var index = 0
    var state: State = .code
    var blockDepth = 0
    var output = ""

    init(source: String) {
        characters = Array(source)
        output.reserveCapacity(source.count)
    }

    mutating func run() -> String {
        while index < characters.count {
            switch state {
            case .code: stepCode()
            case .lineComment: stepLineComment()
            case .blockComment: stepBlockComment()
            case .string: stepString()
            case .multilineString: stepMultilineString()
            }
        }
        return output
    }

    // MARK: Cursor

    private func peek(_ offset: Int) -> Character? {
        let target = index + offset
        return target < characters.count ? characters[target] : nil
    }

    private var current: Character { characters[index] }

    private mutating func take(_ count: Int, as replacement: String) {
        output += replacement
        index += count
    }

    // MARK: States

    private mutating func stepCode() {
        if current == "/", peek(1) == "/" {
            state = .lineComment
            take(2, as: "  ")
        } else if current == "/", peek(1) == "*" {
            state = .blockComment
            blockDepth = 1
            take(2, as: "  ")
        } else if current == "\"", peek(1) == "\"", peek(2) == "\"" {
            state = .multilineString
            take(3, as: "\"\"\"")
        } else {
            if current == "\"" { state = .string }
            take(1, as: String(current))
        }
    }

    private mutating func stepLineComment() {
        if current == "\n" {
            state = .code
            take(1, as: "\n")
        } else {
            take(1, as: " ")
        }
    }

    private mutating func stepBlockComment() {
        if current == "/", peek(1) == "*" {
            blockDepth += 1
            take(2, as: "  ")
        } else if current == "*", peek(1) == "/" {
            blockDepth -= 1
            take(2, as: "  ")
            if blockDepth == 0 { state = .code }
        } else {
            take(1, as: current == "\n" ? "\n" : " ")
        }
    }

    private mutating func stepString() {
        // An escape carries its next character along, so `\"` does not end
        // the literal.
        if current == "\\", let next = peek(1) {
            take(2, as: "\\\(next)")
            return
        }
        if current == "\"" || current == "\n" { state = .code }
        take(1, as: String(current))
    }

    private mutating func stepMultilineString() {
        if current == "\"", peek(1) == "\"", peek(2) == "\"" {
            state = .code
            take(3, as: "\"\"\"")
        } else {
            take(1, as: String(current))
        }
    }
}
