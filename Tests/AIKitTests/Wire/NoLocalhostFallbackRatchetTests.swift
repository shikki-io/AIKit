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

    // MARK: - Literals (shared with W24's fixture rule)

    /// One string literal found in a file: its VALUE, and the line it opened
    /// on.
    struct Hit: Sendable, Equatable {
        /// The literal's value — escapes resolved, so `"{\"a\":1}"` reads as
        /// `{"a":1}` and can be handed to a JSON parser.
        let text: String
        /// 1-indexed, and the line of the OPENING delimiter, so a multiline
        /// literal is reported where a reader will look for it.
        let line: Int
    }

    /// Every string literal in `code`, in all three Swift spellings.
    ///
    /// W24's fixture rule needs the literals themselves, where W21's rules
    /// needed only a substring match, so this is the one piece of scanning the
    /// two rules share rather than each having its own idea of what a literal
    /// is — two enforcers of one idea being the defect BR-EC-02 names.
    ///
    /// `code` is expected to have been through ``codeOnly`` already.
    static func stringLiterals(in code: String) -> [Hit] {
        var scanner = StringLiteralScanner(source: code)
        return scanner.run()
    }

    /// `Tests/`, found from this file rather than from the working directory.
    static var testsRoot: URL {
        URL(fileURLWithPath: #filePath)  // Tests/AIKitTests/Wire/<self>.swift
            .deletingLastPathComponent()  // Wire
            .deletingLastPathComponent()  // AIKitTests
            .deletingLastPathComponent()  // Tests
    }

    /// The repo root — the parent of `Sources/` and `Tests/`.
    static var repoRoot: URL { testsRoot.deletingLastPathComponent() }

    /// `file` as a repo-relative path, which is how a declaration names it and
    /// how a failure message should read.
    static func relativeToRepo(_ file: URL) -> String {
        let root = repoRoot.path
        return file.path.hasPrefix(root)
            ? String(file.path.dropFirst(root.count + 1))
            : file.path
    }
}

// MARK: - StringLiteralScanner

/// Pulls the string literals out of Swift source, resolving escapes.
///
/// Three spellings, and each one matters to the rules above:
///
///   * `"…"` with escapes — the form a JSON payload is usually written in, and
///     the one whose `\"` must become `"` before a JSON parser sees it;
///   * `"""…"""` multiline — how the captured replies are written;
///   * `#"…"#` raw, any number of `#` — how a payload with quotes in it is
///     written when escaping every one would be unreadable.
///
/// An INTERPOLATION becomes the placeholder `INTERP`. Without that, a payload
/// built as `"{\"model\":\"\(name)\"}"` would carry `\(` — not a legal
/// JSON escape — so `JSONSerialization` would reject it and the fixture rule
/// would quietly let an interpolated hand-written payload through. The
/// placeholder keeps such a payload parseable, and therefore catchable.
private struct StringLiteralScanner {

    let characters: [Character]
    var index = 0
    var line = 1
    var hits: [NoLocalhostFallbackRatchetTests.Hit] = []

    init(source: String) {
        characters = Array(source)
    }

    mutating func run() -> [NoLocalhostFallbackRatchetTests.Hit] {
        while index < characters.count {
            if characters[index] == "\n" {
                line += 1
                index += 1
            } else if let hashes = rawOpener() {
                readRaw(hashes: hashes)
            } else if matches("\"\"\"") {
                readDelimited(open: 3, terminator: "\"\"\"", raw: false, stopsAtNewline: false)
            } else if characters[index] == "\"" {
                readDelimited(open: 1, terminator: "\"", raw: false, stopsAtNewline: true)
            } else {
                index += 1
            }
        }
        return hits
    }

    // MARK: Cursor

    private func peek(_ offset: Int) -> Character? {
        let target = index + offset
        return target < characters.count ? characters[target] : nil
    }

    private func matches(_ text: String) -> Bool {
        text.enumerated().allSatisfy { peek($0.offset) == $0.element }
    }

    /// The number of `#` opening a raw literal here, or nil.
    private func rawOpener() -> Int? {
        guard characters[index] == "#" else { return nil }
        var hashes = 0
        while peek(hashes) == "#" { hashes += 1 }
        return peek(hashes) == "\"" ? hashes : nil
    }

    // MARK: Reading

    private mutating func readRaw(hashes: Int) {
        let closing = "\"" + String(repeating: "#", count: hashes)
        // `#"""…"""#` opens with three quotes; `#"…"#` with one.
        let opensMultiline = (0..<3).allSatisfy { peek(hashes + $0) == "\"" }
        index += hashes + (opensMultiline ? 3 : 1)
        readBody(terminator: closing, raw: true, stopsAtNewline: false)
    }

    private mutating func readDelimited(open: Int, terminator: String, raw: Bool, stopsAtNewline: Bool) {
        index += open
        readBody(terminator: terminator, raw: raw, stopsAtNewline: stopsAtNewline)
    }

    private mutating func readBody(terminator: String, raw: Bool, stopsAtNewline: Bool) {
        let startLine = line
        var body = ""
        while index < characters.count {
            if matches(terminator) {
                index += terminator.count
                hits.append(.init(text: body, line: startLine))
                return
            }
            let character = characters[index]
            if character == "\n" {
                if stopsAtNewline {
                    // An unterminated literal: not this scanner's business to
                    // diagnose, and the compiler has already refused the file.
                    return
                }
                line += 1
                body.append(character)
                index += 1
            } else if !raw, character == "\\" {
                body += escape()
            } else {
                body.append(character)
                index += 1
            }
        }
    }

    /// Resolves one backslash escape, including an interpolation.
    private mutating func escape() -> String {
        guard let next = peek(1) else {
            index += 1
            return "\\"
        }
        if next == "(" {
            skipInterpolation()
            return "INTERP"
        }
        index += 2
        switch next {
        case "\"": return "\""
        case "\\": return "\\"
        case "n": return "\n"
        case "t": return "\t"
        case "r": return "\r"
        case "/": return "/"
        // Anything else (`\u`, `\0`, …) is kept as written: a JSON parser
        // reads `\u0041` itself, and no rule here depends on the rest.
        default: return "\\" + String(next)
        }
    }

    /// Steps over `\( … )`, counting nested parentheses.
    private mutating func skipInterpolation() {
        index += 2  // past `\(`
        var depth = 1
        while index < characters.count, depth > 0 {
            switch characters[index] {
            case "(": depth += 1
            case ")": depth -= 1
            case "\n": line += 1
            default: break
            }
            index += 1
        }
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
    enum State { case code, lineComment, blockComment, string, multilineString, rawString }

    let characters: [Character]
    var index = 0
    var state: State = .code
    var blockDepth = 0
    /// How many `#` opened the raw literal being scanned, so `"` + that many
    /// closes it. W24: `Tests/` is full of `#"…"#`, and without this state a
    /// `//` inside one was read as a comment and the rest of the line vanished.
    var rawHashes = 0
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
            case .rawString: stepRawString()
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
        if current == "#", let hashes = rawLiteralOpener() {
            state = .rawString
            rawHashes = hashes
            // The opener is emitted verbatim: it is code, and its content is a
            // literal that `stringLiterals` still has to be able to find.
            take(hashes + 1, as: String(repeating: "#", count: hashes) + "\"")
        } else if current == "/", peek(1) == "/" {
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

    /// The number of `#` opening a raw literal here, or nil if this `#` starts
    /// something else (`#expect`, `#filePath`, `#if`).
    private func rawLiteralOpener() -> Int? {
        var hashes = 0
        while peek(hashes) == "#" { hashes += 1 }
        return peek(hashes) == "\"" ? hashes : nil
    }

    /// Inside `#"…"#`. Nothing escapes in a raw literal, so the only thing to
    /// look for is `"` followed by the same number of `#` that opened it.
    /// That also terminates `#"""…"""#` correctly, at the last of the three
    /// quotes.
    private mutating func stepRawString() {
        if current == "\"", (1...rawHashes).allSatisfy({ peek($0) == "#" }) {
            state = .code
            take(rawHashes + 1, as: "\"" + String(repeating: "#", count: rawHashes))
        } else {
            take(1, as: String(current))
        }
    }
}

// MARK: - The operator's server is spelled once

/// W24, carried from W21 (AIKit#18 r4500587739): *"I see too many
/// `http://127.0.0.1:1234` in all the tests we did not have a SSoT for that?
/// Instead of hardcoded the url of my lms?"*
///
/// The rule is narrow on purpose, and the narrowness is the point. W21's rule
/// in `Sources/` forbids a localhost FALLBACK — an absent address answered
/// with the local machine. In `Tests/` that shape is legitimate everywhere,
/// because an address is an INPUT to most of these suites. What is not
/// legitimate is re-typing the operator's own LM Studio: when it moves, every
/// copy has to be found, and the ones that are missed keep passing against a
/// server that is no longer there.
///
/// So this scans for one thing — ``fixtureServerHostPort`` — and anything
/// else about addresses stays W21's business. The files below hold it because
/// the address IS their subject; every other file takes it from
/// ``fixtureServerRoot``.
@Suite("The fixture server address is declared once, not re-typed per suite")
struct FixtureAddressSSoTTests {

    /// Where the operator's LM Studio address may be written, and why.
    struct Declared {
        let path: String
        let count: Int
        let reason: String
    }

    /// Counts measured at the W24 commit by the rule below, not estimated.
    static let declared: [Declared] = [
        Declared(
            path: "Tests/AIKitTests/Support/WireFixtures.swift",
            count: 2,
            reason: "The SSoT itself: `fixtureServerRoot` and the `fixtureServerHostPort` taken from it."
        ),
        Declared(
            path: "Tests/AIKitTests/Wire/AIServerAddressTests.swift",
            count: 12,
            reason: """
                `AIServerAddress(validating:)` is the subject here, so its inputs are addresses \
                by definition — the good root, the same root with a trailing slash, the `/v1` \
                that must be refused, the bare `host:port` that must not be assumed to be http.
                """
        ),
        Declared(
            path: "Tests/AIKitTests/Wire/AIServerAddressReachTests.swift",
            count: 1,
            reason: """
                `AIServerAddress.reach` is the subject: the loopback rows are the classification \
                being asserted, and replacing them with a constant would hide WHICH address was \
                classified.
                """
        ),
        Declared(
            path: "Tests/AIKitTests/Settings/AIProviderRepositoryTests.swift",
            count: 9,
            reason: """
                A settings FILE is the subject. These addresses live inside TOML documents the \
                repository has to parse, plus the list of bad addresses it has to refuse, so they \
                are test data rather than a reference to the operator's server.
                """
        ),
        Declared(
            path: "Tests/AIKitTests/Wire/NoLocalhostFallbackRatchetTests.swift",
            count: 6,
            reason: """
                This file: the four expressions W21 deleted and the two shapes its rule must not \
                flag are quoted so a rotted detector goes red here rather than passing forever \
                (BR-EC-02). Declaring the enforcer to its own rule is the honest way round.
                """
        ),
    ]

    static let byPath: [String: Declared] = Dictionary(
        uniqueKeysWithValues: declared.map { ($0.path, $0) })

    @Test("no suite re-types the operator's LM Studio address")
    func theFixtureAddressIsNotRetyped() throws {
        var offenders: [String] = []
        var counted: [String: Int] = [:]

        for file in try NoLocalhostFallbackRatchetTests.swiftFiles(
            under: NoLocalhostFallbackRatchetTests.testsRoot
        ) {
            let path = NoLocalhostFallbackRatchetTests.relativeToRepo(file)
            let code = NoLocalhostFallbackRatchetTests.codeOnly(
                try String(contentsOf: file, encoding: .utf8))
            let hits = NoLocalhostFallbackRatchetTests.stringLiterals(in: code)
                .filter { $0.text.contains(fixtureServerHostPort) }
            guard !hits.isEmpty else { continue }
            counted[path] = hits.count

            if Self.byPath[path] == nil {
                offenders.append("\(path):\(hits[0].line) — \(hits.count) occurrence(s)")
            }
        }

        #expect(
            offenders.isEmpty,
            """
            W24: the operator's LM Studio address is `fixtureServerRoot` in \
            WireFixtures.swift, and a suite that needs it interpolates that \
            (`"\\(fixtureServerRoot)/v1/chat/completions"`) or uses \
            `fixtureServerHostPort` for a bare host:port. If the address is \
            genuinely the SUBJECT of a new suite, declare it above with its \
            reason. Offenders: \(offenders.joined(separator: " | "))
            """
        )

        for row in Self.declared {
            let found = counted[row.path] ?? 0
            #expect(
                found == row.count,
                "\(row.path) declares \(row.count) occurrence(s) of the fixture address and holds \(found)"
            )
        }
    }

    @Test("the rule still recognises the address it is for")
    func theAddressRuleStillMatches() {
        // BR-EC-02 again: a rule that stopped matching would pass forever.
        for spelling in [
            #"let a = "http://127.0.0.1:1234""#,
            #"let a = "http://127.0.0.1:1234/v1/chat/completions""#,
            #"let a = "127.0.0.1:1234""#,
        ] {
            let hits = NoLocalhostFallbackRatchetTests.stringLiterals(
                in: NoLocalhostFallbackRatchetTests.codeOnly(spelling)
            ).filter { $0.text.contains(fixtureServerHostPort) }
            #expect(!hits.isEmpty, "the address rule no longer recognises: \(spelling)")
        }

        // And what it must NOT flag: the interpolated form every migrated
        // suite now uses, a different port on the same host, and a comment.
        for allowed in [
            #"let a = "\(fixtureServerRoot)/v1/chat/completions""#,
            #"let a = "http://127.0.0.1:8080""#,
            #"// it used to be "http://127.0.0.1:1234" here"#,
        ] {
            let hits = NoLocalhostFallbackRatchetTests.stringLiterals(
                in: NoLocalhostFallbackRatchetTests.codeOnly(allowed)
            ).filter { $0.text.contains(fixtureServerHostPort) }
            #expect(hits.isEmpty, "the address rule flags something it should not: \(allowed)")
        }
    }
}
