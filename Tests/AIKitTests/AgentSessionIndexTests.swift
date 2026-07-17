import Foundation
import Testing

@testable import AIKit

// MARK: - ClaudeSessionIndexTests
//
// Spec: shi-crash-resilience-db-ssot-session-index-auto-recovery-2026-07-17.md
// Wave W1 — "Global session picker (kill the CWD trap)".
//
// Covers:
//   AC-R1: scan() lists all jsonl sessions under a projects root; no cwd filter.
//   AC-R2: entries carry topic + size + age; picker payload is deterministic.
//   AC-R3: sessions above the summary threshold flag as large (via caller).
//   AC-S3: index is a pure filesystem read — no @db, no NATS, no subprocess.
//
// Fixtures are written to a per-test tmp dir mirroring the
// `~/.claude/projects/<slug>/<uuid>.jsonl` layout, so tests are hermetic and
// do not depend on the operator's real transcript store.

@Suite("ClaudeSessionIndex — W1 global picker (crash-resilience spec)")
struct ClaudeSessionIndexTests {

    // MARK: - scan()

    @Test("scan() returns entries newest-first, one per jsonl file")
    func scanNewestFirst() throws {
        let root = try makeTempProjectsRoot()
        defer { try? FileManager.default.removeItem(atPath: root) }

        let older = try writeSession(
            root: root,
            slug: "-tmp-project-a",
            sessionID: "aaaaaaaa-1111-1111-1111-111111111111",
            cwd: "/tmp/project-a",
            userMessage: "first task from A",
            mtime: Date(timeIntervalSinceNow: -3_600)
        )
        let newer = try writeSession(
            root: root,
            slug: "-tmp-project-b",
            sessionID: "bbbbbbbb-2222-2222-2222-222222222222",
            cwd: "/tmp/project-b",
            userMessage: "second task from B",
            mtime: Date(timeIntervalSinceNow: -60)
        )

        let entries = ClaudeSessionIndex.scan(root: root)
        #expect(entries.count == 2)
        #expect(entries.first?.jsonlPath == newer)
        #expect(entries.last?.jsonlPath == older)
    }

    @Test("scan() reads cwd from the jsonl header when present (AC-R1 defense)")
    func scanReadsCwdField() throws {
        let root = try makeTempProjectsRoot()
        defer { try? FileManager.default.removeItem(atPath: root) }

        _ = try writeSession(
            root: root,
            slug: "-nonsense-slug-that-cannot-decode",
            sessionID: "cccccccc-3333-3333-3333-333333333333",
            cwd: "/Users/anyone/does-not-need-to-exist",
            userMessage: "hello",
            mtime: Date()
        )

        let entries = ClaudeSessionIndex.scan(root: root)
        #expect(entries.count == 1)
        // The stored cwd wins over slug-decode fallback.
        #expect(entries.first?.cwd == "/Users/anyone/does-not-need-to-exist")
    }

    @Test("scan() extracts the first user message as topic (AC-R2)")
    func scanExtractsTopic() throws {
        let root = try makeTempProjectsRoot()
        defer { try? FileManager.default.removeItem(atPath: root) }

        _ = try writeSession(
            root: root,
            slug: "-tmp-topic",
            sessionID: "dddddddd-4444-4444-4444-444444444444",
            cwd: "/tmp/topic",
            userMessage: "fix the flaky ballot test\nsecond line ignored",
            mtime: Date()
        )

        let entries = ClaudeSessionIndex.scan(root: root)
        #expect(entries.count == 1)
        #expect(entries.first?.firstMessageTopic?.hasPrefix("fix the flaky ballot test") == true)
    }

    @Test("scan() respects maxAge (older entries dropped)")
    func scanRespectsMaxAge() throws {
        let root = try makeTempProjectsRoot()
        defer { try? FileManager.default.removeItem(atPath: root) }

        let ancient = try writeSession(
            root: root,
            slug: "-tmp-ancient",
            sessionID: "eeeeeeee-5555-5555-5555-555555555555",
            cwd: "/tmp/ancient",
            userMessage: "old work",
            mtime: Date(timeIntervalSinceNow: -7 * 86_400)
        )
        _ = try writeSession(
            root: root,
            slug: "-tmp-fresh",
            sessionID: "ffffffff-6666-6666-6666-666666666666",
            cwd: "/tmp/fresh",
            userMessage: "new work",
            mtime: Date(timeIntervalSinceNow: -60)
        )

        let entries = ClaudeSessionIndex.scan(root: root, maxAge: 3_600)
        #expect(entries.count == 1)
        #expect(entries.first?.sessionID.hasPrefix("ffff") == true)
        _ = ancient
    }

    @Test("scan() respects limit (returns at most N entries)")
    func scanRespectsLimit() throws {
        let root = try makeTempProjectsRoot()
        defer { try? FileManager.default.removeItem(atPath: root) }

        for i in 0..<5 {
            _ = try writeSession(
                root: root,
                slug: "-tmp-batch-\(i)",
                sessionID: "batch\(i)aa-0000-0000-0000-000000000000",
                cwd: "/tmp/batch-\(i)",
                userMessage: "task \(i)",
                mtime: Date(timeIntervalSinceNow: -Double(i * 60))
            )
        }

        let entries = ClaudeSessionIndex.scan(root: root, limit: 3)
        #expect(entries.count == 3)
    }

    @Test("scan() returns [] when root does not exist (never throws)")
    func scanMissingRoot() {
        let entries = ClaudeSessionIndex.scan(root: "/tmp/definitely-not-a-directory-\(UUID().uuidString)")
        #expect(entries.isEmpty)
    }

    // MARK: - decodeSlug()

    @Test("decodeSlug walks real directory structure (dot-prefix handling)")
    func decodeSlugWalksRealDirs() throws {
        // Build /tmp/csi-decode-XYZ/a/.hidden/leaf and decode "-tmp-csi-decode-XYZ-a--hidden-leaf"
        let uuid = UUID().uuidString.prefix(8)
        let base = "/tmp/csi-decode-\(uuid)"
        let leaf = "\(base)/a/.hidden/leaf"
        try FileManager.default.createDirectory(atPath: leaf, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(atPath: base) }

        // Slug: replace '/' with '-' and '.' with an extra '-' (double-dash marker).
        // Path "/tmp/csi-decode-XYZ/a/.hidden/leaf" → tokens
        //   "" · "tmp" · "csi" · "decode" · "XYZ" · "a" · "" · "hidden" · "leaf"
        let slug = "-tmp-csi-decode-\(uuid)-a--hidden-leaf"
        let decoded = ClaudeSessionIndex.decodeSlug(slug)
        #expect(decoded == leaf)
    }

    @Test("decodeSlug returns nil for a slug that matches no real directory")
    func decodeSlugNoMatch() {
        let out = ClaudeSessionIndex.decodeSlug("-definitely-not-a-real-path-\(UUID().uuidString)")
        #expect(out == nil)
    }

    @Test("decodeSlug rejects non-slug input (no leading dash)")
    func decodeSlugRejectsNonSlug() {
        #expect(ClaudeSessionIndex.decodeSlug("no-leading-dash") == nil)
    }

    // MARK: - extractStringField()

    @Test("extractStringField parses standard JSON escapes")
    func extractStringFieldParsesEscapes() {
        let line = "{\"type\":\"user\",\"content\":\"line one\\nline two\\ttabbed\\\"quoted\\\" end\"}"
        let out = ClaudeSessionIndex.extractStringField(json: line, key: "content")
        #expect(out == "line one\nline two\ttabbed\"quoted\" end")
    }

    @Test("extractStringField returns nil when key missing")
    func extractStringFieldMissing() {
        let line = "{\"type\":\"user\"}"
        #expect(ClaudeSessionIndex.extractStringField(json: line, key: "content") == nil)
    }

    // MARK: - shortenTopic()

    @Test("shortenTopic collapses whitespace and clips to the limit")
    func shortenTopicClips() {
        let raw = String(repeating: "abc  ", count: 60)
        let out = ClaudeSessionIndex.shortenTopic(raw, limit: 40)
        #expect(out.count == 40)
        #expect(out.hasSuffix("…"))
    }

    // MARK: - Entry helpers

    @Test("ageString buckets seconds / minutes / hours / days")
    func ageStringBuckets() {
        let base = Date()
        let make: (TimeInterval) -> AgentSessionEntry = { seconds in
            AgentSessionEntry(
                sessionID: "x",
                jsonlPath: "/tmp/x.jsonl",
                projectSlug: "-tmp",
                cwd: "/tmp",
                mtime: base.addingTimeInterval(-seconds),
                sizeBytes: 0,
                firstMessageTopic: nil
            )
        }
        #expect(make(5).ageString(now: base) == "5s")
        #expect(make(120).ageString(now: base) == "2m")
        #expect(make(3_600 * 2).ageString(now: base) == "2h")
        #expect(make(86_400 * 3).ageString(now: base) == "3d")
    }

    @Test("sizeString scales to KB / MB")
    func sizeStringScales() {
        let mk: (Int64) -> AgentSessionEntry = { bytes in
            AgentSessionEntry(
                sessionID: "x",
                jsonlPath: "/tmp/x.jsonl",
                projectSlug: "-tmp",
                cwd: nil,
                mtime: Date(),
                sizeBytes: bytes,
                firstMessageTopic: nil
            )
        }
        #expect(mk(500).sizeString == "500 B")
        #expect(mk(2_048).sizeString.contains("KB"))
        #expect(mk(5_000_000).sizeString.contains("MB"))
    }

    // MARK: - Fixture helpers

    /// Create a fresh temp directory to act as `~/.claude/projects`.
    private func makeTempProjectsRoot() throws -> String {
        let path = NSTemporaryDirectory() + "csi-\(UUID().uuidString)"
        try FileManager.default.createDirectory(atPath: path, withIntermediateDirectories: true)
        return path
    }

    /// Write a fake session .jsonl and set its mtime. Returns the jsonl path.
    @discardableResult
    private func writeSession(
        root: String,
        slug: String,
        sessionID: String,
        cwd: String,
        userMessage: String,
        mtime: Date
    ) throws -> String {
        let dir = "\(root)/\(slug)"
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        let path = "\(dir)/\(sessionID).jsonl"

        let escaped =
            userMessage
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
            .replacingOccurrences(of: "\n", with: "\\n")
            .replacingOccurrences(of: "\t", with: "\\t")

        let line1 = "{\"type\":\"summary\",\"cwd\":\"\(cwd)\",\"sessionId\":\"\(sessionID)\"}\n"
        let line2 =
            "{\"type\":\"user\",\"cwd\":\"\(cwd)\",\"message\":{\"role\":\"user\"},\"content\":\"\(escaped)\"}\n"
        try (line1 + line2).write(toFile: path, atomically: true, encoding: .utf8)

        try FileManager.default.setAttributes(
            [.modificationDate: mtime],
            ofItemAtPath: path
        )
        return path
    }
}
