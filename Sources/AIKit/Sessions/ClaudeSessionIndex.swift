// Migrated from shikki ShiKit (PR #1401 review, third pass): the AI-provider
// SPM owns the session-store contract AND the claude implementation. Origin:
// shi-crash-resilience-ssot-2026-07-17 W1; shikki consumes via re-export once
// its AIKit pin bumps (extract-before-consume bridge).

import Foundation

// MARK: - AgentSessionEntry
//
// Spec: shi-crash-resilience-db-ssot-session-index-auto-recovery-2026-07-17.md
// Wave W1 — "Global session picker (kill the CWD trap)"
//
// AC-R1: `shi r` from ANY directory must NEVER print "Nothing to resume" while
//        resumable jsonl files exist under `~/.claude/projects/<cwd-slug>/*.jsonl`.
// AC-R2: multiple sessions in one dir → picker list with topic + size + age.
// AC-S3: lazy mtime scan at `shi r` time — no duplication of jsonl content into @db.
//
// ClaudeSessionIndex is a pure, filesystem-only, sync scanner. It does NOT:
//   - Talk to @db (that's the SessionRow / SessionRegistry surface).
//   - Talk to NATS (this is a read-only picker source).
//   - Spawn subprocesses.
// It reads directory listings + first-line file metadata only. Safe from any
// context (subagent, hook, TUI, etc.).
//
// PROVIDER BOUNDARY (PR #1401 review): the on-disk layout scanned here is
// claude-code-SPECIFIC (`~/.claude/projects/<slug>/<uuid>.jsonl`). Consumers
// must depend on `AgentSessionIndexProvider` (ShiKit/Agent/
// AgentSessionIndexProvider.swift) — a gemma-4 / KIMI-K3 / any-provider
// session store plugs in as another conformer. The provider family's final
// home is the AIKit SPM (FJ-Studios/AIKit); it migrates there with the
// extraction epic (plan 34bd228b).

// MARK: - AgentSessionEntry

/// One resumable claude-code session on disk.
public struct AgentSessionEntry: Equatable, Sendable {
    /// The claude session UUID (parsed from the .jsonl file's basename).
    public let sessionID: String

    /// Absolute path to the .jsonl transcript file.
    public let jsonlPath: String

    /// The slug directory name under `~/.claude/projects/`
    /// (e.g. `-Users-jeoffrey--shikki`).
    public let projectSlug: String

    /// Best-effort resolved cwd. Sourced from the `cwd` field inside the
    /// first jsonl line when available; falls back to a slug-decoded path
    /// if any candidate exists on disk. `nil` when no candidate matches.
    public let cwd: String?

    /// File modification time.
    public let mtime: Date

    /// Total .jsonl size in bytes.
    public let sizeBytes: Int64

    /// First user-message snippet (~120 chars, whitespace-collapsed).
    /// `nil` when the file has no user-message or is unreadable.
    public let firstMessageTopic: String?

    public init(
        sessionID: String,
        jsonlPath: String,
        projectSlug: String,
        cwd: String?,
        mtime: Date,
        sizeBytes: Int64,
        firstMessageTopic: String?
    ) {
        self.sessionID = sessionID
        self.jsonlPath = jsonlPath
        self.projectSlug = projectSlug
        self.cwd = cwd
        self.mtime = mtime
        self.sizeBytes = sizeBytes
        self.firstMessageTopic = firstMessageTopic
    }

    /// Age in seconds relative to `now` (default: current wall clock).
    public func ageSeconds(now: Date = Date()) -> TimeInterval {
        max(0, now.timeIntervalSince(mtime))
    }

    /// Compact age label (`"5s"`, `"2m"`, `"2h"`, `"3d"`).
    /// Self-contained: AIKit depends on Foundation only — consumers with the
    /// ShikkiCore formatter family may format `mtime`/`sizeBytes` themselves.
    public func ageString(now: Date = Date()) -> String {
        let s = Int(ageSeconds(now: now))
        if s < 60 { return "\(s)s" }
        if s < 3_600 { return "\(s / 60)m" }
        if s < 86_400 { return "\(s / 3_600)h" }
        return "\(s / 86_400)d"
    }

    /// Fine-grained byte label (`"812 B"`, `"4.7 KB"`, `"1.2 MB"`, `"2.10 GB"`).
    public var sizeString: String {
        let kb = Double(sizeBytes) / 1_024
        if kb < 1 { return "\(sizeBytes) B" }
        if kb < 1_024 { return String(format: "%.1f KB", kb) }
        let mb = kb / 1_024
        if mb < 1_024 { return String(format: "%.1f MB", mb) }
        return String(format: "%.2f GB", mb / 1_024)
    }
}

// MARK: - ClaudeSessionIndex

/// Filesystem scanner over `~/.claude/projects/<slug>/<uuid>.jsonl` transcripts.
public enum ClaudeSessionIndex {

    /// Default projects root (`~/.claude/projects`).
    public static var defaultRoot: String {
        "\(NSHomeDirectory())/.claude/projects"
    }

    // MARK: - Scan

    /// Scan every .jsonl transcript under `root` and return entries sorted newest-first.
    ///
    /// - Parameters:
    ///   - root: Projects root. Defaults to `~/.claude/projects`.
    ///   - maxAge: When non-nil, entries older than `maxAge` seconds are dropped.
    ///   - limit: When non-nil, at most `limit` entries are returned (after sorting).
    ///   - now: Injectable clock for `maxAge` filtering (tests).
    /// - Returns: entries sorted by mtime descending (newest first).
    public static func scan(
        root: String? = nil,
        maxAge: TimeInterval? = nil,
        limit: Int? = nil,
        now: Date = Date()
    ) -> [AgentSessionEntry] {
        let rootPath = root ?? defaultRoot
        let fm = FileManager.default

        guard let projectDirs = try? fm.contentsOfDirectory(atPath: rootPath) else {
            return []
        }

        var results: [AgentSessionEntry] = []
        for slug in projectDirs {
            let dir = "\(rootPath)/\(slug)"
            var isDir: ObjCBool = false
            guard fm.fileExists(atPath: dir, isDirectory: &isDir), isDir.boolValue else {
                continue
            }
            guard let files = try? fm.contentsOfDirectory(atPath: dir) else { continue }
            for file in files where file.hasSuffix(".jsonl") {
                let path = "\(dir)/\(file)"
                guard let entry = makeEntry(jsonlPath: path, projectSlug: slug) else { continue }
                if let maxAge, entry.ageSeconds(now: now) > maxAge { continue }
                results.append(entry)
            }
        }

        results.sort { $0.mtime > $1.mtime }
        if let limit, limit >= 0, results.count > limit {
            results = Array(results.prefix(limit))
        }
        return results
    }

    // MARK: - Entry construction

    /// Build one `AgentSessionEntry` from a jsonl transcript path.
    /// Returns nil when the file cannot be stat'd or has no valid basename.
    public static func makeEntry(jsonlPath: String, projectSlug: String) -> AgentSessionEntry? {
        let fm = FileManager.default
        guard let attrs = try? fm.attributesOfItem(atPath: jsonlPath) else { return nil }
        let mtime = (attrs[.modificationDate] as? Date) ?? Date(timeIntervalSince1970: 0)
        let size = (attrs[.size] as? NSNumber)?.int64Value ?? 0

        let basename = (jsonlPath as NSString).lastPathComponent
        guard basename.hasSuffix(".jsonl") else { return nil }
        let sessionID = String(basename.dropLast(".jsonl".count))
        guard !sessionID.isEmpty else { return nil }

        // Best-effort probe: read first ~64 KB to look for cwd + first user message.
        let (cwd, topic) = probeHeader(path: jsonlPath, maxBytes: 64 * 1_024)
        let resolvedCwd = cwd ?? decodeSlug(projectSlug)

        return AgentSessionEntry(
            sessionID: sessionID,
            jsonlPath: jsonlPath,
            projectSlug: projectSlug,
            cwd: resolvedCwd,
            mtime: mtime,
            sizeBytes: size,
            firstMessageTopic: topic
        )
    }

    // MARK: - Slug decoding

    /// Best-effort decode of a claude project slug back into an absolute path.
    ///
    /// Claude writes the slug as `path.replace(/\//g, '-')`, replacing every path
    /// separator with `-`. The mapping is lossy — a dash in an original path segment
    /// is indistinguishable from a separator. We walk from the filesystem root and
    /// consume as many trailing dashes as possible to match a real directory.
    ///
    /// Returns nil when no candidate matches an existing directory (caller should
    /// fall back to the raw slug or the cwd inside the jsonl).
    public static func decodeSlug(
        _ slug: String,
        fileManager: FileManager = .default
    ) -> String? {
        guard slug.hasPrefix("-") else { return nil }
        let stripped = String(slug.dropFirst())
        guard !stripped.isEmpty else { return "/" }

        // Split on '-' and greedy-match longest existing prefix.
        // Recombine remaining tokens with '-' since they may be part of one segment.
        let tokens = stripped.split(separator: "-", omittingEmptySubsequences: false).map(String.init)
        return walk(tokens: tokens, index: 0, prefix: "/", fileManager: fileManager)
    }

    private static func walk(
        tokens: [String],
        index: Int,
        prefix: String,
        fileManager: FileManager
    ) -> String? {
        if index >= tokens.count { return prefix }

        // Try consuming as many contiguous tokens as possible, longest match first.
        // Empty tokens (from `--` sequences) indicate a `.` prefix on the next segment.
        var dotPrefix = ""
        var idx = index
        while idx < tokens.count, tokens[idx].isEmpty {
            dotPrefix += "."
            idx += 1
        }
        if idx >= tokens.count {
            // Trailing empty tokens don't form a real path segment.
            return nil
        }

        // Try greedy longest match: from tokens[idx..<end] joined by '-'.
        var end = tokens.count
        while end > idx {
            let segment = dotPrefix + tokens[idx..<end].joined(separator: "-")
            let candidate = joinPath(prefix, segment)
            var isDir: ObjCBool = false
            if fileManager.fileExists(atPath: candidate, isDirectory: &isDir), isDir.boolValue {
                if let deeper = walk(
                    tokens: tokens,
                    index: end,
                    prefix: candidate,
                    fileManager: fileManager
                ) {
                    return deeper
                }
            }
            end -= 1
        }
        return nil
    }

    private static func joinPath(_ a: String, _ b: String) -> String {
        (a as NSString).appendingPathComponent(b)
    }

    // MARK: - Header probe

    /// Read the first `maxBytes` of the jsonl and extract `(cwd, firstUserTopic)`.
    ///
    /// jsonl files are one JSON object per line. We do a naive scan for the first
    /// `"cwd":"..."` field and for the first `"type":"user"` entry's message content.
    /// This is intentionally cheap — full parsing is deferred to consumers who need it.
    static func probeHeader(path: String, maxBytes: Int) -> (cwd: String?, topic: String?) {
        guard let fh = FileHandle(forReadingAtPath: path) else { return (nil, nil) }
        defer { try? fh.close() }

        let data = fh.readData(ofLength: maxBytes)
        guard let text = String(data: data, encoding: .utf8), !text.isEmpty else {
            return (nil, nil)
        }

        var cwd: String?
        var topic: String?

        for line in text.split(separator: "\n", omittingEmptySubsequences: true) {
            let str = String(line)
            if cwd == nil, let extracted = extractStringField(json: str, key: "cwd") {
                cwd = extracted
            }
            if topic == nil, str.contains("\"type\":\"user\"") {
                if let content = extractStringField(json: str, key: "content") {
                    topic = shortenTopic(content)
                }
            }
            if cwd != nil, topic != nil { break }
        }

        return (cwd, topic)
    }

    /// Extract `"<key>":"<value>"` from a JSON line, handling standard escapes.
    /// Returns nil when the key is absent or the value is not a string.
    static func extractStringField(json: String, key: String) -> String? {
        let needle = "\"\(key)\":\""
        guard let range = json.range(of: needle) else { return nil }
        var iter = range.upperBound  // needle ends after the opening quote of the value
        var result = ""
        while iter < json.endIndex {
            let ch = json[iter]
            if ch == "\\" {
                let next = json.index(after: iter)
                guard next < json.endIndex else { break }
                let esc = json[next]
                switch esc {
                case "n": result.append("\n")
                case "t": result.append("\t")
                case "r": result.append("\r")
                case "\"": result.append("\"")
                case "\\": result.append("\\")
                case "/": result.append("/")
                default: result.append(esc)
                }
                iter = json.index(after: next)
            } else if ch == "\"" {
                return result
            } else {
                result.append(ch)
                iter = json.index(after: iter)
            }
        }
        return nil
    }

    /// Collapse whitespace and clip to ~120 chars for the picker preview.
    static func shortenTopic(_ raw: String, limit: Int = 120) -> String {
        let collapsed =
            raw
            .replacingOccurrences(of: "\n", with: " ")
            .replacingOccurrences(of: "\t", with: " ")
            .split(separator: " ", omittingEmptySubsequences: true)
            .joined(separator: " ")
        if collapsed.count <= limit { return collapsed }
        return String(collapsed.prefix(limit - 1)) + "…"
    }
}
