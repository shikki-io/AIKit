// MARK: - BattleStore
// SQLite persistence for battle results (spec §5 Phase 3 deliverable 7 +
// BR-18 + BR-20).
//
// Lives in the SAME SQLite enclave as the embeddings (`~/.shikki/vision/
// embeddings.sqlite`) — new table, same file. This matches BR-20 (local
// enclave, no ShikkiDB) and the Phase 3 dispatch brief.
//
// BR-18 schema:
//
//   CREATE TABLE vision_battles (
//       battle_id TEXT PRIMARY KEY,
//       task_id TEXT NOT NULL,
//       task_category TEXT NOT NULL,
//       provider_a TEXT NOT NULL,
//       provider_b TEXT NOT NULL,
//       winner TEXT NOT NULL,
//       metrics_json TEXT NOT NULL,
//       timestamp DATETIME NOT NULL
//   );
//
// `metrics_json` is a `BattleMetrics` value encoded as JSON — keeping the
// schema stable across v1 while letting new fields (e.g. VJEPAInferenceMode
// per BR-19) ride along without migrations.

#if canImport(SQLite3)
import Foundation
import SQLite3

// MARK: - Errors

public enum BattleStoreError: Error, Sendable, CustomStringConvertible {
    case openFailed(String)
    case queryFailed(String)
    case prepareFailed(String)
    case encodeFailed(String)
    case decodeFailed(String)

    public var description: String {
        switch self {
        case .openFailed(let m):     "BattleStore open failed: \(m)"
        case .queryFailed(let m):    "BattleStore query failed: \(m)"
        case .prepareFailed(let m):  "BattleStore prepare failed: \(m)"
        case .encodeFailed(let m):   "BattleStore encode failed: \(m)"
        case .decodeFailed(let m):   "BattleStore decode failed: \(m)"
        }
    }
}

// MARK: - Store

/// Thread-safe SQLite-backed store for `BattleMetrics` records.
///
/// Shares a file with `EmbeddingStore` by design — they are two tables in
/// the same local enclave (BR-20). Safe to open both concurrently because
/// sqlite3 is built with `SQLITE_OPEN_FULLMUTEX` and the connection is
/// serialised under an internal `NSLock`.
public final class BattleStore: @unchecked Sendable {

    private let db: OpaquePointer
    private let lock = NSLock()
    public let path: String

    // MARK: Default path (same file as embeddings — BR-20)

    public static let defaultPath: String = EmbeddingStore.defaultPath

    // MARK: Init

    public init(path: String = BattleStore.defaultPath) throws {
        self.path = path

        if path != ":memory:" {
            let parent = (path as NSString).deletingLastPathComponent
            if !FileManager.default.fileExists(atPath: parent) {
                try FileManager.default.createDirectory(
                    atPath: parent,
                    withIntermediateDirectories: true
                )
            }
        }

        var handle: OpaquePointer?
        let flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX
        let result = sqlite3_open_v2(path, &handle, flags, nil)
        guard result == SQLITE_OK, let opened = handle else {
            let msg = handle.flatMap { String(cString: sqlite3_errmsg($0)) } ?? "unknown"
            if let h = handle { sqlite3_close(h) }
            throw BattleStoreError.openFailed(msg)
        }
        db = opened
        try executeRaw("PRAGMA journal_mode=WAL")
        try createSchema()
    }

    deinit {
        sqlite3_close(db)
    }

    // MARK: Schema (BR-18)

    private func createSchema() throws {
        let ddl = """
        CREATE TABLE IF NOT EXISTS vision_battles (
            battle_id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL,
            task_category TEXT NOT NULL,
            provider_a TEXT NOT NULL,
            provider_b TEXT NOT NULL,
            winner TEXT NOT NULL,
            metrics_json TEXT NOT NULL,
            timestamp DATETIME NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_vision_battles_category
            ON vision_battles(task_category);
        CREATE INDEX IF NOT EXISTS idx_vision_battles_timestamp
            ON vision_battles(timestamp);
        """
        try executeRaw(ddl)
    }

    /// Returns the list of column names in `vision_battles`. Backs the
    /// Phase 3 schema-gate test.
    public func dumpSchema() throws -> [String] {
        lock.lock(); defer { lock.unlock() }
        var cols: [String] = []
        var stmt: OpaquePointer?
        let sql = "PRAGMA table_info(vision_battles)"
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK, let s = stmt else {
            throw BattleStoreError.prepareFailed(String(cString: sqlite3_errmsg(db)))
        }
        defer { sqlite3_finalize(s) }
        while sqlite3_step(s) == SQLITE_ROW {
            if let namePtr = sqlite3_column_text(s, 1) {
                cols.append(String(cString: namePtr))
            }
        }
        return cols
    }

    // MARK: Insert

    /// Insert one battle record. Idempotent on `battle_id` (REPLACE).
    public func insert(_ metrics: BattleMetrics) throws {
        lock.lock(); defer { lock.unlock() }

        let jsonData: Data
        do {
            let enc = JSONEncoder()
            enc.dateEncodingStrategy = .iso8601
            enc.outputFormatting = [.sortedKeys]
            jsonData = try enc.encode(metrics)
        } catch {
            throw BattleStoreError.encodeFailed(String(describing: error))
        }
        guard let json = String(data: jsonData, encoding: .utf8) else {
            throw BattleStoreError.encodeFailed("non-utf8 metrics_json")
        }

        let sql = """
        INSERT OR REPLACE INTO vision_battles
          (battle_id, task_id, task_category, provider_a, provider_b,
           winner, metrics_json, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK, let s = stmt else {
            throw BattleStoreError.prepareFailed(String(cString: sqlite3_errmsg(db)))
        }
        defer { sqlite3_finalize(s) }

        let transient = unsafeBitCast(-1, to: sqlite3_destructor_type.self)
        _ = metrics.battleId.withCString { sqlite3_bind_text(s, 1, $0, -1, transient) }
        _ = metrics.taskId.withCString { sqlite3_bind_text(s, 2, $0, -1, transient) }
        _ = metrics.taskCategory.withCString { sqlite3_bind_text(s, 3, $0, -1, transient) }
        _ = metrics.providerA.providerName.withCString { sqlite3_bind_text(s, 4, $0, -1, transient) }
        _ = metrics.providerB.providerName.withCString { sqlite3_bind_text(s, 5, $0, -1, transient) }
        _ = metrics.winner.rawValue.withCString { sqlite3_bind_text(s, 6, $0, -1, transient) }
        _ = json.withCString { sqlite3_bind_text(s, 7, $0, -1, transient) }

        let iso = ISO8601DateFormatter()
        iso.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        let ts = iso.string(from: metrics.timestamp)
        _ = ts.withCString { sqlite3_bind_text(s, 8, $0, -1, transient) }

        guard sqlite3_step(s) == SQLITE_DONE else {
            throw BattleStoreError.queryFailed(String(cString: sqlite3_errmsg(db)))
        }
    }

    public func insertAll(_ records: [BattleMetrics]) throws {
        for r in records { try insert(r) }
    }

    // MARK: Fetch

    public func fetch(battleId: String) throws -> BattleMetrics? {
        lock.lock(); defer { lock.unlock() }
        let sql = "SELECT metrics_json FROM vision_battles WHERE battle_id = ? LIMIT 1"
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK, let s = stmt else {
            throw BattleStoreError.prepareFailed(String(cString: sqlite3_errmsg(db)))
        }
        defer { sqlite3_finalize(s) }
        let transient = unsafeBitCast(-1, to: sqlite3_destructor_type.self)
        _ = battleId.withCString { sqlite3_bind_text(s, 1, $0, -1, transient) }
        let rc = sqlite3_step(s)
        if rc == SQLITE_DONE { return nil }
        guard rc == SQLITE_ROW else {
            throw BattleStoreError.queryFailed(String(cString: sqlite3_errmsg(db)))
        }
        return try decodeMetrics(s)
    }

    public func all() throws -> [BattleMetrics] {
        lock.lock(); defer { lock.unlock() }
        let sql = "SELECT metrics_json FROM vision_battles ORDER BY timestamp ASC"
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK, let s = stmt else {
            throw BattleStoreError.prepareFailed(String(cString: sqlite3_errmsg(db)))
        }
        defer { sqlite3_finalize(s) }
        var out: [BattleMetrics] = []
        while sqlite3_step(s) == SQLITE_ROW {
            out.append(try decodeMetrics(s))
        }
        return out
    }

    public func count() throws -> Int {
        lock.lock(); defer { lock.unlock() }
        var stmt: OpaquePointer?
        let sql = "SELECT COUNT(*) FROM vision_battles"
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK, let s = stmt else {
            throw BattleStoreError.prepareFailed(String(cString: sqlite3_errmsg(db)))
        }
        defer { sqlite3_finalize(s) }
        guard sqlite3_step(s) == SQLITE_ROW else {
            throw BattleStoreError.queryFailed(String(cString: sqlite3_errmsg(db)))
        }
        return Int(sqlite3_column_int64(s, 0))
    }

    // MARK: - Execute raw

    private func executeRaw(_ sql: String) throws {
        var err: UnsafeMutablePointer<CChar>?
        if sqlite3_exec(db, sql, nil, nil, &err) != SQLITE_OK {
            let msg = err.map { String(cString: $0) } ?? "unknown"
            sqlite3_free(err)
            throw BattleStoreError.queryFailed(msg)
        }
    }

    // MARK: - Decode

    private func decodeMetrics(_ s: OpaquePointer) throws -> BattleMetrics {
        guard let ptr = sqlite3_column_text(s, 0) else {
            throw BattleStoreError.decodeFailed("null metrics_json")
        }
        let json = String(cString: ptr)
        guard let data = json.data(using: .utf8) else {
            throw BattleStoreError.decodeFailed("non-utf8 metrics_json")
        }
        let dec = JSONDecoder()
        dec.dateDecodingStrategy = .iso8601
        do {
            return try dec.decode(BattleMetrics.self, from: data)
        } catch {
            throw BattleStoreError.decodeFailed(String(describing: error))
        }
    }
}

#endif
