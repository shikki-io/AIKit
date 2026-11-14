// kagami-scope: exempt — AIKit (moved from Irori) declares its scope in .kagami/scopes.override.toml (separate spec). File touched only to add resolver-exempt marker for cross-package layering; no behavioral change. See feedback_scope-miss-is-quality-gate-failure-not-caveat for the umbrella rule.
// MARK: - EmbeddingStore
// Local SQLite enclave for V-JEPA embeddings (spec BR-20).
//
// Schema per spec §5 Phase 2 deliverable 3:
//
//   CREATE TABLE video_embeddings (
//       id INTEGER PRIMARY KEY AUTOINCREMENT,
//       video_id TEXT NOT NULL,
//       source_path TEXT NOT NULL,
//       embedding BLOB NOT NULL,
//       frame_count INTEGER NOT NULL,
//       feature_dim INTEGER NOT NULL,
//       model_version TEXT NOT NULL,
//       created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
//   );
//
// Raw SQL only (no ORM per workspace rule). BLOB stores little-endian
// float32 row-major embedding data — the same memory layout the MLX /
// Reference backends already produce.
//
// Default path is ~/.shikki/vision/embeddings.sqlite. Tests pass
// ":memory:" for hermetic runs.

#if canImport(SQLite3)
import Foundation
import SQLite3

// MARK: - Errors

public enum EmbeddingStoreError: Error, Sendable, CustomStringConvertible {
    case openFailed(String)
    case queryFailed(String)
    case prepareFailed(String)
    case corrupt(String)

    public var description: String {
        switch self {
        case .openFailed(let m):    "EmbeddingStore open failed: \(m)"
        case .queryFailed(let m):   "EmbeddingStore query failed: \(m)"
        case .prepareFailed(let m): "EmbeddingStore prepare failed: \(m)"
        case .corrupt(let m):       "EmbeddingStore corrupt row: \(m)"
        }
    }
}

// MARK: - Record

public struct VideoEmbeddingRecord: Sendable, Equatable {
    public let id: Int64
    public let videoId: String
    public let sourcePath: String
    public let embedding: VJEPAEmbedding
    public let modelVersion: String
    public let createdAt: String

    public init(
        id: Int64,
        videoId: String,
        sourcePath: String,
        embedding: VJEPAEmbedding,
        modelVersion: String,
        createdAt: String
    ) {
        self.id = id
        self.videoId = videoId
        self.sourcePath = sourcePath
        self.embedding = embedding
        self.modelVersion = modelVersion
        self.createdAt = createdAt
    }
}

// MARK: - Store

/// Thread-safe SQLite-backed store for V-JEPA embeddings (BR-20).
///
/// Uses the C sqlite3 API directly, one connection per instance, guarded by
/// an internal lock. Not an actor for the same reason as `SQLiteStore` in
/// ShikkiTestRunner — synchronous init + throwing mutators are simpler and
/// match the existing workspace pattern.
public final class EmbeddingStore: @unchecked Sendable {

    private let db: OpaquePointer
    private let lock = NSLock()
    public let path: String

    // MARK: Default path (BR-20)

    public static let defaultPath: String = {
        (NSHomeDirectory() as NSString).appendingPathComponent(".shikki/vision/embeddings.sqlite") // resolver-exempt: AIKit package cannot import ShiKit; ~/.shikki/vision/ canonical per BR-20
    }()

    // MARK: Init

    /// Open or create the embedding store at the given path.
    ///
    /// - Parameter path: filesystem path, or `:memory:` for a hermetic
    ///   in-memory store (used in tests).
    public init(path: String = EmbeddingStore.defaultPath) throws {
        self.path = path

        // Ensure parent directory exists for non-memory paths.
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
            throw EmbeddingStoreError.openFailed(msg)
        }
        db = opened
        try executeRaw("PRAGMA journal_mode=WAL")
        try createSchema()
    }

    deinit {
        sqlite3_close(db)
    }

    // MARK: Schema

    private func createSchema() throws {
        let ddl = """
        CREATE TABLE IF NOT EXISTS video_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT NOT NULL,
            source_path TEXT NOT NULL,
            embedding BLOB NOT NULL,
            frame_count INTEGER NOT NULL,
            feature_dim INTEGER NOT NULL,
            model_version TEXT NOT NULL,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_video_embeddings_video_id
            ON video_embeddings(video_id);
        CREATE INDEX IF NOT EXISTS idx_video_embeddings_created_at
            ON video_embeddings(created_at);
        """
        try executeRaw(ddl)
    }

    /// Returns the list of column names in `video_embeddings`. Used by the
    /// schema test; also handy for `shi vision status`.
    public func dumpSchema() throws -> [String] {
        lock.lock(); defer { lock.unlock() }
        var cols: [String] = []
        var stmt: OpaquePointer?
        let sql = "PRAGMA table_info(video_embeddings)"
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK, let s = stmt else {
            throw EmbeddingStoreError.prepareFailed(String(cString: sqlite3_errmsg(db)))
        }
        defer { sqlite3_finalize(s) }
        while sqlite3_step(s) == SQLITE_ROW {
            // PRAGMA table_info: cid, name, type, notnull, dflt_value, pk
            if let namePtr = sqlite3_column_text(s, 1) {
                cols.append(String(cString: namePtr))
            }
        }
        return cols
    }

    // MARK: Insert

    /// Insert one embedding row. Returns the generated primary key.
    @discardableResult
    public func insert(
        videoId: String,
        sourcePath: String,
        embedding: VJEPAEmbedding,
        modelVersion: String
    ) throws -> Int64 {
        lock.lock(); defer { lock.unlock() }

        let sql = """
        INSERT INTO video_embeddings
          (video_id, source_path, embedding, frame_count, feature_dim, model_version)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK, let s = stmt else {
            throw EmbeddingStoreError.prepareFailed(String(cString: sqlite3_errmsg(db)))
        }
        defer { sqlite3_finalize(s) }

        let transient = unsafeBitCast(-1, to: sqlite3_destructor_type.self)

        _ = videoId.withCString { sqlite3_bind_text(s, 1, $0, -1, transient) }
        _ = sourcePath.withCString { sqlite3_bind_text(s, 2, $0, -1, transient) }

        let blob = Self.encodeEmbedding(embedding)
        _ = blob.withUnsafeBytes { raw -> Int32 in
            guard let base = raw.baseAddress else { return SQLITE_OK }
            return sqlite3_bind_blob(s, 3, base, Int32(blob.count), transient)
        }

        sqlite3_bind_int64(s, 4, Int64(embedding.frameCount))
        sqlite3_bind_int64(s, 5, Int64(embedding.featureDim))
        _ = modelVersion.withCString { sqlite3_bind_text(s, 6, $0, -1, transient) }

        guard sqlite3_step(s) == SQLITE_DONE else {
            throw EmbeddingStoreError.queryFailed(String(cString: sqlite3_errmsg(db)))
        }
        return sqlite3_last_insert_rowid(db)
    }

    // MARK: Fetch one

    public func fetch(id: Int64) throws -> VideoEmbeddingRecord? {
        lock.lock(); defer { lock.unlock() }

        let sql = """
        SELECT id, video_id, source_path, embedding, frame_count, feature_dim,
               model_version, created_at
          FROM video_embeddings
         WHERE id = ?
         LIMIT 1
        """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK, let s = stmt else {
            throw EmbeddingStoreError.prepareFailed(String(cString: sqlite3_errmsg(db)))
        }
        defer { sqlite3_finalize(s) }
        sqlite3_bind_int64(s, 1, id)
        let rc = sqlite3_step(s)
        if rc == SQLITE_DONE { return nil }
        guard rc == SQLITE_ROW else {
            throw EmbeddingStoreError.queryFailed(String(cString: sqlite3_errmsg(db)))
        }
        return try Self.decodeRow(s)
    }

    // MARK: Fetch all

    public func all() throws -> [VideoEmbeddingRecord] {
        lock.lock(); defer { lock.unlock() }

        let sql = """
        SELECT id, video_id, source_path, embedding, frame_count, feature_dim,
               model_version, created_at
          FROM video_embeddings
         ORDER BY id ASC
        """
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK, let s = stmt else {
            throw EmbeddingStoreError.prepareFailed(String(cString: sqlite3_errmsg(db)))
        }
        defer { sqlite3_finalize(s) }

        var rows: [VideoEmbeddingRecord] = []
        while sqlite3_step(s) == SQLITE_ROW {
            rows.append(try Self.decodeRow(s))
        }
        return rows
    }

    // MARK: Count

    public func count() throws -> Int {
        lock.lock(); defer { lock.unlock() }
        var stmt: OpaquePointer?
        let sql = "SELECT COUNT(*) FROM video_embeddings"
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK, let s = stmt else {
            throw EmbeddingStoreError.prepareFailed(String(cString: sqlite3_errmsg(db)))
        }
        defer { sqlite3_finalize(s) }
        guard sqlite3_step(s) == SQLITE_ROW else {
            throw EmbeddingStoreError.queryFailed(String(cString: sqlite3_errmsg(db)))
        }
        return Int(sqlite3_column_int64(s, 0))
    }

    // MARK: Execute raw (schema DDL)

    private func executeRaw(_ sql: String) throws {
        var err: UnsafeMutablePointer<CChar>?
        if sqlite3_exec(db, sql, nil, nil, &err) != SQLITE_OK {
            let msg = err.map { String(cString: $0) } ?? "unknown"
            sqlite3_free(err)
            throw EmbeddingStoreError.queryFailed(msg)
        }
    }

    // MARK: - BLOB codec: [Float] ↔ Data (little-endian float32)

    static func encodeEmbedding(_ e: VJEPAEmbedding) -> Data {
        e.data.withUnsafeBufferPointer { buf in
            Data(buffer: buf)
        }
    }

    static func decodeEmbedding(data: Data, frameCount: Int, featureDim: Int) throws -> VJEPAEmbedding {
        let expectedCount = frameCount * featureDim
        let byteCount = expectedCount * MemoryLayout<Float>.size
        guard data.count == byteCount else {
            throw EmbeddingStoreError.corrupt(
                "blob size mismatch: expected \(byteCount) bytes, got \(data.count)"
            )
        }
        let floats = data.withUnsafeBytes { raw -> [Float] in
            let ptr = raw.bindMemory(to: Float.self)
            return Array(ptr.prefix(expectedCount))
        }
        return VJEPAEmbedding(frameCount: frameCount, featureDim: featureDim, data: floats)
    }

    // MARK: Row decoder

    private static func decodeRow(_ s: OpaquePointer) throws -> VideoEmbeddingRecord {
        let id = sqlite3_column_int64(s, 0)
        let videoId = String(cString: sqlite3_column_text(s, 1))
        let sourcePath = String(cString: sqlite3_column_text(s, 2))

        let blobPtr = sqlite3_column_blob(s, 3)
        let blobLen = Int(sqlite3_column_bytes(s, 3))
        let blobData: Data
        if let raw = blobPtr, blobLen > 0 {
            blobData = Data(bytes: raw, count: blobLen)
        } else {
            blobData = Data()
        }

        let frameCount = Int(sqlite3_column_int64(s, 4))
        let featureDim = Int(sqlite3_column_int64(s, 5))
        let modelVersion = String(cString: sqlite3_column_text(s, 6))
        let createdAt = String(cString: sqlite3_column_text(s, 7))

        let embedding = try decodeEmbedding(
            data: blobData,
            frameCount: frameCount,
            featureDim: featureDim
        )
        return VideoEmbeddingRecord(
            id: id,
            videoId: videoId,
            sourcePath: sourcePath,
            embedding: embedding,
            modelVersion: modelVersion,
            createdAt: createdAt
        )
    }
}

#endif
