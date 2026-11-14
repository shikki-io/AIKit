import CoreKit
import Foundation

// MARK: - AIProviderRepository

/// The provider list as a CoreKit `Repository`: rows in, rows out, keyed by
/// the id the operator wrote.
///
/// BR-AIL-20. Why a repository at all, rather than handing `AIProviderSettings`
/// around: the settings document is a FILE, and a file is one backend. The
/// fleet's layer for "a store of records whose backend is not the caller's
/// business" is `RepositoryProtocol`, so a host that keeps provider rows in
/// Postgres, in a plist or behind a sync service substitutes its own conformer
/// and nothing above changes. `list()` returns FILE ORDER, which is also the
/// fallback order (BR-AIL-11).
///
/// An `actor` because `save` and `delete` mutate it. Reads are `await`ed, which
/// every caller already is: a provider list is read at assembly time, not in a
/// hot loop.
public actor AIProviderRepository: RepositoryProtocol {

    public typealias ModelType = AIProviderEntry

    // MARK: - State

    /// Rows in file order. An ARRAY and not a dictionary: the order is part of
    /// the configuration, and a dictionary would lose it on the way in.
    private var entries: [AIProviderEntry]
    /// The document these rows came from, for an error that must name it.
    private let source: String

    // MARK: - Init

    /// Holds a settings document already loaded and validated.
    public init(settings: AIProviderSettings, source: String = "the provider settings") {
        self.entries = settings.providers
        self.source = source
    }

    /// Loads the document the HOST chose, validating every address once
    /// (BR-AIL-18) and refusing the whole file on the first bad row.
    ///
    /// - Throws: ``AIKitError/settings(file:entry:reason:)``.
    public static func loading(from url: URL) throws(AIKitError) -> AIProviderRepository {
        AIProviderRepository(settings: try AIProviderSettings.load(from: url), source: url.path)
    }

    // MARK: - RepositoryProtocol

    /// The row stored under `id`.
    ///
    /// - Throws: ``AIKitError/settings(file:entry:reason:)`` naming the file,
    ///   because an id nobody configured is answered by an edit to that file.
    public func get(_ id: AIProviderEntry.EntryID) async throws -> AIProviderEntry {
        guard let entry = entries.first(where: { $0.id == id }) else {
            throw AIKitError.settings(
                file: source,
                entry: id,
                reason: "no [[providers]] row carries this id. Configured: "
                    + (entries.isEmpty ? "none" : entries.map(\.id).joined(separator: ", "))
            )
        }
        return entry
    }

    /// Every row, IN FILE ORDER.
    public func list() async throws -> [AIProviderEntry] { entries }

    /// Inserts or replaces the row under `id`.
    ///
    /// A replacement keeps the row's POSITION, because the position is the
    /// fallback order: re-saving an entry to change its model must not move it
    /// behind the cloud provider that was below it.
    public func save(_ id: AIProviderEntry.EntryID, data: AIProviderEntry) async throws {
        if let index = entries.firstIndex(where: { $0.id == id }) {
            entries[index] = data
        } else {
            entries.append(data)
        }
    }

    /// Removes the row under `id`.
    ///
    /// - Throws: as ``get(_:)`` — deleting a row that is not there would
    ///   otherwise read as a successful delete of something else.
    public func delete(_ id: AIProviderEntry.EntryID) async throws {
        guard let index = entries.firstIndex(where: { $0.id == id }) else {
            _ = try await get(id)
            return
        }
        entries.remove(at: index)
    }

    // MARK: - Reading

    /// The document these rows came from.
    public var fileName: String { source }

    /// Rows of one family, in file order.
    public func list(family: AIProviderFamily) -> [AIProviderEntry] {
        entries.filter { $0.family == family }
    }

    /// The rows as a settings document again, so a host can write them back.
    public func settings() -> AIProviderSettings {
        AIProviderSettings(providers: entries)
    }
}
