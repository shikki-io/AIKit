import Foundation

/// One provider as the operator configured it: who it is, what it speaks,
/// where it is, and which model to ask for.
///
/// This is one `[[providers]]` row of the settings document. It is the ONLY
/// thing ``HTTPAIProvider`` needs, which is what makes BR-AIL-15's one entry
/// point possible: `complete` and `stream` read a family that was resolved
/// once, at construction, from this value.
///
/// BR-AIL-20. The address is an ``AIServerAddress``, so it was validated when
/// settings loaded and cannot be a string that falls back to localhost. The
/// secret is an `apiKeyRef` — a reference a broker resolves — and never the
/// key itself: a row that carried the key would print it in every log that
/// prints an entry.
///
/// `Codable` and the `RepositoryProtocol` that loads these from TOML arrive in
/// W22 of spec ai-layer-aikit-core-2026-10-31, together with the per-family
/// options struct. This wave introduces the row because the provider is built
/// from it; it does not yet decide where the document lives.
public struct AIProviderEntry: Sendable, Equatable {

    // MARK: - Defaults

    /// How many bytes one line of a streamed reply may reach before the reader
    /// gives up on it.
    ///
    /// Stated once, here. The reader this replaces had no bound at all: it kept
    /// a `Data` buffer, appended every chunk, and re-decoded the WHOLE buffer
    /// on each one (`OpenAIProvider.processBuffer`). A 200 reply carrying a
    /// newline-free error body therefore grew without limit and was re-scanned
    /// per chunk (AIKit#18, panel finding on `OpenAIProvider.swift:131`).
    ///
    /// 1 MiB is far above any SSE line a model produces — the captured chunks
    /// are a few hundred bytes — and far below a size that matters to a host.
    public static let defaultStreamLineByteCap = 1 << 20

    /// How long one request may take, in seconds.
    ///
    /// 300 rather than `URLSession`'s 60: a local model loading cold needs more
    /// than a minute, and a request that dies at 60 seconds reads as a dead
    /// server rather than a slow one — the misreading the fallback rule exists
    /// to avoid.
    public static let defaultRequestTimeout: TimeInterval = 300

    // MARK: - Properties

    /// The id this provider reports, and the value every answer it gives
    /// carries in ``AIResponse/providerID`` (BR-AIL-06).
    public let id: String
    /// What this provider is, and for an HTTP one what its server speaks.
    public let family: AIProviderFamily
    /// The server root. Validated when settings loaded (BR-AIL-18).
    public let address: AIServerAddress
    /// The model name as the server knows it.
    public let model: String
    /// A broker reference to the secret, never the secret (BR-AIL-20).
    public let apiKeyRef: String?
    /// Seconds one request may take. Defaults to
    /// ``AIProviderEntry/defaultRequestTimeout``.
    public let requestTimeout: TimeInterval
    /// Byte ceiling for one line of a streamed reply. Defaults to
    /// ``AIProviderEntry/defaultStreamLineByteCap``.
    public let streamLineByteCap: Int

    // MARK: - Init

    public init(
        id: String,
        family: AIProviderFamily,
        address: AIServerAddress,
        model: String,
        apiKeyRef: String? = nil,
        requestTimeout: TimeInterval = AIProviderEntry.defaultRequestTimeout,
        streamLineByteCap: Int = AIProviderEntry.defaultStreamLineByteCap
    ) {
        self.id = id
        self.family = family
        self.address = address
        self.model = model
        self.apiKeyRef = apiKeyRef
        self.requestTimeout = requestTimeout
        self.streamLineByteCap = streamLineByteCap
    }

    /// Builds an entry from a configured address string, validating it.
    ///
    /// - Throws: ``AIKitError/invalidServerAddress(value:reason:)`` naming the
    ///   value, so a settings mistake is reported where it was written.
    public init(
        id: String,
        family: AIProviderFamily,
        address: String,
        model: String,
        apiKeyRef: String? = nil,
        requestTimeout: TimeInterval = AIProviderEntry.defaultRequestTimeout,
        streamLineByteCap: Int = AIProviderEntry.defaultStreamLineByteCap
    ) throws(AIKitError) {
        self.init(
            id: id,
            family: family,
            address: try AIServerAddress(validating: address),
            model: model,
            apiKeyRef: apiKeyRef,
            requestTimeout: requestTimeout,
            streamLineByteCap: streamLineByteCap
        )
    }
}

// MARK: - AIRequestTarget

/// Where one request went, and as whom.
///
/// Carried by every error a request produces (BR-AIL-19), so a failure names
/// the entry the operator configured, the address it reached and the route it
/// posted to — rather than a literal host written in the error text, which is
/// what `AIWireStandard` did (AIKit#18 r4493433754).
public struct AIRequestTarget: Sendable, Equatable, Hashable, CustomStringConvertible {

    /// ``AIProviderEntry/id`` of the entry the request was built from.
    public let entryID: String
    /// The server it was sent to.
    public let address: AIServerAddress
    /// The route it posted to, as the endpoint enum declares it.
    public let route: String

    public init(entryID: String, address: AIServerAddress, route: String) {
        self.entryID = entryID
        self.address = address
        self.route = route
    }

    /// Builds a target from an entry and the route that was called.
    public init(entry: AIProviderEntry, route: String) {
        self.init(entryID: entry.id, address: entry.address, route: route)
    }

    public var description: String { "\(entryID) at \(address)\(route)" }
}
