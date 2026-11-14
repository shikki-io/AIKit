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
/// W22 of spec ai-layer-aikit-core-2026-10-31 made it `Codable`, so this type
/// IS the `[[providers]]` row rather than something assembled from one: the
/// TOML keys below are the file's keys. AIKit still does not decide where that
/// document lives — the host passes its URL (``AIProviderSettings``).
public struct AIProviderEntry: Sendable, Equatable, Codable {

    // MARK: - EntryID

    /// The key an ``AIProviderRepository`` stores a row under: the entry's own
    /// ``AIProviderEntry/id``, as the operator wrote it.
    ///
    /// Deliberately `String` and not ``AIProviderID``. That enum is a CLOSED
    /// routing vocabulary of 17 known backends; a settings id is whatever the
    /// operator typed, and production already emits ids outside it
    /// (`lmstudio`, `mlx/<model>`). Keying the repository by the enum would
    /// refuse those rows. The spec's W22 text names `AIProviderID` for this
    /// key; it is the same intent under a name that is already taken by a
    /// different thing.
    public typealias EntryID = String

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
    /// The server root, for a family reached over HTTP. Validated when
    /// settings loaded (BR-AIL-18).
    ///
    /// nil for a LOCAL family, which has no server. Read it through
    /// ``serverAddress()``, which names the row when it is missing.
    public let address: AIServerAddress?
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
    /// What this family carries beyond the fields above.
    public let options: AIProviderOptions

    // MARK: - Init

    public init(
        id: String,
        family: AIProviderFamily,
        address: AIServerAddress?,
        model: String,
        apiKeyRef: String? = nil,
        requestTimeout: TimeInterval = AIProviderEntry.defaultRequestTimeout,
        streamLineByteCap: Int = AIProviderEntry.defaultStreamLineByteCap,
        options: AIProviderOptions = .none
    ) {
        self.id = id
        self.family = family
        self.address = address
        self.model = model
        self.apiKeyRef = apiKeyRef
        self.requestTimeout = requestTimeout
        self.streamLineByteCap = streamLineByteCap
        self.options = options
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
        streamLineByteCap: Int = AIProviderEntry.defaultStreamLineByteCap,
        options: AIProviderOptions = .none
    ) throws(AIKitError) {
        self.init(
            id: id,
            family: family,
            address: try AIServerAddress(validating: address),
            model: model,
            apiKeyRef: apiKeyRef,
            requestTimeout: requestTimeout,
            streamLineByteCap: streamLineByteCap,
            options: options
        )
    }

    // MARK: - Reading the address

    /// The server root this entry points at.
    ///
    /// - Throws: ``AIKitError/settings(file:entry:reason:)`` when there is
    ///   none. A row of a LOCAL family carries no address — `cli-subprocess`
    ///   and `mlx-in-process` run on this host — so the field is optional, and
    ///   this is the accessor every wire path goes through. The alternative
    ///   was a required field with a loopback placeholder, which is the
    ///   `?? "127.0.0.1"` BR-AIL-18 exists to forbid.
    public func serverAddress() throws(AIKitError) -> AIServerAddress {
        guard let address else {
            throw .settings(
                file: "the provider settings",
                entry: id,
                reason: "family '\(family.rawValue)' is reached over HTTP and the row names no address"
            )
        }
        return address
    }

    // MARK: - Codable

    /// The keys of one `[[providers]]` row, snake_case as TOML is written.
    enum CodingKeys: String, CodingKey {
        case id
        case family
        case address
        case model
        case apiKeyRef = "api_key_ref"
        case requestTimeout = "request_timeout"
        case streamLineByteCap = "stream_line_byte_cap"
    }

    /// Decodes one row, VALIDATING its address as it goes (BR-AIL-18).
    ///
    /// The address is a `String` on the wire and an ``AIServerAddress`` in
    /// memory, and this is the one crossing: no row becomes an entry without
    /// passing ``AIServerAddress/init(validating:)``. An HTTP family must name
    /// one; a local family must not, because it has no server.
    ///
    /// `family` is read through ``AIProviderFamily/init(kind:)``, which is the
    /// ONE `kind` vocabulary (BR-AIL-20) — so `openai-compatible`, the value
    /// all seven of the operator's existing profiles carry, still decodes.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let file = Self.fileName(in: decoder)
        let id = try container.decode(String.self, forKey: .id)
        let kind = try container.decode(String.self, forKey: .family)

        guard let family = AIProviderFamily(kind: kind) else {
            throw AIKitError.settings(
                file: file,
                entry: id,
                reason: "family '\(kind)' is not one AIKit serves. Write one of: "
                    + AIProviderFamily.allCases.map(\.rawValue).joined(separator: ", ")
            )
        }

        let address: AIServerAddress?
        if let text = try container.decodeIfPresent(String.self, forKey: .address) {
            guard family.isHTTP else {
                throw AIKitError.settings(
                    file: file,
                    entry: id,
                    reason: "family '\(kind)' runs on this host and names an address ('\(text)'). "
                        + "Drop the address, or write an HTTP family."
                )
            }
            do {
                address = try AIServerAddress(validating: text)
            } catch {
                // Re-thrown against the FILE and the ROW, so the operator is
                // told where to edit as well as what is wrong. The address
                // error's own sentence is kept whole.
                throw AIKitError.settings(file: file, entry: id, reason: error.localizedDescription)
            }
        } else {
            guard !family.isHTTP else {
                throw AIKitError.settings(
                    file: file,
                    entry: id,
                    reason: "family '\(kind)' is reached over HTTP and the row names no address"
                )
            }
            address = nil
        }

        let options: AIProviderOptions
        if family.carriesCLIOptions {
            do {
                options = .cliSubprocess(try AICLISubprocessOptions(from: decoder))
            } catch let error as AIKitError {
                throw error
            } catch {
                throw AIKitError.settings(file: file, entry: id, reason: "\(error)")
            }
        } else {
            options = .none
        }

        self.init(
            id: id,
            family: family,
            address: address,
            model: try container.decode(String.self, forKey: .model),
            apiKeyRef: try container.decodeIfPresent(String.self, forKey: .apiKeyRef),
            requestTimeout: try Self.decodeSeconds(from: container) ?? Self.defaultRequestTimeout,
            streamLineByteCap: try container.decodeIfPresent(Int.self, forKey: .streamLineByteCap)
                ?? Self.defaultStreamLineByteCap,
            options: options
        )
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encode(family.rawValue, forKey: .family)
        try container.encodeIfPresent(address?.description, forKey: .address)
        try container.encode(model, forKey: .model)
        try container.encodeIfPresent(apiKeyRef, forKey: .apiKeyRef)
        try container.encode(requestTimeout, forKey: .requestTimeout)
        try container.encode(streamLineByteCap, forKey: .streamLineByteCap)
        if case .cliSubprocess(let options) = options {
            try options.encode(to: encoder)
        }
    }

    /// `request_timeout`, written as a whole number or with decimals.
    ///
    /// Both, because `request_timeout = 600` is an INTEGER in TOML and
    /// TOMLKit reports a type it did not find as `keyNotFound` — so decoding
    /// `TimeInterval` alone refused the whole document over a value the
    /// operator wrote correctly. The seconds are a `TimeInterval` either way.
    private static func decodeSeconds(
        from container: KeyedDecodingContainer<CodingKeys>
    ) throws -> TimeInterval? {
        if let seconds = try? container.decodeIfPresent(TimeInterval.self, forKey: .requestTimeout) {
            return seconds
        }
        if let whole = try container.decodeIfPresent(Int.self, forKey: .requestTimeout) {
            return TimeInterval(whole)
        }
        return nil
    }

    /// The settings file an error should name, as the loader put it on the
    /// decoder.
    ///
    /// ``AIProviderSettings`` sets it, so a row that fails deep in a nested
    /// decode still reports the path the operator edits.
    static func fileName(in decoder: any Decoder) -> String {
        decoder.userInfo[AIProviderSettings.fileNameKey] as? String ?? "the provider settings"
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

    /// Builds a target from an entry, the address its request actually went
    /// to, and the route that was called.
    ///
    /// The address is passed rather than read off the entry because
    /// ``AIProviderEntry/address`` is optional — a local family has none — and
    /// a target exists only for a request that WAS sent somewhere. The caller
    /// has already resolved it through ``AIProviderEntry/serverAddress()``.
    public init(entry: AIProviderEntry, address: AIServerAddress, route: String) {
        self.init(entryID: entry.id, address: address, route: route)
    }

    /// The platform the process that sent the request is running on.
    ///
    /// Not stored, and deliberately not part of ``Equatable``: it is a property
    /// of the running host rather than of the request, so two targets that
    /// name the same entry, address and route are the same target whichever
    /// machine printed them.
    public var hostPlatform: HostPlatform { .current }

    /// `<entry> at <address><route>, from <platform>`.
    ///
    /// The platform is in the text because of AIKit#18 r4500322517: *"you have
    /// the current target where it run, iOS, macOS, tvOS, Linux, windows …
    /// it's not because lms can run it's running in local, you can have
    /// several mac on the same network or the lms server running on a gaming
    /// computer you gonna use even if shikki is running on a MacBook Air"*.
    ///
    /// "Local" names the SERVER's reach (``AIServerAddress/reach``), never the
    /// client's whereabouts, so a loopback address and an iPhone client are not
    /// a contradiction — they are two different facts, and a log that carries
    /// only the first cannot say which device failed. The settings file and
    /// entry are the other half of that context and arrive with
    /// ``AIKitError/settings(file:entry:reason:)`` (W22).
    public var description: String { "\(entryID) at \(address)\(route), from \(hostPlatform.rawValue)" }
}

// MARK: - AIProviderOptions

/// What a `[[providers]]` row carries beyond id, family, address, model and
/// the key reference.
///
/// A CASE per shape rather than a bag of optionals, so a row cannot be read as
/// carrying CLI fields and an address at once, and so adding a family's options
/// later is a case rather than another optional every other family ignores.
/// Only `cli-subprocess` has any today: the HTTP families are fully described
/// by their address and model, and the on-device ones by their model.
public enum AIProviderOptions: Sendable, Equatable {
    /// Nothing beyond the common fields.
    case none
    /// A `cli-subprocess` row's own fields.
    case cliSubprocess(AICLISubprocessOptions)

    /// The CLI fields, when this row has them.
    public var cliSubprocess: AICLISubprocessOptions? {
        guard case .cliSubprocess(let options) = self else { return nil }
        return options
    }
}

// MARK: - AICLISubprocessOptions

/// A `cli-subprocess` row's own fields: what to spawn, how, and where.
///
/// This is the settings half of `CLISubprocessProviderConfig`, which parsed the
/// same keys out of an untyped `[String: Any]` with its own `kind` check. The
/// two were the "two `kind` parsers unaware of each other" the panel found
/// (AIKit#18); this type is the one that reads a row now, and that config type
/// is built FROM it.
///
/// Available on every platform on purpose. iOS has no subprocess API, so no
/// provider can be built here (W25 refuses it with a typed error naming the
/// platform) — but settings written on a Mac and synced to a phone must still
/// DECODE, or one `cli-subprocess` row would refuse the whole document and the
/// phone would have no providers at all.
public struct AICLISubprocessOptions: Sendable, Equatable, Codable {

    /// The binary to spawn, looked up on `PATH`.
    public let binary: String
    /// Arguments, with `{prompt}` and `{model}` placeholders (``ArgsTemplate``).
    public let argsTemplate: [String]
    /// Short-name to canonical-id map (``ModelAliasMap``).
    public let modelAliases: [String: String]
    /// Extra environment variables for the CHILD process.
    ///
    /// Not a read of this process's environment: these are values the operator
    /// wrote in settings and AIKit hands down. BR-AIL-20 forbids a library
    /// reading its own environment, not passing one on.
    public let environment: [String: String]
    /// Working directory for the child; nil inherits the caller's.
    public let workingDirectory: String?

    enum CodingKeys: String, CodingKey {
        case binary
        case argsTemplate = "args_template"
        case modelAliases = "model_aliases"
        case environment = "env"
        case workingDirectory = "cwd"
    }

    public init(
        binary: String,
        argsTemplate: [String] = [],
        modelAliases: [String: String] = [:],
        environment: [String: String] = [:],
        workingDirectory: String? = nil
    ) {
        self.binary = binary
        self.argsTemplate = argsTemplate
        self.modelAliases = modelAliases
        self.environment = environment
        self.workingDirectory = workingDirectory
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let binary = try container.decodeIfPresent(String.self, forKey: .binary) ?? ""
        guard !binary.isEmpty else {
            throw AIKitError.settings(
                file: AIProviderEntry.fileName(in: decoder),
                entry: try? decoder.container(keyedBy: AIProviderEntry.CodingKeys.self)
                    .decode(String.self, forKey: .id),
                reason: "a cli-subprocess row names no binary to spawn"
            )
        }
        self.init(
            binary: binary,
            argsTemplate: try container.decodeIfPresent([String].self, forKey: .argsTemplate) ?? [],
            modelAliases: try container.decodeIfPresent([String: String].self, forKey: .modelAliases) ?? [:],
            environment: try container.decodeIfPresent([String: String].self, forKey: .environment) ?? [:],
            workingDirectory: try container.decodeIfPresent(String.self, forKey: .workingDirectory)
        )
    }
}
