import Foundation
import TOMLKit

#if canImport(Network)
import Dispatch
import Network
#endif

// MARK: - The network path a download would run over

/// What the device's current network path costs, as far as it can be observed.
///
/// W33 of spec ai-layer-aikit-core-2026-10-31, BR-AIL-23. Operator,
/// 2026-11-12: *"did we manage the fact to check the size of the model and the
/// free space available on the device … especially if you are on mobile device
/// not on wifi"*.
///
/// Three observable answers and one honest absence:
///
/// - ``unrestricted`` — the path costs nothing extra and limits nothing.
/// - ``expensive`` — `NWPath.isExpensive`: cellular, or a personal hotspot.
/// - ``constrained`` — `NWPath.isConstrained`: Low Data Mode is on for this
///   path, so the system is asking every client to be frugal.
/// - ``unknown`` — there is no `Network` framework on this platform (Linux,
///   Windows), so the path is not observable here at all.
///
/// A path that is BOTH expensive and constrained reports ``expensive``. That
/// loses nothing operationally: both need the same explicit allow, and the
/// stronger of the two is the one an operator pays a bill for. It is also the
/// shape BR-AIL-23 names — two consent reasons, not four.
public enum ModelDownloadNetwork: String, Sendable, Equatable, CaseIterable, Codable {

    /// Wired, or Wi-Fi with Low Data Mode off.
    case unrestricted

    /// Cellular or a hotspot: bytes here are metered.
    case expensive

    /// Low Data Mode: the system has asked clients to limit traffic.
    case constrained

    /// Not observable on this platform.
    ///
    /// NOT a synonym for "fine": it means AIKit measured nothing. What follows
    /// from it is stated once, in ``ModelDownloadPlan/decision``.
    case unknown

    /// The case the two `NWPath` flags describe.
    ///
    /// A factory rather than a memberwise case, so `(false, false)` has exactly
    /// one spelling — ``unrestricted`` — instead of being representable twice.
    public static func observed(isExpensive: Bool, isConstrained: Bool) -> Self {
        if isExpensive { return .expensive }
        if isConstrained { return .constrained }
        return .unrestricted
    }

    /// Whether this path needs the operator's explicit allow before bytes move.
    public var needsConsent: Bool {
        switch self {
        case .expensive, .constrained: true
        case .unrestricted, .unknown: false
        }
    }

    /// Why consent is needed, or `nil` when it is not.
    public var consentReason: ModelDownloadConsentReason? {
        switch self {
        case .expensive: .expensiveNetwork
        case .constrained: .constrainedNetwork
        case .unrestricted, .unknown: nil
        }
    }
}

// MARK: - The settings that govern a download

/// The `[models.download]` table of the host's settings document (BR-AIL-23).
///
/// ```toml
/// [models.download]
/// allow_expensive_network = false      # the default: never spend on cellular silently
/// free_space_margin_bytes = 536870912  # 512 MiB kept free beyond the model
/// ```
///
/// It lives in the SAME document as the `[[providers]]` array
/// (``AIProviderSettings``) and is read separately, because the two answer
/// different questions and a caller that only downloads models has no reason
/// to hold a provider list. AIKit does not decide where the file is: the host
/// passes the `URL`, exactly as it does for the provider list (BR-AIL-20).
///
/// Both keys have a default, and a document with no `[models.download]` table
/// is not an error — it is the default policy, written down nowhere. What IS
/// an error is a key present with a value that is not one (a margin below
/// zero), because that is an edit the operator made and meant.
public struct ModelDownloadSettings: Sendable, Equatable, Codable {

    /// Bytes kept free beyond what a download needs, when the operator has not
    /// said otherwise.
    ///
    /// 512 MiB. The number is a judgment, and the reason it is not zero is
    /// that a volume filled to its last byte by a model download takes the
    /// host's own writes down with it — the manifest, the receipt, a log. It is
    /// a settings key precisely so a device with a different idea of "nearly
    /// full" does not need a new AIKit.
    public static let defaultFreeSpaceMarginBytes: Int64 = 512 * 1024 * 1024

    /// Whether a download may start on an expensive or constrained path with no
    /// per-download consent.
    ///
    /// `false` by default: BR-AIL-23's "never started silently".
    public var allowExpensiveNetwork: Bool

    /// Bytes to keep free beyond the download itself.
    public var freeSpaceMarginBytes: Int64

    public init(
        allowExpensiveNetwork: Bool = false,
        freeSpaceMarginBytes: Int64 = Self.defaultFreeSpaceMarginBytes
    ) {
        self.allowExpensiveNetwork = allowExpensiveNetwork
        self.freeSpaceMarginBytes = freeSpaceMarginBytes
    }

    // MARK: - Codable

    enum CodingKeys: String, CodingKey {
        case allowExpensiveNetwork = "allow_expensive_network"
        case freeSpaceMarginBytes = "free_space_margin_bytes"
    }

    /// Decodes the table, each key independently optional.
    ///
    /// A key the operator did not write takes its default. A key they DID
    /// write is read strictly: a margin below zero is refused rather than
    /// clamped, because a clamp would silently run a policy nobody authored.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        allowExpensiveNetwork =
            try container.decodeIfPresent(Bool.self, forKey: .allowExpensiveNetwork) ?? false
        let margin =
            try container.decodeIfPresent(Int64.self, forKey: .freeSpaceMarginBytes)
            ?? Self.defaultFreeSpaceMarginBytes
        guard margin >= 0 else {
            throw DecodingError.dataCorruptedError(
                forKey: .freeSpaceMarginBytes,
                in: container,
                debugDescription: "a free-space margin cannot be negative; got \(margin)"
            )
        }
        freeSpaceMarginBytes = margin
    }

    // MARK: - Loading

    /// The settings document, as the host chose to place it.
    ///
    /// A MISSING file is the default policy and not an error: the download
    /// policy is optional in a way a provider list is not — a host with no
    /// settings file still has providers injected, and still downloads models.
    /// A file that exists and cannot be read IS an error, for the same reason
    /// ``AIProviderSettings/load(from:)`` treats it as one: an unreadable file
    /// the operator wrote is a configuration that is not being honoured.
    public static func load(from url: URL) throws(AIKitError) -> ModelDownloadSettings {
        guard FileManager.default.fileExists(atPath: url.path) else {
            return ModelDownloadSettings()
        }
        let text: String
        do {
            text = try String(contentsOf: url, encoding: .utf8)
        } catch {
            throw .settings(
                file: url.path,
                entry: Self.tableName,
                reason: "it could not be read: \(error.localizedDescription)"
            )
        }
        return try parse(toml: text, file: url.path)
    }

    /// Reads the `[models.download]` table out of a TOML document.
    ///
    /// - Parameters:
    ///   - toml: the whole document, `[[providers]]` rows and all. Anything
    ///     that is not this table is ignored here and read by its own type.
    ///   - file: the path to NAME in an error, passed rather than derived so a
    ///     caller holding only the text still produces something an operator
    ///     can act on.
    public static func parse(
        toml: String, file: String
    ) throws(AIKitError) -> ModelDownloadSettings {
        do {
            let document = try TOMLDecoder().decode(ModelDownloadDocument.self, from: toml)
            return document.models?.download ?? ModelDownloadSettings()
        } catch let error as TOMLParseError {
            throw .settings(file: file, entry: nil, reason: "it is not valid TOML: \(error)")
        } catch {
            throw .settings(file: file, entry: Self.tableName, reason: "\(error)")
        }
    }

    /// The table's name, for an error a human reads.
    static let tableName = "models.download"
}

// MARK: - Reaching the table

/// Just enough of the settings document to reach `[models.download]`.
///
/// File-private and outside ``ModelDownloadSettings`` rather than nested inside
/// it: the nesting would be three levels deep, and the two shapes here exist
/// only to spell the table's path for the decoder.
private struct ModelDownloadDocument: Decodable {

    /// The `[models]` table.
    struct Models: Decodable {
        let download: ModelDownloadSettings?
    }

    let models: Models?
}

// MARK: - Consent

/// A per-download override of ``ModelDownloadSettings/allowExpensiveNetwork``
/// (BR-AIL-23: *"a per-download consent that overrides it once"*).
///
/// One value covers both consent reasons, because it answers one question the
/// operator was asked once — *spend on this path for this model?* — and a
/// consent that distinguished cellular from Low Data Mode would be two
/// questions for one tap.
public enum ModelDownloadConsent: String, Sendable, Equatable, CaseIterable, Codable {

    /// Nothing was granted; the setting decides. The default everywhere.
    case none

    /// The operator allowed THIS download on an expensive or constrained path.
    case allowExpensiveNetwork
}

/// Why a download is waiting for the operator.
public enum ModelDownloadConsentReason: String, Sendable, Equatable, CaseIterable, Codable {

    /// The path is metered (cellular, hotspot).
    case expensiveNetwork

    /// The path is in Low Data Mode.
    case constrainedNetwork

    /// One line an interface can show as it asks.
    public var question: String {
        switch self {
        case .expensiveNetwork:
            "This network is metered. Download anyway?"
        case .constrainedNetwork:
            "Low Data Mode is on for this network. Download anyway?"
        }
    }
}

// MARK: - Refusal

/// Why a download will not be started at all.
///
/// One case today, and a case rather than a sentence for the reason every other
/// refusal in this kit is one: an interface has to show the two numbers, and
/// parsing them back out of prose is how a message becomes an API nobody
/// declared.
public enum ModelDownloadRefusal: Sendable, Equatable {

    /// The volume cannot hold the download and still keep the margin.
    ///
    /// - Parameters:
    ///   - needed: bytes still to fetch PLUS the margin — what must be free for
    ///     this download to be safe, not just possible.
    ///   - free: what the volume reported available for important usage.
    case insufficientSpace(needed: Int64, free: Int64)
}

// MARK: - Decision

/// What to do with a planned download, before a byte is fetched (BR-AIL-23).
public enum ModelDownloadDecision: Sendable, Equatable {

    /// Start it.
    case proceed

    /// Do not start it, and say why.
    case refuse(ModelDownloadRefusal)

    /// Ask the operator first.
    case needsConsent(ModelDownloadConsentReason)
}

// MARK: - The plan

/// Everything known about a download BEFORE it starts: what is left to fetch,
/// what the volume has, and what the network costs.
///
/// W33 of spec ai-layer-aikit-core-2026-10-31, BR-AIL-23. What this replaces is
/// measured: `DefaultModelRegistry.download` and `MLXHubDownloader.download`
/// both began transferring immediately. The Hugging Face tree has published
/// every file's `size` (and `lfs.size`) since W13 — ``ModelStore`` reads it to
/// verify digests — and nothing compared that total with the free space on the
/// volume it was about to be written to. On a phone, nothing looked at the
/// network either: a 4 GB model started downloading over cellular exactly as
/// it would over Wi-Fi.
///
/// A plan is a VALUE, with no side effects and nothing to tear down. That is
/// deliberate: an interface can hold one and render it (*"needs 4.1 GB, 2.3 GB
/// free"*) without having begun anything, which is the whole point of planning
/// first, and a test can state one directly instead of arranging a disk.
public struct ModelDownloadPlan: Sendable, Equatable {

    /// The repository this plan is for, named in every error it throws.
    public let repositoryID: String

    /// The files that still have to be fetched, in the order they were given.
    public let pending: [ModelFile]

    /// The files that are already held and verified, so they are not counted.
    ///
    /// Kept rather than discarded: *"4.1 GB, of which 3.8 GB you already have"*
    /// is a different sentence from *"300 MB"*, and only one of them explains
    /// why a 4 GB model is a short download.
    public let held: Set<String>

    /// Bytes to fetch: the sum of ``pending``, verified files subtracted.
    public let bytesToFetch: Int64

    /// Bytes the destination volume reports available for important usage, or
    /// `nil` when the volume could not be asked.
    public let bytesFree: Int64?

    /// Bytes to keep free beyond the download itself.
    public let marginBytes: Int64

    /// What the network path costs, as far as this platform can observe.
    public let network: ModelDownloadNetwork

    /// Whether settings already allow an expensive or constrained path.
    public let expensiveNetworkAllowed: Bool

    public init(
        repositoryID: String,
        pending: [ModelFile],
        held: Set<String>,
        bytesToFetch: Int64,
        bytesFree: Int64?,
        marginBytes: Int64,
        network: ModelDownloadNetwork,
        expensiveNetworkAllowed: Bool
    ) {
        self.repositoryID = repositoryID
        self.pending = pending
        self.held = held
        self.bytesToFetch = bytesToFetch
        self.bytesFree = bytesFree
        self.marginBytes = marginBytes
        self.network = network
        self.expensiveNetworkAllowed = expensiveNetworkAllowed
    }

    // MARK: - Reading

    /// What must be free for this download to be safe: the bytes to fetch plus
    /// the margin.
    ///
    /// Zero when there is nothing to fetch — a margin is a reserve for a
    /// write, and no write is about to happen.
    public var bytesNeeded: Int64 {
        bytesToFetch > 0 ? bytesToFetch + marginBytes : 0
    }

    /// Whether the volume can hold it, or `nil` when the volume did not answer.
    ///
    /// `nil` is not `false`. Three states, because two would force a guess:
    /// see ``decision``.
    public var fits: Bool? {
        bytesFree.map { $0 >= bytesNeeded }
    }

    /// Whether anything has to be fetched at all.
    public var isEmpty: Bool { bytesToFetch <= 0 }

    // MARK: - The decision

    /// Whether this download may start, must be refused, or needs an answer.
    ///
    /// The order is space, then network, and it matters: a download that cannot
    /// fit must not first ask the operator whether to spend on cellular for it.
    ///
    /// Two judgment calls are recorded here rather than left implicit.
    ///
    /// **An unreadable volume proceeds.** `bytesFree == nil` means the volume
    /// did not answer — an unusual filesystem, a platform whose Foundation
    /// does not carry the key. Refusing every download on such a volume is a
    /// worse failure than the one being prevented: the download would have
    /// worked, and the operator has no setting to overrule it with. It
    /// proceeds, and ``bytesFree`` stays `nil` so an interface says *"free
    /// space unknown"* instead of inventing a number. The cost is the original
    /// failure mode on that volume alone: a transfer that fills a disk and
    /// fails at the filesystem.
    ///
    /// **An unobservable network proceeds.** ``ModelDownloadNetwork/unknown``
    /// is Linux and Windows, where there is no `Network` framework. Nothing is
    /// known to be metered there, and the alternative — refusing every download
    /// on a Linux host because its path cannot be classified — would stop CI
    /// and every server deployment for a protection aimed at a phone. So the
    /// setting is the only input left, and with nothing observed to override it
    /// the download runs. This is the reading of BR-AIL-23's *"it reports
    /// `unknown` and the decision falls to the setting"*, stated plainly here
    /// so it can be argued with.
    public var decision: ModelDownloadDecision {
        guard !isEmpty else { return .proceed }

        if let free = bytesFree, free < bytesNeeded {
            return .refuse(.insufficientSpace(needed: bytesNeeded, free: free))
        }

        guard !expensiveNetworkAllowed, let reason = network.consentReason else {
            return .proceed
        }
        return .needsConsent(reason)
    }

    /// Throws unless this download may start now.
    ///
    /// - Parameter consent: an answer the operator gave for THIS download,
    ///   which overrides the setting once and never persists.
    /// - Throws: ``AIKitError/insufficientSpace(model:neededBytes:freeBytes:)``
    ///   carrying both figures, or
    ///   ``AIKitError/downloadNeedsConsent(model:reason:network:)`` naming what
    ///   has to be answered. Typed cases, not sentences, so an interface can
    ///   render the numbers (BR-AIL-23).
    public func authorize(consent: ModelDownloadConsent = .none) throws(AIKitError) {
        switch decision {
        case .proceed:
            return

        case .refuse(.insufficientSpace(let needed, let free)):
            throw .insufficientSpace(
                model: repositoryID, neededBytes: needed, freeBytes: free)

        case .needsConsent(let reason):
            guard consent == .allowExpensiveNetwork else {
                throw .downloadNeedsConsent(
                    model: repositoryID, reason: reason, network: network)
            }
        }
    }

    /// One line for a log or a CLI, with the figures a human needs.
    public var summary: String {
        let fetch = DownloadProgress.formatBytes(bytesToFetch)
        let free = bytesFree.map(DownloadProgress.formatBytes) ?? "unknown"
        let heldNote = held.isEmpty ? "" : ", \(held.count) file(s) already verified"
        return
            "\(repositoryID): \(fetch) to fetch\(heldNote), \(free) free, network \(network.rawValue)"
    }
}

// MARK: - The planner

/// Builds a ``ModelDownloadPlan`` by asking the volume and the network path.
///
/// The two probes are injected closures rather than protocols, for the reason
/// `MLXHubDownloader`'s index and transfer are: both have exactly one
/// production implementation, both are a single call, and a test needs to state
/// an answer rather than conform a type. The defaults are the real ones, so a
/// host that configures nothing still gets a measured plan, and the suite that
/// exercises the defaults exercises the shipped code (BR-EC-02).
public struct ModelDownloadPlanner: Sendable {

    /// Bytes available for important usage on the volume holding a directory,
    /// or `nil` when it could not be asked.
    public typealias FreeSpaceProbe = @Sendable (URL) -> Int64?

    /// What the current network path costs.
    public typealias NetworkProbe = @Sendable () async -> ModelDownloadNetwork

    /// The policy this planner applies.
    public let settings: ModelDownloadSettings

    private let freeSpace: FreeSpaceProbe
    private let networkPath: NetworkProbe

    /// - Parameters:
    ///   - settings: the `[models.download]` policy. Defaults to the shipped
    ///     one — expensive paths need consent, 512 MiB margin.
    ///   - freeSpace: how to ask a volume. Defaults to
    ///     ``availableCapacity(at:)``.
    ///   - networkPath: how to classify the path. Defaults to
    ///     ``observedNetwork(timeout:)``.
    public init(
        settings: ModelDownloadSettings = ModelDownloadSettings(),
        freeSpace: FreeSpaceProbe? = nil,
        networkPath: NetworkProbe? = nil
    ) {
        self.settings = settings
        self.freeSpace = freeSpace ?? { Self.availableCapacity(at: $0) }
        self.networkPath = networkPath ?? { await Self.observedNetwork() }
    }

    /// Plan a fetch of `files` into `directory`.
    ///
    /// - Parameters:
    ///   - repository: the repository id, for the plan's errors.
    ///   - files: every file the download covers, as the hub published them.
    ///   - alreadyHeld: repository-relative paths the caller has already
    ///     verified. Supplied by the caller because it is the caller that holds
    ///     the store (``ModelStore/heldPaths(among:ofRepository:)``), and
    ///     subtracted HERE so the arithmetic has one home.
    ///   - directory: where the bytes will be written. Its volume is the one
    ///     asked for free space, which is why this is the resolved destination
    ///     and not a root a planner guessed.
    public func plan(
        repository: String,
        files: [ModelFile],
        alreadyHeld: Set<String> = [],
        into directory: URL
    ) async -> ModelDownloadPlan {
        let pending = files.filter { !alreadyHeld.contains($0.path) }
        return ModelDownloadPlan(
            repositoryID: repository,
            pending: pending,
            held: alreadyHeld.intersection(files.map(\.path)),
            bytesToFetch: pending.reduce(0) { $0 + max($1.sizeBytes, 0) },
            bytesFree: freeSpace(directory),
            marginBytes: settings.freeSpaceMarginBytes,
            network: await networkPath(),
            expensiveNetworkAllowed: settings.allowExpensiveNetwork
        )
    }

    // MARK: - The real probes

    /// Bytes the volume holding `url` reports available for important usage.
    ///
    /// `volumeAvailableCapacityForImportantUsage` and not
    /// `volumeAvailableCapacity`: the former is what the system will actually
    /// free up for a user-visible download (it counts purgeable caches), and
    /// the latter under-reports on a Mac with a large purgeable pool, which
    /// would refuse downloads that fit.
    ///
    /// The URL need not exist yet — a model's directory is created by the
    /// write, not by resolution (CoreKit BR-RP-01) — so the nearest existing
    /// ancestor is asked instead. It is on the same volume by construction.
    ///
    /// On a platform whose Foundation does not carry the important-usage key,
    /// the plain available capacity is read instead. `nil` means the volume did
    /// not answer at all; see ``ModelDownloadPlan/decision``.
    public static func availableCapacity(at url: URL) -> Int64? {
        let probe = nearestExistingAncestor(of: url)
        #if canImport(Darwin)
        let values = try? probe.resourceValues(
            forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        return values?.volumeAvailableCapacityForImportantUsage
        #else
        let values = try? probe.resourceValues(forKeys: [.volumeAvailableCapacityKey])
        return values?.volumeAvailableCapacity.map(Int64.init)
        #endif
    }

    /// The deepest existing directory at or above `url`.
    ///
    /// Walks with `deletingLastPathComponent`, bounded by the number of
    /// components rather than by `path != "/"`: the root's spelling is not the
    /// same on every platform, and a loop that waits for one spelling is a
    /// hang on the others.
    static func nearestExistingAncestor(of url: URL) -> URL {
        var candidate = url.standardizedFileURL
        for _ in 0...candidate.pathComponents.count {
            if FileManager.default.fileExists(atPath: candidate.path) { return candidate }
            let parent = candidate.deletingLastPathComponent().standardizedFileURL
            if parent == candidate { break }
            candidate = parent
        }
        return candidate
    }

    /// The current network path, classified.
    ///
    /// `NWPathMonitor` reports through a handler, so this waits for the first
    /// report and then stops monitoring. `currentPath` is NOT read instead:
    /// Apple's documentation does not say what it holds before `start(queue:)`,
    /// and a value whose meaning is unstated is not evidence.
    ///
    /// The wait is bounded. A monitor that never reports would otherwise hang
    /// a download that was only trying to be careful, and the honest answer in
    /// that case is ``ModelDownloadNetwork/unknown`` — which proceeds — not a
    /// stall. `timeout` is a parameter so a test can state it; nothing in
    /// production passes one.
    ///
    /// Off Apple platforms there is no `Network` framework, and this is
    /// ``ModelDownloadNetwork/unknown`` at compile time.
    public static func observedNetwork(
        timeout: Duration = .seconds(2)
    ) async -> ModelDownloadNetwork {
        #if canImport(Network)
        let monitor = NWPathMonitor()
        let reports = AsyncStream<ModelDownloadNetwork> { continuation in
            monitor.pathUpdateHandler = { path in
                continuation.yield(
                    .observed(isExpensive: path.isExpensive, isConstrained: path.isConstrained))
            }
            // The monitor is cancelled when the stream ends, including when the
            // task below is cancelled by the deadline: the observer's lifetime
            // is the stream's, so there is no path on which it outlives this
            // call.
            continuation.onTermination = { _ in monitor.cancel() }
            monitor.start(queue: DispatchQueue(label: "aikit.model-download.network-path"))
        }

        return await withTaskGroup(of: ModelDownloadNetwork?.self) { group in
            group.addTask {
                for await report in reports { return report }
                return nil
            }
            group.addTask {
                try? await Task.sleep(for: timeout)
                return nil
            }
            // One level of optional at a time: `next()` answers
            // `ModelDownloadNetwork??`, where the outer `nil` means "no child
            // finished" and the inner one means "the child that finished had
            // no answer". Both end at `.unknown`, and collapsing them with a
            // coalescing operator reads as if there were only one.
            var answer: ModelDownloadNetwork?
            if let reported = await group.next() {
                answer = reported
            }
            group.cancelAll()
            return answer ?? .unknown
        }
        #else
        return .unknown
        #endif
    }
}
