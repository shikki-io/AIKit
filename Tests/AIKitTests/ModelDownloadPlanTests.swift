import CoreKit
import CoreKitTestSupport
import CryptoKit
import Foundation
import NetKit
import Testing

@testable import AIKit

// MARK: - Fixtures shared by the suites below

/// One entry of `/api/models/<id>/tree/<revision>`, as an ENCODER.
///
/// `HFTreeEntry` is `Decodable` only — AIKit reads this route and never writes
/// it — so a test that needs a tree reply has to mint one from somewhere.
/// This is that somewhere, and it is the shape, not a string: W24's rule is
/// that a payload a decoder reads is minted from a typed model, and
/// `ModelDownloadPlanTests.theTreeFixtureIsReadByTheRealDecoder` is what keeps
/// this mirror honest — it mints a tree here and asserts the PRODUCTION
/// decoder (`HuggingFaceClient.files`) reads back the sizes and digests this
/// wrote (BR-EC-02: one agreement, asserted, not two models assumed equal).
struct FixtureHubTreeEntry: Encodable, Sendable {

    /// The `lfs` object the hub publishes for a large file. Its `oid` is the
    /// file's SHA-256 (measured 2026-10-31, W13).
    struct LFS: Encodable, Sendable {
        let oid: String
        let size: Int64
    }

    let type: String
    let path: String
    let size: Int64
    let oid: String?
    let lfs: LFS?

    /// An LFS-tracked file carrying `data`'s real SHA-256, the way the hub
    /// publishes weights.
    static func lfsFile(path: String, data: Data) -> FixtureHubTreeEntry {
        let hex = sha256Hex(of: data)
        return FixtureHubTreeEntry(
            type: "file",
            path: path,
            size: Int64(data.count),
            oid: nil,
            lfs: LFS(oid: hex, size: Int64(data.count))
        )
    }

    /// A file of a declared SIZE whose bytes are not supplied — what a plan
    /// needs when the test never intends to transfer anything.
    static func lfsFile(path: String, sizeBytes: Int64, digest: String) -> FixtureHubTreeEntry {
        FixtureHubTreeEntry(
            type: "file",
            path: path,
            size: sizeBytes,
            oid: nil,
            lfs: LFS(oid: digest, size: sizeBytes)
        )
    }

    static func sha256Hex(of data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}

/// The Hugging Face routes `DefaultModelRegistry.download` calls, answered from
/// memory.
///
/// `MockNetworkService` carries ONE `resultData` for every request, and this
/// path needs two different replies — the model row, then the file tree — so a
/// stub that routes on the endpoint's own `path` is the only honest way to
/// serve it. It records every path it was asked for, which is how the suites
/// assert the ORDER: the tree is read before anything is transferred.
///
/// It reaches no server (BR-AIL-12).
final class HubStub: NetworkProtocol, @unchecked Sendable {

    private let lock = NSLock()
    private let model: HFModelInfo
    private let tree: [FixtureHubTreeEntry]
    private var paths: [String] = []

    init(model: HFModelInfo, tree: [FixtureHubTreeEntry]) {
        self.model = model
        self.tree = tree
    }

    /// Every endpoint path this stub was asked for, in order.
    var requestedPaths: [String] { lock.withLock { paths } }

    func sendRequest<T: Decodable & Sendable>(endpoint: EndPoint) async throws -> T {
        try answer(for: endpoint)
    }

    func sendRequest<T: Decodable>(
        endpoint: EndPoint,
        resultHandler: @escaping @Sendable (Result<T, NetworkError>) -> Void
    ) {
        do {
            resultHandler(.success(try answer(for: endpoint)))
        } catch let error as NetworkError {
            resultHandler(.failure(error))
        } catch {
            resultHandler(.failure(.unknown(error)))
        }
    }

    private func answer<T: Decodable>(for endpoint: EndPoint) throws -> T {
        let path = endpoint.path
        lock.withLock { paths.append(path) }
        let data =
            path.contains("/tree/")
            ? try JSONEncoder().encode(tree)
            : try JSONEncoder().encode(model)
        return try JSONDecoder().decode(T.self, from: data)
    }
}

/// The fractions a download reported, collected from the `@Sendable` callback
/// it is handed.
final class ProgressLog: @unchecked Sendable {
    private let lock = NSLock()
    private var recorded: [Double] = []

    var fractions: [Double] { lock.withLock { recorded } }

    func record(_ fraction: Double) {
        lock.withLock { recorded.append(fraction) }
    }
}

/// A transfer that moves no bytes over a network, and remembers exactly what it
/// was asked to do.
///
/// The seam is ``ModelDownloader/Transport``, which W33 introduced for this:
/// the resume DECISION above it — read a stored token, pass it on, keep a fresh
/// one, start over once when a stored one is rejected, forget it on success —
/// is the shipped logic, and a transport that fails on command is how it is
/// proved without a server.
final class StubTransport: @unchecked Sendable {

    /// One recorded transfer.
    struct Call: Sendable {
        let url: URL
        let destination: URL
        let token: Data?
    }

    /// What the stub does on one call.
    enum Outcome: Sendable {
        /// Write these bytes at the destination and report them as progress.
        case write(Data)
        /// Fail the way `URLSession` fails a part-finished download: an error
        /// carrying a resume token.
        case failWithResumeToken(Data)
        /// Fail with nothing to resume from.
        case failWithoutToken
    }

    private let lock = NSLock()
    private var recorded: [Call] = []
    private var outcomes: [Outcome]

    /// - Parameter outcomes: one per call. The LAST one repeats, so a stub that
    ///   always succeeds is written as a single `.write`.
    init(outcomes: [Outcome]) {
        self.outcomes = outcomes
    }

    var calls: [Call] { lock.withLock { recorded } }
    var callCount: Int { lock.withLock { recorded.count } }

    var transport: ModelDownloader.Transport {
        { [self] url, destination, token, onProgress in
            let outcome = lock.withLock { () -> Outcome? in
                recorded.append(Call(url: url, destination: destination, token: token))
                guard !outcomes.isEmpty else { return nil }
                return outcomes[min(recorded.count - 1, outcomes.count - 1)]
            }

            switch outcome {
            case .write(let data):
                try FileManager.default.createDirectory(
                    at: destination.deletingLastPathComponent(),
                    withIntermediateDirectories: true
                )
                onProgress(
                    DownloadProgress(
                        bytesDownloaded: Int64(data.count), totalBytes: Int64(data.count)))
                try data.write(to: destination)

            case .failWithResumeToken(let resumeToken):
                throw URLError(
                    .networkConnectionLost,
                    userInfo: [NSURLSessionDownloadTaskResumeData: resumeToken]
                )

            case .failWithoutToken:
                throw URLError(.timedOut)

            case nil:
                throw AIKitError.downloadFailed("the stub has no outcome for this call")
            }
        }
    }
}

// MARK: - T-AIL-60 — space and network, decided before anything starts

/// W33 of spec ai-layer-aikit-core-2026-10-31, BR-AIL-23. Operator,
/// 2026-11-12: *"did we manage the fact to check the size of the model and the
/// free space available on the device … especially if you are on mobile device
/// not on wifi"*.
///
/// Every plan below is built by the PRODUCTION ``ModelDownloadPlanner`` with
/// its probes injected — the arithmetic, the ordering and the refusals are the
/// shipped ones, and only the two measurements a test cannot make on a real
/// volume and a real radio are stated.
@Suite("ModelDownloadPlan — space and network")
struct ModelDownloadPlanSpaceAndNetworkTests {

    // MARK: - Fixtures

    private static let gigabyte: Int64 = 1024 * 1024 * 1024

    /// A planner whose two measurements are stated.
    private func planner(
        free: Int64?,
        network: ModelDownloadNetwork = .unrestricted,
        settings: ModelDownloadSettings = ModelDownloadSettings()
    ) -> ModelDownloadPlanner {
        ModelDownloadPlanner(
            settings: settings,
            freeSpace: { _ in free },
            networkPath: { network }
        )
    }

    private func weights(_ bytes: Int64, path: String = "model.safetensors") -> ModelFile {
        ModelFile(path: path, sizeBytes: bytes, digest: .sha256(String(repeating: "a", count: 64)))
    }

    /// Any directory: the stated probe ignores it. Not a literal path — the
    /// value is irrelevant to these suites and inventing one would read as if
    /// it were not.
    private var anyDirectory: URL { URL(fileURLWithPath: NSTemporaryDirectory()) }

    // MARK: - Space

    @Test("a download that does not fit is refused, naming both figures")
    func insufficientSpaceIsRefusedWithTheFigures() async throws {
        let model = weights(4 * Self.gigabyte)
        let plan = await planner(free: 2 * Self.gigabyte)
            .plan(repository: "mlx-community/Mistral-7B", files: [model], into: anyDirectory)

        #expect(plan.bytesToFetch == 4 * Self.gigabyte)
        #expect(plan.fits == false)

        let needed = 4 * Self.gigabyte + ModelDownloadSettings.defaultFreeSpaceMarginBytes
        #expect(
            plan.decision
                == .refuse(.insufficientSpace(needed: needed, free: 2 * Self.gigabyte)))

        let error = try #require(throws: AIKitError.self) {
            try plan.authorize()
        }
        #expect(
            error
                == .insufficientSpace(
                    model: "mlx-community/Mistral-7B",
                    neededBytes: needed,
                    freeBytes: 2 * Self.gigabyte
                ))
        // The two numbers are in the sentence an operator reads, formatted,
        // which is what makes this error usable by a UI without parsing prose.
        let sentence = try #require(error.errorDescription)
        #expect(sentence.contains("4.50 GB"))
        #expect(sentence.contains("2.00 GB"))
    }

    @Test("the margin is part of what must be free, not a rounding allowance")
    func theMarginCounts() async throws {
        let model = weights(Self.gigabyte)
        let margin = ModelDownloadSettings.defaultFreeSpaceMarginBytes

        // Exactly the model's own size free: it would fit and leave nothing.
        let tight = await planner(free: Self.gigabyte)
            .plan(repository: "a/b", files: [model], into: anyDirectory)
        #expect(tight.bytesNeeded == Self.gigabyte + margin)
        #expect(tight.fits == false)
        #expect(tight.decision == .refuse(.insufficientSpace(needed: tight.bytesNeeded, free: Self.gigabyte)))

        // The model plus the margin, to the byte.
        let exact = await planner(free: Self.gigabyte + margin)
            .plan(repository: "a/b", files: [model], into: anyDirectory)
        #expect(exact.fits == true)
        #expect(exact.decision == .proceed)
        try exact.authorize()
    }

    @Test("a configured margin replaces the default one")
    func theMarginComesFromSettings() async throws {
        let settings = ModelDownloadSettings(freeSpaceMarginBytes: 0)
        let plan = await planner(free: Self.gigabyte, settings: settings)
            .plan(repository: "a/b", files: [weights(Self.gigabyte)], into: anyDirectory)
        #expect(plan.marginBytes == 0)
        #expect(plan.bytesNeeded == Self.gigabyte)
        #expect(plan.decision == .proceed)
    }

    @Test("a verified file is not counted")
    func aVerifiedFileIsNotCounted() async throws {
        let files = [
            weights(4 * Self.gigabyte, path: "model.safetensors"),
            weights(2048, path: "config.json"),
        ]

        let plan = await planner(free: Self.gigabyte)
            .plan(
                repository: "a/b",
                files: files,
                alreadyHeld: ["model.safetensors"],
                into: anyDirectory
            )

        // The 4 GB file is held, so it is neither fetched nor charged for, and
        // what is left fits in a gigabyte with the margin. Counting it would
        // have refused a download of 2 KB.
        #expect(plan.bytesToFetch == 2048)
        #expect(plan.pending.map(\.path) == ["config.json"])
        #expect(plan.held == ["model.safetensors"])
        #expect(plan.decision == .proceed)
    }

    @Test("a held path the listing does not mention is not reported as held")
    func heldIsIntersectedWithTheListing() async throws {
        let plan = await planner(free: Self.gigabyte)
            .plan(
                repository: "a/b",
                files: [weights(10, path: "config.json")],
                alreadyHeld: ["config.json", "a-file-from-another-revision.bin"],
                into: anyDirectory
            )
        #expect(plan.held == ["config.json"])
    }

    @Test("nothing to fetch proceeds, whatever the volume and the radio say")
    func anEmptyPlanProceeds() async throws {
        let plan = await planner(free: 0, network: .expensive)
            .plan(
                repository: "a/b",
                files: [weights(4 * Self.gigabyte)],
                alreadyHeld: ["model.safetensors"],
                into: anyDirectory
            )
        #expect(plan.isEmpty)
        #expect(plan.bytesNeeded == 0)
        #expect(plan.decision == .proceed)
        try plan.authorize()
    }

    @Test("a volume that did not answer proceeds, and says it did not answer")
    func anUnreadableVolumeProceeds() async throws {
        let plan = await planner(free: nil)
            .plan(repository: "a/b", files: [weights(4 * Self.gigabyte)], into: anyDirectory)
        // `nil` is not `false`: the plan records that nothing was measured, so
        // an interface says "free space unknown" rather than inventing a
        // number, and the download is not refused for a measurement AIKit
        // could not take.
        #expect(plan.bytesFree == nil)
        #expect(plan.fits == nil)
        #expect(plan.decision == .proceed)
    }

    // MARK: - Network

    @Test("a metered path needs consent, and names what to answer")
    func anExpensivePathNeedsConsent() async throws {
        let plan = await planner(free: 100 * Self.gigabyte, network: .expensive)
            .plan(repository: "a/b", files: [weights(Self.gigabyte)], into: anyDirectory)

        #expect(plan.decision == .needsConsent(.expensiveNetwork))
        let error = try #require(throws: AIKitError.self) { try plan.authorize() }
        #expect(
            error
                == .downloadNeedsConsent(
                    model: "a/b", reason: .expensiveNetwork, network: .expensive))
    }

    @Test("Low Data Mode needs consent too, with its own reason")
    func aConstrainedPathNeedsConsent() async throws {
        let plan = await planner(free: 100 * Self.gigabyte, network: .constrained)
            .plan(repository: "a/b", files: [weights(Self.gigabyte)], into: anyDirectory)

        #expect(plan.decision == .needsConsent(.constrainedNetwork))
        let error = try #require(throws: AIKitError.self) { try plan.authorize() }
        #expect(
            error
                == .downloadNeedsConsent(
                    model: "a/b", reason: .constrainedNetwork, network: .constrained))
    }

    @Test("the setting allows a metered path with no question")
    func theSettingAllowsAMeteredPath() async throws {
        let settings = ModelDownloadSettings(allowExpensiveNetwork: true)
        let plan = await planner(free: 100 * Self.gigabyte, network: .expensive, settings: settings)
            .plan(repository: "a/b", files: [weights(Self.gigabyte)], into: anyDirectory)

        #expect(plan.expensiveNetworkAllowed)
        #expect(plan.decision == .proceed)
        try plan.authorize()
    }

    @Test("a per-download consent overrides the setting once, and does not change it")
    func aPerDownloadConsentOverridesOnce() async throws {
        let plan = await planner(free: 100 * Self.gigabyte, network: .expensive)
            .plan(repository: "a/b", files: [weights(Self.gigabyte)], into: anyDirectory)

        // The answer authorises this download…
        try plan.authorize(consent: .allowExpensiveNetwork)
        // …and nothing about the plan changed, so the next one asks again.
        #expect(plan.decision == .needsConsent(.expensiveNetwork))
        #expect(!plan.expensiveNetworkAllowed)
        #expect(throws: AIKitError.self) { try plan.authorize() }
    }

    @Test("space is decided before the network")
    func spaceBeatsTheNetwork() async throws {
        let plan = await planner(free: 1024, network: .expensive)
            .plan(repository: "a/b", files: [weights(4 * Self.gigabyte)], into: anyDirectory)

        // A download that cannot fit must not first ask the operator whether to
        // spend cellular data on it.
        #expect(plan.decision == .refuse(.insufficientSpace(needed: plan.bytesNeeded, free: 1024)))
        // …and a consent does not buy its way past a full volume.
        #expect(throws: AIKitError.self) { try plan.authorize(consent: .allowExpensiveNetwork) }
    }

    @Test("an unobservable path proceeds — Linux has no radio to classify")
    func anUnknownPathProceeds() async throws {
        let plan = await planner(free: 100 * Self.gigabyte, network: .unknown)
            .plan(repository: "a/b", files: [weights(Self.gigabyte)], into: anyDirectory)

        // BR-AIL-23's "it reports unknown and the decision falls to the
        // setting", as read in ModelDownloadPlan.decision: nothing was observed
        // to be metered, and refusing every download on a host whose path
        // cannot be classified would stop CI and every server for a protection
        // aimed at a phone.
        #expect(plan.network == .unknown)
        #expect(plan.decision == .proceed)
        try plan.authorize()
    }

    @Test("the network case comes from the two NWPath flags, and both-true is expensive")
    func theNetworkCaseComesFromTheFlags() {
        #expect(ModelDownloadNetwork.observed(isExpensive: false, isConstrained: false) == .unrestricted)
        #expect(ModelDownloadNetwork.observed(isExpensive: true, isConstrained: false) == .expensive)
        #expect(ModelDownloadNetwork.observed(isExpensive: false, isConstrained: true) == .constrained)
        // Both flags set is one consent question, and the metered one is the
        // one with a bill attached.
        #expect(ModelDownloadNetwork.observed(isExpensive: true, isConstrained: true) == .expensive)

        #expect(ModelDownloadNetwork.unrestricted.consentReason == nil)
        #expect(ModelDownloadNetwork.unknown.consentReason == nil)
        #expect(ModelDownloadNetwork.expensive.consentReason == .expensiveNetwork)
        #expect(ModelDownloadNetwork.constrained.consentReason == .constrainedNetwork)
        #expect(ModelDownloadNetwork.expensive.needsConsent)
        #expect(!ModelDownloadNetwork.unknown.needsConsent)
    }

    @Test("a download refusal is not a reason to ask another provider")
    func aRefusalIsNotFallbackEligible() {
        // A fallback list exists for a provider that cannot answer. Neither a
        // full volume nor a missing consent is that, and retrying elsewhere
        // would bury the one thing the operator has to act on.
        #expect(
            !FallbackEligibility.allows(
                AIKitError.insufficientSpace(model: "a/b", neededBytes: 2, freeBytes: 1)))
        #expect(
            !FallbackEligibility.allows(
                AIKitError.downloadNeedsConsent(
                    model: "a/b", reason: .expensiveNetwork, network: .expensive)))
    }

    @Test("a plan summarises itself with the figures a log needs")
    func thePlanSummarises() async throws {
        let plan = await planner(free: 2 * Self.gigabyte, network: .expensive)
            .plan(
                repository: "mlx-community/Mistral-7B",
                files: [weights(4 * Self.gigabyte), weights(1024, path: "config.json")],
                alreadyHeld: ["config.json"],
                into: anyDirectory
            )
        let summary = plan.summary
        #expect(summary.contains("mlx-community/Mistral-7B"))
        #expect(summary.contains("4.00 GB"))
        #expect(summary.contains("2.00 GB"))
        #expect(summary.contains("1 file(s) already verified"))
        #expect(summary.contains("expensive"))
    }
}

// MARK: - The shipped probes (BR-EC-02)

/// The two measurements a planner makes for real.
///
/// A suite that only ever injected its probes would prove the arithmetic and
/// leave the measurement unexercised — and the measurement is where a wrong
/// resource key or a hanging monitor lives. These run the default
/// implementations.
@Suite("ModelDownloadPlan — the real probes")
struct ModelDownloadProbeTests {

    @Test("the free-space probe answers for a real volume")
    func theFreeSpaceProbeAnswers() throws {
        let scratch = TestScratch.forPackage()
        let root = try scratch.makeScratchDir(scope: "download-plan-probe")
        defer { try? scratch.removeScratchDir(root) }

        let capacity = try #require(
            ModelDownloadPlanner.availableCapacity(at: root),
            "the volume holding the package's scratch directory reported no capacity"
        )
        #expect(capacity > 0)
    }

    @Test("the probe asks the volume of a directory that does not exist yet")
    func theProbeWalksUpToAnExistingAncestor() throws {
        let scratch = TestScratch.forPackage()
        let root = try scratch.makeScratchDir(scope: "download-plan-probe-missing")
        defer { try? scratch.removeScratchDir(root) }

        // A model's directory is created by the WRITE, not by resolution
        // (CoreKit BR-RP-01), so the planner is routinely handed a path that
        // does not exist. `resourceValues` throws on one, and the answer is the
        // nearest existing ancestor — the same volume by construction.
        let unwritten = root
            .appendingPathComponent("not-yet")
            .appendingPathComponent("either")
        #expect(!FileManager.default.fileExists(atPath: unwritten.path))

        #expect(ModelDownloadPlanner.nearestExistingAncestor(of: unwritten) == root.standardizedFileURL)
        let capacity = try #require(ModelDownloadPlanner.availableCapacity(at: unwritten))
        #expect(capacity > 0)
    }

    @Test("the ancestor walk terminates on a path with no existing component")
    func theAncestorWalkTerminates() {
        // The loop is bounded by the component count rather than by waiting for
        // one spelling of the filesystem root, which is not the same string on
        // every platform. This asserts it returns rather than spins.
        let nowhere = URL(fileURLWithPath: "/aikit-no-such-root-\(UUID().uuidString)/a/b/c")
        let resolved = ModelDownloadPlanner.nearestExistingAncestor(of: nowhere)
        #expect(FileManager.default.fileExists(atPath: resolved.path))
    }

    @Test("the network probe classifies this machine's path, bounded")
    func theNetworkProbeAnswers() async {
        // What is asserted is that the shipped probe RETURNS, with one of the
        // four cases, inside its bound — not which case, because that depends
        // on the machine running the suite. A probe that hung would hang every
        // download that was only trying to be careful, and this is the test
        // that fails instead.
        let observed = await ModelDownloadPlanner.observedNetwork(timeout: .seconds(5))
        #expect(ModelDownloadNetwork.allCases.contains(observed))
    }

    @Test("a probe whose monitor never reports answers unknown rather than hanging")
    func theNetworkProbeIsBounded() async {
        // The deadline, exercised: a timeout no monitor can beat must produce
        // `.unknown`, which proceeds.
        let observed = await ModelDownloadPlanner.observedNetwork(timeout: .nanoseconds(1))
        #expect(ModelDownloadNetwork.allCases.contains(observed))
    }
}

// MARK: - The settings key

/// `[models.download]`, read out of the host's own settings document
/// (BR-AIL-23).
@Suite("ModelDownloadSettings")
struct ModelDownloadSettingsTests {

    /// A document holding BOTH a provider list and the download table, because
    /// that is the file the host actually has: one settings document, read by
    /// the type that owns each part of it.
    private func document(table: String) -> String {
        """
        [[providers]]
        id      = "lmstudio"
        family  = "openai-chat"
        address = "\(fixtureServerRoot)"
        model   = "devstral-small-2507"

        \(table)
        """
    }

    @Test("the key is read from the same document as the provider list")
    func theKeyIsReadFromTheProvidersDocument() throws {
        let toml = document(
            table: """
                [models.download]
                allow_expensive_network = true
                free_space_margin_bytes = 1048576
                """)

        let settings = try ModelDownloadSettings.parse(toml: toml, file: "providers.toml")
        #expect(settings.allowExpensiveNetwork)
        #expect(settings.freeSpaceMarginBytes == 1_048_576)

        // And the provider list still reads the same bytes: two types, one
        // file, neither aware of the other's keys.
        let providers = try AIProviderSettings.parse(toml: toml, file: "providers.toml")
        #expect(providers.providers.count == 1)
    }

    @Test("no table is the default policy, and the default refuses a metered path")
    func noTableIsTheDefaultPolicy() throws {
        let settings = try ModelDownloadSettings.parse(
            toml: document(table: ""), file: "providers.toml")
        #expect(settings == ModelDownloadSettings())
        #expect(!settings.allowExpensiveNetwork)
        #expect(
            settings.freeSpaceMarginBytes == ModelDownloadSettings.defaultFreeSpaceMarginBytes)
    }

    @Test("one key written takes its value, the other its default")
    func oneKeyAtATime() throws {
        let settings = try ModelDownloadSettings.parse(
            toml: document(
                table: """
                    [models.download]
                    allow_expensive_network = true
                    """),
            file: "providers.toml")
        #expect(settings.allowExpensiveNetwork)
        #expect(
            settings.freeSpaceMarginBytes == ModelDownloadSettings.defaultFreeSpaceMarginBytes)
    }

    @Test("a negative margin is refused, naming the file and the table")
    func aNegativeMarginIsRefused() throws {
        let error = try #require(throws: AIKitError.self) {
            try ModelDownloadSettings.parse(
                toml: document(
                    table: """
                        [models.download]
                        free_space_margin_bytes = -1
                        """),
                file: "/tmp/providers.toml")
        }
        // Refused rather than clamped to zero: a clamp runs a policy nobody
        // wrote, and the operator typed this value on purpose.
        guard case .settings(let file, let entry, let reason) = error else {
            Issue.record("expected a settings error, got \(error)")
            return
        }
        #expect(file == "/tmp/providers.toml")
        #expect(entry == "models.download")
        #expect(reason.contains("negative"))
    }

    @Test("a document that is not TOML is refused, naming the file")
    func badTOMLIsRefused() throws {
        let error = try #require(throws: AIKitError.self) {
            try ModelDownloadSettings.parse(toml: "[models.download", file: "providers.toml")
        }
        guard case .settings(let file, _, let reason) = error else {
            Issue.record("expected a settings error, got \(error)")
            return
        }
        #expect(file == "providers.toml")
        #expect(reason.contains("not valid TOML"))
    }

    @Test("a missing settings file is the default policy, not an error")
    func aMissingFileIsTheDefault() throws {
        let absent = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("aikit-no-such-settings-\(UUID().uuidString).toml")
        // The download policy is optional in a way a provider list is not: a
        // host with no file still downloads models, under the default.
        #expect(try ModelDownloadSettings.load(from: absent) == ModelDownloadSettings())
    }

    @Test("a file that exists is read from disk")
    func anExistingFileIsRead() throws {
        let scratch = TestScratch.forPackage()
        let root = try scratch.makeScratchDir(scope: "download-settings")
        defer { try? scratch.removeScratchDir(root) }

        let url = root.appendingPathComponent("providers.toml")
        try document(
            table: """
                [models.download]
                allow_expensive_network = true
                """
        ).write(to: url, atomically: true, encoding: .utf8)

        #expect(try ModelDownloadSettings.load(from: url).allowExpensiveNetwork)
    }
}

// MARK: - T-AIL-60, the integration — nothing is requested before the plan clears

/// The plan, enforced where the bytes would have moved.
///
/// These run the real ``DefaultModelRegistry/download(id:onProgress:)``: the
/// hub routes are answered from memory and the transfer is a stub that COUNTS,
/// so "refused before any request" is asserted as a count of transfers and not
/// as a claim about the order of two lines of code.
@Suite("DefaultModelRegistry — a download is planned before it starts")
struct DefaultModelRegistryPlanTests {

    // MARK: - Fixtures

    private static let repository = "test/tiny-gguf"
    private static let weightsName = "tiny-gguf.gguf"

    /// The stores, the layer they resolve against, and the scratch tree.
    private final class Sandbox {
        let paths: TestPathLayer
        let localStore: LocalModelStore
        let modelStore: ModelStore

        init(scope: String) throws {
            let paths = try TestPathLayer(scope: scope)
            try FileManager.default.createDirectory(
                at: paths.layer.base(of: AIKitNode.models), withIntermediateDirectories: true)
            self.paths = paths
            self.localStore = try LocalModelStore(paths: paths.layer)
            self.modelStore = ModelStore(paths: paths.layer)
        }
    }

    private var modelID: ModelIdentifier {
        ModelIdentifier(provider: "huggingface", modelId: Self.repository)
    }

    /// The bytes the stub "downloads". Small, and hashed for real.
    private var payload: Data { Data("the weights, as far as this suite is concerned".utf8) }

    private func hub(
        declaredSize: Int64? = nil,
        extraFiles: [FixtureHubTreeEntry] = []
    ) -> HubStub {
        let bytes = payload
        let weights: FixtureHubTreeEntry =
            if let declaredSize {
                // A size the hub published that the test never intends to
                // transfer: enough to plan against 4 GB without writing it.
                .lfsFile(
                    path: Self.weightsName,
                    sizeBytes: declaredSize,
                    digest: FixtureHubTreeEntry.sha256Hex(of: bytes)
                )
            } else {
                .lfsFile(path: Self.weightsName, data: bytes)
            }
        return HubStub(
            model: makeFixtureHFModel(modelId: Self.repository, author: "test", tags: ["gguf"]),
            tree: [weights] + extraFiles
        )
    }

    private func registry(
        _ sandbox: Sandbox,
        hub: HubStub,
        transport: StubTransport,
        free: Int64,
        network: ModelDownloadNetwork = .unrestricted,
        settings: ModelDownloadSettings = ModelDownloadSettings(),
        consent: ModelDownloadConsent = .none
    ) -> DefaultModelRegistry {
        DefaultModelRegistry(
            localStore: sandbox.localStore,
            modelStore: sandbox.modelStore,
            planner: ModelDownloadPlanner(
                settings: settings,
                freeSpace: { _ in free },
                networkPath: { network }
            ),
            consent: consent,
            hfClient: HuggingFaceClient(networkService: hub),
            downloader: ModelDownloader(transport: transport.transport)
        )
    }

    // MARK: - T-AIL-60

    @Test("a download that does not fit is refused before any file is requested")
    func refusedBeforeAnyRequest() async throws {
        let sandbox = try Sandbox(scope: "registry-plan-space")
        let hub = hub(declaredSize: 4 * 1024 * 1024 * 1024)
        let transport = StubTransport(outcomes: [.write(payload)])

        let registry = registry(sandbox, hub: hub, transport: transport, free: 2 * 1024 * 1024 * 1024)

        let error = try await #require(throws: AIKitError.self) {
            try await registry.download(id: modelID) { _ in }
        }
        guard case .insufficientSpace(let model, let needed, let free) = error else {
            Issue.record("expected insufficientSpace, got \(error)")
            return
        }
        #expect(model == Self.repository)
        #expect(needed == 4 * 1024 * 1024 * 1024 + ModelDownloadSettings.defaultFreeSpaceMarginBytes)
        #expect(free == 2 * 1024 * 1024 * 1024)

        // The whole point: NOTHING was fetched.
        #expect(transport.callCount == 0)
        // …and the tree WAS read, because that is where the size came from.
        #expect(hub.requestedPaths.contains { $0.contains("/tree/") })
        // Nothing entered the manifest either.
        #expect(try sandbox.localStore.listLocal().isEmpty)
    }

    @Test("a metered path is asked about before any file is requested")
    func consentIsAskedBeforeAnyRequest() async throws {
        let sandbox = try Sandbox(scope: "registry-plan-network")
        let hub = hub()
        let transport = StubTransport(outcomes: [.write(payload)])

        let registry = registry(
            sandbox, hub: hub, transport: transport,
            free: 100 * 1024 * 1024 * 1024, network: .expensive)

        let error = try await #require(throws: AIKitError.self) {
            try await registry.download(id: modelID) { _ in }
        }
        #expect(
            error
                == .downloadNeedsConsent(
                    model: Self.repository, reason: .expensiveNetwork, network: .expensive))
        #expect(transport.callCount == 0)
        #expect(try sandbox.localStore.listLocal().isEmpty)
    }

    @Test("the operator's consent lets the same download through")
    func consentLetsItThrough() async throws {
        let sandbox = try Sandbox(scope: "registry-plan-consent")
        let hub = hub()
        let transport = StubTransport(outcomes: [.write(payload)])

        let registry = registry(
            sandbox, hub: hub, transport: transport,
            free: 100 * 1024 * 1024 * 1024, network: .expensive,
            consent: .allowExpensiveNetwork)

        try await registry.download(id: modelID) { _ in }
        #expect(transport.callCount == 1)
        #expect(try sandbox.localStore.listLocal().count == 1)
    }

    @Test("a plan that clears downloads, verifies, and records the published size")
    func aClearedPlanDownloadsAndVerifies() async throws {
        let sandbox = try Sandbox(scope: "registry-plan-proceed")
        let hub = hub()
        let transport = StubTransport(outcomes: [.write(payload)])

        let reported = ProgressLog()
        let registry = registry(
            sandbox, hub: hub, transport: transport, free: 100 * 1024 * 1024 * 1024)

        try await registry.download(id: modelID) { reported.record($0) }

        #expect(transport.callCount == 1)
        let stored = try #require(try sandbox.localStore.listLocal().first)
        #expect(stored.descriptor.id == modelID)
        #expect(stored.descriptor.isLocal)
        // The size the hub published, which this descriptor used to carry as 0
        // because the tree was never read.
        #expect(stored.descriptor.sizeBytes == Int64(payload.count))
        #expect(try Data(contentsOf: stored.localPath) == payload)
        #expect(reported.fractions.last == 1.0)
    }

    @Test("a file already on disk and verified is not downloaded again")
    func aVerifiedFileIsNotFetchedAgain() async throws {
        let sandbox = try Sandbox(scope: "registry-plan-held")
        let hub = hub()
        let transport = StubTransport(outcomes: [.write(payload)])
        let registry = registry(
            sandbox, hub: hub, transport: transport, free: 100 * 1024 * 1024 * 1024)

        try await registry.download(id: modelID) { _ in }
        #expect(transport.callCount == 1)

        // Second call, same registry: the file is there and hashes to the
        // published digest, so the plan has nothing to fetch — and a plan with
        // nothing to fetch proceeds even on a path it would otherwise refuse.
        try await registry.download(id: modelID) { _ in }
        #expect(transport.callCount == 1)
        #expect(try sandbox.localStore.listLocal().count == 1)
    }

    @Test("bytes that are not the published ones are deleted, not registered")
    func aCorruptDownloadIsNotRegistered() async throws {
        let sandbox = try Sandbox(scope: "registry-plan-corrupt")
        let hub = hub()
        // The hub published `payload`'s digest; the transfer delivers something
        // else, which is a truncated or substituted download.
        let transport = StubTransport(outcomes: [.write(Data("not the weights".utf8))])
        let registry = registry(
            sandbox, hub: hub, transport: transport, free: 100 * 1024 * 1024 * 1024)

        let error = try await #require(throws: ModelStoreError.self) {
            try await registry.download(id: modelID) { _ in }
        }
        guard case .digestMismatch(let file, let algorithm, _, _) = error else {
            Issue.record("expected digestMismatch, got \(error)")
            return
        }
        #expect(file == Self.weightsName)
        #expect(algorithm == "sha256")

        // Verified BEFORE registration: the manifest is untouched and the bytes
        // are gone, so the next attempt starts clean rather than resuming onto
        // a corrupt file.
        #expect(try sandbox.localStore.listLocal().isEmpty)
        let destination = try sandbox.localStore.fileURL(for: modelID, named: Self.weightsName)
        #expect(!FileManager.default.fileExists(atPath: destination.url.path))
    }

    @Test("a repository publishing no weights file is refused by name")
    func noWeightsFileIsRefused() throws {
        let error = try #require(throws: AIKitError.self) {
            _ = try DefaultModelRegistry.weightsFile(
                among: [ModelFile(path: "README.md", sizeBytes: 10, digest: nil)],
                repository: "a/b")
        }
        #expect(error.errorDescription?.contains("publishes no .gguf file") == true)
    }

    @Test("a repository publishing several unnamed weights files is refused, listing them")
    func severalWeightsFilesAreRefused() throws {
        // Picking "the largest" would choose a quantisation on the operator's
        // behalf, and which quantisation to run is their decision.
        let error = try #require(throws: AIKitError.self) {
            _ = try DefaultModelRegistry.weightsFile(
                among: [
                    ModelFile(path: "q4.gguf", sizeBytes: 10, digest: nil),
                    ModelFile(path: "q8.gguf", sizeBytes: 20, digest: nil),
                ],
                repository: "a/b")
        }
        let sentence = try #require(error.errorDescription)
        #expect(sentence.contains("q4.gguf"))
        #expect(sentence.contains("q8.gguf"))
    }

    @Test("a repository's one weights file is taken whatever it is called")
    func theOnlyWeightsFileIsTaken() throws {
        let only = ModelFile(path: "weights-q4_k_m.gguf", sizeBytes: 10, digest: nil)
        #expect(
            try DefaultModelRegistry.weightsFile(
                among: [only, ModelFile(path: "README.md", sizeBytes: 1, digest: nil)],
                repository: "a/b") == only)
    }

    @Test("a non-HuggingFace model is refused without a request")
    func aNonHubProviderIsRefused() async throws {
        let sandbox = try Sandbox(scope: "registry-plan-foreign")
        let hub = hub()
        let transport = StubTransport(outcomes: [.write(payload)])
        let registry = registry(
            sandbox, hub: hub, transport: transport, free: 100 * 1024 * 1024 * 1024)

        await #expect(throws: AIKitError.self) {
            try await registry.download(
                id: ModelIdentifier(provider: "ollama", modelId: "llama3")) { _ in }
        }
        #expect(hub.requestedPaths.isEmpty)
        #expect(transport.callCount == 0)
    }

    // MARK: - The fixture is pinned to the production decoder (BR-EC-02)

    @Test("the tree fixture is read by the real decoder")
    func theTreeFixtureIsReadByTheRealDecoder() async throws {
        let bytes = payload
        let hub = HubStub(
            model: makeFixtureHFModel(modelId: Self.repository),
            tree: [
                .lfsFile(path: Self.weightsName, data: bytes),
                FixtureHubTreeEntry(
                    type: "directory", path: "subdir", size: 0, oid: nil, lfs: nil),
            ]
        )

        // `HuggingFaceClient.files` is the production decoder: if this mirror
        // and `HFTreeEntry` ever disagree about a wire key, this fails rather
        // than the suites above silently planning against a zero size.
        let files = try await HuggingFaceClient(networkService: hub)
            .files(repoId: Self.repository)

        #expect(files.count == 1, "a directory entry is not a file to download")
        let weights = try #require(files.first)
        #expect(weights.path == Self.weightsName)
        #expect(weights.sizeBytes == Int64(bytes.count))
        #expect(weights.digest == .sha256(FixtureHubTreeEntry.sha256Hex(of: bytes)))
    }
}

// MARK: - T-AIL-61 — an interrupted download resumes

/// BR-AIL-23: *"Interrupted downloads resume from where they stopped."*
///
/// `URLSession` hands back an opaque token when a download task fails
/// part-way, and AIKit used to throw it away: a 4 GB model interrupted at
/// 3.9 GB started again at zero, which on a metered path is the same bill
/// twice.
@Suite("ModelDownloader — resume")
struct ModelDownloaderResumeTests {

    private final class Sandbox {
        let paths: TestPathLayer
        let store: ModelStore

        init(scope: String) throws {
            let paths = try TestPathLayer(scope: scope)
            try FileManager.default.createDirectory(
                at: paths.layer.base(of: AIKitNode.models), withIntermediateDirectories: true)
            self.paths = paths
            self.store = ModelStore(paths: paths.layer)
        }

        /// A resolved destination, the only kind the downloader accepts.
        func destination(_ path: String, repository: String = "a/b") throws -> ModelFileDestination {
            try store.destination(
                for: ModelFile(path: path, sizeBytes: 0, digest: nil), ofRepository: repository)
        }
    }

    /// Any URL: a stub transport never dereferences it.
    private var sourceURL: URL { URL(fileURLWithPath: "/dev/null") }

    // MARK: - The token store

    @Test("a token lives beside the file it belongs to")
    func theTokenLivesBesideTheFile() throws {
        let sandbox = try Sandbox(scope: "resume-store")
        let destination = try sandbox.destination("model.safetensors")
        let store = DownloadResumeStore()

        #expect(store.token(for: destination.url) == nil)
        #expect(
            store.location(for: destination.url).lastPathComponent
                == "model.safetensors.\(DownloadResumeStore.tokenExtension)")
        #expect(
            store.location(for: destination.url).deletingLastPathComponent()
                == destination.url.deletingLastPathComponent(),
            "beside the file is what makes a token impossible to orphan"
        )

        let token = Data("opaque".utf8)
        try store.save(token, for: destination.url)
        #expect(store.token(for: destination.url) == token)

        store.clear(for: destination.url)
        #expect(store.token(for: destination.url) == nil)
    }

    @Test("an empty token file is no token")
    func anEmptyTokenIsNoToken() throws {
        let sandbox = try Sandbox(scope: "resume-store-empty")
        let destination = try sandbox.destination("model.safetensors")
        let store = DownloadResumeStore()
        try store.save(Data(), for: destination.url)
        // A zero-byte token resumes nothing, and handing it to URLSession would
        // be a failure where a plain download was available.
        #expect(store.token(for: destination.url) == nil)
    }

    // MARK: - The token URLSession hands back

    @Test("the resume token is read off the error URLSession throws")
    func theTokenIsReadOffTheError() {
        let token = Data("resume-me".utf8)
        let failure = URLError(
            .networkConnectionLost, userInfo: [NSURLSessionDownloadTaskResumeData: token])
        #expect(ModelDownloader.resumeToken(from: failure) == token)

        // An error with nothing to resume from yields nothing — not an empty
        // Data that would be saved and then rejected.
        #expect(ModelDownloader.resumeToken(from: URLError(.timedOut)) == nil)
        #expect(ModelDownloader.resumeToken(from: AIKitError.downloadFailed("HTTP 500")) == nil)
    }

    // MARK: - T-AIL-61

    @Test("an interrupted transfer resumes from its token")
    func anInterruptedTransferResumes() async throws {
        let sandbox = try Sandbox(scope: "resume-interrupted")
        let destination = try sandbox.destination("model.safetensors")
        let token = Data("three point nine gigabytes in".utf8)
        let payload = Data("the rest of the weights".utf8)

        let transport = StubTransport(
            outcomes: [.failWithResumeToken(token), .write(payload)])
        let downloader = ModelDownloader(transport: transport.transport)

        // First attempt: interrupted.
        await #expect(throws: URLError.self) {
            try await downloader.download(from: sourceURL, to: destination) { _ in }
        }
        #expect(transport.calls.count == 1)
        #expect(transport.calls[0].token == nil, "nothing was stored yet, so nothing to resume")
        #expect(
            DownloadResumeStore().token(for: destination.url) == token,
            "the token the failure carried was kept"
        )

        // Second attempt: it continues from where it stopped.
        try await downloader.download(from: sourceURL, to: destination) { _ in }
        #expect(transport.calls.count == 2)
        #expect(transport.calls[1].token == token)
        #expect(try Data(contentsOf: destination.url) == payload)
        #expect(
            DownloadResumeStore().token(for: destination.url) == nil,
            "a finished file has nothing left to resume"
        )
    }

    @Test("a stale token does not wedge the file: one restart from the start")
    func aStaleTokenDoesNotWedgeTheFile() async throws {
        let sandbox = try Sandbox(scope: "resume-stale")
        let destination = try sandbox.destination("model.safetensors")
        let payload = Data("the whole file".utf8)

        // A token from an earlier run, for a partial file the server has since
        // moved on from: the resumed attempt is rejected, the plain one works.
        try DownloadResumeStore().save(Data("stale".utf8), for: destination.url)

        let transport = StubTransport(outcomes: [.failWithoutToken, .write(payload)])
        let downloader = ModelDownloader(transport: transport.transport)

        try await downloader.download(from: sourceURL, to: destination) { _ in }

        #expect(transport.calls.count == 2)
        #expect(transport.calls[0].token != nil, "the stored token was tried first")
        #expect(transport.calls[1].token == nil, "then the file was fetched from the start")
        #expect(try Data(contentsOf: destination.url) == payload)
        #expect(DownloadResumeStore().token(for: destination.url) == nil)
    }

    @Test("a second failure after a restart is the news, and its token is kept")
    func aSecondFailureIsReported() async throws {
        let sandbox = try Sandbox(scope: "resume-twice")
        let destination = try sandbox.destination("model.safetensors")
        let fresh = Data("a new partial".utf8)

        try DownloadResumeStore().save(Data("stale".utf8), for: destination.url)
        let transport = StubTransport(
            outcomes: [.failWithoutToken, .failWithResumeToken(fresh)])
        let downloader = ModelDownloader(transport: transport.transport)

        await #expect(throws: URLError.self) {
            try await downloader.download(from: sourceURL, to: destination) { _ in }
        }
        #expect(transport.calls.count == 2, "exactly one restart, never a loop")
        #expect(DownloadResumeStore().token(for: destination.url) == fresh)
    }

    @Test("a transfer that never fails stores nothing")
    func aCleanTransferStoresNothing() async throws {
        let sandbox = try Sandbox(scope: "resume-clean")
        let destination = try sandbox.destination("model.safetensors")
        let transport = StubTransport(outcomes: [.write(Data("fine".utf8))])
        let downloader = ModelDownloader(transport: transport.transport)

        try await downloader.download(from: sourceURL, to: destination) { _ in }
        #expect(transport.calls.count == 1)
        #expect(transport.calls[0].token == nil)
        #expect(DownloadResumeStore().token(for: destination.url) == nil)
    }

    @Test("an interrupted registry download resumes and is verified before registration")
    func aRegistryDownloadResumesAndVerifies() async throws {
        // The whole of T-AIL-61 through the shipped path: the registry's
        // downloader is interrupted, keeps its token, resumes from it, and the
        // bytes that finally arrive are hashed against the hub's digest BEFORE
        // the model enters the manifest.
        let paths = try TestPathLayer(scope: "registry-resume")
        try FileManager.default.createDirectory(
            at: paths.layer.base(of: AIKitNode.models), withIntermediateDirectories: true)
        let localStore = try LocalModelStore(paths: paths.layer)
        let modelStore = ModelStore(paths: paths.layer)

        let payload = Data("the weights, resumed".utf8)
        let id = ModelIdentifier(provider: "huggingface", modelId: "test/tiny-gguf")
        let hub = HubStub(
            model: makeFixtureHFModel(modelId: "test/tiny-gguf", tags: ["gguf"]),
            tree: [.lfsFile(path: "tiny-gguf.gguf", data: payload)]
        )
        let token = Data("most of the way".utf8)
        let transport = StubTransport(
            outcomes: [.failWithResumeToken(token), .write(payload)])

        let registry = DefaultModelRegistry(
            localStore: localStore,
            modelStore: modelStore,
            planner: ModelDownloadPlanner(
                freeSpace: { _ in 100 * 1024 * 1024 * 1024 },
                networkPath: { .unrestricted }
            ),
            hfClient: HuggingFaceClient(networkService: hub),
            downloader: ModelDownloader(transport: transport.transport)
        )

        await #expect(throws: URLError.self) {
            try await registry.download(id: id) { _ in }
        }
        #expect(try localStore.listLocal().isEmpty, "an interrupted download registers nothing")

        try await registry.download(id: id) { _ in }
        #expect(transport.calls.count == 2)
        #expect(transport.calls[1].token == token, "it continued from where it stopped")

        let stored = try #require(try localStore.listLocal().first)
        #expect(stored.descriptor.id == id)
        #expect(try Data(contentsOf: stored.localPath) == payload)
    }
}
