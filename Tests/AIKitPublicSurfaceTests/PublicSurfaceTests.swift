import AIKit
import AIKitTestSupport
import Foundation
import Testing

/// T-AIL-52 and T-AIL-53, W27 of spec ai-layer-aikit-core-2026-10-31,
/// BR-AIL-21: AIKit's public surface is four doors, and adding a fifth means
/// editing a checked-in file.
///
/// Operator, 2026-11-12: *"the public api surface of the AIKit package should
/// be thin with one system to talk to ai providers, one list of providers type
/// we managed and the hugging face endpoint to go to grab some provider to dl
/// them"*.
///
/// WHY THE SYMBOL GRAPH AND NOT A GREP OVER SOURCES. A grep for `public` reads
/// the keyword; this reads what the compiler EXPORTS. The two differ in both
/// directions: a `public` member of an internal type is not exported, and a
/// conformance a public type gains exports members no source file spells —
/// `Codable` alone adds `init(from:)` and `encode(to:)` to the promise this
/// package makes for a major version. Only the graph has both.
///
/// WHY ITS OWN TEST TARGET. `scripts/public-api.sh` runs two `swift build`
/// invocations, and a cold one compiles mlx-swift's Metal kernels: measured in
/// this worktree, folding this suite into `AIKitTests` took the whole-target
/// `AIKitAll` scope from 11s to 1m19s, and a 7-core compile beside tests that
/// assert a wall-clock budget is a flake waiting to happen — one appeared on
/// the first run (`FallbackProviderChainTests` "the timeout is the whole
/// chain's budget", run FE7289AE, spending a 5s budget on 0.2s of stub delay).
/// `AIKitLiveTests` and `AIKitMLXTests` are separated for the same reason: a
/// scope that needs something expensive gets its own target, so the unit
/// suites never pay for it.
///
/// It imports `AIKit` WITHOUT `@testable`, which is part of the claim: every
/// type this suite names is reachable through a door, by a consumer, with no
/// test-only access.
///
/// The graph comes from `scripts/public-api.sh`, which is the SAME script a
/// human runs with `--write` to regenerate `AIKitPublicAPI.txt` (BR-EC-02: a
/// generator and its enforcer share one code path, and this test is where that
/// agreement is asserted). The script runs `swift build --target AIKit` and
/// `--target AIKitMLX` with `-emit-symbol-graph` under ITS OWN scratch path, so
/// it does not contend with the `swift test` build that is running it.
@Suite("AIKit's public surface is four doors (W27)")
struct PublicSurfaceTests {

    // MARK: - Running the one code path

    private static let packageRoot = URL(filePath: #filePath)
        .deletingLastPathComponent()  // AIKitPublicSurfaceTests
        .deletingLastPathComponent()  // Tests
        .deletingLastPathComponent()  // package root

    /// Every line `scripts/public-api.sh` renders for this build.
    ///
    /// Run ONCE for the whole suite: the script builds two targets, and three
    /// tests asking for it separately would build them three times.
    private static let rendered: Result<[String], any Error> = {
        do {
            return .success(try runScript())
        } catch {
            return .failure(error)
        }
    }()

    private static func runScript() throws -> [String] {
        let script = packageRoot.appending(path: "scripts/public-api.sh")
        let process = Process()
        process.executableURL = URL(filePath: "/bin/bash")
        process.arguments = [script.path]
        process.currentDirectoryURL = packageRoot
        let out = Pipe()
        process.standardOutput = out
        // The script's own build chatter goes to stderr; let it through to the
        // test log rather than into the rendering.
        process.standardError = FileHandle.standardError

        try process.run()
        // Read BEFORE waiting: a full pipe buffer with nobody reading it is a
        // deadlock, and 1000-plus lines is past the 64 KB default.
        let data = try out.fileHandleForReading.readToEnd() ?? Data()
        process.waitUntilExit()

        guard process.terminationStatus == 0 else {
            throw SurfaceFailure.scriptFailed(status: process.terminationStatus)
        }
        let lines = (String(data: data, encoding: .utf8) ?? "")
            .split(separator: "\n", omittingEmptySubsequences: true)
            .map(String.init)
        guard !lines.isEmpty else {
            throw SurfaceFailure.scriptRenderedNothing
        }
        return lines
    }

    private enum SurfaceFailure: Error, CustomStringConvertible {
        case scriptFailed(status: Int32)
        case scriptRenderedNothing

        var description: String {
            switch self {
            case .scriptFailed(let status):
                "scripts/public-api.sh exited \(status) — see its output above"
            case .scriptRenderedNothing:
                "scripts/public-api.sh rendered no symbol at all, which is never right"
            }
        }
    }

    /// The allowlist's symbol lines. `#` lines are the review notes BR-AIL-21
    /// item 2 requires for a symbol that is public without being a door; they
    /// are not part of the surface.
    private static func allowlist() throws -> [String] {
        let text = try String(contentsOf: packageRoot.appending(path: "AIKitPublicAPI.txt"), encoding: .utf8)
        return text
            .split(separator: "\n", omittingEmptySubsequences: true)
            .map(String.init)
            .filter { !$0.hasPrefix("#") && !$0.trimmingCharacters(in: .whitespaces).isEmpty }
    }

    // MARK: - T-AIL-52

    @Test("The public symbol graph of AIKit and AIKitMLX equals AIKitPublicAPI.txt")
    func symbolGraphEqualsAllowlist() throws {
        let built = Set(try Self.rendered.get())
        let allowed = Set(try Self.allowlist())

        let added = built.subtracting(allowed).sorted()
        let removed = allowed.subtracting(built).sorted()

        #expect(
            added.isEmpty,
            """
            \(added.count) symbol(s) are public in this build and NOT in AIKitPublicAPI.txt.

            \(added.joined(separator: "\n"))

            Each one is a promise for a major version. Either make it internal, \
            or add it with `scripts/public-api.sh --write` and say in a `#` note \
            which of the four doors it belongs to (BR-AIL-21).
            """
        )
        #expect(
            removed.isEmpty,
            """
            \(removed.count) symbol(s) are in AIKitPublicAPI.txt and are NOT public in this build.

            \(removed.joined(separator: "\n"))

            A removal is a source break for every consumer. Intended? Regenerate \
            with `scripts/public-api.sh --write` in the wave that tags the major.
            """
        )
    }

    // MARK: - T-AIL-53

    /// A provider API, by the name it goes by. BR-AIL-21: *"Every provider API
    /// (endpoints, codecs, the HTTP provider, the CLI parsers, the stream
    /// parsers, the fallback internals) is `internal`."*
    ///
    /// Matched on the symbol's own path rather than on a file, because a type
    /// can move file and keep its promise. `AIProviderFamily` and
    /// `AICLISubprocessOptions` carry a family's NAME in settings and are door
    /// 1, so the patterns below are the implementation nouns only.
    static let providerAPINames: [(pattern: String, what: String)] = [
        ("EndPoint", "a route enum"),
        ("Codec", "a wire codec"),
        ("HTTPAIProvider", "the HTTP provider"),
        ("AIHTTPTransport", "the transport seam"),
        ("URLSessionHTTPTransport", "the production transport"),
        ("TimeoutNetworkService", "a NetKit service"),
        ("WireCodecCapabilities", "a codec's declaration"),
        ("WireReplyContext", "a codec's reply context"),
        ("Parser", "a CLI or stream parser"),
        ("StreamEvent", "a parsed stream event"),
        ("StreamParsing", "the stream-parser protocol"),
        ("FallbackProviderChain", "the fallback chain"),
        ("AgentProviding", "the prompt-in text-out family"),
        ("AgentProviderAdapter", "the adapter over it"),
        ("HuggingFace", "the Hugging Face client or endpoint"),
        ("ModelDownloader", "the downloader behind the catalogue"),
        ("LMStudioProvider", "LM Studio's own provider"),
        ("OpenAICompatibleEngine", "the OpenAI-compatible engine"),
        ("CoreMLEngine", "the Core ML engine"),
        ("MLXEngine", "the MLX subprocess engine"),
        ("MLXVideoProvider", "the MLX video provider"),
        ("MLXInProcessProvider", "the on-device provider"),
        ("MLXContainerSession", "its MLX session"),
        ("MLXHubDownloader", "its downloader"),
        ("MCPEngine", "the MCP engine"),
        ("MCPProvider", "the MCP provider"),
        ("CLISubprocessProvider", "the CLI subprocess provider"),
        ("WhisperCLIProvider", "the Whisper CLI provider"),
        ("ProviderHealthCheck", "the health check"),
        ("ProviderModelCatalog", "the provider model catalogue"),
        ("ArgsTemplate", "the CLI args template"),
        ("ShellRunner", "the shell runner"),
        ("Mock", "a test double"),
    ]

    @Test(
        "No provider API is public",
        arguments: PublicSurfaceTests.providerAPINames
    )
    func providerAPIsAreInternal(_ pattern: String, _ what: String) throws {
        // The dotted path, without the module and kind columns: a match on the
        // kind word (say `method`) would be meaningless.
        let paths = try Self.rendered.get().compactMap { line -> String? in
            let columns = line.split(separator: "  ", omittingEmptySubsequences: true)
            return columns.count >= 3 ? String(columns[2]) : nil
        }
        let offenders = paths.filter { $0.contains(pattern) }.sorted()

        #expect(
            offenders.isEmpty,
            """
            \(pattern) — \(what) — is exported:

            \(offenders.joined(separator: "\n"))

            BR-AIL-21: a caller reaches a provider through AIProvider and names \
            it in settings. It never holds the thing that talks a wire.
            """
        )
    }

    @Test("AIKitMLX exports its family registration and nothing else")
    func mlxExportsOnlyItsRegistration() throws {
        let mlx = try Self.rendered.get()
            .filter { $0.hasPrefix("AIKitMLX  ") }
            .map { $0.replacingOccurrences(of: "AIKitMLX  ", with: "") }
            .sorted()

        #expect(
            mlx == [
                "enum  AIKitMLX",
                "type.method  AIKitMLX.register(into:store:device:)",
                "type.property  AIKitMLX.family",
            ],
            """
            AIKitMLX exports \(mlx.count) symbol(s):

            \(mlx.joined(separator: "\n"))

            BR-AIL-21 item 3: one registration. A host that constructs an \
            MLXInProcessProvider by hand has a provider outside settings — \
            nothing ranks it and nothing records why it was unavailable \
            (BR-AIL-15, BR-AIL-20).
            """
        )
    }

    // MARK: - The doubles are exported by AIKitTestSupport

    /// The other half of T-AIL-53. ``MockProductBoundaryTests`` proves the
    /// `AIKit` product does NOT carry a mock; this proves `AIKitTestSupport`
    /// does — which is what makes that boundary a move rather than a deletion.
    ///
    /// This file imports `AIKitTestSupport` WITHOUT `@testable`, so naming the
    /// three types below is itself the assertion: a mock that stopped being
    /// public there would not compile here.
    @Test("AIKitTestSupport exports the three doubles, through a plain import")
    func testSupportExportsTheMocks() async {
        let provider: any AIProvider = MockAIProvider(id: "m", displayName: "m", capabilities: .textGeneration)
        let registry: any ModelRegistry = MockModelRegistry()
        let engine: any RuntimeEngine = MockRuntimeEngine()

        #expect(provider.id == "m")
        #expect(try! await registry.listModels(filter: nil).isEmpty)
        #expect(engine.id == "mock-engine")
    }
}
