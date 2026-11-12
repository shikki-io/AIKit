import CoreKit
import Foundation
import Testing

@testable import AIKit

// MARK: - AIKitAssemblyTests
//
// T-AIL-46, W22 of spec ai-layer-aikit-core-2026-10-31 (BR-AIL-20): the
// production and mock assemblies register ONE provider per settings row, in
// settings order, under the same names — differing only in what a row becomes.
// And no library initializer reads an environment variable: the three that used
// to configure providers are gone from Sources, which the ratchet at the bottom
// of this file is what keeps true.
//
// `DI.configure` mutates a PROCESS-WIDE default container, so every test here
// builds its own `Container` and assembles into it directly. Nothing in this
// suite touches `Container.default`, and nothing reaches a real provider
// (BR-AIL-12): the factory is injected with a stub transport.

@Suite("AIKitAssembly — settings to registrations")
struct AIKitAssemblyTests {

    // MARK: - Fixtures

    private static let threeProviders = """
        [[providers]]
        id      = "lmstudio-local"
        family  = "openai-chat"
        address = "http://127.0.0.1:1234"
        model   = "devstral-small-2507"

        [[providers]]
        id          = "claude-api"
        family      = "anthropic-messages"
        address     = "https://api.anthropic.com"
        model       = "fixture-anthropic-model"
        api_key_ref = "broker://anthropic/api-key"

        [[providers]]
        id      = "lmstudio-desk"
        family  = "lmstudio-native"
        address = "http://192.168.1.20:1234"
        model   = "qwen3-coder-30b"
        """

    private static let fixtureFile = "/operator/settings/providers.toml"

    private static func settings(_ toml: String = threeProviders) throws -> AIProviderSettings {
        try AIProviderSettings.parse(toml: toml, file: fixtureFile)
    }

    /// A factory whose transport is a stub, so nothing it builds can reach a
    /// server, and whose credential resolver is a dictionary rather than a
    /// broker.
    private static func stubFactory(
        credentials: [String: String] = ["broker://anthropic/api-key": TestCredential.anthropic]
    ) -> AIProviderFactory {
        AIProviderFactory(
            credentials: { credentials[$0] },
            transport: { _ in MockHTTPTransport() }
        )
    }

    private static func assembled(
        _ environment: DIEnvironment,
        toml: String = threeProviders,
        factory: AIProviderFactory? = nil
    ) throws -> Container {
        let container = Container(name: "aikit-w22-\(UUID().uuidString)")
        AIKitAssembly(
            settings: try settings(toml),
            factory: factory ?? stubFactory(),
            source: fixtureFile
        ).assemble(container: container, environment: environment)
        return container
    }

    // MARK: - T-AIL-46: production and mock differ only in what a row becomes

    @Test(
        "both environments register one provider per row, in settings order",
        arguments: [DIEnvironment.production, .mock]
    )
    func oneProviderPerEntry(_ environment: DIEnvironment) throws {
        let container = try Self.assembled(environment)
        let providers = try container.resolve([any AIProvider].self)

        #expect(providers.count == 3)
        // Order, because the file's order IS the fallback order (BR-AIL-11) and
        // local-first is the operator's decision, not a rule in code.
        #expect(providers.map(\.id) == ["lmstudio-local", "claude-api", "lmstudio-desk"])
    }

    @Test(
        "both environments register each provider under the row's own id",
        arguments: [DIEnvironment.production, .mock]
    )
    func eachProviderIsNamed(_ environment: DIEnvironment) throws {
        let container = try Self.assembled(environment)
        for id in ["lmstudio-local", "claude-api", "lmstudio-desk"] {
            let provider = try container.resolve((any AIProvider).self, name: id)
            #expect(provider.id == id)
        }
    }

    @Test("production builds the real provider for each family, from settings alone")
    func productionRegistersRealProviders() throws {
        // Every one of the four HTTP families has a production caller
        // (T-AIL-42): a row naming any of them builds a working provider here
        // with no new code. Three of them had none at all before the rework.
        let container = try Self.assembled(.production)
        let providers = try container.resolve([any AIProvider].self)

        let http = providers.compactMap { $0 as? HTTPAIProvider }
        #expect(http.count == 3)
        #expect(http.map(\.entry.family) == [.openAIChat, .anthropicMessages, .lmStudioNative])
        // The codec is resolved from the family ONCE, at construction.
        #expect(http.map(\.codec.family) == [.openAIChat, .anthropicMessages, .lmStudioNative])
        #expect(!providers.contains { $0 is MockAIProvider })
    }

    @Test("mock registers a MockAIProvider for each row, carrying the row's id")
    func mockRegistersMocks() throws {
        let container = try Self.assembled(.mock)
        let providers = try container.resolve([any AIProvider].self)

        #expect(providers.allSatisfy { $0 is MockAIProvider })
        #expect(providers.map(\.id) == ["lmstudio-local", "claude-api", "lmstudio-desk"])
        #expect(!providers.contains { $0 is HTTPAIProvider })
    }

    @Test("a mock answer still names who answered")
    func aMockAnswerNamesItself() async throws {
        // BR-AIL-06. A mock that answered anonymously would let a routing test
        // pass against a response no real provider can produce.
        let container = try Self.assembled(.mock)
        let provider = try container.resolve((any AIProvider).self, name: "claude-api")
        let response = try await provider.complete(
            request: AIRequest(messages: [AIMessage(role: .user, content: "hello")])
        )
        #expect(response.providerID == "claude-api")
    }

    @Test("the credential arrives from the injected resolver, and the row holds only a reference")
    func theCredentialIsResolved() throws {
        // BR-AIL-20: a row carries an `api_key_ref` and a broker resolves it
        // before the provider is built. AIKit never reads a secret store, and
        // the resolver here is a dictionary.
        let container = try Self.assembled(.production)
        let cloud = try #require(
            try container.resolve((any AIProvider).self, name: "claude-api") as? HTTPAIProvider
        )
        #expect(cloud.credential == TestCredential.anthropic)
        #expect(cloud.entry.apiKeyRef == "broker://anthropic/api-key")

        let local = try #require(
            try container.resolve((any AIProvider).self, name: "lmstudio-local") as? HTTPAIProvider
        )
        #expect(local.credential == nil, "a row with no api_key_ref gets no credential")
    }

    @Test("a host with no broker gets providers with no credential, not an invented one")
    func noBrokerNoCredential() throws {
        let container = try Self.assembled(.production, factory: AIProviderFactory(transport: { _ in
            MockHTTPTransport()
        }))
        let cloud = try #require(
            try container.resolve((any AIProvider).self, name: "claude-api") as? HTTPAIProvider
        )
        #expect(cloud.credential == nil)
    }

    // MARK: - The rest of the registrations

    @Test("the settings document and its repository are registered")
    func theDocumentIsRegistered() async throws {
        let container = try Self.assembled(.production)
        #expect(try container.resolve(AIProviderSettings.self).providers.count == 3)

        let repository = try container.resolve(AIProviderRepository.self)
        #expect(try await repository.list().map(\.id) == ["lmstudio-local", "claude-api", "lmstudio-desk"])
        #expect(await repository.fileName == Self.fixtureFile)
    }

    @Test("the router is the ordered chain over those providers")
    func theRouterIsOrdered() throws {
        let container = try Self.assembled(.production)
        let router = try container.resolve(ProviderRouter.self)
        #expect(router.orderedProviders.map(\.id) == ["lmstudio-local", "claude-api", "lmstudio-desk"])
    }

    @Test("the registered providers are the SAME instances the list holds")
    func oneInstancePerRow() throws {
        // Built once, not once per registration closure: a factory call per
        // closure would hand a caller resolving by name a different provider
        // instance than the one in the list, each with its own transport.
        let container = try Self.assembled(.mock)
        let listed = try container.resolve([any AIProvider].self)
        let named = try container.resolve((any AIProvider).self, name: "lmstudio-local")
        #expect((listed.first as? MockAIProvider) === (named as? MockAIProvider))
    }

    // MARK: - A row this host cannot serve

    @Test("a row this host cannot serve is recorded, and the others still work")
    func anUnservableRowDoesNotTakeTheOthersDown() throws {
        // W25: a typed refusal naming the row, never a crash and never a
        // silent drop. Settings written on a Mac and synced to a phone keep
        // every HTTP row; `mlx-in-process` needs the AIKitMLX product, which
        // this target does not link.
        let container = try Self.assembled(
            .production,
            toml: """
                [[providers]]
                id      = "lmstudio-local"
                family  = "openai-chat"
                address = "http://127.0.0.1:1234"
                model   = "devstral-small-2507"

                [[providers]]
                id     = "on-device"
                family = "mlx-in-process"
                model  = "qwen2.5-0.5b-instruct-4bit"
                """
        )

        let set = try container.resolve(AIProviderSet.self)
        #expect(set.providers.map(\.id) == ["lmstudio-local"])
        #expect(set.unavailable.map(\.providerID) == ["on-device"])
        #expect(set.unavailable[0].cause.contains("AIKitMLX"))

        // Resolving the row that could not be built reports WHY, naming the
        // file and the row — not "not registered".
        do {
            _ = try container.resolve((any AIProvider).self, name: "on-device")
            Issue.record("Expected the row's own refusal")
        } catch {
            #expect("\(error)".contains("on-device"))
        }
    }

    @Test("a registered family builder serves a row AIKit cannot serve alone")
    func aRegisteredBuilderServesItsFamily() throws {
        // How `AIKitMLX.register(into:)` will plug in (W27): on-device
        // inference stays an optional target (BR-AIL-14) instead of a
        // dependency AIKit links to satisfy a settings case.
        let factory = Self.stubFactory().registering(.mlxInProcess) { entry, _ in
            MockAIProvider(id: entry.id, displayName: "stub on-device")
        }
        let container = try Self.assembled(
            .production,
            toml: """
                [[providers]]
                id     = "on-device"
                family = "mlx-in-process"
                model  = "qwen2.5-0.5b-instruct-4bit"
                """,
            factory: factory
        )
        #expect(try container.resolve(AIProviderSet.self).unavailable.isEmpty)
        #expect(try container.resolve([any AIProvider].self).map(\.id) == ["on-device"])
    }

    @Test("a document where nothing can be built is an error carrying every cause")
    func nothingBuildableIsAnError() throws {
        let document = """
            [[providers]]
            id     = "on-device"
            family = "mlx-in-process"
            model  = "m"

            [[providers]]
            id     = "also-on-device"
            family = "coreml"
            model  = "m"
            """
        // The factory is where that verdict is reached, so it is asserted
        // there: a settings file that yields no provider at all is a
        // configuration error, not an empty fleet.
        do {
            _ = try Self.stubFactory().makeAll(try Self.settings(document))
            Issue.record("Expected allProvidersFailed")
        } catch let error as AIKitError {
            guard case .allProvidersFailed(let attempts) = error else {
                Issue.record("Expected allProvidersFailed, got \(error)")
                return
            }
            #expect(attempts.map(\.providerID) == ["on-device", "also-on-device"])
        }
    }

    // MARK: - T-AIL-46: no library initializer reads an environment variable

    /// Every `Sources` file, as text, keyed by its repo-relative path.
    private static func sourceFiles() throws -> [(path: String, text: String)] {
        let root = URL(filePath: #filePath)
            .deletingLastPathComponent()  // Settings
            .deletingLastPathComponent()  // AIKitTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // package root
            .appending(path: "Sources")

        let enumerator = try #require(FileManager.default.enumerator(atPath: root.path))
        var files: [(String, String)] = []
        for case let relative as String in enumerator where relative.hasSuffix(".swift") {
            let url = root.appending(path: relative)
            files.append(("Sources/" + relative, try String(contentsOf: url, encoding: .utf8)))
        }
        #expect(files.count > 50, "the source scan found almost nothing, so it is not scanning Sources")
        return files
    }

    @Test("the three environment variables that used to configure providers are gone from Sources")
    func theProviderEnvironmentVariablesAreGone() throws {
        // BR-AIL-20, and the panel finding on `LMStudioProvider.swift:62,66`
        // and `MLXEngine.swift:44`: a library initializer read its host's
        // environment and then guessed — `http://127.0.0.1:1234`,
        // `local-model`, port 8080 — so a host that set nothing got a provider
        // pointing at its own machine and a model name no server has.
        for variable in ["LMSTUDIO_URL", "LMSTUDIO_MODEL", "MLX_LM_BASE_URL"] {
            for file in try Self.sourceFiles() {
                #expect(
                    !file.text.contains("environment[\"\(variable)\"]"),
                    "\(file.path) reads \(variable); it is settings, through AIProviderEntry"
                )
            }
        }
    }

    /// The only environment reads `Sources` is allowed to contain, and what
    /// each one is for.
    ///
    /// A RATCHET: this list only shrinks. Both entries read `PATH` to find an
    /// installed binary, which is the host's own answer to "where is `claude`"
    /// and is not provider configuration — W25 owns that lookup and made it
    /// cross-platform. Adding a read means editing this list, so it shows up
    /// in review.
    static let allowedEnvironmentReads: [(path: String, variable: String)] = [
        ("Sources/AIKit/Providers/CLISubprocessProvider.swift", "PATH"),
        ("Sources/AIKit/Providers/WhisperCLIProvider.swift", "PATH"),
    ]

    @Test("Sources reads no environment variable beyond the two PATH lookups")
    func sourcesReadNoOtherEnvironment() throws {
        // The spec's T-AIL-46 asks for `ProcessInfo` to be absent from Sources
        // outright. It is not, and two of the three remaining uses are
        // legitimate and not environment reads at all:
        // `DeviceCapability` reads `physicalMemory` to decide whether a model
        // fits, and the two below read `PATH`. So this asserts the rule the
        // business rule actually states — a library initializer reads no
        // environment variable — as an allowlist that can only shrink.
        var found: [(path: String, variable: String)] = []
        for file in try Self.sourceFiles() {
            for line in file.text.split(separator: "\n", omittingEmptySubsequences: false) {
                let code = line.trimmingCharacters(in: .whitespaces)
                // A comment reads nothing.
                guard !code.hasPrefix("//"), !code.hasPrefix("///"), let after = code.range(of: "environment[\"") else { continue }
                let rest = code[after.upperBound...]
                guard let close = rest.range(of: "\"") else { continue }
                found.append((file.path, String(rest[rest.startIndex..<close.lowerBound])))
            }
        }

        let sorted = found.sorted { ($0.path, $0.variable) < ($1.path, $1.variable) }
        let allowed = Self.allowedEnvironmentReads.sorted { ($0.path, $0.variable) < ($1.path, $1.variable) }
        #expect(
            sorted.map { "\($0.path): \($0.variable)" } == allowed.map { "\($0.path): \($0.variable)" },
            "Sources reads an environment variable that is not on the allowlist"
        )
    }

    @Test("no provider type reads the environment in its initializer")
    func noProviderInitializerReadsTheEnvironment() throws {
        // The rule stated as the files it is about: the provider and engine
        // types a settings row builds.
        let providerFiles = [
            "Sources/AIKit/Providers/LMStudioProvider.swift",
            "Sources/AIKit/Engines/MLXEngine.swift",
            "Sources/AIKit/Engines/HTTPAIProvider.swift",
            "Sources/AIKit/Engines/OpenAICompatibleEngine.swift",
            "Sources/AIKit/Settings/AIProviderFactory.swift",
            "Sources/AIKit/Settings/AIProviderRepository.swift",
            "Sources/AIKit/Settings/AIProviderSettings.swift",
            "Sources/AIKit/DI/AIKitAssembly.swift",
        ]
        let scanned = try Self.sourceFiles().filter { providerFiles.contains($0.path) }
        #expect(scanned.count == providerFiles.count, "a file in the list was renamed or removed")

        for file in scanned {
            for line in file.text.split(separator: "\n", omittingEmptySubsequences: false) {
                let code = line.trimmingCharacters(in: .whitespaces)
                guard !code.hasPrefix("//"), !code.hasPrefix("///") else { continue }
                #expect(
                    !code.contains("ProcessInfo.processInfo.environment"),
                    "\(file.path) reads the environment: \(code)"
                )
            }
        }
    }
}
