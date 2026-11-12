import CoreKit
import Foundation
import Testing

@testable import AIKit

// MARK: - AIProviderRepositoryTests
//
// T-AIL-45, W22 of spec ai-layer-aikit-core-2026-10-31 (BR-AIL-18, BR-AIL-20):
// a TOML `[[providers]]` document decodes to entries IN FILE ORDER, and a bad
// address refuses the whole file naming the entry.
//
// No test reaches a real provider or a real file (BR-AIL-12): every document
// here is text, and the one that is written to disk goes to a per-test
// temporary directory.

@Suite("AIProviderRepository — settings in, entries out")
struct AIProviderRepositoryTests {

    // MARK: - Fixtures

    /// The operator's shape: local first, then a cloud entry with a key
    /// REFERENCE, then a second local server on the LAN.
    static let threeProviders = """
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
        request_timeout = 600
        stream_line_byte_cap = 4096
        """

    private static let fixtureFile = "/operator/settings/providers.toml"

    private static func makeSUT(_ toml: String) throws -> AIProviderRepository {
        AIProviderRepository(
            settings: try AIProviderSettings.parse(toml: toml, file: fixtureFile),
            source: fixtureFile
        )
    }

    /// The three parts of the `AIKitError.settings` a document refused with.
    private struct Refusal {
        let file: String
        let entry: String?
        let reason: String
    }

    /// The refusal a document produced.
    private static func refusal(of toml: String) throws -> Refusal {
        do {
            _ = try AIProviderSettings.parse(toml: toml, file: fixtureFile)
            throw RefusalExpected()
        } catch let error as AIKitError {
            guard case .settings(let file, let entry, let reason) = error else {
                throw RefusalExpected()
            }
            return Refusal(file: file, entry: entry, reason: reason)
        }
    }

    private struct RefusalExpected: Error {}

    // MARK: - T-AIL-45: a document decodes in file order

    @Test("a TOML [[providers]] document decodes to entries in file order")
    func loadsFromTOML() async throws {
        let repository = try Self.makeSUT(Self.threeProviders)
        let entries = try await repository.list()

        // ORDER, not a set: the file's order is the fallback order (BR-AIL-11),
        // so a repository that returned a dictionary's values would have lost
        // the operator's local-first decision.
        #expect(entries.map(\.id) == ["lmstudio-local", "claude-api", "lmstudio-desk"])
        #expect(
            entries.map(\.family) == [.openAIChat, .anthropicMessages, .lmStudioNative]
        )
    }

    @Test("every field of a row arrives typed, and the address is validated")
    func everyFieldArrives() async throws {
        let repository = try Self.makeSUT(Self.threeProviders)

        let local = try await repository.get("lmstudio-local")
        #expect(local.address?.scheme == "http")
        #expect(local.address?.host == "127.0.0.1")
        #expect(local.address?.port == 1234)
        #expect(local.model == "devstral-small-2507")
        #expect(local.apiKeyRef == nil)
        // Absent keys take the stated defaults, not zero.
        #expect(local.requestTimeout == AIProviderEntry.defaultRequestTimeout)
        #expect(local.streamLineByteCap == AIProviderEntry.defaultStreamLineByteCap)

        let desk = try await repository.get("lmstudio-desk")
        #expect(desk.requestTimeout == 600)
        #expect(desk.streamLineByteCap == 4096)
    }

    @Test("a row carries a key REFERENCE, never a key")
    func theRowCarriesAReference() async throws {
        // BR-AIL-20. A row that held the secret would print it in every log
        // line that prints an entry, and `api_key_ref` is the only field a
        // document has for one.
        let repository = try Self.makeSUT(Self.threeProviders)
        let cloud = try await repository.get("claude-api")
        #expect(cloud.apiKeyRef == "broker://anthropic/api-key")
        #expect(!Self.threeProviders.contains("api_key ="), "a document has no field for a literal key")
    }

    @Test("the spellings already in operators' files still decode")
    func theOldKindSpellingsDecode() async throws {
        // All 7 of the operator's existing provider profiles carry
        // `openai-compatible`, written when nothing read `kind` at all. The ONE
        // vocabulary maps it to the chat-completions family rather than failing
        // a configuration that was valid yesterday.
        let repository = try Self.makeSUT(
            """
            [[providers]]
            id      = "legacy"
            family  = "openai-compatible"
            address = "http://127.0.0.1:1234"
            model   = "m"
            """
        )
        #expect(try await repository.get("legacy").family == .openAIChat)
    }

    // MARK: - T-AIL-45: a bad address refuses the file, naming the entry

    @Test("a bad address refuses the whole file and names the entry and the value")
    func aBadAddressRefusesTheFile() throws {
        let refusal = try Self.refusal(of:
            """
            [[providers]]
            id      = "good"
            family  = "openai-chat"
            address = "http://127.0.0.1:1234"
            model   = "m"

            [[providers]]
            id      = "bad"
            family  = "openai-chat"
            address = "https://proxy.corp/openai/v1"
            model   = "m"
            """
        )
        #expect(refusal.file == Self.fixtureFile)
        #expect(refusal.entry == "bad")
        #expect(refusal.reason.contains("/openai/v1"))
    }

    @Test(
        "each address the panel executed against the wrong server is refused, naming the value",
        arguments: [
            "127.0.0.1:1234",
            "not a url at all",
            "https://proxy.corp/openai/v1",
            "http:///v1",
            "http://:1234",
            "ftp://127.0.0.1:1234",
            "http://user:secret@127.0.0.1:1234",
            "http://127.0.0.1:1234?key=x",
            "",
        ]
    )
    func everyBadAddressIsRefused(_ address: String) throws {
        // Six of these ten were EXECUTED against the wrong server by the shape
        // this replaces, one of them carrying a bearer key (AIKit#18
        // r4500281951). Here each one refuses the document.
        let refusal = try Self.refusal(of:
            """
            [[providers]]
            id      = "entry-under-test"
            family  = "openai-chat"
            address = "\(address)"
            model   = "m"
            """
        )
        #expect(refusal.entry == "entry-under-test")
        #expect(refusal.file == Self.fixtureFile)
    }

    @Test("an HTTP row with no address is refused rather than defaulted to this machine")
    func anHTTPRowNeedsAnAddress() throws {
        // The `?? "127.0.0.1"` this replaces turned a row with no address into
        // a request against the local machine.
        let refusal = try Self.refusal(of:
            """
            [[providers]]
            id     = "no-address"
            family = "openai-chat"
            model  = "m"
            """
        )
        #expect(refusal.entry == "no-address")
        #expect(refusal.reason.contains("names no address"))
    }

    @Test("an unknown family is refused, and the error lists the ones that exist")
    func anUnknownFamilyIsRefused() throws {
        let refusal = try Self.refusal(of:
            """
            [[providers]]
            id      = "mystery"
            family  = "gemini-whatever"
            address = "https://example.test"
            model   = "m"
            """
        )
        #expect(refusal.entry == "mystery")
        for family in AIProviderFamily.allCases {
            #expect(refusal.reason.contains(family.rawValue))
        }
    }

    @Test("two rows under one id are refused")
    func duplicateIDsAreRefused() throws {
        // The second row silently wins every lookup keyed by id, so the first
        // is configured and unreachable.
        let refusal = try Self.refusal(of:
            """
            [[providers]]
            id      = "lmstudio"
            family  = "openai-chat"
            address = "http://127.0.0.1:1234"
            model   = "a"

            [[providers]]
            id      = "lmstudio"
            family  = "openai-chat"
            address = "http://127.0.0.1:1235"
            model   = "b"
            """
        )
        #expect(refusal.entry == "lmstudio")
        #expect(refusal.reason.contains("one provider"))
    }

    @Test("text that is not TOML is refused as such, naming the file")
    func notTOMLIsRefused() throws {
        let refusal = try Self.refusal(of: "this is not [ toml")
        #expect(refusal.file == Self.fixtureFile)
        #expect(refusal.entry == nil)
    }

    // MARK: - Local families

    @Test("a local row carries no address, and a local row that names one is refused")
    func aLocalRowHasNoAddress() async throws {
        let repository = try Self.makeSUT(
            """
            [[providers]]
            id            = "claude-code"
            family        = "cli-subprocess"
            model         = "sonnet"
            binary        = "claude"
            args_template = ["-p", "--model", "{model}"]
            model_aliases = { sonnet = "fixture-sonnet-canonical" }
            cwd           = "/workspace"
            env           = { CLAUDE_HOME = "/tmp/claude" }
            """
        )

        let entry = try await repository.get("claude-code")
        #expect(entry.address == nil, "a subprocess has no server")
        let options = try #require(entry.options.cliSubprocess)
        #expect(options.binary == "claude")
        #expect(options.argsTemplate == ["-p", "--model", "{model}"])
        #expect(options.modelAliases["sonnet"] == "fixture-sonnet-canonical")
        #expect(options.workingDirectory == "/workspace")
        #expect(options.environment == ["CLAUDE_HOME": "/tmp/claude"])

        // Reading it names the row rather than returning a placeholder.
        #expect(throws: AIKitError.self) { try entry.serverAddress() }

        let refusal = try Self.refusal(of:
            """
            [[providers]]
            id      = "claude-code"
            family  = "cli-subprocess"
            address = "http://127.0.0.1:1234"
            model   = "sonnet"
            binary  = "claude"
            """
        )
        #expect(refusal.entry == "claude-code")
        #expect(refusal.reason.contains("runs on this host"))
    }

    @Test("a cli-subprocess row with no binary is refused")
    func aCLIRowNeedsABinary() throws {
        let refusal = try Self.refusal(of:
            """
            [[providers]]
            id     = "claude-code"
            family = "cli-subprocess"
            model  = "sonnet"
            """
        )
        #expect(refusal.entry == "claude-code")
        #expect(refusal.reason.contains("binary"))
    }

    @Test("the typed CLI options build the config that used to parse them itself")
    func theOptionsBuildTheConfig() async throws {
        // The two `kind` parsers are one now: this row was read by
        // `AIProviderEntry`'s decoder, and `CLISubprocessProviderConfig` is
        // built FROM it instead of parsing the same keys out of a dictionary.
        #if canImport(ShellKit)
            let repository = try Self.makeSUT(
                """
                [[providers]]
                id            = "claude-code"
                family        = "cli-subprocess"
                model         = "sonnet"
                binary        = "claude"
                args_template = ["-p"]
                """
            )
            let entry = try await repository.get("claude-code")
            let options = try #require(entry.options.cliSubprocess)
            let config = CLISubprocessProviderConfig(options: options, defaultModel: entry.model)
            #expect(config.binary == "claude")
            #expect(config.defaultModel == "sonnet")
            // And back again, so the two shapes agree in both directions.
            #expect(config.options == options)
            // The literal is derived from the family, so the vocabularies
            // cannot drift.
            #expect(CLISubprocessProviderConfig.kindLiteral == AIProviderFamily.cliSubprocess.rawValue)
        #endif
    }

    // MARK: - RepositoryProtocol

    @Test("it is a CoreKit Repository keyed by the id the operator wrote")
    func itIsACoreKitRepository() async throws {
        let repository: any RepositoryProtocol<AIProviderEntry.EntryID, AIProviderEntry> =
            try Self.makeSUT(Self.threeProviders)
        #expect(try await repository.list().count == 3)
        #expect(try await repository.get("claude-api").id == "claude-api")
    }

    @Test("an id nobody configured is an error naming the file and what IS configured")
    func anUnknownIDNamesTheFile() async throws {
        let repository = try Self.makeSUT(Self.threeProviders)
        do {
            _ = try await repository.get("ollama-local")
            Issue.record("Expected a settings error")
        } catch let error as AIKitError {
            guard case .settings(let file, let entry, let reason) = error else {
                Issue.record("Expected .settings, got \(error)")
                return
            }
            #expect(file == Self.fixtureFile)
            #expect(entry == "ollama-local")
            #expect(reason.contains("lmstudio-local"))
        }
    }

    @Test("saving a row keeps its position, because the position is the fallback order")
    func savingKeepsThePosition() async throws {
        let repository = try Self.makeSUT(Self.threeProviders)
        let local = try await repository.get("lmstudio-local")

        try await repository.save(
            local.id,
            data: AIProviderEntry(
                id: local.id,
                family: local.family,
                address: local.address,
                model: "a-different-model"
            )
        )

        let entries = try await repository.list()
        #expect(entries.map(\.id) == ["lmstudio-local", "claude-api", "lmstudio-desk"])
        #expect(entries[0].model == "a-different-model")
    }

    @Test("a new row is appended, and deleting one that is not there is an error")
    func savingAndDeleting() async throws {
        let repository = try Self.makeSUT(Self.threeProviders)
        try await repository.save(
            "ollama-local",
            data: try AIProviderEntry(
                id: "ollama-local",
                family: .openAIChat,
                address: "http://127.0.0.1:11434",
                model: "llama3"
            )
        )
        #expect(try await repository.list().map(\.id).last == "ollama-local")

        try await repository.delete("ollama-local")
        #expect(try await repository.list().count == 3)

        await #expect(throws: AIKitError.self) {
            try await repository.delete("ollama-local")
        }
    }

    @Test("the rows round trip back to a document")
    func theRowsRoundTrip() async throws {
        // A host that edits providers has to write them back, so an entry that
        // decoded must encode — including the address, which is a validated
        // type in memory and a string on the wire.
        let repository = try Self.makeSUT(Self.threeProviders)
        let written = try JSONEncoder().encode(await repository.settings())
        let read = try JSONDecoder().decode(AIProviderSettings.self, from: written)
        #expect(read.providers.map(\.id) == ["lmstudio-local", "claude-api", "lmstudio-desk"])
        #expect(read.providers[2].streamLineByteCap == 4096)
        #expect(read == (await repository.settings()))
    }

    // MARK: - Loading from the URL the HOST chose

    @Test("it loads the document the host points it at, and names the path when it cannot")
    func itLoadsTheHostsFile() async throws {
        // AIKit does not decide where this file lives (BR-AIL-20): the host
        // passes a URL. The directory here is this test's own, so nothing
        // touches the operator's home.
        let directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appending(path: "aikit-w22-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let file = directory.appending(path: "providers.toml")
        try Data(Self.threeProviders.utf8).write(to: file)

        let repository = try AIProviderRepository.loading(from: file)
        #expect(try await repository.list().count == 3)
        #expect(await repository.fileName == file.path)

        let missing = directory.appending(path: "absent.toml")
        do {
            _ = try AIProviderRepository.loading(from: missing)
            Issue.record("Expected a settings error")
        } catch let error as AIKitError {
            guard case .settings(let named, _, _) = error else {
                Issue.record("Expected .settings, got \(error)")
                return
            }
            #expect(named == missing.path)
        }
    }
}
