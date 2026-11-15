import AIKitTestSupport
import CoreKit
import Foundation
import Testing

@testable import AIKit

// MARK: - HTTPEmbeddingProviderTests
//
// T-AIL-62, BR-AIL-24 / BR-AIL-25. The embeddings route, exercised through the
// transport seam — never the network (BR-AIL-12).
//
// WHAT THE OPERATOR ASKED FOR, and where each answer is, because this suite
// replaces one he rejected line by line on AIKit#24:
//
// 1. `AIEmbeddingWireShape.swift:25` — *"EndPoint is the SSoT for end points
//    and per topics, not mixing several systems"*. The route asserted below is
//    `OpenAIEndPoint.embeddings`, a case on the enum that already owns `/v1`.
//    No route string is spelled in this file; `routeIsTheEndPointEnums` reads
//    it off the enum, so the assertion cannot drift from the SSoT.
// 2. `AIEmbeddingProvider.swift:58` — *"it's the endpoint protocol that has to
//    manage this, not a trick"*. There is no base-URL normalisation to test
//    any more: `addressWithAPathIsRefused` pins that an address carrying `/v1`
//    is REFUSED where it is configured, instead of trimmed where it is used.
// 3. `AIEmbeddingProviderTests.swift:27` — *"don't we have a Codable typed
//    model for this fixture?"*. Every payload here is minted from
//    `EmbeddingsResponseDTO` by `mintEmbeddingsReply`. This file contains zero
//    raw JSON literals, which `FixtureLiteralRatchetTests` enforces.
// 4. `:29` — *"we are supposed to feed from settings like providers.toml made
//    for testing, from the DI and the right Path system"*. Every provider below
//    is resolved out of a `Container` that `AIKitAssembly` assembled from a
//    `providers.toml` document, written to a CoreKit `TestScratch` directory
//    and read through `AIProviderSettings.load(from:)`. No test constructs a
//    provider by hand, and nothing is written to `NSTemporaryDirectory()` or to
//    the operator's home.
// 5. `:32` — *"hardcoded settings need to be against the minimum settings we
//    manage, not against a working system"*. That document is
//    `minimumEmbeddingSettingsTOML` — five keys, the ones AIKit cannot do
//    without — and not a copy of `~/.shikki/settings/providers.toml`.

@Suite("HTTPEmbeddingProvider — settings to vectors")
struct HTTPEmbeddingProviderTests {

    // MARK: - Fixtures

    /// A settings document on disk, the container assembled from it, and the
    /// stub transport every provider in it was built with.
    ///
    /// A class so `TestPathLayer`'s `deinit` sweeps the scratch tree: a `defer`
    /// in every test is the line that gets forgotten.
    private final class Sandbox {
        let paths: TestPathLayer
        /// The document the HOST points AIKit at — AIKit never guesses a path.
        let settingsFile: URL
        let settings: AIProviderSettings
        let transport: MockHTTPTransport
        let container: Container

        init(
            toml: String = minimumEmbeddingSettingsTOML(),
            scope: String = "embeddings-settings",
            environment: DIEnvironment = .production
        ) throws {
            paths = try TestPathLayer(scope: scope)
            settingsFile = paths.root.appending(path: "providers.toml")
            try Data(toml.utf8).write(to: settingsFile)

            // The production reader, on the production path layer's root.
            settings = try AIProviderSettings.load(from: settingsFile)

            transport = MockHTTPTransport()
            let stub = transport
            container = Container(name: "aikit-embeddings-\(UUID().uuidString)")
            AIKitAssembly(
                settings: settings,
                factory: AIProviderFactory(
                    credentials: { _ in nil },
                    transport: { _ in stub }
                ),
                source: settingsFile.path,
                mockProvider: AIKitAssembly.mockProvider,
                mockEmbedder: AIKitAssembly.mockEmbedder
            ).assemble(container: container, environment: environment)
        }

        /// The embedder the settings row declares, straight out of DI.
        func embedder(_ id: String = "lmstudio") throws -> any AIEmbeddingProviding {
            try container.resolve((any AIEmbeddingProviding).self, name: id)
        }

        /// Answers the next call with a minted reply for `indices`.
        func answer(indices: [Int], status: Int = 200) throws {
            transport.answer(json: try mintEmbeddingsReply(indices: indices), status: status)
        }
    }

    /// The texts a batch of `count` embeds. Distinct, so a mispairing is
    /// visible in a failure message.
    private static func texts(_ count: Int) -> [String] {
        (0..<count).map { "fixture text \($0)" }
    }

    // MARK: - T-AIL-62: the route is the EndPoint enum's, and nothing else's

    @Test("the route is OpenAIEndPoint.embeddings, read off the enum rather than spelled here")
    func routeIsTheEndPointEnums() async throws {
        let sandbox = try Sandbox()
        try sandbox.answer(indices: [0])

        _ = try await sandbox.embedder().embed(text: "emaki")

        let sent = try #require(sandbox.transport.sentRequest)
        let endpoint = OpenAIEndPoint.embeddings(
            address: try makeFixtureAddress(),
            credential: nil,
            body: EmbeddingsRequest(model: fixtureEmbeddingModel, input: ["emaki"])
        )
        // The path comes from the SSoT. If the enum's route moves, this test
        // moves with it — which is the whole point of there being one.
        #expect(sent.url?.path == endpoint.path)
        #expect(sent.httpMethod == endpoint.method.rawValue)
        #expect(endpoint.path == OpenAIEndPoint.routes.first { $0.method == .POST && $0.path.hasSuffix("embeddings") }?.path)
    }

    @Test("the address carries no route, so there is no base URL to normalise")
    func addressWithAPathIsRefused() throws {
        // What `AIEmbeddingProvider.serverRoot` used to trim at call time is
        // refused here, where it was configured, naming the path (BR-AIL-09).
        let refusal = #expect(throws: AIKitError.self) {
            _ = try AIProviderSettings.parse(
                toml: minimumEmbeddingSettingsTOML(root: fixtureServerRoot + "/v1"),
                file: "the minimum document"
            )
        }
        let described = try #require(refusal?.errorDescription)
        #expect(described.contains("/v1"))
    }

    // MARK: - The model comes from settings, never from the caller (BR-AIL-25)

    @Test("the body's model is the row's embedding_model, not its chat model")
    func modelComesFromTheEmbeddingModelKey() async throws {
        let sandbox = try Sandbox()
        try sandbox.answer(indices: [0])

        _ = try await sandbox.embedder().embed(text: "emaki")

        let body = try #require(try sandbox.transport.sentBodyObject())
        #expect(body["model"] as? String == fixtureEmbeddingModel)
        // The chat model is a DIFFERENT value on the same row, and embedding
        // with it answers a vector in another space with no error at all.
        #expect(body["model"] as? String != fakeReplyModel)
        #expect(body["input"] as? [String] == ["emaki"], "input is always a list: one vector per element")
    }

    @Test("a row with no embedding_model has no embedder, and says what to add")
    func aRowThatServesNoEmbeddingsHasNoEmbedder() throws {
        let sandbox = try Sandbox(toml: minimumMixedSettingsTOML(), scope: "embeddings-mixed")

        let embedders = try sandbox.container.resolve([any AIEmbeddingProviding].self)
        #expect(embedders.map(\.id) == ["lmstudio"], "only the declaring row gets one")
        #expect(throws: (any Error).self) {
            _ = try sandbox.container.resolve((any AIEmbeddingProviding).self, name: "chat-only")
        }

        // And the row itself explains the edit, rather than quietly using the
        // chat model.
        let chatOnly = try #require(sandbox.settings.providers.first { $0.id == "chat-only" })
        let refusal = #expect(throws: AIKitError.self) { _ = try chatOnly.embeddingModelName() }
        let described = try #require(refusal?.errorDescription)
        #expect(described.contains("embedding_model"))
    }

    @Test("the embedder reports the model it embeds with, so a stored vector can name it")
    func embedderNamesItsModel() throws {
        let sandbox = try Sandbox()
        #expect(try sandbox.embedder().embeddingModel == fixtureEmbeddingModel)
        #expect(try sandbox.embedder().id == "lmstudio")
    }

    // MARK: - Batching and order

    @Test("a batch answered out of order is restored to INPUT order by index")
    func batchIsRestoredToInputOrder() async throws {
        let sandbox = try Sandbox()
        // The server answers 2, 0, 1 — which it is allowed to do.
        try sandbox.answer(indices: [2, 0, 1])

        let vectors = try await sandbox.embedder().embed(texts: Self.texts(3))

        #expect(vectors == (0..<3).map { mintEmbeddingVector(for: $0) })
    }

    @Test("an empty input list is answered without a request")
    func emptyInputMakesNoCall() async throws {
        let sandbox = try Sandbox()
        #expect(try await sandbox.embedder().embed(texts: []).isEmpty)
        #expect(sandbox.transport.sentRequests.isEmpty)
    }

    // MARK: - The call answers or refuses

    @Test("a 2xx carrying no vector refuses as emptyReply, naming the target")
    func noVectorRefuses() async throws {
        let sandbox = try Sandbox()
        try sandbox.answer(indices: [])

        let embedder = try sandbox.embedder()
        await #expect(throws: AIKitError.self) { _ = try await embedder.embed(texts: Self.texts(1)) }
    }

    @Test("a short batch refuses rather than mispairing vectors with inputs")
    func shortBatchRefuses() async throws {
        let sandbox = try Sandbox()
        try sandbox.answer(indices: [0])

        let embedder = try sandbox.embedder()
        await #expect(
            throws: AIKitError.embeddingCountMismatch(provider: "lmstudio", expected: 3, received: 1)
        ) {
            _ = try await embedder.embed(texts: Self.texts(3))
        }
    }

    @Test("a count mismatch is NOT eligible for fallback: another embedder is another vector space")
    func aCountMismatchDoesNotFallBack() {
        #expect(
            FallbackEligibility.allows(
                AIKitError.embeddingCountMismatch(provider: "lmstudio", expected: 3, received: 1)
            ) == false
        )
    }

    @Test("a non-2xx refuses with the status and the server's own text")
    func nonSuccessStatusRefuses() async throws {
        let sandbox = try Sandbox()
        try sandbox.answer(indices: [0], status: 503)

        let embedder = try sandbox.embedder()
        let refusal = await #expect(throws: AIKitError.self) {
            _ = try await embedder.embed(texts: Self.texts(1))
        }
        guard case .http(let status, _, let target) = try #require(refusal) else {
            Issue.record("a non-2xx must arrive as .http, not as a decode failure")
            return
        }
        #expect(status == 503)
        #expect(target.route == OpenAIEndPoint.routes.first { $0.path.hasSuffix("embeddings") }?.path)
    }

    @Test("a 2xx that does not decode is replyUndecodable, not an empty answer")
    func anUndecodableReplyRefuses() async throws {
        let sandbox = try Sandbox()
        // A reply shaped like the OTHER surface's: minted from the chat DTO, so
        // this is still one statement of a wire shape and not a hand-written
        // payload.
        sandbox.transport.answer(json: try mintReply(family: .openAIChat))

        let embedder = try sandbox.embedder()
        let refusal = await #expect(throws: AIKitError.self) {
            _ = try await embedder.embed(texts: Self.texts(1))
        }
        guard case .replyUndecodable = try #require(refusal) else {
            Issue.record("a reply AIKit cannot read must say so, not answer no vectors")
            return
        }
    }

    // MARK: - request_timeout is honoured, by the layer that owns it

    @Test("the row's request_timeout lands on the request the transport builds")
    func requestTimeoutIsHonoured() async throws {
        // #24 raced the call in a task group because `EndPoint` has no timeout
        // member. The seam is the TRANSPORT: `URLSessionHTTPTransport` is built
        // with the entry's timeout and puts it on every request it makes, and
        // `MockHTTPTransport` does the same so this is the production path.
        let seconds: TimeInterval = 7
        let sandbox = try Sandbox(scope: "embeddings-timeout")
        sandbox.transport.timeout = seconds
        try sandbox.answer(indices: [0])

        _ = try await sandbox.embedder().embed(text: "emaki")

        #expect(try #require(sandbox.transport.sentRequest).timeoutInterval == seconds)
    }

    @Test("a minimum document leaves request_timeout at the declared default")
    func theMinimumDocumentUsesTheDefault() throws {
        let sandbox = try Sandbox(scope: "embeddings-default-timeout")
        let entry = try #require(sandbox.settings.providers.first)
        #expect(entry.requestTimeout == AIProviderEntry.defaultRequestTimeout)
        #expect(entry.streamLineByteCap == AIProviderEntry.defaultStreamLineByteCap)
    }

    // MARK: - The fixture is pinned to the model, both ways (T-AIL-49)

    @Test("the minted embeddings reply round trips through the DTO that reads it")
    func theMintedReplyRoundTrips() throws {
        let json = try mintEmbeddingsReply(indices: [1, 0])
        let decoded = try JSONDecoder().decode(
            EmbeddingsResponseDTO.self, from: Data(json.utf8))

        #expect(decoded.model == fixtureEmbeddingModel)
        #expect(decoded.data.map(\.index) == [1, 0], "the fixture keeps the server's order")
        #expect(decoded.vectorsInInputOrder == [mintEmbeddingVector(for: 0), mintEmbeddingVector(for: 1)])
        // Back to text, and equal: the pin's second direction, which is what
        // catches a field that moved on only one side.
        #expect(try mintJSONText(decoded) == json)
    }

    @Test("a reply that reports no usage reads as nil, never as zero tokens")
    func noUsageIsNilNotZero() throws {
        let json = try mintEmbeddingsReply(indices: [0], promptTokens: nil)
        let decoded = try JSONDecoder().decode(EmbeddingsResponseDTO.self, from: Data(json.utf8))
        #expect(decoded.usage == nil)
    }

    // MARK: - The settings row survives a round trip through the document

    @Test("embedding_model round trips through the TOML document")
    func embeddingModelRoundTripsThroughTOML() throws {
        let settings = try AIProviderSettings.parse(
            toml: minimumEmbeddingSettingsTOML(), file: "the minimum document")
        let entry = try #require(settings.providers.first)
        #expect(entry.embeddingModel == fixtureEmbeddingModel)
        #expect(entry.model == fakeReplyModel, "the chat model is a separate key and keeps its value")

        // A host that edits providers writes them back, so a row that decoded
        // must encode — and the new key has to survive that, or an edit to any
        // other field would silently drop the embedding model.
        let written = try JSONEncoder().encode(settings)
        let read = try JSONDecoder().decode(AIProviderSettings.self, from: written)
        #expect(read.providers.first?.embeddingModel == fixtureEmbeddingModel)
        #expect(read == settings)
    }

    // MARK: - Both environments register the same names (T-AIL-46)

    @Test(
        "production and mock register an embedder for the same rows, under the same names",
        arguments: [DIEnvironment.production, .mock]
    )
    func bothEnvironmentsRegisterTheSameEmbedders(_ environment: DIEnvironment) throws {
        let sandbox = try Sandbox(
            toml: minimumMixedSettingsTOML(),
            scope: "embeddings-\(environment)",
            environment: environment
        )
        let embedders = try sandbox.container.resolve([any AIEmbeddingProviding].self)
        #expect(embedders.map(\.id) == ["lmstudio"])
        #expect(embedders.first?.embeddingModel == fixtureEmbeddingModel)
        #expect(try sandbox.embedder().id == "lmstudio")
    }

    // MARK: - A family with no embeddings route says so, before any call

    @Test(
        "a family whose API documents no embeddings route is refused when the provider is built",
        arguments: [AIProviderFamily.anthropicMessages, .lmStudioNative]
    )
    func aFamilyWithNoEmbeddingsRouteIsRefused(_ family: AIProviderFamily) throws {
        let entry = try makeFixtureEntry(
            family: family, id: "no-embeddings", embeddingModel: fixtureEmbeddingModel)
        #expect(throws: AIKitError.unsupported(field: "embeddings", family: family)) {
            _ = try HTTPEmbeddingProvider(entry: entry, network: MockHTTPTransport())
        }
    }
}
