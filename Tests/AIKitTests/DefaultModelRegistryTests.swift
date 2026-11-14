import CoreKit
import Foundation
import Testing

@testable import AIKit
@testable import NetKit

@Suite("DefaultModelRegistry")
struct DefaultModelRegistryTests {

    /// The manifest store, the verified file store, and the layer both resolve
    /// against.
    ///
    /// A class that HOLDS the `TestPathLayer` rather than a tuple that returns
    /// it: the layer owns the scratch tree and sweeps it in `deinit`, so
    /// dropping it would delete the directory the stores are still resolving
    /// against. W23 of spec ai-layer-aikit-core-2026-10-31, and the shape every
    /// other suite in this target uses.
    ///
    /// Both stores come from ONE layer, which is the point: W33 gave the
    /// registry a `ModelStore` so a download can be planned and verified, and
    /// two stores over two roots is the duplicate-4 GB defect W13 measured.
    private final class Fixture {
        let paths: TestPathLayer
        let localStore: LocalModelStore
        let modelStore: ModelStore

        /// The scratch root, for a file a test pretends to have downloaded.
        var root: URL { paths.root }

        init(scope: String = "default-model-registry") throws {
            let paths = try TestPathLayer(scope: scope)
            try FileManager.default.createDirectory(
                at: paths.layer.base(of: AIKitNode.models), withIntermediateDirectories: true)
            self.paths = paths
            self.localStore = try LocalModelStore(paths: paths.layer)
            self.modelStore = ModelStore(paths: paths.layer)
        }
    }

    private func sampleDescriptor(
        provider: String = "huggingface",
        modelId: String = "test/model-7b"
    ) -> ModelDescriptor {
        ModelDescriptor(
            id: ModelIdentifier(provider: provider, modelId: modelId),
            name: "model-7b",
            author: "test",
            description: "A test model",
            capabilities: .textGeneration,
            format: .gguf,
            parameters: "7B",
            sizeBytes: 4_000_000_000,
            architecture: "llama",
            domain: .llm,
            isLocal: true,
            huggingFaceId: modelId
        )
    }

    @Test("listModels returns local models")
    func listModelsReturnsLocal() async throws {
        let fixture = try Fixture()
        let dir = fixture.root

        let descriptor = sampleDescriptor()
        let fakePath = dir.appendingPathComponent("model.gguf")
        FileManager.default.createFile(atPath: fakePath.path, contents: Data(count: 1024))
        try fixture.localStore.register(descriptor, at: fakePath)

        let mock = MockNetworkService()
        let client = HuggingFaceClient(networkService: mock)
        let registry = DefaultModelRegistry(
            localStore: fixture.localStore,
            modelStore: fixture.modelStore,
            hfClient: client
        )

        let models = try await registry.listModels(filter: nil)
        #expect(models.count == 1)
        #expect(models[0].id == descriptor.id)
    }

    @Test("listModels applies filter")
    func listModelsFilters() async throws {
        let fixture = try Fixture()
        let dir = fixture.root

        let descriptor = sampleDescriptor()
        let fakePath = dir.appendingPathComponent("model.gguf")
        FileManager.default.createFile(atPath: fakePath.path, contents: Data(count: 1024))
        try fixture.localStore.register(descriptor, at: fakePath)

        let mock = MockNetworkService()
        let client = HuggingFaceClient(networkService: mock)
        let registry = DefaultModelRegistry(
            localStore: fixture.localStore,
            modelStore: fixture.modelStore,
            hfClient: client
        )

        // Filter that doesn't match
        let filter = ModelFilter(format: .mlx)
        let models = try await registry.listModels(filter: filter)
        #expect(models.isEmpty)
    }

    @Test("getModel finds local model first")
    func getModelFindsLocal() async throws {
        let fixture = try Fixture()
        let dir = fixture.root

        let descriptor = sampleDescriptor()
        let fakePath = dir.appendingPathComponent("model.gguf")
        FileManager.default.createFile(atPath: fakePath.path, contents: Data(count: 1024))
        try fixture.localStore.register(descriptor, at: fakePath)

        let mock = MockNetworkService()
        // Intentionally don't set resultData — if it hits network, it would fail
        let client = HuggingFaceClient(networkService: mock)
        let registry = DefaultModelRegistry(
            localStore: fixture.localStore,
            modelStore: fixture.modelStore,
            hfClient: client
        )

        let model = try await registry.getModel(id: descriptor.id)
        #expect(model.id == descriptor.id)
        #expect(model.name == "model-7b")
        // Verify no network call was made (local-first)
        #expect(mock.capturedRequests.isEmpty)
    }

    @Test("getModel falls back to HuggingFace for remote")
    func getModelFallsBackToHF() async throws {
        let fixture = try Fixture()

        let mock = MockNetworkService()
        mock.resultData = try mintHFModelReply(makeFixtureHFModel(tags: ["gguf"]))

        let client = HuggingFaceClient(networkService: mock)
        let registry = DefaultModelRegistry(
            localStore: fixture.localStore,
            modelStore: fixture.modelStore,
            hfClient: client
        )

        let id = ModelIdentifier(provider: "huggingface", modelId: "nvidia/nemotron-3-nano-4b")
        let model = try await registry.getModel(id: id)
        #expect(model.id.modelId == "nvidia/nemotron-3-nano-4b")
        #expect(model.author == "nvidia")
    }

    @Test("getModel throws for unknown non-HF provider")
    func getModelThrowsForUnknownProvider() async throws {
        let fixture = try Fixture()

        let mock = MockNetworkService()
        let client = HuggingFaceClient(networkService: mock)
        let registry = DefaultModelRegistry(
            localStore: fixture.localStore,
            modelStore: fixture.modelStore,
            hfClient: client
        )

        let id = ModelIdentifier(provider: "unknown", modelId: "some-model")
        await #expect(throws: AIKitError.self) {
            try await registry.getModel(id: id)
        }
    }

    @Test("deleteLocal removes from store")
    func deleteLocalRemoves() async throws {
        let fixture = try Fixture()
        let dir = fixture.root

        let descriptor = sampleDescriptor()
        let fakePath = dir.appendingPathComponent("model.gguf")
        FileManager.default.createFile(atPath: fakePath.path, contents: Data(count: 1024))
        try fixture.localStore.register(descriptor, at: fakePath)

        let mock = MockNetworkService()
        let client = HuggingFaceClient(networkService: mock)
        let registry = DefaultModelRegistry(
            localStore: fixture.localStore,
            modelStore: fixture.modelStore,
            hfClient: client
        )

        #expect(try fixture.localStore.isDownloaded(descriptor.id))
        try await registry.deleteLocal(id: descriptor.id)
        #expect(!(try fixture.localStore.isDownloaded(descriptor.id)))
    }

    @Test("search delegates to HuggingFace client")
    func searchDelegatesToHF() async throws {
        let fixture = try Fixture()

        let mock = MockNetworkService()
        mock.resultData = try mintHFSearchReply([
            makeFixtureHFModel(
                modelId: "test/llama-7b",
                author: "test",
                tags: ["gguf"],
                downloads: 100,
                likes: 10,
                lastModified: nil,
                library: nil
            ),
        ])

        let client = HuggingFaceClient(networkService: mock)
        let registry = DefaultModelRegistry(
            localStore: fixture.localStore,
            modelStore: fixture.modelStore,
            hfClient: client
        )

        let results = try await registry.search(query: "llama", format: .gguf)
        #expect(results.count == 1)
        #expect(results[0].id.modelId == "test/llama-7b")
        #expect(results[0].format == .gguf)
    }

    @Test("search without format uses gguf default")
    func searchWithoutFormat() async throws {
        let fixture = try Fixture()

        let mock = MockNetworkService()
        mock.resultData = try mintHFSearchReply([])

        let client = HuggingFaceClient(networkService: mock)
        let registry = DefaultModelRegistry(
            localStore: fixture.localStore,
            modelStore: fixture.modelStore,
            hfClient: client
        )

        let results = try await registry.search(query: "anything", format: nil)
        #expect(results.isEmpty)
        #expect(mock.capturedRequests.count == 1)
    }
}
