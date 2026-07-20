import Foundation
import Testing
@testable import AIKit
@testable import NetKit

@Suite("DefaultModelRegistry")
struct DefaultModelRegistryTests {

    private func makeTempStore() -> (LocalModelStore, URL) {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("aikit-registry-tests-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        return (LocalModelStore(basePath: tempDir), tempDir)
    }

    private func cleanup(_ dir: URL) {
        try? FileManager.default.removeItem(at: dir)
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
        let (store, dir) = makeTempStore()
        defer { cleanup(dir) }

        let descriptor = sampleDescriptor()
        let fakePath = dir.appendingPathComponent("model.gguf")
        FileManager.default.createFile(atPath: fakePath.path, contents: Data(count: 1024))
        try store.register(descriptor, at: fakePath)

        let mock = MockNetworkService()
        let client = HuggingFaceClient(networkService: mock)
        let registry = DefaultModelRegistry(localStore: store, hfClient: client)

        let models = try await registry.listModels(filter: nil)
        #expect(models.count == 1)
        #expect(models[0].id == descriptor.id)
    }

    @Test("listModels applies filter")
    func listModelsFilters() async throws {
        let (store, dir) = makeTempStore()
        defer { cleanup(dir) }

        let descriptor = sampleDescriptor()
        let fakePath = dir.appendingPathComponent("model.gguf")
        FileManager.default.createFile(atPath: fakePath.path, contents: Data(count: 1024))
        try store.register(descriptor, at: fakePath)

        let mock = MockNetworkService()
        let client = HuggingFaceClient(networkService: mock)
        let registry = DefaultModelRegistry(localStore: store, hfClient: client)

        // Filter that doesn't match
        let filter = ModelFilter(format: .mlx)
        let models = try await registry.listModels(filter: filter)
        #expect(models.isEmpty)
    }

    @Test("getModel finds local model first")
    func getModelFindsLocal() async throws {
        let (store, dir) = makeTempStore()
        defer { cleanup(dir) }

        let descriptor = sampleDescriptor()
        let fakePath = dir.appendingPathComponent("model.gguf")
        FileManager.default.createFile(atPath: fakePath.path, contents: Data(count: 1024))
        try store.register(descriptor, at: fakePath)

        let mock = MockNetworkService()
        // Intentionally don't set resultData — if it hits network, it would fail
        let client = HuggingFaceClient(networkService: mock)
        let registry = DefaultModelRegistry(localStore: store, hfClient: client)

        let model = try await registry.getModel(id: descriptor.id)
        #expect(model.id == descriptor.id)
        #expect(model.name == "model-7b")
        // Verify no network call was made (local-first)
        #expect(mock.capturedRequests.isEmpty)
    }

    @Test("getModel falls back to HuggingFace for remote")
    func getModelFallsBackToHF() async throws {
        let (store, dir) = makeTempStore()
        defer { cleanup(dir) }

        let mock = MockNetworkService()
        mock.resultData = """
        {
            "modelId": "nvidia/nemotron-3-nano-4b",
            "author": "nvidia",
            "tags": ["gguf"],
            "downloads": 50000,
            "likes": 1200,
            "pipeline_tag": "text-generation",
            "library_name": "transformers"
        }
        """.data(using: .utf8)!

        let client = HuggingFaceClient(networkService: mock)
        let registry = DefaultModelRegistry(localStore: store, hfClient: client)

        let id = ModelIdentifier(provider: "huggingface", modelId: "nvidia/nemotron-3-nano-4b")
        let model = try await registry.getModel(id: id)
        #expect(model.id.modelId == "nvidia/nemotron-3-nano-4b")
        #expect(model.author == "nvidia")
    }

    @Test("getModel throws for unknown non-HF provider")
    func getModelThrowsForUnknownProvider() async throws {
        let (store, dir) = makeTempStore()
        defer { cleanup(dir) }

        let mock = MockNetworkService()
        let client = HuggingFaceClient(networkService: mock)
        let registry = DefaultModelRegistry(localStore: store, hfClient: client)

        let id = ModelIdentifier(provider: "unknown", modelId: "some-model")
        await #expect(throws: AIKitError.self) {
            try await registry.getModel(id: id)
        }
    }

    @Test("deleteLocal removes from store")
    func deleteLocalRemoves() async throws {
        let (store, dir) = makeTempStore()
        defer { cleanup(dir) }

        let descriptor = sampleDescriptor()
        let fakePath = dir.appendingPathComponent("model.gguf")
        FileManager.default.createFile(atPath: fakePath.path, contents: Data(count: 1024))
        try store.register(descriptor, at: fakePath)

        let mock = MockNetworkService()
        let client = HuggingFaceClient(networkService: mock)
        let registry = DefaultModelRegistry(localStore: store, hfClient: client)

        #expect(store.isDownloaded(descriptor.id))
        try await registry.deleteLocal(id: descriptor.id)
        #expect(!store.isDownloaded(descriptor.id))
    }

    @Test("search delegates to HuggingFace client")
    func searchDelegatesToHF() async throws {
        let (store, dir) = makeTempStore()
        defer { cleanup(dir) }

        let mock = MockNetworkService()
        mock.resultData = """
        [
            {
                "modelId": "test/llama-7b",
                "author": "test",
                "tags": ["gguf"],
                "downloads": 100,
                "likes": 10,
                "pipeline_tag": "text-generation"
            }
        ]
        """.data(using: .utf8)!

        let client = HuggingFaceClient(networkService: mock)
        let registry = DefaultModelRegistry(localStore: store, hfClient: client)

        let results = try await registry.search(query: "llama", format: .gguf)
        #expect(results.count == 1)
        #expect(results[0].id.modelId == "test/llama-7b")
        #expect(results[0].format == .gguf)
    }

    @Test("search without format uses gguf default")
    func searchWithoutFormat() async throws {
        let (store, dir) = makeTempStore()
        defer { cleanup(dir) }

        let mock = MockNetworkService()
        mock.resultData = "[]".data(using: .utf8)!

        let client = HuggingFaceClient(networkService: mock)
        let registry = DefaultModelRegistry(localStore: store, hfClient: client)

        let results = try await registry.search(query: "anything", format: nil)
        #expect(results.isEmpty)
        #expect(mock.capturedRequests.count == 1)
    }
}
