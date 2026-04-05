import Foundation
import Testing
@testable import AIKit

@Suite("LocalModelStore")
struct LocalModelStoreTests {

    private func makeTempStore() -> (LocalModelStore, URL) {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("aikit-tests-\(UUID().uuidString)")
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

    @Test("List returns empty on fresh store")
    func listEmpty() {
        let (store, dir) = makeTempStore()
        defer { cleanup(dir) }

        let models = store.listLocal()
        #expect(models.isEmpty)
    }

    @Test("Register + list returns model")
    func registerAndList() throws {
        let (store, dir) = makeTempStore()
        defer { cleanup(dir) }

        let descriptor = sampleDescriptor()
        let fakePath = dir.appendingPathComponent("fake.gguf")
        FileManager.default.createFile(atPath: fakePath.path, contents: Data(repeating: 0, count: 1024))

        try store.register(descriptor, at: fakePath)

        let models = store.listLocal()
        #expect(models.count == 1)
        #expect(models[0].descriptor.id == descriptor.id)
        #expect(models[0].descriptor.name == "model-7b")
    }

    @Test("isDownloaded returns true after register")
    func isDownloadedAfterRegister() throws {
        let (store, dir) = makeTempStore()
        defer { cleanup(dir) }

        let descriptor = sampleDescriptor()
        let id = descriptor.id
        let fakePath = dir.appendingPathComponent("fake.gguf")
        FileManager.default.createFile(atPath: fakePath.path, contents: Data(count: 512))

        #expect(!store.isDownloaded(id))

        try store.register(descriptor, at: fakePath)

        #expect(store.isDownloaded(id))
    }

    @Test("Delete removes model")
    func deleteRemovesModel() throws {
        let (store, dir) = makeTempStore()
        defer { cleanup(dir) }

        let descriptor = sampleDescriptor()
        let fakePath = dir.appendingPathComponent("deletable.gguf")
        FileManager.default.createFile(atPath: fakePath.path, contents: Data(count: 256))

        try store.register(descriptor, at: fakePath)
        #expect(store.listLocal().count == 1)

        try store.delete(descriptor.id)
        #expect(store.listLocal().isEmpty)
        #expect(!store.isDownloaded(descriptor.id))
    }

    @Test("Delete throws for unknown model")
    func deleteUnknownThrows() {
        let (store, dir) = makeTempStore()
        defer { cleanup(dir) }

        let unknownId = ModelIdentifier(provider: "unknown", modelId: "no-model")
        #expect(throws: AIKitError.self) {
            try store.delete(unknownId)
        }
    }

    @Test("Total size computes correctly")
    func totalSizeComputes() throws {
        let (store, dir) = makeTempStore()
        defer { cleanup(dir) }

        let desc1 = sampleDescriptor(modelId: "test/model-a")
        let desc2 = sampleDescriptor(modelId: "test/model-b")

        let path1 = dir.appendingPathComponent("a.gguf")
        let path2 = dir.appendingPathComponent("b.gguf")

        let data1 = Data(repeating: 0xAA, count: 2048)
        let data2 = Data(repeating: 0xBB, count: 4096)
        FileManager.default.createFile(atPath: path1.path, contents: data1)
        FileManager.default.createFile(atPath: path2.path, contents: data2)

        try store.register(desc1, at: path1)
        try store.register(desc2, at: path2)

        let total = store.totalSizeBytes()
        #expect(total == 2048 + 4096)
    }

    @Test("modelPath returns provider-scoped path")
    func modelPathFormat() {
        let (store, dir) = makeTempStore()
        defer { cleanup(dir) }

        let id = ModelIdentifier(provider: "huggingface", modelId: "nvidia/nemotron-3")
        let path = store.modelPath(for: id)

        #expect(path.path.contains("huggingface"))
        #expect(path.path.contains("nvidia_nemotron-3"))
    }
}
