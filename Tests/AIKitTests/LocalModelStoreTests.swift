import CoreKit
import CoreKitTestSupport
import Foundation
import Testing

@testable import AIKit

/// W23 of spec ai-layer-aikit-core-2026-10-31: the store holds an injected
/// CoreKit `PathResolving` and asks it where the manifest and each model live.
/// There is no base-path parameter to pass, optional or otherwise
/// (AIKit#18 r4500478370), and nothing hardcoded for it to drift from
/// (r4500481612).
///
/// Every suite here builds the layer with ``AIKitPaths/layer(root:nodeRoots:)``,
/// the call a host makes, over a run-scoped `TestScratch` root — so the tests
/// differ from production by that root and by nothing else.
@Suite("LocalModelStore")
struct LocalModelStoreTests {

    /// The store and the layer it resolves against.
    private final class Fixture {
        let paths: TestPathLayer
        let store: LocalModelStore
        /// A scratch directory for files a test pretends to have downloaded.
        let scratchFiles: URL

        init(scope: String = "local-model-store") throws {
            let paths = try TestPathLayer(scope: scope)
            let scratchFiles = paths.root.appendingPathComponent("downloaded")
            try FileManager.default.createDirectory(
                at: scratchFiles, withIntermediateDirectories: true)
            self.paths = paths
            self.scratchFiles = scratchFiles
            self.store = try LocalModelStore(paths: paths.layer)
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

    @Test("List returns empty on fresh store")
    func listEmpty() throws {
        let fixture = try Fixture()
        #expect(try fixture.store.listLocal().isEmpty)
    }

    @Test("Register + list returns model")
    func registerAndList() throws {
        let fixture = try Fixture()

        let descriptor = sampleDescriptor()
        let fakePath = fixture.scratchFiles.appendingPathComponent("fake.gguf")
        try Data(repeating: 0, count: 1024).write(to: fakePath)

        try fixture.store.register(descriptor, at: fakePath)

        let models = try fixture.store.listLocal()
        #expect(models.count == 1)
        #expect(models[0].descriptor.id == descriptor.id)
        #expect(models[0].descriptor.name == "model-7b")
    }

    @Test("isDownloaded returns true after register")
    func isDownloadedAfterRegister() throws {
        let fixture = try Fixture()

        let descriptor = sampleDescriptor()
        let id = descriptor.id
        let fakePath = fixture.scratchFiles.appendingPathComponent("fake.gguf")
        try Data(count: 512).write(to: fakePath)

        #expect(!(try fixture.store.isDownloaded(id)))

        try fixture.store.register(descriptor, at: fakePath)

        #expect(try fixture.store.isDownloaded(id))
    }

    @Test("Delete removes model")
    func deleteRemovesModel() throws {
        let fixture = try Fixture()

        let descriptor = sampleDescriptor()
        let fakePath = fixture.scratchFiles.appendingPathComponent("deletable.gguf")
        try Data(count: 256).write(to: fakePath)

        try fixture.store.register(descriptor, at: fakePath)
        #expect(try fixture.store.listLocal().count == 1)

        try fixture.store.delete(descriptor.id)
        #expect(try fixture.store.listLocal().isEmpty)
        #expect(!(try fixture.store.isDownloaded(descriptor.id)))
        #expect(!FileManager.default.fileExists(atPath: fakePath.path))
    }

    @Test("Delete throws for unknown model")
    func deleteUnknownThrows() throws {
        let fixture = try Fixture()

        let unknownId = ModelIdentifier(provider: "unknown", modelId: "no-model")
        #expect(throws: AIKitError.self) {
            try fixture.store.delete(unknownId)
        }
    }

    @Test("Total size computes correctly")
    func totalSizeComputes() throws {
        let fixture = try Fixture()

        let desc1 = sampleDescriptor(modelId: "test/model-a")
        let desc2 = sampleDescriptor(modelId: "test/model-b")

        let path1 = fixture.scratchFiles.appendingPathComponent("a.gguf")
        let path2 = fixture.scratchFiles.appendingPathComponent("b.gguf")

        try Data(repeating: 0xAA, count: 2048).write(to: path1)
        try Data(repeating: 0xBB, count: 4096).write(to: path2)

        try fixture.store.register(desc1, at: path1)
        try fixture.store.register(desc2, at: path2)

        #expect(try fixture.store.totalSizeBytes() == 2048 + 4096)
    }

    /// A corrupt manifest must throw `LocalModelStoreError.corruptManifest`.
    ///
    /// Returning `[]` was the previous behaviour: silent empty hides the
    /// corruption and, worse, the next write would overwrite all recorded models
    /// with an empty list. The caller must see the error and decide.
    @Test("Corrupt manifest throws corruptManifest, not empty")
    func corruptManifestThrows() throws {
        let fixture = try Fixture()

        // Written at the location the OWNER declares, read back from there:
        // the test does not know where that is either.
        let manifestURL = fixture.store.manifestURL
        try FileManager.default.createDirectory(
            at: manifestURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data("not valid json {{{{".utf8).write(to: manifestURL)

        #expect(throws: LocalModelStoreError.corruptManifest) {
            try fixture.store.listLocal()
        }
        #expect(throws: LocalModelStoreError.corruptManifest) {
            try fixture.store.totalSizeBytes()
        }
    }

    /// A missing manifest is NOT an error — fresh store, no file yet.
    @Test("Missing manifest returns empty, does not throw")
    func missingManifestIsEmpty() throws {
        let fixture = try Fixture()
        #expect(!FileManager.default.fileExists(atPath: fixture.store.manifestURL.path))
        #expect(try fixture.store.listLocal().isEmpty)
    }

    @Test("The manifest is a round trip through the owner's declared document")
    func manifestRoundTripsThroughTheOwner() throws {
        let fixture = try Fixture()

        let descriptor = sampleDescriptor()
        let fakePath = fixture.scratchFiles.appendingPathComponent("round.gguf")
        try Data(count: 777).write(to: fakePath)
        try fixture.store.register(descriptor, at: fakePath)

        // A SECOND store over the same layer reads what the first wrote: the
        // file is the source of truth, not any instance's memory, and both
        // resolve it from `LocalModelManifest` rather than from a path either
        // one was handed.
        let second = try LocalModelStore(paths: fixture.paths.layer)
        let models = try second.listLocal()
        #expect(models.count == 1)
        #expect(models[0].sizeOnDisk == 777)
        #expect(models[0].localPath == fakePath)
        #expect(second.manifestURL == fixture.store.manifestURL)
    }

    @Test("modelPath returns a provider-scoped path under the models node")
    func modelPathFormat() throws {
        let fixture = try Fixture()

        let id = ModelIdentifier(provider: "huggingface", modelId: "nvidia/nemotron-3")
        let path = try fixture.store.modelPath(for: id)

        #expect(path.path.contains("huggingface"))
        #expect(path.lastPathComponent == "nvidia_nemotron-3")
        #expect(
            PathContainment.isContained(
                path, in: fixture.paths.layer.base(of: AIKitNode.models))
        )
    }

    @Test("A file inside a model's directory is resolved, never appended to")
    func fileURLIsResolved() throws {
        let fixture = try Fixture()

        let id = ModelIdentifier(provider: "huggingface", modelId: "nvidia/nemotron-3")
        let destination = try fixture.store.fileURL(for: id, named: "model.gguf")

        #expect(destination.url.lastPathComponent == "model.gguf")
        // Compared as PATHS: `deletingLastPathComponent()` leaves a trailing
        // separator that `standardizedFileURL` keeps.
        #expect(
            destination.url.deletingLastPathComponent().standardizedFileURL.path
                == (try fixture.store.modelPath(for: id)).standardizedFileURL.path
        )

        // A name that is not a name is refused where it is resolved.
        for name in ["../escape.gguf", "sub/../../escape", ""] {
            #expect(throws: VirtualPathError.self) {
                try fixture.store.fileURL(for: id, named: name)
            }
        }
    }

    @Test("A layer without AIKit's owners refuses to build the store, by name")
    func anUnregisteredManifestIsRefused() throws {
        let scratch = TestScratch.forPackage()
        let root = try scratch.makeScratchDir(scope: "local-store-unregistered")
        defer { try? scratch.removeScratchDir(root) }

        let bare = try PathLayer(root: root)
        let error = #expect(throws: VirtualPathError.self) {
            try LocalModelStore(paths: bare)
        }
        #expect(error == .unregisteredOwner(String(reflecting: LocalModelManifest.self)))
    }
}
