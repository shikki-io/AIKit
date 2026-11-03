import Foundation
import Testing
@testable import AIKit

@Suite("CoreMLEngine")
struct CoreMLEngineTests {

    private static let directory = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
        .appendingPathComponent("aikit-coreml-engine-tests", isDirectory: true)

    @Test("CoreML engine has correct id and formats")
    func engineIdentity() {
        let engine = CoreMLEngine(modelsDirectory: Self.directory)
        #expect(engine.id == "coreml")
        #expect(engine.displayName == "Apple CoreML")
        #expect(engine.supportedFormats == [.coreml])
    }

    #if canImport(CoreML)
    @Test("isAvailable returns true on Apple platforms")
    func availableOnApple() {
        let engine = CoreMLEngine(modelsDirectory: Self.directory)
        #expect(engine.isAvailable)
    }

    @Test("availableModelIDs lists the .mlmodelc bundles, sorted, ignoring other entries")
    func availableModelIDsListsBundles() throws {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .appendingPathComponent("aikit-coreml-discover-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        for name in ["whisper-small.mlmodelc", "ocr.mlmodelc", "notes.txt", "other.mlpackage"] {
            try FileManager.default.createDirectory(
                at: dir.appendingPathComponent(name), withIntermediateDirectories: true)
        }
        let engine = CoreMLEngine(modelsDirectory: dir)
        #expect(engine.availableModelIDs() == ["ocr", "whisper-small"])
    }

    @Test("availableModelIDs is empty when the directory is missing")
    func availableModelIDsMissingDirectory() {
        let missing = Self.directory.appendingPathComponent("does-not-exist-\(UUID().uuidString)")
        #expect(CoreMLEngine(modelsDirectory: missing).availableModelIDs().isEmpty)
    }

    @Test("Custom models directory is respected")
    func customModelsDirectory() {
        let customDir = URL(fileURLWithPath: "/tmp/custom-coreml-models")
        let engine = CoreMLEngine(modelsDirectory: customDir)
        #expect(engine.modelsDirectory == customDir)
    }
    #else
    @Test("isAvailable returns false off Apple platforms")
    func unavailableOffPlatform() {
        let engine = CoreMLEngine(modelsDirectory: Self.directory)
        #expect(!engine.isAvailable)
    }
    #endif

    @Test("Loaded models starts empty")
    func loadedModelsEmpty() {
        let engine = CoreMLEngine(modelsDirectory: Self.directory)
        #expect(engine.loadedModels().isEmpty)
    }

    @Test("loadModel throws for nonexistent model")
    func loadModelThrowsForMissing() async {
        let engine = CoreMLEngine(modelsDirectory: URL(fileURLWithPath: "/tmp/__nonexistent_aikit_dir__"))
        let descriptor = ModelDescriptor(
            id: ModelIdentifier(provider: "coreml", modelId: "nonexistent"),
            name: "Nonexistent",
            author: "test",
            description: "Does not exist",
            capabilities: .vision,
            format: .coreml,
            parameters: "unknown",
            sizeBytes: 0,
            architecture: "unknown",
            domain: .vision
        )

        await #expect(throws: AIKitError.self) {
            _ = try await engine.loadModel(descriptor)
        }
    }
}
