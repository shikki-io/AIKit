import Foundation
import Testing
@testable import AIKit

@Suite("CoreMLEngine")
struct CoreMLEngineTests {

    @Test("CoreML engine has correct id and formats")
    func engineIdentity() {
        let engine = CoreMLEngine()
        #expect(engine.id == "coreml")
        #expect(engine.displayName == "Apple CoreML")
        #expect(engine.supportedFormats == [.coreml])
    }

    #if canImport(CoreML)
    @Test("isAvailable returns true on Apple platforms")
    func availableOnApple() {
        let engine = CoreMLEngine()
        #expect(engine.isAvailable)
    }

    @Test("Default models directory is ~/.aikit/coreml-models/")
    func defaultModelsDirectory() {
        let engine = CoreMLEngine()
        let home = FileManager.default.homeDirectoryForCurrentUser
        let expected = home.appendingPathComponent(".aikit/coreml-models", isDirectory: true)
        #expect(engine.modelsDirectory == expected)
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
        let engine = CoreMLEngine()
        #expect(!engine.isAvailable)
    }
    #endif

    @Test("Loaded models starts empty")
    func loadedModelsEmpty() {
        let engine = CoreMLEngine()
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
