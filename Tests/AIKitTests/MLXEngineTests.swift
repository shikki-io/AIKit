import Foundation
import Testing
@testable import AIKit

@Suite("MLXEngine")
struct MLXEngineTests {

    @Test("MLX engine has correct id and formats")
    func engineIdentity() {
        let engine = MLXEngine()
        #expect(engine.id == "mlx")
        #expect(engine.displayName == "Apple MLX")
        #expect(engine.supportedFormats == [.mlx, .gguf])
    }

    #if canImport(Darwin) && arch(arm64)
    @Test("isAvailable returns true on Apple Silicon")
    func availableOnAppleSilicon() {
        let engine = MLXEngine()
        #expect(engine.isAvailable)
    }
    #else
    @Test("isAvailable returns false on non-Apple-Silicon")
    func unavailableOffPlatform() {
        let engine = MLXEngine()
        #expect(!engine.isAvailable)
    }
    #endif

    @Test("VideoGenerationOptions defaults are sensible")
    func videoOptionsDefaults() {
        let opts = VideoGenerationOptions.default
        #expect(opts.width == 480)
        #expect(opts.height == 320)
        #expect(opts.numFrames == 49)
        #expect(opts.fps == 24)
        #expect(opts.numInferenceSteps == 30)
        #expect(opts.guidanceScale == 5.0)
        #expect(opts.seed == nil)
        #expect(opts.outputPath == nil)
        #expect(opts.inputImagePath == nil)
    }

    @Test("VideoGenerationOptions hd720p preset")
    func videoOptionsHD() {
        let opts = VideoGenerationOptions.hd720p
        #expect(opts.width == 1280)
        #expect(opts.height == 720)
        #expect(opts.numFrames == 97)
        #expect(opts.fps == 24)
        #expect(opts.numInferenceSteps == 50)
        #expect(opts.guidanceScale == 5.0)
    }

    @Test("ShellRunner.commandExists finds /usr/bin/env via 'env'")
    func shellRunnerFindsEnv() {
        #expect(ShellRunner.commandExists("env"))
    }

    @Test("ShellRunner.commandExists returns false for nonexistent")
    func shellRunnerMissing() {
        #expect(!ShellRunner.commandExists("__nonexistent_command_aikit_test__"))
    }

    @Test("ShellRunner.run executes simple command")
    func shellRunnerRun() async throws {
        let result = try await ShellRunner.run("/bin/echo", arguments: ["hello"])
        #expect(result.exitCode == 0)
        #expect(result.stdout.trimmingCharacters(in: .whitespacesAndNewlines) == "hello")
    }

    @Test("Video model catalog has expected entries")
    func videoModelCatalog() {
        #if canImport(Darwin)
        let catalog = MLXEngine.videoModelCatalog
        #expect(catalog.count == 4)

        let ids = catalog.map(\.id.modelId)
        #expect(ids.contains("wan2.1-t2v-1.3b"))
        #expect(ids.contains("wan2.1-t2v-14b"))
        #expect(ids.contains("ltx-2"))
        #expect(ids.contains("wan2.1-i2v-14b"))

        // All video models should have video domain.
        for model in catalog {
            #expect(model.domain == .video)
            #expect(model.capabilities.contains(.videoGeneration))
        }
        #endif
    }

    @Test("MLXVideoProvider capabilities include videoGeneration")
    func mlxVideoProviderCapabilities() {
        #if canImport(Darwin)
        let provider = MLXVideoProvider(
            id: "test",
            displayName: "Test Video",
            pythonPath: "/usr/bin/python3",
            modelName: "wan2.1-t2v-1.3b"
        )
        #expect(provider.capabilities.contains(.videoGeneration))
        #expect(provider.capabilities.contains(.vision))
        #endif
    }

    @Test("Loaded models starts empty")
    func loadedModelsEmpty() {
        let engine = MLXEngine()
        #expect(engine.loadedModels().isEmpty)
    }
}
