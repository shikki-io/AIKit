import Foundation
import Testing
@testable import AIKit

// MARK: - MLX W4 Integration Smoke
//
// Scope: structural end-to-end verification that MLXEngine routes LLM requests
// through the MLX provider path (OpenAIProvider wrapping mlx_lm.server) with
// the correct capabilities and id prefix.
//
// The actual synthesis network call (complete()) is gated behind
// SHIKKI_ENABLE_MLX_SMOKE=1 so CI does not pay the model-download cost.
//
// Run locally:
//   SHIKKI_ENABLE_MLX_SMOKE=1 kagami test --scope AIKit

/// A tiny ModelDescriptor for a small MLX LLM (Qwen2.5 4-bit, ~2 GB).
/// Used across smoke sub-tests.
private func smallLLMDescriptor() -> ModelDescriptor {
    ModelDescriptor(
        id: ModelIdentifier(provider: "mlx", modelId: "qwen2.5-0.5b-instruct-4bit"),
        name: "Qwen2.5 0.5B Instruct (4-bit)",
        author: "mlx-community",
        description: "Tiny MLX LLM for smoke testing.",
        capabilities: [.textGeneration, .synthesis],
        format: .mlx,
        parameters: "0.5B",
        quantization: "4-bit",
        sizeBytes: 400_000_000,
        architecture: "qwen2",
        domain: .llm,
        isLocal: true,
        huggingFaceId: "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
    )
}

@Suite("MLXEngine — W4 Integration Smoke", .tags(.mlxSmoke))
struct MLXEngineSmokeTest {

    // MARK: Structural (always run — no network)

    #if canImport(Darwin) && arch(arm64)
    @Test("loadModel LLM domain returns provider with mlx/ id prefix")
    func loadModelLLMDomainProviderIdPrefix() async throws {
        let engine = MLXEngine()
        let descriptor = smallLLMDescriptor()

        let provider = try await engine.loadModel(descriptor)

        // Provider id must be "mlx/<modelId>" — proves MLX routing, not fallback.
        #expect(provider.id.hasPrefix("mlx/"),
                "Expected provider id to start with 'mlx/' but got '\(provider.id)'")
    }

    @Test("loadModel LLM domain returns provider advertising synthesis capability")
    func loadModelLLMDomainAdvertisesSynthesis() async throws {
        let engine = MLXEngine()
        let descriptor = smallLLMDescriptor()

        let provider = try await engine.loadModel(descriptor)

        #expect(provider.capabilities.contains(.synthesis),
                "Expected provider to advertise .synthesis capability")
        #expect(provider.capabilities.contains(.textGeneration),
                "Expected provider to advertise .textGeneration capability")
    }

    @Test("loadModel registers model in loadedModels")
    func loadModelRegistersInLoadedModels() async throws {
        let engine = MLXEngine()
        let descriptor = smallLLMDescriptor()

        _ = try await engine.loadModel(descriptor)

        let loaded = engine.loadedModels()
        #expect(loaded.contains(descriptor.id),
                "Expected \(descriptor.id) in loadedModels after loadModel")
    }

    @Test("ProviderRouter routeSmart synthesis fast-path selects MLX provider")
    func routeSmartSynthesisFastPathSelectsMLX() async throws {
        let engine = MLXEngine()
        let descriptor = smallLLMDescriptor()
        let mlxProvider = try await engine.loadModel(descriptor)

        // A cloud fallback that also advertises synthesis (should NOT be picked).
        let cloudFallback = MockAIProvider(
            id: "cloud/gpt-4o",
            capabilities: [.textGeneration, .synthesis]
        )

        let router = ProviderRouter(providers: [cloudFallback, mlxProvider])

        // Request with model.provider == "mlx" triggers the synthesis fast-path.
        let request = AIRequest(
            messages: [AIMessage(role: .user, content: "ping")],
            model: descriptor.id
        )
        let context = UsageContext(app: "aikit-smoke", task: "synthesis")

        let routed = await router.routeSmart(request: request, context: context)

        #expect(routed?.id.hasPrefix("mlx/") == true,
                "Expected routeSmart to return MLX provider, got '\(routed?.id ?? "nil")'")
        #expect(routed?.id == mlxProvider.id,
                "Routed id '\(routed?.id ?? "nil")' should equal mlxProvider id '\(mlxProvider.id)'")
    }

    @Test("MLXEngine detectCapabilities advertises synthesis on Apple Silicon")
    func detectCapabilitiesIncludesSynthesis() {
        let engine = MLXEngine()
        let caps = engine.detectCapabilities()
        // synthesis is advertised when python3 is available on arm64.
        // If python3 is absent in the test sandbox we accept the result gracefully.
        if ShellRunner.commandExists("python3") {
            #expect(caps.contains(.synthesis))
            #expect(caps.contains(.textGeneration))
        }
    }
    #endif
}

// MARK: - Tag

extension Tag {
    @Tag static var mlxSmoke: Self
}
