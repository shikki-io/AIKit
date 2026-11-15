import Foundation
import Testing

@testable import AIKit

// Needs a running mlx_lm.server, so it sits in its own target and scope:
// the default gate must never depend on a model being present.

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

@Suite("MLXEngine — live synthesis")
struct MLXEngineLiveSmokeTests {
    // MARK: Gated live synthesis call

    #if canImport(Darwin) && arch(arm64)
        @Test("live synthesis: loadModel + complete returns non-empty output via MLX")
        func liveSynthesisOutputNonEmpty() async throws {
            // This test requires mlx_lm.server running at the address below;
            // `MLX_LM_BASE_URL` overrides it. `shi doctor --check mlx-lm --fix`
            // provisions it (W3).
            //
            // The environment read lives HERE, in a live test, and no longer in
            // `MLXEngine` (W22, BR-AIL-20): a library initializer must not read
            // its host's environment, while a test that needs a running server
            // is exactly the caller allowed to ask the host where it put one.
            let engine = MLXEngine(
                serverAddress: try AIServerAddress(
                    validating: ProcessInfo.processInfo.environment["MLX_LM_BASE_URL"]
                        ?? "http://127.0.0.1:8080"
                )
            )
            let descriptor = smallLLMDescriptor()

            let provider = try await engine.loadModel(descriptor)

            let request = AIRequest(
                messages: [AIMessage(role: .user, content: "Reply with one word: hello")],
                maxTokens: 32,
                model: descriptor.id
            )

            let response = try await withTimeout(seconds: 60) {
                try await provider.complete(request: request)
            }

            #expect(
                !response.content.isEmpty,
                "Expected non-empty synthesis output from MLX provider")
            #expect(
                response.model.isEmpty == false,
                "Expected model name in response")
        }
    #endif
}

// MARK: - Timeout helper (prevents hang per [[bash-poll-loops-must-have-timeout]])

private func withTimeout<T: Sendable>(
    seconds: Double,
    operation: @escaping @Sendable () async throws -> T
) async throws -> T {
    try await withThrowingTaskGroup(of: T.self) { group in
        group.addTask { try await operation() }
        group.addTask {
            try await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))
            throw CancellationError()
        }
        let result = try await group.next()!
        group.cancelAll()
        return result
    }
}
