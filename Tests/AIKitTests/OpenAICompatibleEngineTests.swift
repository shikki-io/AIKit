import Foundation
import NetKit
import Testing

@testable import AIKit

@Suite("OpenAICompatibleEngine")
struct OpenAICompatibleEngineTests {

    // MARK: - Preset Defaults

    @Test("LM Studio preset has correct defaults")
    func lmStudioPresetDefaults() throws {
        let engine = try OpenAICompatibleEngine.lmStudio()
        #expect(engine.id == "lmstudio")
        #expect(engine.displayName == "LM Studio")
        #expect(engine.isAvailable)
        #expect(engine.supportedFormats.contains(.gguf))
        #expect(engine.supportedFormats.contains(.mlx))
        #expect(engine.supportedFormats.contains(.api))
    }

    @Test("Ollama preset has correct defaults")
    func ollamaPresetDefaults() throws {
        let engine = try OpenAICompatibleEngine.ollama()
        #expect(engine.id == "ollama")
        #expect(engine.displayName == "Ollama")
        #expect(engine.isAvailable)
        #expect(engine.supportedFormats.contains(.gguf))
    }

    @Test("OpenAI preset includes API key")
    func openAIPresetIncludesKey() throws {
        let engine = try OpenAICompatibleEngine.openAI(apiKey: TestCredential.openAI)
        #expect(engine.id == "openai")
        #expect(engine.displayName == "OpenAI")
    }

    @Test("Anthropic preset configures correctly")
    func anthropicPresetDefaults() throws {
        let engine = try OpenAICompatibleEngine.anthropic(apiKey: TestCredential.anthropic)
        #expect(engine.id == "anthropic")
        #expect(engine.displayName == "Anthropic")
    }

    // MARK: - Model Discovery

    @Test("discoverModels parses /v1/models response")
    func discoverModelsParsesList() async throws {
        let mock = MockHTTPTransport()
        let modelsJSON = """
            {
                "data": [
                    {"id": "gpt-4", "owned_by": "openai", "created": 1700000000},
                    {"id": "text-embedding-ada-002", "owned_by": "openai", "created": 1699000000}
                ]
            }
            """
        mock.body = Data(modelsJSON.utf8)

        let engine = try OpenAICompatibleEngine.openAI(apiKey: TestCredential.openAI, transport: mock)
        let models = try await engine.discoverModels()

        #expect(models.count == 2)
        #expect(models[0].name == "gpt-4")
        #expect(models[0].id.provider == "openai")
        #expect(models[0].author == "openai")
        #expect(models[0].capabilities.contains(.textGeneration))
        #expect(models[1].name == "text-embedding-ada-002")
        #expect(models[1].capabilities.contains(.embedding))
        #expect(models[1].domain == .embedding)
    }

    @Test("discoverModels detects vision models")
    func discoverModelsDetectsVision() async throws {
        let mock = MockHTTPTransport()
        let modelsJSON = """
            {"data": [{"id": "llava-1.5-7b", "owned_by": "local"}]}
            """
        mock.body = Data(modelsJSON.utf8)

        let engine = try OpenAICompatibleEngine.lmStudio(transport: mock)
        let models = try await engine.discoverModels()

        #expect(models.count == 1)
        #expect(models[0].capabilities.contains(.vision))
        #expect(models[0].isLocal)
    }

    // MARK: - W21: where the server is decides what "local" means

    @Test(
        "a model on this network is local, wherever on this network it is",
        arguments: [
            "http://127.0.0.1:1234",  // this machine
            "http://192.168.1.42:1234",  // the Mac on the desk
            "http://studio.local:1234",  // the same Mac, as Bonjour spells it
            "http://10.0.0.7:11434",  // a server in the rack
        ])
    func aModelOnThisNetworkIsLocal(root: String) async throws {
        // T-AIL-43 at the consumer. `isLocal` used to be
        // `host == "127.0.0.1" || host == "localhost"`, so the three
        // non-loopback rows here were reported as CLOUD models in the
        // catalogue the operator picks from — a LAN Mac running LM Studio read
        // as something that costs money and sends the prompt off the network
        // (AIKit#18, panel finding on `OpenAICompatibleEngine.swift:62`).
        let mock = MockHTTPTransport()
        mock.body = Data(#"{"data": [{"id": "devstral", "owned_by": "local"}]}"#.utf8)

        let engine = try OpenAICompatibleEngine(id: "lan", baseURL: root, transport: mock)
        let models = try await engine.discoverModels()

        #expect(try #require(models.first).isLocal, "\(root) is on this network")
    }

    @Test("a model on a cloud server is not local")
    func aCloudModelIsNotLocal() async throws {
        let mock = MockHTTPTransport()
        mock.body = Data(#"{"data": [{"id": "gpt-4", "owned_by": "openai"}]}"#.utf8)

        let engine = try OpenAICompatibleEngine.openAI(apiKey: TestCredential.openAI, transport: mock)
        let models = try await engine.discoverModels()

        #expect(!(try #require(models.first).isLocal))
    }

    // MARK: - Load / Unload

    @Test("loadModel creates provider and tracks it")
    func loadModelCreatesProvider() async throws {
        let mock = MockHTTPTransport()
        let engine = try OpenAICompatibleEngine.lmStudio(transport: mock)

        let descriptor = ModelDescriptor(
            id: ModelIdentifier(provider: "lmstudio", modelId: "qwen2.5-7b"),
            name: "qwen2.5-7b",
            author: "local",
            description: "Test",
            capabilities: .textGeneration,
            format: .gguf,
            parameters: "7B",
            sizeBytes: 0,
            architecture: "qwen",
            domain: .llm
        )

        let provider = try await engine.loadModel(descriptor)
        #expect(provider.id == "lmstudio/qwen2.5-7b")
        #expect(engine.loadedModels().count == 1)
        #expect(engine.loadedModels().first?.modelId == "qwen2.5-7b")

        try await engine.unloadModel(descriptor.id)
        #expect(engine.loadedModels().isEmpty)
    }

    @Test("unloadModel is safe for unknown model")
    func unloadUnknownModel() async throws {
        let engine = try OpenAICompatibleEngine.lmStudio(transport: MockHTTPTransport())
        try await engine.unloadModel(ModelIdentifier(provider: "lmstudio", modelId: "nonexistent"))
        #expect(engine.loadedModels().isEmpty)
    }

    // MARK: - Provider Complete

    @Test("Provider complete() sends correct request and parses response")
    func providerCompleteParsesResponse() async throws {
        let mock = MockHTTPTransport()
        let responseJSON = """
            {
                "id": "chatcmpl-123",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello, world!"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
            }
            """
        mock.body = Data(responseJSON.utf8)

        let engine = try OpenAICompatibleEngine.lmStudio(transport: mock)
        let descriptor = ModelDescriptor(
            id: ModelIdentifier(provider: "lmstudio", modelId: "test-model"),
            name: "test-model",
            author: "local",
            description: "Test",
            capabilities: .textGeneration,
            format: .api,
            parameters: "7B",
            sizeBytes: 0,
            architecture: "unknown",
            domain: .llm
        )

        let provider = try await engine.loadModel(descriptor)
        let request = AIRequest(
            messages: [AIMessage(role: .user, content: "Hi")],
            systemPrompt: "You are helpful.",
            temperature: 0.3,
            maxTokens: 512
        )

        let response = try await provider.complete(request: request)
        #expect(response.content == "Hello, world!")
        #expect(response.model == "test-model")
        #expect(response.tokensUsed.prompt == 10)
        #expect(response.tokensUsed.completion == 5)
        #expect(response.tokensUsed.total == 15)
        #expect(response.latencyMs >= 0)

        // Verify endpoint was called correctly
        #expect(mock.sentRequests.count == 1)
        let urlRequest = try #require(mock.sentRequests.first)
        #expect(urlRequest.url?.path == "/v1/chat/completions")
        #expect(urlRequest.httpMethod == "POST")
    }

    @Test("Provider complete() includes API key in header")
    func providerCompleteIncludesAPIKey() async throws {
        let mock = MockHTTPTransport()
        let responseJSON = """
            {"id":"x","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}
            """
        mock.body = Data(responseJSON.utf8)

        let engine = try OpenAICompatibleEngine.openAI(apiKey: TestCredential.other, transport: mock)
        let descriptor = ModelDescriptor(
            id: ModelIdentifier(provider: "openai", modelId: "gpt-4"),
            name: "gpt-4",
            author: "openai",
            description: "Test",
            capabilities: .textGeneration,
            format: .api,
            parameters: "unknown",
            sizeBytes: 0,
            architecture: "unknown",
            domain: .llm
        )

        let provider = try await engine.loadModel(descriptor)
        _ = try await provider.complete(request: AIRequest(messages: [AIMessage(role: .user, content: "test")]))

        let urlRequest = try #require(mock.sentRequests.first)
        #expect(urlRequest.allHTTPHeaderFields?["Authorization"] == "Bearer \(TestCredential.other)")
    }

    @Test("Provider complete() handles tool calls in response")
    func providerCompleteHandlesToolCalls() async throws {
        let mock = MockHTTPTransport()
        let responseJSON = """
            {
                "id": "chatcmpl-456",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": "{\\"city\\":\\"Paris\\"}"}
                        }]
                    },
                    "finish_reason": "tool_calls"
                }],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}
            }
            """
        mock.body = Data(responseJSON.utf8)

        let engine = try OpenAICompatibleEngine.lmStudio(transport: mock)
        let descriptor = ModelDescriptor(
            id: ModelIdentifier(provider: "lmstudio", modelId: "tool-model"),
            name: "tool-model",
            author: "local",
            description: "Test",
            capabilities: [.textGeneration, .toolUse],
            format: .api,
            parameters: "7B",
            sizeBytes: 0,
            architecture: "unknown",
            domain: .llm
        )

        let provider = try await engine.loadModel(descriptor)
        let response = try await provider.complete(
            request: AIRequest(
                messages: [AIMessage(role: .user, content: "What's the weather?")],
                tools: [
                    AITool(name: "get_weather", description: "Get weather", parametersSchema: "{\"type\":\"object\"}")
                ]
            ))

        #expect(response.content.isEmpty)
        #expect(response.toolCalls?.count == 1)
        #expect(response.toolCalls?.first?.name == "get_weather")
        #expect(response.toolCalls?.first?.arguments == "{\"city\":\"Paris\"}")
    }

    @Test("Provider complete() throws on empty choices")
    func providerCompleteThrowsOnEmptyChoices() async throws {
        let mock = MockHTTPTransport()
        let responseJSON = """
            {"id":"x","choices":[],"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}
            """
        mock.body = Data(responseJSON.utf8)

        let engine = try OpenAICompatibleEngine.lmStudio(transport: mock)
        let descriptor = ModelDescriptor(
            id: ModelIdentifier(provider: "lmstudio", modelId: "empty"),
            name: "empty",
            author: "local",
            description: "Test",
            capabilities: .textGeneration,
            format: .api,
            parameters: "7B",
            sizeBytes: 0,
            architecture: "unknown",
            domain: .llm
        )

        let provider = try await engine.loadModel(descriptor)
        await #expect(throws: AIKitError.self) {
            _ = try await provider.complete(request: AIRequest(messages: [AIMessage(role: .user, content: "test")]))
        }
    }

    @Test("Provider status is always ready")
    func providerStatusIsReady() async throws {
        let mock = MockHTTPTransport()
        mock.body = Data(
            """
            {"id":"x","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}
            """.utf8)

        let engine = try OpenAICompatibleEngine.lmStudio(transport: mock)
        let descriptor = ModelDescriptor(
            id: ModelIdentifier(provider: "lmstudio", modelId: "test"),
            name: "test",
            author: "local",
            description: "Test",
            capabilities: .textGeneration,
            format: .api,
            parameters: "7B",
            sizeBytes: 0,
            architecture: "unknown",
            domain: .llm
        )
        let provider = try await engine.loadModel(descriptor)
        let status = await provider.status
        #expect(status == .ready)
    }
}
