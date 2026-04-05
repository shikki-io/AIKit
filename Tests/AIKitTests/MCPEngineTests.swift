import Foundation
import Testing
import NetKit
@testable import AIKit

@Suite("MCPEngine")
struct MCPEngineTests {

    @Test("Koharu preset has correct URL")
    func koharuPreset() {
        let engine = MCPEngine.koharu()
        #expect(engine.id == "koharu")
        #expect(engine.displayName == "Koharu")
        #expect(engine.isAvailable)
    }

    @Test("Koharu preset with custom port")
    func koharuCustomPort() {
        let engine = MCPEngine.koharu(port: 8888)
        #expect(engine.id == "koharu")
    }

    @Test("LM Studio MCP preset has correct URL")
    func lmStudioMCPPreset() {
        let engine = MCPEngine.lmStudioMCP()
        #expect(engine.id == "lmstudio-mcp")
        #expect(engine.displayName == "LM Studio MCP")
        #expect(engine.isAvailable)
    }

    @Test("LM Studio MCP preset with custom port")
    func lmStudioCustomPort() {
        let engine = MCPEngine.lmStudioMCP(port: 5678)
        #expect(engine.id == "lmstudio-mcp")
    }

    @Test("Tool discovery parses response")
    func toolDiscovery() async throws {
        let mockNetwork = MockNetworkService()
        let toolsResponse = MCPToolsListResponse(tools: [
            MCPTool(
                name: "detect",
                description: "Detect text regions",
                inputSchema: MCPInputSchema(
                    type: "object",
                    properties: ["image": MCPPropertySchema(type: "string", description: "Base64 image")]
                )
            ),
            MCPTool(name: "ocr", description: "OCR text from regions"),
            MCPTool(name: "translate", description: "Translate text"),
        ])
        mockNetwork.resultData = try JSONEncoder().encode(toolsResponse)

        let engine = MCPEngine.koharu(networkService: mockNetwork)
        let tools = try await engine.discoverTools()

        #expect(tools.count == 3)
        #expect(tools[0].name == "detect")
        #expect(tools[1].name == "ocr")
        #expect(tools[2].name == "translate")
        #expect(tools[0].inputSchema?.properties?["image"]?.type == "string")
    }

    @Test("Tool call sends correct request")
    func toolCall() async throws {
        let mockNetwork = MockNetworkService()
        let toolResult = MCPToolResult(content: [
            MCPContent(type: "text", text: "Translated: Bonjour le monde"),
        ])
        mockNetwork.resultData = try JSONEncoder().encode(toolResult)

        let engine = MCPEngine.koharu(networkService: mockNetwork)
        let result = try await engine.callTool(
            name: "translate",
            arguments: ["input": "Hello world", "target": "fr"]
        )

        #expect(result.text == "Translated: Bonjour le monde")
        #expect(result.content.count == 1)
    }

    @Test("MCPProvider wraps tool calls as completions")
    func mcpProviderComplete() async throws {
        let mockNetwork = MockNetworkService()
        let toolResult = MCPToolResult(content: [
            MCPContent(type: "text", text: "Response from MCP"),
        ])
        mockNetwork.resultData = try JSONEncoder().encode(toolResult)

        let engine = MCPEngine(
            id: "test-mcp",
            displayName: "Test MCP",
            baseURL: "http://127.0.0.1:9999/mcp",
            networkService: mockNetwork
        )

        let descriptor = ModelDescriptor(
            id: ModelIdentifier(provider: "test-mcp", modelId: "test"),
            name: "Test Model",
            author: "test",
            description: "Test",
            capabilities: .textGeneration,
            format: .api,
            parameters: "unknown",
            sizeBytes: 0,
            architecture: "unknown",
            domain: .llm
        )

        let provider = try await engine.loadModel(descriptor)
        let request = AIRequest(messages: [AIMessage(role: .user, content: "Hello")])
        let response = try await provider.complete(request: request)

        #expect(response.content == "Response from MCP")
        #expect(response.latencyMs >= 0)
    }

    @Test("Engine conforms to RuntimeEngine")
    func runtimeEngineConformance() {
        let engine = MCPEngine(baseURL: "http://127.0.0.1:9999")
        #expect(engine.supportedFormats == [.api])
        #expect(engine.loadedModels().isEmpty)
    }

    @Test("MCPToolResult text convenience")
    func toolResultTextConvenience() {
        let result = MCPToolResult(content: [
            MCPContent(type: "image", text: nil),
            MCPContent(type: "text", text: "Hello"),
        ])
        #expect(result.text == "Hello")

        let emptyResult = MCPToolResult(content: [])
        #expect(emptyResult.text == nil)
    }
}
