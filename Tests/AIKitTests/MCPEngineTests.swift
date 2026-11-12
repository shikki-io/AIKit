import Foundation
import NetKit
import Testing

@testable import AIKit

@Suite("MCPEngine")
struct MCPEngineTests {

    @Test("Koharu preset has correct URL")
    func koharuPreset() {
        let engine = MCPEngine.localKoharu()
        #expect(engine.id == "koharu")
        #expect(engine.displayName == "Koharu")
        #expect(engine.isAvailable)
    }

    @Test("Koharu preset with custom port")
    func koharuCustomPort() {
        let engine = MCPEngine.localKoharu(port: 8888)
        #expect(engine.id == "koharu")
    }

    @Test("LM Studio MCP preset has correct URL")
    func lmStudioMCPPreset() {
        let engine = MCPEngine.localLMStudio()
        #expect(engine.id == "lmstudio-mcp")
        #expect(engine.displayName == "LM Studio MCP")
        #expect(engine.isAvailable)
    }

    @Test("LM Studio MCP preset with custom port")
    func lmStudioCustomPort() {
        let engine = MCPEngine.localLMStudio(port: 5678)
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

        let engine = MCPEngine.localKoharu(networkService: mockNetwork)
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
            MCPContent(type: "text", text: "Translated: Bonjour le monde")
        ])
        mockNetwork.resultData = try JSONEncoder().encode(toolResult)

        let engine = MCPEngine.localKoharu(networkService: mockNetwork)
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
            MCPContent(type: "text", text: "Response from MCP")
        ])
        mockNetwork.resultData = try JSONEncoder().encode(toolResult)

        let engine = MCPEngine(
            id: "test-mcp",
            displayName: "Test MCP",
            baseURL: "\(fixtureMCPServerRoot)/mcp",
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
        let engine = MCPEngine(baseURL: fixtureMCPServerRoot)
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

    // MARK: - MCPEndPoint

    @Test("MCPEndPoint.toolsList is a bodyless GET on /tools/list at the server")
    func endPointToolsList() throws {
        let endpoint = MCPEndPoint.toolsList(server: try MCPServerLocation(validating: fixtureMCPServerRoot))
        #expect(endpoint.method == .GET)
        #expect(endpoint.path == "/tools/list")
        #expect(endpoint.host == "127.0.0.1")
        #expect(endpoint.port == 9999)
        #expect(endpoint.scheme == "http")
        #expect(endpoint.body == nil)
        #expect(endpoint.apiPath.isEmpty, "a server at the root has no mount path")
    }

    @Test("a mounted MCP server keeps its mount path, and the tool routes hang off it")
    func endPointHonoursAMountPath() throws {
        // An MCP server is addressed WITH its mount (`/mcp`), unlike the four
        // AI wire families whose routes are absolute. That is why it has its
        // own validated location type and not `AIServerAddress`.
        let endpoint = MCPEndPoint.toolsList(
            server: try MCPServerLocation(validating: "\(fixtureMCPServerRoot)/mcp")
        )
        #expect(endpoint.apiPath == "/mcp")
        #expect(try NetworkService().createRequest(endPoint: endpoint).url?.path == "/mcp/tools/list")
    }

    @Test(
        "an MCP address with no scheme or host is refused, never defaulted",
        arguments: [
            "", "127.0.0.1:9999", "not a url", "ftp://host/mcp",
        ])
    func aBadMCPAddressIsRefused(input: String) throws {
        // What this replaces defaulted a missing scheme to `http` and a missing
        // host to `127.0.0.1`, sending the call to the local machine instead of
        // reporting the mistake.
        #expect(throws: AIKitError.self) { _ = try MCPServerLocation(validating: input) }
    }

    @Test("a trailing slash is not a mount path")
    func aTrailingSlashIsNotAMount() throws {
        #expect(try MCPServerLocation(validating: "http://host:9999/").mountPath.isEmpty)
        #expect(try MCPServerLocation(validating: "http://host:9999/mcp/").mountPath == "/mcp")
    }

    @Test("MCPEndPoint.callTool is a POST on /tools/call carrying name and arguments")
    func endPointCallTool() throws {
        let endpoint = MCPEndPoint.callTool(
            server: try MCPServerLocation(validating: fixtureMCPServerRoot),
            name: "ocr",
            arguments: ["page": "1"]
        )
        #expect(endpoint.method == .POST)
        #expect(endpoint.path == "/tools/call")
        let body = try #require(endpoint.body)
        #expect(body["name"] as? String == "ocr")
        #expect(body["arguments"] as? [String: String] == ["page": "1"])
    }

    @Test("MCPCallToolRequest encodes the exact bytes the hand-written literal did")
    func callToolBodyByteParity() throws {
        let arguments = ["page": "1", "lang": "fr"]
        let legacy: [String: Any] = ["name": "ocr", "arguments": arguments]
        let legacyData = try JSONSerialization.data(withJSONObject: legacy, options: [.sortedKeys])

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let dtoData = try encoder.encode(MCPCallToolRequest(name: "ocr", arguments: arguments))
        #expect(dtoData == legacyData)

        // And the other way: the legacy bytes decode into the DTO.
        let decoded = try JSONDecoder().decode(MCPCallToolRequest.self, from: legacyData)
        #expect(decoded == MCPCallToolRequest(name: "ocr", arguments: arguments))
    }
}
