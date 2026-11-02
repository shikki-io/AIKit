import Foundation
import Logging
import NetKit

private let endpointLogger = Logger(label: "aikit.endpoints.mcp")

/// Generic MCP (Model Context Protocol) client engine.
/// Connects to any MCP server exposing tools — Koharu, LM Studio MCP, custom servers.
public final class MCPEngine: RuntimeEngine, @unchecked Sendable {
    public let id: String
    public let displayName: String
    public let supportedFormats: [ModelFormat] = [.api]

    private let baseURL: String
    private let networkService: any NetworkProtocol

    public init(
        id: String = "mcp",
        displayName: String = "MCP Server",
        baseURL: String,
        networkService: (any NetworkProtocol)? = nil
    ) {
        self.id = id
        self.displayName = displayName
        self.baseURL = baseURL
        self.networkService = networkService ?? NetworkService()
    }

    public var isAvailable: Bool {
        // MCP engines are available if configured — actual health is checked at call time.
        true
    }

    // MARK: - MCP Tool Discovery

    /// Discover tools from the MCP server.
    public func discoverTools() async throws -> [MCPTool] {
        let parsed = URLComponents(string: baseURL)
        let endpoint = MCPToolsListEndPoint(
            host: parsed?.host ?? "127.0.0.1",
            port: parsed?.port,
            scheme: parsed?.scheme ?? "http"
        )
        let response: MCPToolsListResponse = try await networkService.sendRequest(endpoint: endpoint)
        return response.tools
    }

    /// Call an MCP tool by name with arguments.
    public func callTool(name: String, arguments: [String: String]) async throws -> MCPToolResult {
        let parsed = URLComponents(string: baseURL)
        let body = MCPCallToolRequest(name: name, arguments: arguments)
        let endpoint = try MCPCallToolEndPoint(
            host: parsed?.host ?? "127.0.0.1",
            port: parsed?.port,
            scheme: parsed?.scheme ?? "http",
            requestBody: body
        )
        return try await networkService.sendRequest(endpoint: endpoint)
    }

    // MARK: - RuntimeEngine

    public func loadModel(_ descriptor: ModelDescriptor) async throws -> any AIProvider {
        MCPProvider(
            id: "\(id)/\(descriptor.id.modelId)",
            displayName: "\(displayName) — \(descriptor.name)",
            capabilities: descriptor.capabilities,
            engine: self
        )
    }

    public func unloadModel(_ id: ModelIdentifier) async throws {
        // MCP engines are stateless — nothing to unload.
    }

    public func loadedModels() -> [ModelIdentifier] {
        // MCP engines are stateless — no loaded models to track.
        []
    }
}

// MARK: - MCP DTOs

/// A tool exposed by an MCP server.
public struct MCPTool: Sendable, Codable, Equatable {
    public let name: String
    public let description: String
    public let inputSchema: MCPInputSchema?

    public init(name: String, description: String, inputSchema: MCPInputSchema? = nil) {
        self.name = name
        self.description = description
        self.inputSchema = inputSchema
    }
}

/// JSON Schema-like description of a tool's input.
public struct MCPInputSchema: Sendable, Codable, Equatable {
    public let type: String
    public let properties: [String: MCPPropertySchema]?

    public init(type: String = "object", properties: [String: MCPPropertySchema]? = nil) {
        self.type = type
        self.properties = properties
    }
}

/// Schema for a single property in an MCP tool input.
public struct MCPPropertySchema: Sendable, Codable, Equatable {
    public let type: String
    public let description: String?

    public init(type: String, description: String? = nil) {
        self.type = type
        self.description = description
    }
}

/// Result from an MCP tool call.
public struct MCPToolResult: Sendable, Codable, Equatable {
    public let content: [MCPContent]

    public init(content: [MCPContent]) {
        self.content = content
    }

    /// Convenience: first text content value.
    public var text: String? {
        content.first { $0.type == "text" }?.text
    }
}

/// A content block in an MCP response.
public struct MCPContent: Sendable, Codable, Equatable {
    public let type: String
    public let text: String?

    public init(type: String = "text", text: String? = nil) {
        self.type = type
        self.text = text
    }
}

// MARK: - Internal DTOs

struct MCPToolsListResponse: Sendable, Codable {
    let tools: [MCPTool]
}

struct MCPCallToolRequest: Sendable, Codable {
    let name: String
    let arguments: [String: String]
}

// MARK: - MCP Endpoints

struct MCPToolsListEndPoint: EndPoint {
    let host: String
    let port: Int?
    let scheme: String

    var apiPath: String { "" }
    var apiFilePath: String { "" }
    var path: String { "/tools/list" }
    var method: RequestMethod { .GET }
    var header: [String: String]? { ["Content-Type": "application/json"] }
    var body: [String: Any]? { nil }
    var queryParams: [String: Any]? { nil }
}

struct MCPCallToolEndPoint: EndPoint {
    let host: String
    let port: Int?
    let scheme: String
    let requestBody: MCPCallToolRequest
    /// JSON-encoded body, verified at init to be a JSON object.
    ///
    /// `Data` is `Sendable`; `[String: Any]` is not. The body is encoded once
    /// here so encoding failures reach the caller as a thrown error rather than
    /// silently producing a bodyless request.
    private let _bodyData: Data

    /// - Throws: An encoding error when `requestBody` cannot be serialised to JSON.
    init(
        host: String,
        port: Int?,
        scheme: String,
        requestBody: MCPCallToolRequest
    ) throws {
        self.host = host
        self.port = port
        self.scheme = scheme
        self.requestBody = requestBody
        let data = try JSONEncoder().encode(requestBody)
        guard try JSONSerialization.jsonObject(with: data) is [String: Any] else {
            throw EncodingError.invalidValue(
                requestBody,
                .init(codingPath: [], debugDescription: "MCPCallToolRequest did not encode to a JSON object")
            )
        }
        self._bodyData = data
    }

    var apiPath: String { "" }
    var apiFilePath: String { "" }
    var path: String { "/tools/call" }
    var method: RequestMethod { .POST }
    var header: [String: String]? { ["Content-Type": "application/json"] }
    var queryParams: [String: Any]? { nil }
    var body: [String: Any]? {
        // Safety: _bodyData was verified to be a JSON object at init time.
        (try? JSONSerialization.jsonObject(with: _bodyData)) as? [String: Any]
    }
}
