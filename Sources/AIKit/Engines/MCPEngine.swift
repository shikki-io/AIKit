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
        networkService: any NetworkProtocol = NetworkService()
    ) {
        self.id = id
        self.displayName = displayName
        self.baseURL = baseURL
        self.networkService = networkService
    }

    public var isAvailable: Bool {
        // MCP engines are available if configured — actual health is checked at call time.
        true
    }

    // MARK: - MCP Tool Discovery

    /// The configured server as an `AIWireServer`; scheme and host default as before.
    private var server: AIWireServer {
        let parsed = URLComponents(string: baseURL)
        return AIWireServer(
            scheme: parsed?.scheme ?? "http",
            host: parsed?.host ?? "127.0.0.1",
            port: parsed?.port
        )
    }

    /// Discover tools from the MCP server.
    public func discoverTools() async throws -> [MCPTool] {
        let response: MCPToolsListResponse = try await networkService.sendRequest(
            endpoint: MCPEndPoint.toolsList(server: server)
        )
        return response.tools
    }

    /// Call an MCP tool by name with arguments.
    public func callTool(name: String, arguments: [String: String]) async throws -> MCPToolResult {
        try await networkService.sendRequest(
            endpoint: MCPEndPoint.callTool(server: server, name: name, arguments: arguments)
        )
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

// MARK: - MCPEndPoint

/// Every call this engine makes to an MCP server, one case each — the same
/// shape as NetKit's `UmamiEndPoint`: the server travels with the case, so the
/// endpoint holds no state.
enum MCPEndPoint: EndPoint {
    /// `GET /tools/list` — discover the tools a server exposes.
    case toolsList(server: AIWireServer)
    /// `POST /tools/call` — run one tool with string arguments.
    case callTool(server: AIWireServer, name: String, arguments: [String: String])

    private var server: AIWireServer {
        switch self {
        case .toolsList(let server): server
        case .callTool(let server, _, _): server
        }
    }

    var scheme: String { server.scheme }
    var host: String { server.host }
    var port: Int? { server.port }
    var apiPath: String { "" }
    var apiFilePath: String { "" }

    var path: String {
        switch self {
        case .toolsList: "/tools/list"
        case .callTool: "/tools/call"
        }
    }

    var method: RequestMethod {
        switch self {
        case .toolsList: .GET
        case .callTool: .POST
        }
    }

    var header: [String: String]? { ["Content-Type": "application/json"] }
    var queryParams: [String: Any]? { nil }

    var body: [String: Any]? {
        switch self {
        case .toolsList: nil
        case .callTool(_, let name, let arguments): ["name": name, "arguments": arguments]
        }
    }
}
