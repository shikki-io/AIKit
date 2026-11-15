import Foundation
import Logging
import NetKit

private let endpointLogger = Logger(label: "aikit.endpoints.mcp")

/// Generic MCP (Model Context Protocol) client engine.
/// Connects to any MCP server exposing tools — Koharu, LM Studio MCP, custom servers.
final class MCPEngine: RuntimeEngine, @unchecked Sendable {
    let id: String
    let displayName: String
    let supportedFormats: [ModelFormat] = [.api]

    private let baseURL: String
    private let networkService: any NetworkProtocol

    init(
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

    var isAvailable: Bool {
        // MCP engines are available if configured — actual health is checked at call time.
        true
    }

    // MARK: - MCP Tool Discovery

    /// The configured server, validated.
    ///
    /// Throwing on purpose. What this replaces defaulted a missing scheme to
    /// `http` and a missing host to `127.0.0.1`, so a misconfigured MCP server
    /// address sent its call to the local machine instead of reporting the
    /// mistake (AIKit#18, panel finding on `MCPEngine.swift:37-41`).
    ///
    /// NOT an ``AIServerAddress``: an MCP server is addressed WITH its mount
    /// path (`http://127.0.0.1:9999/mcp`), and `/tools/list` hangs off that
    /// mount. BR-AIL-18 governs the four AI wire families, whose routes are
    /// absolute and whose addresses are therefore roots; folding MCP into that
    /// rule would reject every real MCP URL. The mount travels as the
    /// endpoint's `apiPath`, which is what that `EndPoint` member is for.
    private func server() throws(AIKitError) -> MCPServerLocation {
        try MCPServerLocation(validating: baseURL)
    }

    /// Discover tools from the MCP server.
    func discoverTools() async throws -> [MCPTool] {
        let response: MCPToolsListResponse = try await networkService.sendRequest(
            endpoint: MCPEndPoint.toolsList(server: try server())
        )
        return response.tools
    }

    /// Call an MCP tool by name with arguments.
    func callTool(name: String, arguments: [String: String]) async throws -> MCPToolResult {
        try await networkService.sendRequest(
            endpoint: MCPEndPoint.callTool(server: try server(), name: name, arguments: arguments)
        )
    }

    // MARK: - RuntimeEngine

    func loadModel(_ descriptor: ModelDescriptor) async throws -> any AIProvider {
        MCPProvider(
            id: "\(id)/\(descriptor.id.modelId)",
            displayName: "\(displayName) — \(descriptor.name)",
            capabilities: descriptor.capabilities,
            engine: self
        )
    }

    func unloadModel(_ id: ModelIdentifier) async throws {
        // MCP engines are stateless — nothing to unload.
    }

    func loadedModels() -> [ModelIdentifier] {
        // MCP engines are stateless — no loaded models to track.
        []
    }
}

// MARK: - MCP DTOs

/// A tool exposed by an MCP server.
struct MCPTool: Sendable, Codable, Equatable {
    let name: String
    let description: String
    let inputSchema: MCPInputSchema?

    init(name: String, description: String, inputSchema: MCPInputSchema? = nil) {
        self.name = name
        self.description = description
        self.inputSchema = inputSchema
    }
}

/// JSON Schema-like description of a tool's input.
struct MCPInputSchema: Sendable, Codable, Equatable {
    let type: String
    let properties: [String: MCPPropertySchema]?

    init(type: String = "object", properties: [String: MCPPropertySchema]? = nil) {
        self.type = type
        self.properties = properties
    }
}

/// Schema for a single property in an MCP tool input.
struct MCPPropertySchema: Sendable, Codable, Equatable {
    let type: String
    let description: String?

    init(type: String, description: String? = nil) {
        self.type = type
        self.description = description
    }
}

/// Result from an MCP tool call.
struct MCPToolResult: Sendable, Codable, Equatable {
    let content: [MCPContent]

    init(content: [MCPContent]) {
        self.content = content
    }

    /// Convenience: first text content value.
    var text: String? {
        content.first { $0.type == "text" }?.text
    }
}

/// A content block in an MCP response.
struct MCPContent: Sendable, Codable, Equatable {
    let type: String
    let text: String?

    init(type: String = "text", text: String? = nil) {
        self.type = type
        self.text = text
    }
}

// MARK: - Internal DTOs

struct MCPToolsListResponse: Sendable, Codable {
    let tools: [MCPTool]
}

/// Body of `POST /tools/call`.
struct MCPCallToolRequest: Codable, Sendable, Equatable {
    let name: String
    let arguments: [String: String]
}

// MARK: - MCPServerLocation

/// Where an MCP server is mounted: scheme, host, port, and the mount path its
/// tool routes hang off.
///
/// Validated once, with nothing defaulted — the point of the type. A missing
/// scheme or host is an error naming the value, rather than a quiet rewrite to
/// `http://127.0.0.1`.
struct MCPServerLocation: Sendable, Equatable {
    let scheme: String
    let host: String
    let port: Int?
    /// The mount path, with no trailing slash. Empty when the server is
    /// mounted at the root.
    let mountPath: String

    /// - Throws: ``AIKitError/invalidServerAddress(value:reason:)``.
    init(validating text: String) throws(AIKitError) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw .invalidServerAddress(value: text, reason: "it is empty")
        }
        guard let components = URLComponents(string: trimmed) else {
            throw .invalidServerAddress(value: trimmed, reason: "it is not a URL")
        }
        guard let scheme = components.scheme?.lowercased(), scheme == "http" || scheme == "https" else {
            throw .invalidServerAddress(
                value: trimmed,
                reason: "it names no http or https scheme — write http:// in front of the host"
            )
        }
        guard let host = components.host, !host.isEmpty else {
            throw .invalidServerAddress(value: trimmed, reason: "it names no host")
        }
        self.scheme = scheme
        self.host = host
        self.port = components.port
        var path = components.path
        while path.hasSuffix("/") { path.removeLast() }
        self.mountPath = path
    }

    var description: String {
        var text = "\(scheme)://\(host)"
        if let port { text += ":\(port)" }
        return text + mountPath
    }
}

// MARK: - MCPEndPoint

/// Every call this engine makes to an MCP server, one case each — the same
/// shape as NetKit's `UmamiEndPoint`: the server travels with the case, so the
/// endpoint holds no state.
enum MCPEndPoint: EndPoint {
    /// `GET /tools/list` — discover the tools a server exposes.
    case toolsList(server: MCPServerLocation)
    /// `POST /tools/call` — run one tool with string arguments.
    case callTool(server: MCPServerLocation, name: String, arguments: [String: String])

    private var server: MCPServerLocation {
        switch self {
        case .toolsList(let server): server
        case .callTool(let server, _, _): server
        }
    }

    var scheme: String { server.scheme }
    var host: String { server.host }
    var port: Int? { server.port }
    /// The server's mount path. The tool routes below hang off it, so a server
    /// at `http://host:9999/mcp` is called at `/mcp/tools/list`.
    var apiPath: String { server.mountPath }
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
        // Safety: `bodyObject()` only fails when the DTO cannot encode, and a
        // name plus string arguments always can; engines call it directly.
        (try? bodyObject())
    }

    /// The request body as a JSON object; nil for a call that has none.
    func bodyObject() throws -> [String: Any]? {
        switch self {
        case .toolsList:
            return nil
        case .callTool(_, let name, let arguments):
            return try encodeWireJSONObject(
                MCPCallToolRequest(name: name, arguments: arguments), label: "MCPCallToolRequest")
        }
    }
}
