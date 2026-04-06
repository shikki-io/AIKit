import NetKit

public extension MCPEngine {
    /// Pre-configured for Koharu manga translator MCP server.
    static func koharu(
        port: Int = 9999,
        networkService: (any NetworkProtocol)? = nil
    ) -> MCPEngine {
        MCPEngine(
            id: "koharu",
            displayName: "Koharu",
            baseURL: "http://127.0.0.1:\(port)/mcp",
            networkService: networkService
        )
    }

    /// Pre-configured for LM Studio MCP server.
    static func lmStudioMCP(
        port: Int = 1234,
        networkService: (any NetworkProtocol)? = nil
    ) -> MCPEngine {
        MCPEngine(
            id: "lmstudio-mcp",
            displayName: "LM Studio MCP",
            baseURL: "http://127.0.0.1:\(port)/mcp",
            networkService: networkService
        )
    }
}
