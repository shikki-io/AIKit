import NetKit

// MARK: - What these are, and what they are not
//
// Operator, AIKit#18 r4493375803: "What are those MCP Presets? I don't see all
// the shikki mcps & moto one so I don't get what it it?"
//
// They are TWO CONSTRUCTORS, nothing more: an ``MCPEngine`` already pointed at
// a server whose address and mount path were measured once, so a caller writing
// a demo or a test does not retype `http://127.0.0.1:9999/mcp`. They carry no
// registry, no discovery and no list of the fleet's MCP servers — which is why
// shikki's own MCP servers are absent, and absent on purpose:
//
//  * `shikki-db`, `shikki-cli`, `shikki-chat`, `moto` and `ShikkiMCP` are
//    SHIKKI vocabulary, and BR-AIL-03 of spec ai-layer-aikit-core-2026-10-31
//    keeps AIKit free of it. AIKit does not know what a plan, a spec or a verb
//    is, so it cannot hold the list of servers that serve them. The host names
//    its own servers; this package only knows how to speak to one.
//  * Those servers are also not AI PROVIDERS. They expose tools to a model
//    through a client's MCP config; they do not answer an ``AIRequest``. The
//    two below are the opposite case — an MCP server whose tool is itself the
//    completion — which is the only case ``MCPProvider`` serves.
//
// They are also the wrong shape going forward. A server address is operator
// configuration, and BR-AIL-20 puts configuration in the settings document and
// reaches it through a repository, never in a Swift literal. These survive only
// until the MCP client moves to obyw-one/SwiftMCP (backlog
// 534204d0-fca6-48c4-aabe-0b4c6be3efcf), where the move deletes them rather
// than carrying them over: an MCP server becomes a settings entry like every
// other provider.
//
// The `local` in each name is load-bearing: both default to the loopback
// interface, so neither is a default for anything reachable over a network.

extension MCPEngine {

    /// An engine pointed at a **Koharu manga-translator MCP server running on
    /// this machine**, at its default mount `http://127.0.0.1:<port>/mcp`.
    ///
    /// A convenience for a demo or a test, not configuration: a real
    /// deployment names its server in the settings document. See the note at
    /// the top of this file.
    ///
    /// - Parameter port: the port Koharu listens on; 9999 is its own default.
    static func localKoharu(
        port: Int = 9_999,
        networkService: any NetworkProtocol = NetworkService()
    ) -> MCPEngine {
        MCPEngine(
            id: "koharu",
            displayName: "Koharu",
            baseURL: "http://127.0.0.1:\(port)/mcp",
            networkService: networkService
        )
    }

    /// An engine pointed at **LM Studio's MCP surface on this machine**, at
    /// `http://127.0.0.1:<port>/mcp`.
    ///
    /// Distinct from every other LM Studio entry point in this package: the
    /// four wire families reach LM Studio's *model* APIs
    /// (`/v1/chat/completions`, `/v1/responses`, `/v1/messages`,
    /// `/api/v1/chat`, each a case of its own `EndPoint` enum), while this one
    /// reaches the tool server it can also host. Same process, same port,
    /// different protocol.
    ///
    /// - Parameter port: LM Studio's server port; 1234 is its own default.
    static func localLMStudio(
        port: Int = 1_234,
        networkService: any NetworkProtocol = NetworkService()
    ) -> MCPEngine {
        MCPEngine(
            id: "lmstudio-mcp",
            displayName: "LM Studio MCP",
            baseURL: "http://127.0.0.1:\(port)/mcp",
            networkService: networkService
        )
    }
}
