import Foundation

/// AIProvider that wraps MCP tool calls as AI completions.
/// Translates AIRequest into MCP tool calls and maps responses back to AIResponse.
public struct MCPProvider: AIProvider, Sendable {
    public let id: String
    public let displayName: String
    public let capabilities: AICapabilities

    private let engine: MCPEngine

    public init(
        id: String,
        displayName: String,
        capabilities: AICapabilities,
        engine: MCPEngine
    ) {
        self.id = id
        self.displayName = displayName
        self.capabilities = capabilities
        self.engine = engine
    }

    public var status: AIProviderStatus {
        get async { .ready }
    }

    public func complete(request: AIRequest) async throws -> AIResponse {
        let startTime = ContinuousClock.now

        // Extract the last user message as the prompt.
        guard let lastMessage = request.messages.last(where: { $0.role == .user }) else {
            throw AIKitError.requestFailed("No user message in request")
        }

        // If tools are specified, call the first matching one.
        // Otherwise, use the message content as a generic "completion" tool call.
        let result: MCPToolResult
        if let tools = request.tools, let firstTool = tools.first {
            result = try await engine.callTool(
                name: firstTool.name,
                arguments: ["input": lastMessage.content]
            )
        } else {
            result = try await engine.callTool(
                name: "complete",
                arguments: ["prompt": lastMessage.content]
            )
        }

        let elapsed = startTime.duration(to: .now)
        let latencyMs = Int(elapsed.components.seconds * 1000 + elapsed.components.attoseconds / 1_000_000_000_000_000)

        let content = result.text ?? result.content.compactMap(\.text).joined(separator: "\n")

        // `tokensUsed` is nil/nil because the PROTOCOL reports no counts, not
        // because this provider has not got round to reading them. MCP's
        // `CallToolResult` carries `content`, `structuredContent?`, `isError?`
        // and `_meta?`, and the schema defines no token field anywhere
        // (modelcontextprotocol, schema/2025-06-18/schema.ts, read
        // 2026-11-12). `TokenUsage` documents nil as exactly this — "the
        // provider did not report it" — and never 0, so a budget check cannot
        // read a missing count as free. There is nothing here to fill.
        //
        // FIXME(backlog 97a377f8-be0c-4711-986e-a25d29827ec5): what IS
        // unmanaged is the rest of this mapping, and it is deliberately left to
        // the move to obyw-one/SwiftMCP (backlog
        // 534204d0-fca6-48c4-aabe-0b4c6be3efcf) rather than written twice:
        // `isError: true` is ignored, so a FAILED tool call returns as a
        // successful answer and never triggers a fallback; `structuredContent`
        // and every non-text content block are dropped instead of refused; the
        // request is flattened to its last user message, dropping the system
        // prompt and every earlier turn; with several tools only `tools.first`
        // is called, with an invented argument shape rather than one built from
        // the tool's `inputSchema`; and `toolCalls` is never filled, so the
        // caller cannot see which tool answered. Spec
        // ai-layer-aikit-core-2026-10-31 W26, operator thread AIKit#18
        // r4500255949.
        return AIResponse(
            content: content,
            model: id,
            tokensUsed: TokenUsage(prompt: nil, completion: nil),
            latencyMs: latencyMs,
            providerID: id
        )
    }
}
