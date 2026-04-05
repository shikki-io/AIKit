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

        return AIResponse(
            content: content,
            model: id,
            tokensUsed: TokenUsage(prompt: 0, completion: 0),
            latencyMs: latencyMs
        )
    }
}
