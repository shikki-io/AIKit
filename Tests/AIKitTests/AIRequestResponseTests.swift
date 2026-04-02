import Foundation
import Testing
@testable import AIKit

@Suite("AIRequest and AIResponse")
struct AIRequestResponseTests {

    @Test("AIRequest Codable round-trip")
    func requestCodable() throws {
        let request = AIRequest(
            messages: [
                AIMessage(role: .system, content: "You are helpful."),
                AIMessage(role: .user, content: "Hello"),
            ],
            systemPrompt: "Be concise",
            temperature: 0.5,
            maxTokens: 256,
            model: ModelIdentifier(provider: "openai", modelId: "gpt-5"),
            tools: [AITool(name: "search", description: "Search the web", parametersSchema: "{}")],
            responseFormat: .json
        )

        let encoder = JSONEncoder()
        let decoder = JSONDecoder()
        let data = try encoder.encode(request)
        let decoded = try decoder.decode(AIRequest.self, from: data)

        #expect(decoded.messages.count == 2)
        #expect(decoded.messages[0].role == .system)
        #expect(decoded.messages[1].content == "Hello")
        #expect(decoded.systemPrompt == "Be concise")
        #expect(decoded.temperature == 0.5)
        #expect(decoded.maxTokens == 256)
        #expect(decoded.model?.provider == "openai")
        #expect(decoded.tools?.first?.name == "search")
        #expect(decoded.responseFormat == .json)
    }

    @Test("AIResponse Codable round-trip")
    func responseCodable() throws {
        let response = AIResponse(
            content: "Hello!",
            model: "gpt-5",
            tokensUsed: TokenUsage(prompt: 10, completion: 5),
            latencyMs: 42,
            toolCalls: [AIToolCall(name: "search", arguments: "{\"q\": \"test\"}")]
        )

        let encoder = JSONEncoder()
        let decoder = JSONDecoder()
        let data = try encoder.encode(response)
        let decoded = try decoder.decode(AIResponse.self, from: data)

        #expect(decoded.content == "Hello!")
        #expect(decoded.model == "gpt-5")
        #expect(decoded.tokensUsed.prompt == 10)
        #expect(decoded.tokensUsed.completion == 5)
        #expect(decoded.latencyMs == 42)
        #expect(decoded.toolCalls?.first?.name == "search")
    }

    @Test("TokenUsage total is computed from prompt + completion")
    func tokenUsageTotal() {
        let usage = TokenUsage(prompt: 100, completion: 50)
        #expect(usage.total == 150)
    }

    @Test("TokenUsage Codable encodes total but ignores it on decode")
    func tokenUsageCodable() throws {
        let usage = TokenUsage(prompt: 20, completion: 10)
        let encoder = JSONEncoder()
        let data = try encoder.encode(usage)

        // Verify total is encoded
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        #expect(json?["total"] as? Int == 30)

        // Verify decode recomputes total
        let decoded = try JSONDecoder().decode(TokenUsage.self, from: data)
        #expect(decoded.prompt == 20)
        #expect(decoded.completion == 10)
        #expect(decoded.total == 30)
    }

    @Test("AIMessage roles cover all cases")
    func messageRoles() throws {
        let roles: [AIMessage.Role] = [.system, .user, .assistant, .tool]
        for role in roles {
            let msg = AIMessage(role: role, content: "test")
            let data = try JSONEncoder().encode(msg)
            let decoded = try JSONDecoder().decode(AIMessage.self, from: data)
            #expect(decoded.role == role)
        }
    }
}
