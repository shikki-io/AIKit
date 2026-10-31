import Foundation
import Testing

@testable import AIKit

// MARK: - AgentProviderAdapterTests
//
// T-AIL-05. The adapter presents the narrow `AgentProviding` seam as the typed
// `AIProvider` one, so a prompt-and-text backend — a CLI subprocess, a chain,
// LM Studio — routes like every other provider. What it must not do is lose
// anything on the way through: the text comes back as it was returned, and the
// backend's error arrives as the backend threw it, because that error is what a
// chain reads to decide on a fallback.

/// Records what the adapter handed down, and answers with text or an error.
private final class StubAgent: AgentProviding, @unchecked Sendable {
    private(set) var prompts: [String] = []
    private(set) var timeouts: [TimeInterval] = []
    let output: String
    let error: (any Error)?

    init(output: String = "", error: (any Error)? = nil) {
        self.output = output
        self.error = error
    }

    func run(prompt: String, timeout: TimeInterval) async throws -> String {
        prompts.append(prompt)
        timeouts.append(timeout)
        if let error { throw error }
        return output
    }
}

private func makeSUT(
    agent: any AgentProviding,
    id: String = "claude-code",
    modelName: String = "claude-opus-5",
    timeout: TimeInterval = 90
) -> AgentProviderAdapter {
    AgentProviderAdapter(
        id: id,
        displayName: "Claude Code",
        modelName: modelName,
        timeout: timeout,
        agent: agent
    )
}

@Suite("AgentProviderAdapter — a prompt backend routes like any provider")
struct AgentProviderAdapterTests {

    // MARK: - T-AIL-05

    @Test("both directions: the prompt reaches the backend, the text and the error come back whole")
    func bothDirections() async throws {
        // Down, then up: the backend sees the request's text and the adapter's
        // timeout, and its answer arrives in `content` named by this provider.
        let answering = StubAgent(output: "42")
        let adapter = makeSUT(agent: answering)

        let response = try await adapter.complete(
            request: AIRequest(messages: [AIMessage(role: .user, content: "What is six times seven?")])
        )

        #expect(answering.prompts == ["What is six times seven?"])
        #expect(answering.timeouts == [90])
        #expect(response.content == "42")
        #expect(response.providerID == "claude-code")
        #expect(response.model == "claude-opus-5")
        #expect(response.toolCalls == nil)
        #expect(response.tokensUsed.total == 0)
        #expect(response.latencyMs >= 0)

        // And the other direction for a failure: the backend's own error type
        // arrives, not a wrapper, so `FallbackProviderChain`'s rule still reads
        // the case it was written for.
        let failing = StubAgent(error: LMStudioProvider.LMStudioError.rateLimited)
        let failingAdapter = makeSUT(agent: failing)

        await #expect(throws: LMStudioProvider.LMStudioError.rateLimited) {
            _ = try await failingAdapter.complete(
                request: AIRequest(messages: [AIMessage(role: .user, content: "anything")])
            )
        }
    }

    @Test("a preserved error stays fallback-eligible through the adapter")
    func preservedErrorStaysFallbackEligible() async throws {
        let adapter = makeSUT(agent: StubAgent(error: URLError(.cannotConnectToHost)))

        do {
            _ = try await adapter.complete(
                request: AIRequest(messages: [AIMessage(role: .user, content: "ping")])
            )
            Issue.record("Expected the backend's error to propagate")
        } catch {
            #expect(FallbackProviderChain.isFallbackEligible(error))
        }
    }

    // MARK: - Prompt rendering

    @Test("a single message round trips to exactly its own text")
    func singleMessageRoundTrips() {
        let rendered = AgentProviderAdapter.renderPrompt(
            AIRequest(messages: [AIMessage(role: .user, content: "Summarise this diff.")])
        )
        #expect(rendered == "Summarise this diff.")
    }

    @Test("the system prompt leads, and several turns carry their role")
    func systemPromptLeadsAndTurnsAreLabelled() {
        let rendered = AgentProviderAdapter.renderPrompt(
            AIRequest(
                messages: [
                    AIMessage(role: .user, content: "First?"),
                    AIMessage(role: .assistant, content: "Yes."),
                    AIMessage(role: .user, content: "Second?"),
                ],
                systemPrompt: "You are terse."
            )
        )

        #expect(rendered == """
        You are terse.

        user: First?

        assistant: Yes.

        user: Second?
        """)
    }

    @Test("a system prompt in front of one message still labels nothing")
    func systemPromptWithOneMessageStaysUnlabelled() {
        let rendered = AgentProviderAdapter.renderPrompt(
            AIRequest(
                messages: [AIMessage(role: .user, content: "Go.")],
                systemPrompt: "You are terse."
            )
        )
        #expect(rendered == "You are terse.\n\nGo.")
    }

    // MARK: - Refusals and status

    @Test("media is refused, never dropped")
    func refusesMediaInsteadOfDroppingIt() async throws {
        let agent = StubAgent(output: "should not be reached")
        let adapter = makeSUT(agent: agent)
        let request = AIRequest(
            messages: [AIMessage(role: .user, content: "Transcribe this.")],
            inputs: [AIInput(kind: .audio, url: URL(fileURLWithPath: "/tmp/take.wav"), mimeType: "audio/wav")]
        )

        await #expect(throws: AIKitError.self) {
            _ = try await adapter.complete(request: request)
        }
        #expect(agent.prompts.isEmpty, "the backend must not be called with the media silently dropped")
    }

    @Test("status is ready — a prompt backend has nothing to load")
    func statusIsReady() async {
        let status = await makeSUT(agent: StubAgent(output: "x")).status
        #expect(status == .ready)
    }

    @Test("capabilities are text generation unless the caller declares more")
    func capabilitiesDefaultToText() {
        let adapter = makeSUT(agent: StubAgent(output: "x"))
        #expect(adapter.capabilities == .textGeneration)
        #expect(!adapter.capabilities.contains(.toolUse))
    }

    @Test("streaming falls back to one complete chunk")
    func streamYieldsTheWholeAnswer() async throws {
        let adapter = makeSUT(agent: StubAgent(output: "one shot"))

        var chunks: [AIChunk] = []
        for try await chunk in try await adapter.stream(
            request: AIRequest(messages: [AIMessage(role: .user, content: "go")])
        ) {
            chunks.append(chunk)
        }

        #expect(chunks.count == 1)
        #expect(chunks.first?.delta == "one shot")
        #expect(chunks.first?.isComplete == true)
    }

    // MARK: - Routing

    @Test("the adapter is a provider a router can hold")
    func adapterIsAnAIProvider() async throws {
        let providers: [any AIProvider] = [
            makeSUT(agent: StubAgent(output: "from the CLI"), id: "cli", modelName: "m"),
        ]
        let response = try await providers[0].complete(
            request: AIRequest(messages: [AIMessage(role: .user, content: "q")])
        )
        #expect(response.content == "from the CLI")
        #expect(response.providerID == "cli")
    }
}
