import Foundation

// MARK: - AgentProviderAdapter

/// An `AIProvider` over any `AgentProviding`, so a prompt-and-text backend
/// routes like every other provider.
///
/// `AgentProviding` is the older, narrower seam: give a prompt, get text.
/// `CLISubprocessProvider`, `FallbackProviderChain` and `LMStudioProvider`
/// speak it. The typed family (`AIProvider`, `AIRequest`, `AIResponse`) is the
/// one surface a router sees (BR-AIL-01), so rather than teaching the router a
/// second protocol, this adapter presents the narrow one as the wide one: a
/// `claude -p` subprocess becomes a candidate in the same provider list as an
/// LM Studio server, and `ProviderRouter` needs no case for it.
///
/// This type touches no process API, so it compiles for iOS (BR-AIL-02); the
/// backends that need `Process` stay behind their own platform gate.
public struct AgentProviderAdapter: AIProvider, Sendable {
    public let id: String
    public let displayName: String
    public let capabilities: AICapabilities
    /// The model name every answer reports, as the caller knows it. A prompt
    /// backend does not name the model it used, so this is the only source for
    /// `AIResponse.model`.
    public let modelName: String
    /// Seconds a call may take, passed through to `AgentProviding.run`.
    public let timeout: TimeInterval

    private let agent: any AgentProviding

    /// Present an `AgentProviding` backend as an `AIProvider`.
    /// - Parameters:
    ///   - id: The `AIProvider.id` this adapter reports, and the value every
    ///     answer carries in `AIResponse.providerID`.
    ///   - displayName: Human-readable name.
    ///   - capabilities: What the backend can do. Text generation only by
    ///     default: a prompt-and-text seam returns no structured tool calls and
    ///     carries no media.
    ///   - modelName: Model name to report on the way back.
    ///   - timeout: Seconds a call may take.
    ///   - agent: The backend.
    public init(
        id: String,
        displayName: String,
        capabilities: AICapabilities = .textGeneration,
        modelName: String,
        timeout: TimeInterval,
        agent: any AgentProviding
    ) {
        self.id = id
        self.displayName = displayName
        self.capabilities = capabilities
        self.modelName = modelName
        self.timeout = timeout
        self.agent = agent
    }

    public var status: AIProviderStatus {
        get async { .ready }
    }

    public func complete(request: AIRequest) async throws -> AIResponse {
        guard request.inputs.isEmpty else {
            // Refuse rather than drop: a prompt string cannot carry an audio
            // file, and a silently text-only answer to a request that sent one
            // is the failure that is hardest to see from the outside.
            throw AIKitError.requestFailed(
                "\(displayName) takes a prompt and returns text, so it cannot carry "
                    + "\(request.inputs.count) media input(s). Route media to a provider whose "
                    + "capabilities declare it."
            )
        }

        let startTime = ContinuousClock.now
        // The backend's error travels untouched: whatever a chain or a router
        // reads to decide on a fallback is thrown by the backend, not by this
        // adapter, so wrapping it here would hide it (BR-AIL-07).
        let text = try await agent.run(prompt: Self.renderPrompt(request), timeout: timeout)
        let elapsed = startTime.duration(to: .now)
        let latencyMs = Int(elapsed.components.seconds * 1000 + elapsed.components.attoseconds / 1_000_000_000_000_000)

        return AIResponse(
            content: text,
            model: modelName,
            // A prompt backend reports no usage. Zero is the honest value;
            // a guess from character counts would read as measured.
            tokensUsed: TokenUsage(prompt: 0, completion: 0),
            latencyMs: latencyMs,
            toolCalls: nil,
            providerID: id
        )
    }

    // MARK: - Prompt Rendering

    /// Renders a typed request as the single prompt string an `AgentProviding`
    /// backend takes.
    ///
    /// A prompt backend has one input, so a conversation arrives flattened. The
    /// system prompt leads, then every message in order, blocks separated by a
    /// blank line. A role label is written only when the request carries more
    /// than one message, because several unlabelled turns run together into one
    /// ambiguous block — while a single-message request, which is what every
    /// current caller sends, round trips to exactly its own text.
    static func renderPrompt(_ request: AIRequest) -> String {
        var blocks: [String] = []
        if let systemPrompt = request.systemPrompt, !systemPrompt.isEmpty {
            blocks.append(systemPrompt)
        }

        let labelRoles = request.messages.count > 1
        for message in request.messages {
            blocks.append(labelRoles ? "\(message.role.rawValue): \(message.content)" : message.content)
        }

        return blocks.joined(separator: "\n\n")
    }
}
