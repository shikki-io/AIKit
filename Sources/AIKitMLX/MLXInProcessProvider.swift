import AIKit
import Foundation
import MLX

// MLXLLM is imported for its symbols AND linked for its side effect:
// `loadModelContainer` finds a factory through
// `NSClassFromString("MLXLLM.TrampolineModelFactory")`, so a build that drops
// this dependency still compiles and then fails at load with "no factory".
import MLXLLM
import MLXLMCommon

// MARK: - Errors

/// Errors raised by on-device MLX inference.
///
/// Extracted from shikki-client's `MLXEngineError` in W7 of spec
/// ai-layer-aikit-core-2026-10-31. Both cases name the limit they hit, which is
/// the point: "the model did not load" is not an answer a caller can act on.
public enum MLXProviderError: Error, LocalizedError, Equatable, Sendable {
    /// The device has less free memory than the model needs.
    /// Carries both numbers so the refusal names the limit, per S3.7.
    case insufficientRAM(model: String, required: Double, available: Double)
    /// MLX evaluates on the Apple GPU and has no CPU fallback.
    case unsupportedHardware(String)

    public var errorDescription: String? {
        switch self {
        case .insufficientRAM(let model, let required, let available):
            String(
                format: "Insufficient RAM to load %@: need %.1f GB, have %.1f GB available.",
                model,
                required,
                available
            )
        case .unsupportedHardware(let reason):
            "On-device MLX is unavailable: \(reason)"
        }
    }
}

// MARK: - Session seam

/// A model that is loaded in this process and can answer.
///
/// This is the seam `MLXInProcessProvider` is written against. The shipped
/// implementation is `MLXContainerSession`, which drives `mlx-swift-lm`; a test
/// substitutes its own, because `swift test` does not bundle MLX's metallib and
/// a suite that reached the real runtime would crash rather than fail.
///
/// It carries `AIChunk`, not a new chunk noun, per BR-AIL-01.
public protocol MLXModelSession: Sendable {
    /// Stream the model's answer to a request, one chunk at a time.
    func stream(_ request: AIRequest) -> AsyncThrowingStream<AIChunk, Error>

    /// Release the weights and whatever GPU cache they hold.
    func unload() async
}

/// Loads the weights a descriptor names and returns a session that can answer.
///
/// The second parameter is called with a 0…1 fraction while a multi-gigabyte
/// download runs; that is what `AIProviderStatus.loading(progress:)` reports.
public typealias MLXSessionLoader =
    @Sendable (
        ModelDescriptor,
        @Sendable @escaping (Double) -> Void
    ) async throws -> any MLXModelSession

// MARK: - Provider

/// On-device LLM inference through Apple MLX, served as an `AIProvider`.
///
/// Loads MLX-format weights (from a local directory or from HuggingFace via
/// `mlx-swift-lm`) and evaluates them on the Apple GPU through Metal. No
/// server, no network once the weights are on disk — which is why it is the
/// provider an iPhone can use offline.
///
/// Extracted from shikki-client's `Sources/ShikkiAppKit/Engine/MLXEngine.swift`
/// in W7 of spec ai-layer-aikit-core-2026-10-31. It answers the same `AIRequest`
/// as every remote provider, so `ProviderRouter` routes it like any other
/// (S3.7), and it lives in its own target so an app that wants only remote
/// providers links no MLX at all (BR-AIL-14).
///
/// Which model to serve is the caller's config and never a branch in here
/// (BR-AIL-08): a device class maps to a `ModelDescriptor` upstream, and
/// switching models means choosing another provider — `ProviderRouter`'s job,
/// not a `switchModel` call.
///
/// ```swift
/// let provider = MLXInProcessProvider(model: gemma4B)
/// for try await chunk in try await provider.stream(request: request) {
///     print(chunk.delta, terminator: "")
/// }
/// ```
public actor MLXInProcessProvider: AIProvider {

    // MARK: - AIProvider identity

    nonisolated public let id: String
    nonisolated public let displayName: String
    nonisolated public let capabilities: AICapabilities

    /// The model this provider serves.
    nonisolated public let model: ModelDescriptor

    // MARK: - State

    /// The device measurement every load is refused or allowed against.
    private let device: DeviceCapability
    private let loadSession: MLXSessionLoader
    private var session: (any MLXModelSession)?
    private var state: AIProviderStatus

    /// Whether the weights are in memory right now.
    public var isLoaded: Bool { session != nil }

    // MARK: - Init

    /// Create a provider for one model.
    ///
    /// - Parameters:
    ///   - model: the model to serve. Its `minRAMGB` and `localPath` decide
    ///     whether and from where it loads.
    ///   - device: the device to measure against. Defaults to this device;
    ///     a test passes a described one.
    ///   - id: `AIProvider.id`. Defaults to `mlx/<model id>`.
    ///   - displayName: defaults to the model's name behind "Apple MLX".
    ///   - capabilities: what to advertise to `ProviderRouter`. Defaults to
    ///     text generation and synthesis, matching what an MLX LLM serves.
    ///   - loadSession: how to load the weights. Defaults to
    ///     `MLXContainerSession`, the real in-process runtime.
    public init(
        model: ModelDescriptor,
        device: DeviceCapability = .current(),
        id: String? = nil,
        displayName: String? = nil,
        capabilities: AICapabilities = [.textGeneration, .synthesis],
        loadSession: MLXSessionLoader? = nil
    ) {
        self.model = model
        self.device = device
        self.id = id ?? "mlx/\(model.id.modelId)"
        self.displayName = displayName ?? "Apple MLX — \(model.name)"
        self.capabilities = capabilities
        self.loadSession =
            loadSession ?? { descriptor, progress in
                try await MLXContainerSession.load(descriptor, progress: progress)
            }
        // Reported before anything is asked of it, so a router can skip a
        // provider this device cannot serve instead of discovering it by
        // catching an error.
        self.state = device.canLoad(model) ? .ready : .unavailable
    }

    // MARK: - AIProvider

    /// Current status.
    ///
    /// `.unavailable` means this device cannot serve this model at all — wrong
    /// architecture, or less free RAM than the model's measured minimum.
    /// `AIProviderStatus` has no "generating" case, so a request in flight
    /// still reads `.ready`.
    public var status: AIProviderStatus {
        get async { state }
    }

    public func complete(request: AIRequest) async throws -> AIResponse {
        let startTime = ContinuousClock.now
        var content = ""
        for try await chunk in try await stream(request: request) {
            content += chunk.delta
        }
        let elapsed = startTime.duration(to: .now)
        let latencyMs = Int(
            elapsed.components.seconds * 1000
                + elapsed.components.attoseconds / 1_000_000_000_000_000
        )

        return AIResponse(
            content: content,
            model: modelName,
            // MLX reports token counts in its `.info` event, which the chunk
            // stream does not carry. 0 means "not reported", as it does for an
            // OpenAI-compatible server that omits `usage` — never "zero tokens".
            tokensUsed: TokenUsage(prompt: 0, completion: 0),
            latencyMs: latencyMs,
            providerID: id
        )
    }

    public func stream(request: AIRequest) async throws -> AsyncThrowingStream<AIChunk, Error> {
        let session = try await loadedSession()
        return session.stream(request)
    }

    // MARK: - Model management

    /// Load the weights now, instead of on the first request.
    ///
    /// Callers that want to show a download progress bar before the user types
    /// anything call this; everything else loads lazily.
    ///
    /// - Throws: `MLXProviderError.insufficientRAM` when `DeviceCapability`
    ///   refuses the model, or whatever the session loader throws —
    ///   `.unsupportedHardware` off Apple silicon, a download failure
    ///   otherwise.
    public func load() async throws {
        _ = try await loadedSession()
    }

    /// Drop the weights and free the GPU memory they hold.
    ///
    /// The next request loads again — and is refused again if the device has
    /// meanwhile lost the headroom.
    public func unload() async {
        await session?.unload()
        session = nil
        state = device.canLoad(model) ? .ready : .unavailable
    }

    // MARK: - Private

    /// The model name a response reports: the HuggingFace id when there is one,
    /// because that is what identifies the weights that actually answered.
    private var modelName: String {
        model.huggingFaceId ?? model.id.modelId
    }

    private func loadedSession() async throws -> any MLXModelSession {
        if let session { return session }

        // The refusal is `DeviceCapability`'s verdict, read once: `shortfall`
        // returns the two numbers precisely so the error can name the limit
        // rather than say "it did not fit".
        if let shortfall = device.shortfall(for: model) {
            let error = MLXProviderError.insufficientRAM(
                model: modelName,
                required: shortfall.required,
                available: shortfall.available
            )
            state = .unavailable
            throw error
        }

        state = .loading(progress: 0)
        do {
            let loaded = try await loadSession(model) { [weak self] fraction in
                Task { await self?.report(loadProgress: fraction) }
            }
            session = loaded
            state = .ready
            return loaded
        } catch {
            state = .error(error.localizedDescription)
            throw error
        }
    }

    /// Record download/load progress, ignoring a report that arrives after the
    /// load already finished or failed.
    private func report(loadProgress fraction: Double) {
        guard case .loading = state else { return }
        state = .loading(progress: fraction)
    }
}

// MARK: - The real runtime

/// `MLXModelSession` backed by `mlx-swift-lm` evaluating in this process.
///
/// Holds a `ModelContainer`, which is an actor owning the weights and the
/// tokenizer, and drives its `prepare` / `generate` pair. Those two methods
/// replace the `container.perform { }` closure the client used, which needed
/// `nonisolated(unsafe)` on both the messages and the prepared input.
public struct MLXContainerSession: MLXModelSession {

    private let container: ModelContainer

    /// Wrap a container that is already loaded.
    public init(container: ModelContainer) {
        self.container = container
    }

    /// Load the weights a descriptor names.
    ///
    /// A descriptor with a `localPath` loads from that directory and never
    /// touches the network; otherwise the HuggingFace id is fetched, and the
    /// progress handler reports the download.
    public static func load(
        _ model: ModelDescriptor,
        progress: @Sendable @escaping (Double) -> Void
    ) async throws -> any MLXModelSession {
        // The hardware limit lives here and not in the provider: the provider
        // decides whether this DEVICE has the memory, which is true of any
        // session it is given, while "there is no GPU to evaluate on" is a
        // property of this runtime alone.
        guard hasAppleSiliconGPU else {
            throw MLXProviderError.unsupportedHardware("MLX needs an Apple silicon GPU")
        }

        // Cap the GPU cache at 512 MB before any weights arrive: on a 6 GB
        // iPhone an uncapped cache is what gets the app jetsam-killed.
        MLX.Memory.cacheLimit = 512 * 1024 * 1024

        let container: ModelContainer
        if let directory = model.localPath {
            container = try await loadModelContainer(directory: directory) {
                progress($0.fractionCompleted)
            }
        } else {
            container = try await loadModelContainer(
                id: model.huggingFaceId ?? model.id.modelId
            ) {
                progress($0.fractionCompleted)
            }
        }
        return MLXContainerSession(container: container)
    }

    public func stream(_ request: AIRequest) -> AsyncThrowingStream<AIChunk, Error> {
        let parameters = GenerateParameters(
            maxTokens: request.maxTokens,
            temperature: Float(request.temperature)
        )
        let container = container

        let (stream, continuation) = AsyncThrowingStream.makeStream(of: AIChunk.self)
        let task = Task {
            do {
                // `UserInput` is not Sendable, and `prepare` takes it as
                // `consuming sending`: it has to be built inside this task so
                // the compiler sees a disconnected region it can transfer. The
                // client reached for `nonisolated(unsafe)` on the messages and
                // the prepared input instead; `request` is Sendable and `turns`
                // is pure, so constructing here needs no escape hatch.
                let input = try await container.prepare(
                    input: UserInput(chat: Self.turns(for: request))
                )
                // `maxTokens` in the parameters is what stops generation. The
                // client also counted emitted chunks and `break`-ed on the
                // budget, which broke out of the `switch` and never the loop:
                // dead code, dropped rather than carried over.
                for await event in try await container.generate(
                    input: input,
                    parameters: parameters
                ) {
                    switch event {
                    case .chunk(let text):
                        if !text.isEmpty {
                            continuation.yield(AIChunk(delta: text))
                        }
                    case .info:
                        // Token counts and tok/s; `AIChunk` carries neither,
                        // and `.info` arriving means generation finished.
                        continue
                    case .toolCall:
                        // Tool loops for local models are out of scope (#1859).
                        continue
                    }
                }
                continuation.yield(AIChunk(delta: "", isComplete: true))
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
        continuation.onTermination = { _ in task.cancel() }
        return stream
    }

    public func unload() async {
        MLX.Memory.cacheLimit = 0
        MLX.Memory.clearCache()
    }

    /// MLX evaluates on the Apple GPU; there is no Intel path.
    static var hasAppleSiliconGPU: Bool {
        #if arch(arm64)
            true
        #else
            false
        #endif
    }

    /// Map a request onto MLX's chat turns.
    ///
    /// No persona is invented here: the client injected a hardcoded "You are
    /// Shikki…" system turn when the caller supplied none, which is shikki
    /// vocabulary inside the kit and is forbidden by BR-AIL-03. A persona
    /// arrives as `systemPrompt` or it does not arrive.
    static func turns(for request: AIRequest) -> [Chat.Message] {
        var turns: [Chat.Message] = []

        if let systemPrompt = request.systemPrompt, !systemPrompt.isEmpty {
            turns.append(.system(systemPrompt))
        }

        for message in request.messages {
            switch message.role {
            case .system:
                turns.append(.system(message.content))
            case .user:
                turns.append(.user(message.content))
            case .assistant:
                turns.append(.assistant(message.content))
            case .tool:
                turns.append(.tool(message.content))
            }
        }

        return turns
    }
}
