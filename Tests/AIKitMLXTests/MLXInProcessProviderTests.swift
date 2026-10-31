import Foundation
import MLXLMCommon
import Testing

@testable import AIKit
@testable import AIKitMLX

// MARK: - Doubles

/// Stub `MLXModelSession` — the seam that keeps this suite off the real MLX
/// runtime.
///
/// `swift test` does not bundle MLX's metallib, so a suite that reached the
/// real runtime would crash rather than fail. Everything past the seam
/// (`MLXContainerSession`) is proved by the iOS `xcodebuild` leg of the gate
/// and by a device, never here.
final class MockMLXModelSession: MLXModelSession, @unchecked Sendable {
    private let lock = NSLock()
    private var _requests: [AIRequest] = []
    private var _unloadCount = 0

    /// Deltas this session yields, in order.
    let chunks: [String]
    /// Injected failure, thrown instead of yielding anything.
    let shouldThrow: Error?

    init(chunks: [String] = ["Hello", " world"], shouldThrow: Error? = nil) {
        self.chunks = chunks
        self.shouldThrow = shouldThrow
    }

    /// Every request that reached this session.
    var requests: [AIRequest] { lock.withLock { _requests } }
    /// How many times the provider released it.
    var unloadCount: Int { lock.withLock { _unloadCount } }

    func stream(_ request: AIRequest) -> AsyncThrowingStream<AIChunk, Error> {
        lock.withLock { _requests.append(request) }
        let chunks = chunks
        let shouldThrow = shouldThrow
        return AsyncThrowingStream { continuation in
            if let shouldThrow {
                continuation.finish(throwing: shouldThrow)
                return
            }
            for chunk in chunks {
                continuation.yield(AIChunk(delta: chunk))
            }
            continuation.yield(AIChunk(delta: "", isComplete: true))
            continuation.finish()
        }
    }

    func unload() async {
        lock.withLock { _unloadCount += 1 }
    }
}

/// Stands in for `MLXContainerSession.load`, counting loads so a test can tell
/// a cached session from a reloaded one.
final class MockMLXSessionLoader: @unchecked Sendable {
    let session: MockMLXModelSession
    /// Injected failure, thrown instead of returning a session.
    let shouldThrow: Error?
    /// Fractions handed to the progress callback before returning.
    let progressToReport: [Double]

    private let lock = NSLock()
    private var _descriptors: [ModelDescriptor] = []

    init(
        session: MockMLXModelSession = MockMLXModelSession(),
        shouldThrow: Error? = nil,
        progressToReport: [Double] = []
    ) {
        self.session = session
        self.shouldThrow = shouldThrow
        self.progressToReport = progressToReport
    }

    /// Descriptors this loader was asked for, one per load.
    var descriptors: [ModelDescriptor] { lock.withLock { _descriptors } }
    var callCount: Int { lock.withLock { _descriptors.count } }

    /// The `MLXSessionLoader` to hand the provider.
    var load: MLXSessionLoader {
        { [self] descriptor, progress in
            lock.withLock { _descriptors.append(descriptor) }
            for fraction in progressToReport {
                progress(fraction)
            }
            if let shouldThrow {
                throw shouldThrow
            }
            return session
        }
    }
}

// MARK: - Fixtures

private let gemmaID = "mlx-community/gemma-4-4b-it-4bit"

private func makeModel(
    modelId: String = gemmaID,
    name: String = "Gemma 4 4B Instruct",
    minRAMGB: Double? = 3.2,
    estimatedTokPerSec: [DeviceClass: Double] = [.iPhoneA16: 20, .macM4: 60],
    huggingFaceId: String? = gemmaID,
    localPath: URL? = nil,
    sizeBytes: Int64 = 2_400_000_000
) -> ModelDescriptor {
    ModelDescriptor(
        id: ModelIdentifier(provider: "huggingface", modelId: modelId),
        name: name,
        author: "mlx-community",
        description: "4-bit quantized, MLX-ready",
        capabilities: [.textGeneration, .synthesis],
        format: .mlx,
        parameters: "4B",
        quantization: "Q4",
        sizeBytes: sizeBytes,
        architecture: "gemma",
        domain: .llm,
        isLocal: localPath != nil,
        localPath: localPath,
        huggingFaceId: huggingFaceId,
        tags: ["instruction-following", "reasoning"],
        minRAMGB: minRAMGB,
        estimatedTokPerSec: estimatedTokPerSec
    )
}

private func makeDevice(
    deviceClass: DeviceClass = .iPhoneA16,
    totalRAMGB: Double = 6.0,
    availableRAMGB: Double = 4.5
) -> DeviceCapability {
    DeviceCapability(
        deviceClass: deviceClass,
        totalRAMGB: totalRAMGB,
        availableRAMGB: availableRAMGB
    )
}

private func makeRequest(
    content: String = "Say hello in one word",
    systemPrompt: String? = nil,
    maxTokens: Int = 50
) -> AIRequest {
    AIRequest(
        messages: [AIMessage(role: .user, content: content)],
        systemPrompt: systemPrompt,
        maxTokens: maxTokens
    )
}

// MARK: - T-AIL-14 / T-AIL-15

@Suite("MLXInProcessProvider")
struct MLXInProcessProviderTests {

    /// T-AIL-14 — a stub model container reaches `stream`.
    @Test("streamsThroughAIProvider")
    func streamsThroughAIProvider() async throws {
        let loader = MockMLXSessionLoader(
            session: MockMLXModelSession(chunks: ["Hel", "lo"])
        )
        let provider = MLXInProcessProvider(
            model: makeModel(),
            device: makeDevice(),
            loadSession: loader.load
        )

        var deltas: [String] = []
        var sawCompletion = false
        for try await chunk in try await provider.stream(request: makeRequest(content: "Hi")) {
            if chunk.isComplete {
                sawCompletion = true
            } else {
                deltas.append(chunk.delta)
            }
        }

        #expect(deltas == ["Hel", "lo"])
        #expect(sawCompletion, "the stream must mark its own end")
        // The request reached the session unchanged — the provider routes, it
        // does not rewrite what the caller asked.
        #expect(loader.session.requests.count == 1)
        #expect(loader.session.requests.first?.messages.first?.content == "Hi")
        #expect(loader.callCount == 1)
    }

    /// T-AIL-15 — typed refusal from `DeviceCapability`.
    @Test("refusesWhatTheDeviceCannotLoad")
    func refusesWhatTheDeviceCannotLoad() async throws {
        let loader = MockMLXSessionLoader()
        let provider = MLXInProcessProvider(
            model: makeModel(minRAMGB: 12.0),
            device: makeDevice(totalRAMGB: 6.0, availableRAMGB: 4.0),
            loadSession: loader.load
        )

        do {
            _ = try await provider.stream(request: makeRequest())
            Issue.record("expected the provider to refuse a model this device cannot hold")
        } catch let error as MLXProviderError {
            #expect(
                error == .insufficientRAM(model: gemmaID, required: 12.0, available: 4.0)
            )
            // "Naming the limit" means both numbers are in the message, not
            // just that a typed case was thrown.
            let message = error.errorDescription ?? ""
            #expect(message.contains("12.0"))
            #expect(message.contains("4.0"))
            #expect(message.contains(gemmaID))
        }

        #expect(loader.callCount == 0, "a refused model must never be loaded")
    }

    // MARK: - Status

    @Test("status is unavailable before anything is asked of an oversized model")
    func statusUnavailableUpFront() async {
        let provider = MLXInProcessProvider(
            model: makeModel(minRAMGB: 12.0),
            device: makeDevice(availableRAMGB: 4.0),
            loadSession: MockMLXSessionLoader().load
        )
        #expect(await provider.status == .unavailable)
    }

    @Test("status is ready when the model fits")
    func statusReadyWhenItFits() async {
        let provider = MLXInProcessProvider(
            model: makeModel(minRAMGB: 3.2),
            device: makeDevice(availableRAMGB: 4.5),
            loadSession: MockMLXSessionLoader().load
        )
        #expect(await provider.status == .ready)
    }

    @Test("a load failure surfaces and leaves the status in error")
    func loadFailureSurfaces() async throws {
        let loader = MockMLXSessionLoader(
            shouldThrow: AIKitError.downloadFailed("no network"),
            progressToReport: [0.25, 0.5]
        )
        let provider = MLXInProcessProvider(
            model: makeModel(),
            device: makeDevice(),
            loadSession: loader.load
        )

        await #expect(throws: AIKitError.downloadFailed("no network")) {
            try await provider.load()
        }
        #expect(await provider.status == .error("Download failed: no network"))
        #expect(await provider.isLoaded == false)
    }

    // MARK: - Identity and provenance

    @Test("completeDrainsTheStreamAndNamesWhoAnswered")
    func completeNamesWhoAnswered() async throws {
        let loader = MockMLXSessionLoader(
            session: MockMLXModelSession(chunks: ["on", "-device"])
        )
        let provider = MLXInProcessProvider(
            model: makeModel(),
            device: makeDevice(),
            loadSession: loader.load
        )

        let response = try await provider.complete(request: makeRequest())

        #expect(response.content == "on-device")
        // BR-AIL-06: every response names the provider and the model.
        #expect(response.providerID == "mlx/\(gemmaID)")
        #expect(response.model == gemmaID)
        #expect(response.latencyMs >= 0)
    }

    @Test("default id and display name come from the model")
    func defaultIdentity() {
        let provider = MLXInProcessProvider(
            model: makeModel(),
            device: makeDevice(),
            loadSession: MockMLXSessionLoader().load
        )
        #expect(provider.id == "mlx/\(gemmaID)")
        #expect(provider.displayName == "Apple MLX — Gemma 4 4B Instruct")
        #expect(provider.capabilities.contains(.textGeneration))
        #expect(provider.capabilities.contains(.synthesis))
    }

    @Test("an injected id wins, so two MLX providers can coexist in a router")
    func injectedIdentity() {
        let provider = MLXInProcessProvider(
            model: makeModel(),
            device: makeDevice(),
            id: "mlx/fallback",
            displayName: "Fallback",
            loadSession: MockMLXSessionLoader().load
        )
        #expect(provider.id == "mlx/fallback")
        #expect(provider.displayName == "Fallback")
    }

    // MARK: - Load and unload

    @Test("the session is loaded once and reused across requests")
    func sessionLoadedOnceAndReused() async throws {
        let loader = MockMLXSessionLoader()
        let provider = MLXInProcessProvider(
            model: makeModel(),
            device: makeDevice(),
            loadSession: loader.load
        )

        _ = try await provider.complete(request: makeRequest())
        _ = try await provider.complete(request: makeRequest())

        #expect(loader.callCount == 1, "a second request must not reload the weights")
        #expect(loader.session.requests.count == 2)
        #expect(await provider.isLoaded)
    }

    @Test("unload releases the session and the next request loads again")
    func unloadReleasesAndReloads() async throws {
        let loader = MockMLXSessionLoader()
        let provider = MLXInProcessProvider(
            model: makeModel(),
            device: makeDevice(),
            loadSession: loader.load
        )

        try await provider.load()
        #expect(await provider.isLoaded)

        await provider.unload()
        #expect(await provider.isLoaded == false)
        #expect(loader.session.unloadCount == 1, "the GPU cache must be released, not just dropped")
        #expect(await provider.status == .ready)

        _ = try await provider.complete(request: makeRequest())
        #expect(loader.callCount == 2)
    }

    @Test("nothing loads until something is asked")
    func loadingIsLazy() async {
        let loader = MockMLXSessionLoader()
        _ = MLXInProcessProvider(
            model: makeModel(),
            device: makeDevice(),
            loadSession: loader.load
        )
        #expect(loader.callCount == 0)
    }

    @Test("a model with no measured minimum is not refused")
    func unmeasuredModelIsNotRefused() async throws {
        let loader = MockMLXSessionLoader()
        let provider = MLXInProcessProvider(
            model: makeModel(minRAMGB: nil),
            device: makeDevice(availableRAMGB: 0.2),
            loadSession: loader.load
        )

        // An unmeasured model on a nearly full device still loads: "unmeasured"
        // is not "needs nothing", but refusing it would block every model that
        // has not been through a catalog.
        _ = try await provider.complete(request: makeRequest())
        #expect(loader.callCount == 1)
        #expect(await provider.status == .ready)
    }

    @Test("the loader is handed the model it must load")
    func loaderReceivesTheModel() async throws {
        let loader = MockMLXSessionLoader()
        let model = makeModel(modelId: "mlx-community/gemma-2-2b-it-4bit")
        let provider = MLXInProcessProvider(
            model: model,
            device: makeDevice(),
            loadSession: loader.load
        )

        try await provider.load()
        #expect(loader.descriptors.first?.id == model.id)
    }

    @Test("a session failure reaches the caller instead of an empty answer")
    func sessionFailureSurfaces() async throws {
        let loader = MockMLXSessionLoader(
            session: MockMLXModelSession(shouldThrow: AIKitError.requestFailed("evaluation failed"))
        )
        let provider = MLXInProcessProvider(
            model: makeModel(),
            device: makeDevice(),
            loadSession: loader.load
        )

        await #expect(throws: AIKitError.requestFailed("evaluation failed")) {
            _ = try await provider.complete(request: makeRequest())
        }
    }
}

// MARK: - Request to MLX chat turns

@Suite("MLXContainerSession.turns")
struct MLXContainerSessionTurnsTests {

    @Test("the system prompt leads, then every message in order")
    func systemPromptLeads() {
        let request = AIRequest(
            messages: [
                AIMessage(role: .user, content: "first"),
                AIMessage(role: .assistant, content: "second"),
            ],
            systemPrompt: "You answer briefly"
        )

        let turns = MLXContainerSession.turns(for: request)

        #expect(turns.map(\.role) == [.system, .user, .assistant])
        #expect(turns.map(\.content) == ["You answer briefly", "first", "second"])
    }

    @Test("every AIKit role maps onto an MLX role, tool included")
    func everyRoleMaps() {
        let request = AIRequest(
            messages: [
                AIMessage(role: .system, content: "s"),
                AIMessage(role: .user, content: "u"),
                AIMessage(role: .assistant, content: "a"),
                AIMessage(role: .tool, content: "t"),
            ]
        )

        let turns = MLXContainerSession.turns(for: request)

        #expect(turns.map(\.role) == [.system, .user, .assistant, .tool])
        #expect(turns.map(\.content) == ["s", "u", "a", "t"])
    }

    @Test("no persona is invented when the caller supplies none")
    func noInventedPersona() {
        // BR-AIL-03: AIKit stays free of shikki vocabulary. The client injected
        // a hardcoded "You are Shikki…" turn here; a kit must not name its
        // consumer's product.
        let turns = MLXContainerSession.turns(for: makeRequest(content: "hi"))

        #expect(turns.map(\.role) == [.user])
        #expect(turns.map(\.content) == ["hi"])
    }

    @Test("an empty system prompt adds no turn")
    func emptySystemPromptAddsNothing() {
        let turns = MLXContainerSession.turns(
            for: makeRequest(content: "hi", systemPrompt: "")
        )
        #expect(turns.map(\.role) == [.user])
    }
}

// MARK: - DeviceCapability

@Suite("DeviceCapability")
struct DeviceCapabilityTests {

    @Test("current() returns a capability with positive RAM")
    func currentReturnsValidCapability() {
        let capability = DeviceCapability.current()
        #expect(capability.totalRAMGB > 0)
        #expect(capability.availableRAMGB > 0)
        #expect(capability.availableRAMGB <= capability.totalRAMGB + 0.01)
    }

    @Test("current() detects a Mac class on macOS")
    func currentDetectsMac() {
        let capability = DeviceCapability.current()
        #if os(macOS)
        let macClasses: Set<DeviceClass> = [.macM1, .macM2, .macM3, .macM4, .macM5]
        #expect(
            macClasses.contains(capability.deviceClass),
            "expected a macMx class on macOS, got \(capability.deviceClass.rawValue)"
        )
        #endif
    }

    @Test("canLoad refuses a model needing more RAM than is free")
    func canLoadRefusesOversized() {
        let capability = makeDevice(totalRAMGB: 6.0, availableRAMGB: 4.0)
        #expect(capability.canLoad(makeModel(minRAMGB: 12.0)) == false)
    }

    @Test("canLoad accepts a model that fits")
    func canLoadAcceptsFitting() {
        let capability = makeDevice(deviceClass: .macM4, totalRAMGB: 16.0, availableRAMGB: 12.0)
        #expect(capability.canLoad(makeModel(minRAMGB: 3.0)))
    }

    @Test("canLoad treats an exact fit as loadable")
    func canLoadExactBoundary() {
        let capability = makeDevice(availableRAMGB: 4.5)
        #expect(capability.canLoad(makeModel(minRAMGB: 4.5)))
    }

    @Test("shortfall names both numbers, and is nil when the model fits")
    func shortfallNamesBothNumbers() {
        let capability = makeDevice(totalRAMGB: 6.0, availableRAMGB: 4.0)

        let shortfall = capability.shortfall(for: makeModel(minRAMGB: 12.0))
        #expect(shortfall?.required == 12.0)
        #expect(shortfall?.available == 4.0)

        #expect(capability.shortfall(for: makeModel(minRAMGB: 3.2)) == nil)
        #expect(capability.shortfall(for: makeModel(minRAMGB: nil)) == nil)
    }

    @Test("estimatedTokPerSec reads the model's measurement for this class")
    func estimatedTokPerSecKnown() {
        let capability = makeDevice(deviceClass: .iPhoneA16)
        #expect(capability.estimatedTokPerSec(makeModel()) == 20)
    }

    @Test("estimatedTokPerSec falls back to the floor for an unmeasured class")
    func estimatedTokPerSecFallsBackToFloor() {
        let capability = makeDevice(deviceClass: .iPhoneA15)
        // iPhoneA15 is absent from the fixture's map.
        #expect(
            capability.estimatedTokPerSec(makeModel())
                == DeviceCapability.estimatedTokPerSecFloor
        )
    }
}

// MARK: - The folded descriptor fields

/// These cover `ModelDescriptor`'s new device fields, which W7 folded in from
/// shikki-client's `CatalogEntry`. They live in this wave's suite because this
/// is the wave that added the fields.
@Suite("ModelDescriptor device fields")
struct ModelDescriptorDeviceFieldsTests {

    @Test("a round trip preserves the device fields")
    func roundTripPreservesDeviceFields() throws {
        let model = makeModel(minRAMGB: 3.2, estimatedTokPerSec: [.iPhoneA16: 20, .macM4: 60])
        let data = try JSONEncoder().encode(model)
        let decoded = try JSONDecoder().decode(ModelDescriptor.self, from: data)

        #expect(decoded == model)
        #expect(decoded.minRAMGB == 3.2)
        #expect(decoded.estimatedTokPerSec[.macM4] == 60)
    }

    @Test("a descriptor stored before the device fields existed still decodes")
    func decodesPreW7Payload() throws {
        // LocalModelStore writes descriptors to disk. A registry written before
        // W7 has neither key, and the synthesized decoder would have thrown
        // keyNotFound on `estimatedTokPerSec` for every row in it.
        let stored = """
            {
              "id": { "provider": "huggingface", "modelId": "\(gemmaID)" },
              "name": "Gemma 4 4B Instruct",
              "author": "mlx-community",
              "description": "",
              "capabilities": 2049,
              "format": "mlx",
              "parameters": "4B",
              "sizeBytes": 2400000000,
              "architecture": "gemma",
              "domain": "llm",
              "isLocal": false,
              "tags": []
            }
            """

        let decoded = try JSONDecoder().decode(
            ModelDescriptor.self,
            from: Data(stored.utf8)
        )

        #expect(decoded.minRAMGB == nil)
        #expect(decoded.estimatedTokPerSec.isEmpty)
        #expect(decoded.name == "Gemma 4 4B Instruct")
    }

    @Test("formattedSize reads the byte count")
    func formattedSize() {
        #expect(makeModel(sizeBytes: 2_400_000_000).formattedSize.contains("2"))
        #expect(makeModel(sizeBytes: 0).formattedSize.isEmpty == false)
    }
}
