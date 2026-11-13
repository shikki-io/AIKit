import AIKitTestSupport
import Foundation
import Testing

@testable import AIKit

// MARK: - ProviderRouterPolicyTests (T-AIL-58)
//
// W32 of spec ai-layer-aikit-core-2026-10-31, BR-AIL-22. `candidates(for:
// policy:)` selects by capability, by device fit and by topic — and names every
// provider it leaves out, with the reason.
//
// What this replaces: `route(capabilities:)` returns ONE provider and keeps no
// record of the ones it walked past, so a caller wanting several rebuilt the
// filter and a caller asking "why is my MLX entry not being used?" had nothing
// to read.
//
// No provider here is real and no model is loaded (BR-AIL-12): every device is
// a `DeviceCapability` built from numbers, and every model descriptor is a
// description of weights this test never fetches.

@Suite("ProviderRouter — candidates by capability, device and topic")
struct ProviderRouterPolicyTests {

    // MARK: - Fixtures

    private func request(
        inputs: [AIInput] = [],
        tools: [AITool]? = nil,
        model: ModelIdentifier? = nil
    ) -> AIRequest {
        AIRequest(
            messages: [AIMessage(role: .user, content: "who can serve this?")],
            model: model,
            tools: tools,
            inputs: inputs
        )
    }

    /// A model descriptor with a declared memory floor, or none.
    ///
    /// `minRAMGB: nil` is UNMEASURED, which is a different fact from "needs
    /// nothing" and is tested as such below.
    private func model(_ name: String, minRAMGB: Double?) -> ModelDescriptor {
        ModelDescriptor(
            id: ModelIdentifier(provider: "mlx", modelId: name),
            name: name,
            author: "fixture",
            description: "described here, never loaded",
            capabilities: .textGeneration,
            format: .mlx,
            parameters: "4B",
            sizeBytes: 2_400_000_000,
            architecture: "fixture-arch",
            domain: .llm,
            minRAMGB: minRAMGB
        )
    }

    /// A phone with 3 GB free: small enough that a 12 GB model does not fit and
    /// a 2 GB one does.
    private var smallPhone: DeviceCapability {
        DeviceCapability(deviceClass: .iPhoneA16, totalRAMGB: 6, availableRAMGB: 3)
    }

    private func ready(_ id: String, _ capabilities: AICapabilities = .textGeneration)
        -> MockAIProvider {
        MockAIProvider(id: id, capabilities: capabilities)
    }

    private func router() -> ProviderRouter {
        // The pool is the method's argument here, so the router holds nothing;
        // the "router's own list" path has its own test below.
        ProviderRouter(providers: [])
    }

    // MARK: - The capabilities a request names

    @Test("the required capabilities are the policy's plus what the request carries")
    func requiredCapabilitiesComeFromThePolicyAndTheRequest() {
        let policy = AIRoutingPolicy(capabilities: .synthesis)

        // An audio input needs transcription whether or not the caller said so.
        let audio = policy.requiredCapabilities(
            for: request(inputs: [AIInput(kind: .audio, url: URL(fileURLWithPath: "/tmp/a.wav"))]))
        #expect(audio.contains(.synthesis))
        #expect(audio.contains(.voiceToText))

        // An image, and a video, both need sight.
        let image = policy.requiredCapabilities(
            for: request(inputs: [AIInput(kind: .image, data: Data([0x01]))]))
        #expect(image.contains(.vision))
        let video = policy.requiredCapabilities(
            for: request(inputs: [AIInput(kind: .video, url: URL(fileURLWithPath: "/tmp/a.mp4"))]))
        #expect(video.contains(.vision))

        // Tools need a provider that can call them.
        let tooled = policy.requiredCapabilities(
            for: request(tools: [
                AITool(name: "search", description: "search", parametersSchema: fixtureToolSchema)
            ]))
        #expect(tooled.contains(.toolUse))

        // A document or a plain text artifact rides in the prompt on every
        // family, so requiring a flag for it would reject providers that serve
        // it perfectly well.
        let document = policy.requiredCapabilities(
            for: request(inputs: [AIInput(kind: .document, data: Data([0x02]))]))
        #expect(document == .synthesis)
        let text = policy.requiredCapabilities(
            for: request(inputs: [AIInput(kind: .text, data: Data([0x03]))]))
        #expect(text == .synthesis)

        // An empty tool list is not a tool.
        #expect(policy.requiredCapabilities(for: request(tools: [])) == .synthesis)
    }

    @Test("a provider that does not offer everything the request needs is rejected by name")
    func capabilityFilterKeepsOnlyProvidersThatOfferEverything() async {
        let textOnly = ready("text-only", .textGeneration)
        let transcriber = ready("whisper-cli", [.textGeneration, .voiceToText])

        let plan = await router().candidates(
            for: request(inputs: [AIInput(kind: .audio, url: URL(fileURLWithPath: "/tmp/a.wav"))]),
            policy: AIRoutingPolicy(),
            providers: [textOnly, transcriber]
        )

        #expect(plan.selectedIDs == ["whisper-cli"])
        #expect(plan.rejected.count == 1)
        #expect(plan.rejected[0].providerID == "text-only")
        #expect(
            plan.rejected[0].rejection
                == .missingCapabilities(required: .voiceToText, offered: .textGeneration))
        // The reason names the flag, not a raw option-set integer.
        #expect(plan.rejected[0].rejection.reason.contains("voiceToText"))
    }

    // MARK: - The device filter

    @Test("a local model this device cannot load is rejected, with both numbers")
    func aLocalModelTheDeviceCannotLoadIsRejectedWithItsNumbers() async {
        let onDevice = ready("mlx/big")
        let policy = AIRoutingPolicy(
            capabilities: .textGeneration,
            device: smallPhone,
            localModels: [
                "mlx/big": AILocalModelPlacement(model: model("Big 70B", minRAMGB: 12))
            ]
        )

        let plan = await router().candidates(
            for: request(), policy: policy, providers: [onDevice])

        #expect(plan.isEmpty)
        #expect(
            plan.rejected[0].rejection
                == .deviceCannotLoad(model: "Big 70B", requiredGB: 12, availableGB: 3))
        // The sentence an interface renders: "needs 12.0 GB and 3.0 GB is free".
        let reason = plan.rejected[0].rejection.reason
        #expect(reason.contains("12.0 GB"))
        #expect(reason.contains("3.0 GB"))
        #expect(reason.contains("Big 70B"))
        // Nothing was asked of it — the filter is pure and runs before status.
        #expect(onDevice.completeCallCount == 0)
    }

    @Test("a local model that fits is kept")
    func aLocalModelThatFitsIsKept() async {
        let onDevice = ready("mlx/small")
        let policy = AIRoutingPolicy(
            capabilities: .textGeneration,
            device: smallPhone,
            localModels: [
                "mlx/small": AILocalModelPlacement(model: model("Small 4B", minRAMGB: 2))
            ]
        )

        let plan = await router().candidates(
            for: request(), policy: policy, providers: [onDevice])

        #expect(plan.selectedIDs == ["mlx/small"])
        #expect(plan.rejected.isEmpty)
    }

    @Test("an unmeasured model is not refused: no number is not a big number")
    func anUnmeasuredModelIsNotRefused() async {
        // `DeviceCapability.canLoad` reads a missing `minRAMGB` as unmeasured,
        // never as "needs nothing" and never as a refusal. The filter must not
        // invent a verdict the measurement layer declines to give.
        let onDevice = ready("mlx/uncatalogued")
        let policy = AIRoutingPolicy(
            capabilities: .textGeneration,
            device: smallPhone,
            localModels: [
                "mlx/uncatalogued": AILocalModelPlacement(model: model("Unknown", minRAMGB: nil))
            ]
        )

        let plan = await router().candidates(
            for: request(), policy: policy, providers: [onDevice])

        #expect(plan.selectedIDs == ["mlx/uncatalogued"])
    }

    @Test("a remote provider is never refused for this device's memory")
    func aRemoteProviderIsNeverRefusedForThisDevicesMemory() async {
        // A server's RAM is not the phone's business. A provider with no
        // placement is remote by definition, so the filter has nothing to say
        // about it even on a device that can load almost nothing.
        let cloud = ready("anthropic-cloud")
        let policy = AIRoutingPolicy(capabilities: .textGeneration, device: smallPhone)

        let plan = await router().candidates(for: request(), policy: policy, providers: [cloud])

        #expect(plan.selectedIDs == ["anthropic-cloud"])
    }

    @Test("no device in the policy means no device filter")
    func noDeviceMeansNoDeviceFilter() async {
        // `device` is nil rather than defaulting to `.current()`, so a policy
        // built without one filters nothing instead of silently measuring the
        // machine the caller happens to be on.
        let onDevice = ready("mlx/big")
        let policy = AIRoutingPolicy(
            capabilities: .textGeneration,
            localModels: [
                "mlx/big": AILocalModelPlacement(model: model("Big 70B", minRAMGB: 12))
            ]
        )

        let plan = await router().candidates(
            for: request(), policy: policy, providers: [onDevice])

        #expect(plan.selectedIDs == ["mlx/big"])
    }

    @Test("a cluster is one provider, filtered against its pooled memory")
    func aClusterIsOneProviderWithItsPooledMemory() async {
        // BR-AIL-22's cluster clause: a pool over Thunderbolt is ONE entry with
        // ONE pooled figure, not a new family. The same 120 GB model is refused
        // on the host and kept on the pool, and the only difference between the
        // two cases is the DeviceCapability the placement names.
        let big = model("Big 120B", minRAMGB: 120)
        let host = ready("mlx/local")
        let cluster = ready("mlx/harbor-pool")

        let policy = AIRoutingPolicy(
            capabilities: .textGeneration,
            device: DeviceCapability(deviceClass: .macM4, totalRAMGB: 24, availableRAMGB: 18),
            localModels: [
                "mlx/local": AILocalModelPlacement(model: big),
                "mlx/harbor-pool": AILocalModelPlacement(
                    model: big,
                    device: DeviceCapability(
                        deviceClass: .macM4, totalRAMGB: 512, availableRAMGB: 448)
                ),
            ]
        )

        let plan = await router().candidates(
            for: request(), policy: policy, providers: [host, cluster])

        #expect(plan.selectedIDs == ["mlx/harbor-pool"])
        #expect(plan.rejected.map(\.providerID) == ["mlx/local"])
        #expect(
            plan.rejected[0].rejection
                == .deviceCannotLoad(model: "Big 120B", requiredGB: 120, availableGB: 18))
    }

    // MARK: - Readiness

    @Test("a provider that is not ready is rejected, and its status is in the reason")
    func notReadyIsRejectedWithItsStatus() async {
        let loading = MockAIProvider(
            id: "mlx/loading", capabilities: .textGeneration, status: .loading(progress: 0.4))
        let broken = MockAIProvider(
            id: "lmstudio", capabilities: .textGeneration, status: .error("connection refused"))
        let absent = MockAIProvider(
            id: "coreml", capabilities: .textGeneration, status: .unavailable)

        let plan = await router().candidates(
            for: request(), policy: AIRoutingPolicy(), providers: [loading, broken, absent])

        #expect(plan.isEmpty)
        #expect(plan.rejected.map(\.providerID) == ["mlx/loading", "lmstudio", "coreml"])
        #expect(plan.rejected[0].rejection.reason.contains("loading (40%)"))
        #expect(plan.rejected[1].rejection.reason.contains("connection refused"))
        #expect(plan.rejected[2].rejection.reason.contains("unavailable"))
    }

    @Test("a provider already out on capability or device is never asked for its status")
    func statusIsNotAskedOfAProviderAlreadyOut() async {
        // Why the order of the three filters is fixed: on an HTTP provider
        // `status` is a network round trip, and spending one to learn whether a
        // model this device cannot load is awake buys nothing.
        let reads = StatusReadLog()
        let wrongCapability = CountingStatusProvider(
            id: "text-only", capabilities: .textGeneration, reported: .ready, log: reads)
        let tooBig = CountingStatusProvider(
            id: "mlx/big", capabilities: [.textGeneration, .voiceToText], reported: .ready,
            log: reads)
        let fits = CountingStatusProvider(
            id: "whisper-cli", capabilities: [.textGeneration, .voiceToText], reported: .ready,
            log: reads)

        let policy = AIRoutingPolicy(
            device: smallPhone,
            localModels: [
                "mlx/big": AILocalModelPlacement(model: model("Big 70B", minRAMGB: 12))
            ]
        )
        let plan = await router().candidates(
            for: request(inputs: [AIInput(kind: .audio, url: URL(fileURLWithPath: "/tmp/a.wav"))]),
            policy: policy,
            providers: [wrongCapability, tooBig, fits]
        )

        #expect(plan.selectedIDs == ["whisper-cli"])
        #expect(await reads.asked == ["whisper-cli"])
    }

    // MARK: - The topic ranking

    @Test("the topic's order is honoured, and an unnamed provider keeps its place behind it")
    func topicOrderIsHonoured() async {
        let a = ready("a")
        let b = ready("b")
        let c = ready("c")
        let d = ready("d")

        let plan = await router().candidates(
            for: request(),
            policy: AIRoutingPolicy(
                topic: "security",
                topicRanking: ["security": ["c", "a"]]
            ),
            providers: [a, b, c, d]
        )

        // c and a first, in the topic's order; b and d keep the order they came
        // in, which for a router built from settings is the operator's own.
        #expect(plan.selectedIDs == ["c", "a", "b", "d"])
    }

    @Test("a ranking is an ordering, not an allowlist")
    func aTopicRankingIsNotAnAllowlist() async {
        // A caller's topic table with one incomplete line must not disable
        // providers the operator configured: that failure would look like a
        // routing bug rather than a missing list entry. A caller that wants
        // only a subset asks with only that subset.
        let a = ready("a")
        let b = ready("b")

        let plan = await router().candidates(
            for: request(),
            policy: AIRoutingPolicy(topic: "cost", topicRanking: ["cost": ["b"]]),
            providers: [a, b]
        )

        #expect(plan.selectedIDs == ["b", "a"])
        #expect(plan.rejected.isEmpty)
    }

    @Test("no topic, or a topic the table does not name, keeps the configured order")
    func noTopicKeepsTheConfiguredOrder() async {
        let a = ready("a")
        let b = ready("b")
        let table = ["security": ["b"]]

        let untopiced = await router().candidates(
            for: request(),
            policy: AIRoutingPolicy(topicRanking: table),
            providers: [a, b]
        )
        #expect(untopiced.selectedIDs == ["a", "b"])

        let unknownTopic = await router().candidates(
            for: request(),
            policy: AIRoutingPolicy(topic: "pricing", topicRanking: table),
            providers: [a, b]
        )
        #expect(unknownTopic.selectedIDs == ["a", "b"])

        let emptyRanking = await router().candidates(
            for: request(),
            policy: AIRoutingPolicy(topic: "security", topicRanking: ["security": []]),
            providers: [a, b]
        )
        #expect(emptyRanking.selectedIDs == ["a", "b"])
    }

    @Test("an id repeated in a ranking keeps its best position")
    func aDuplicateInARankingKeepsItsBestPosition() async {
        // A caller's list that mentions a provider twice meant it at its best
        // rank; reading the last occurrence would demote it for being named
        // twice.
        let a = ready("a")
        let b = ready("b")
        let c = ready("c")

        let plan = await router().candidates(
            for: request(),
            policy: AIRoutingPolicy(topic: "t", topicRanking: ["t": ["c", "a", "c"]]),
            providers: [a, b, c]
        )

        #expect(plan.selectedIDs == ["c", "a", "b"])
    }

    @Test("the ranking reorders survivors only, after the filters have run")
    func rankingAppliesAfterTheFilters() async {
        let preferred = MockAIProvider(
            id: "preferred", capabilities: .textGeneration, status: .unavailable)
        let serving = ready("serving")

        let plan = await router().candidates(
            for: request(),
            policy: AIRoutingPolicy(topic: "t", topicRanking: ["t": ["preferred", "serving"]]),
            providers: [serving, preferred]
        )

        // A preference cannot promote a provider that is out: the ranking
        // orders candidates, it does not create them.
        #expect(plan.selectedIDs == ["serving"])
        #expect(plan.rejected.map(\.providerID) == ["preferred"])
    }

    // MARK: - Nothing is silently skipped

    @Test("every provider in the pool is either selected or rejected with a reason")
    func nothingIsSilentlySkipped() async {
        let pool: [any AIProvider] = [
            ready("keeps"),
            ready("wrong-capability", .embedding),
            MockAIProvider(id: "down", capabilities: .textGeneration, status: .unavailable),
            ready("too-big"),
        ]
        let policy = AIRoutingPolicy(
            capabilities: .textGeneration,
            device: smallPhone,
            localModels: [
                "too-big": AILocalModelPlacement(model: model("Big 70B", minRAMGB: 12))
            ]
        )

        let plan = await router().candidates(for: request(), policy: policy, providers: pool)

        #expect(plan.selected.count + plan.rejected.count == pool.count)
        #expect(plan.selectedIDs == ["keeps"])
        #expect(plan.rejected.map(\.providerID) == ["wrong-capability", "down", "too-big"])
        // Each rejection reads as a sentence naming the provider.
        for rejection in plan.rejected {
            #expect(rejection.description.hasPrefix(rejection.providerID))
            #expect(!rejection.rejection.reason.isEmpty)
        }
    }

    @Test("an empty pool is an empty plan, not a thrown error")
    func anEmptyPoolIsAnEmptyPlan() async {
        let plan = await router().candidates(
            for: request(), policy: AIRoutingPolicy(), providers: [])

        #expect(plan.isEmpty)
        #expect(plan.rejected.isEmpty)
    }

    @Test("with no pool given, the router filters its own providers in its own order")
    func defaultsToTheRoutersOwnProviders() async {
        // The order a router built by `AIKitAssembly` holds is the order of the
        // `[[providers]]` rows (BR-AIL-11), so a caller that passes nothing gets
        // the operator's order rather than one re-decided here.
        let first = ready("first")
        let second = ready("second")
        let router = ProviderRouter(providers: [first, second])

        let plan = await router.candidates(for: request(), policy: AIRoutingPolicy())

        #expect(plan.selectedIDs == ["first", "second"])
    }

    @Test("a policy with no required capability keeps a provider that offers nothing")
    func anEmptyRequirementRejectsNobody() async {
        // `AICapabilities([]).contains([])` is true, so an unstated requirement
        // must not accidentally become "must offer nothing" or "must offer
        // everything". The reason text says so too, rather than printing an
        // empty list.
        let bare = MockAIProvider(id: "bare", capabilities: [])

        let plan = await router().candidates(
            for: request(), policy: AIRoutingPolicy(), providers: [bare])

        #expect(plan.selectedIDs == ["bare"])
        #expect(
            AIRoutingRejection.names(of: []).contains("the request asked for no capability"))
    }
}

// MARK: - Test doubles

/// Who was asked for a status, in order.
private actor StatusReadLog {
    private(set) var asked: [String] = []

    func record(_ id: String) { asked.append(id) }
}

/// A provider that records every read of its ``status``.
///
/// `MockAIProvider` stores its status and counts only `complete` and `stream`,
/// so it cannot prove the claim this suite makes about filter ORDER: that a
/// provider already out is never asked.
private struct CountingStatusProvider: AIProvider {
    let id: String
    let capabilities: AICapabilities
    let reported: AIProviderStatus
    let log: StatusReadLog

    var displayName: String { id }

    var status: AIProviderStatus {
        get async {
            await log.record(id)
            return reported
        }
    }

    func complete(request: AIRequest) async throws -> AIResponse {
        AIResponse(
            content: "unused by this suite",
            model: "fixture",
            tokensUsed: TokenUsage(prompt: 1, completion: 1),
            latencyMs: 1,
            providerID: id
        )
    }
}
