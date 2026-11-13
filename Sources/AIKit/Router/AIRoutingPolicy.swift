import Foundation

// MARK: - AILocalModelPlacement

/// A LOCAL provider's model, and the memory that would hold it.
///
/// This is how BR-AIL-22's device filter arrives as DATA. ``AIProvider`` is
/// four members — id, name, capabilities, status — and deliberately says
/// nothing about where it runs or which weights it would load, so the router
/// cannot ask a provider whether this device can hold its model. The caller
/// that built the providers from settings knows both, and hands them over.
///
/// A provider id that has no placement in ``AIRoutingPolicy/localModels`` is
/// treated as REMOTE: a server's memory is not this device's business, and
/// refusing a cloud entry because the phone is small would be wrong. So the map
/// is the declaration of which providers are local — not a hint, the fact.
///
/// ## A cluster is one provider, not a new family
///
/// A cluster that pools memory across machines — Claris Harbor over
/// Thunderbolt 5, exo — is ONE settings entry with ONE pooled-memory figure,
/// which is what ``device`` is for: the placement names the
/// ``DeviceCapability`` whose RAM would actually hold the weights, so a 4-Mac
/// pool is described by its pooled numbers and passes a filter this MacBook
/// would fail. Nothing else in the kit changes, because a cluster answers
/// `complete` and `stream` like any other provider.
public struct AILocalModelPlacement: Sendable, Equatable {

    /// The model this provider would load. Its ``ModelDescriptor/minRAMGB`` is
    /// what the filter reads; a descriptor that carries none is UNMEASURED and
    /// is never refused, exactly as ``DeviceCapability/canLoad(_:)`` says.
    public let model: ModelDescriptor

    /// The device whose memory would hold it, or `nil` for "this host", in
    /// which case ``AIRoutingPolicy/device`` is used.
    public let device: DeviceCapability?

    public init(model: ModelDescriptor, device: DeviceCapability? = nil) {
        self.model = model
        self.device = device
    }
}

// MARK: - AIRoutingRejection

/// Why a provider is not a candidate.
///
/// A CASE carrying its numbers, for the reason ``VideoPresetRefusal`` and
/// ``DeviceCapability/shortfall(for:)`` carry theirs: *"needs 12.0 GB, 4.0 GB
/// free"* tells a caller what to do, where a provider quietly missing from a
/// list tells them nothing and reads as a bug. BR-AIL-22's rule is that nothing
/// is silently skipped, and a reason that cannot be read is a silent skip with
/// extra steps.
public enum AIRoutingRejection: Sendable, Equatable {

    /// The provider does not offer everything the request needs.
    case missingCapabilities(required: AICapabilities, offered: AICapabilities)

    /// This device cannot hold the model this local provider would load.
    ///
    /// Both figures in gigabytes, straight from
    /// ``DeviceCapability/shortfall(for:)`` so the verdict and its numbers
    /// cannot drift apart.
    case deviceCannotLoad(model: String, requiredGB: Double, availableGB: Double)

    /// The provider is not ready to take a request.
    case notReady(AIProviderStatus)

    /// One sentence, for a log line or a `shi providers` row.
    public var reason: String {
        switch self {
        case .missingCapabilities(let required, let offered):
            let missing = required.subtracting(offered)
            return "does not offer \(Self.names(of: missing))"
        case .deviceCannotLoad(let model, let required, let available):
            return String(
                format: "cannot load %@ here: it needs %.1f GB and %.1f GB is free",
                model, required, available
            )
        case .notReady(let status):
            return "is not ready: \(Self.describe(status))"
        }
    }

    /// Capability flags as the names they are declared under.
    ///
    /// `AICapabilities` is an `OptionSet`, so its default description is a raw
    /// integer — a reason reading "does not offer AICapabilities(rawValue: 16)"
    /// is a reason nobody acts on.
    static func names(of capabilities: AICapabilities) -> String {
        let named: [(AICapabilities, String)] = [
            (.textGeneration, "textGeneration"),
            (.translation, "translation"),
            (.ocr, "ocr"),
            (.imageGeneration, "imageGeneration"),
            (.voiceToText, "voiceToText"),
            (.textToVoice, "textToVoice"),
            (.inpainting, "inpainting"),
            (.embedding, "embedding"),
            (.vision, "vision"),
            (.toolUse, "toolUse"),
            (.videoGeneration, "videoGeneration"),
            (.synthesis, "synthesis"),
        ]
        let hits = named.filter { capabilities.contains($0.0) }.map(\.1)
        guard !hits.isEmpty else { return "nothing (the request asked for no capability)" }
        return hits.joined(separator: ", ")
    }

    /// A status as a phrase, keeping the progress a loading provider reports.
    static func describe(_ status: AIProviderStatus) -> String {
        switch status {
        case .ready: "ready"
        case .loading(let progress): String(format: "loading (%.0f%%)", progress * 100)
        case .downloading(let progress, let total):
            String(format: "downloading (%.0f%% of %lld bytes)", progress * 100, total)
        case .error(let message): "error — \(message)"
        case .unavailable: "unavailable on this platform"
        }
    }
}

// MARK: - AIProviderRejection

/// One provider and why it is out.
public struct AIProviderRejection: Sendable, Equatable, CustomStringConvertible {

    /// The ``AIProvider/id`` that was considered.
    public let providerID: String

    /// Why it is not a candidate.
    public let rejection: AIRoutingRejection

    public init(providerID: String, rejection: AIRoutingRejection) {
        self.providerID = providerID
        self.rejection = rejection
    }

    public var description: String { "\(providerID) \(rejection.reason)" }
}

// MARK: - AIRoutingPlan

/// Who qualified, in the order to ask them, and who did not, with reasons.
///
/// Returned whole rather than as a bare `[any AIProvider]` so that the
/// rejections leave the router with the selection. A method that returned only
/// the survivors would make "nothing is silently skipped" impossible to honour:
/// the reasons would have to be logged inside the router, where no caller can
/// render them and no test can read them.
public struct AIRoutingPlan: Sendable {

    /// The providers that qualified, in the order the policy ranks them.
    public let selected: [any AIProvider]

    /// Every provider that did not qualify, in the order they were considered.
    public let rejected: [AIProviderRejection]

    public init(selected: [any AIProvider], rejected: [AIProviderRejection]) {
        self.selected = selected
        self.rejected = rejected
    }

    /// The selected ids, in order — the whole plan in one readable value.
    public var selectedIDs: [String] { selected.map(\.id) }

    /// Whether anything qualified.
    public var isEmpty: Bool { selected.isEmpty }
}

// MARK: - AIRoutingPolicy

/// What a caller wants routed where: the capabilities a request needs, which
/// providers a topic prefers, and what this device can hold (BR-AIL-22).
///
/// A VALUE built from data, and that is the load-bearing part. AIKit stays free
/// of shikki vocabulary (BR-AIL-03) and the stage-to-provider mapping lives in
/// shikki (BR-AIL-04), so a topic is a `String` the caller chose and the
/// ranking is a list the caller wrote. AIKit never learns what `dispatch-impl`
/// means, and shikki never reimplements a parallel dispatch.
///
/// ```swift
/// let policy = AIRoutingPolicy(
///     capabilities: .textGeneration,
///     topic: "security",
///     topicRanking: ["security": ["anthropic-cloud", "lmstudio-local"]],
///     device: .current(),
///     localModels: ["mlx/gemma-4-4b": AILocalModelPlacement(model: gemma)]
/// )
/// let plan = await router.candidates(for: request, policy: policy)
/// let outcomes = try await router.fanOut(request, to: plan.selected, within: .seconds(30))
/// ```
public struct AIRoutingPolicy: Sendable, Equatable {

    // MARK: - Properties

    /// What the caller requires beyond what the request itself implies.
    ///
    /// Unioned with ``requiredCapabilities(for:)``'s reading of the request, so
    /// a caller that states nothing still gets a correct filter and a caller
    /// that states `.synthesis` gets that on top.
    public var capabilities: AICapabilities

    /// The topic this request is about, in the caller's own vocabulary, or
    /// `nil` for no topic.
    public var topic: String?

    /// Topic → provider ids, best first. Caller-owned (BR-AIL-04).
    ///
    /// An ORDERING, never an allowlist. A provider the topic does not name keeps
    /// its place after the ones it does, in the order it was given. That is
    /// deliberate: a ranking that silently dropped every unnamed provider would
    /// let one incomplete line in a caller's topic table disable providers the
    /// operator configured, and the failure would look like a routing bug
    /// rather than a missing list entry. A caller that DOES want only a subset
    /// asks with only that subset — which is a thing it can see itself doing.
    public var topicRanking: [String: [String]]

    /// This device, for the local filter, or `nil` to apply no device filter.
    ///
    /// `nil` rather than defaulting to ``DeviceCapability/current()``: reading
    /// the host's RAM is a measurement, and a value type that takes one in its
    /// initialiser cannot be built in a test for a device it does not run on.
    /// The caller passes `.current()` when it means this machine.
    public var device: DeviceCapability?

    /// Provider id → the local model it would load. Absent means remote.
    ///
    /// See ``AILocalModelPlacement``, including the cluster case.
    public var localModels: [String: AILocalModelPlacement]

    // MARK: - Init

    public init(
        capabilities: AICapabilities = [],
        topic: String? = nil,
        topicRanking: [String: [String]] = [:],
        device: DeviceCapability? = nil,
        localModels: [String: AILocalModelPlacement] = [:]
    ) {
        self.capabilities = capabilities
        self.topic = topic
        self.topicRanking = topicRanking
        self.device = device
        self.localModels = localModels
    }

    // MARK: - Capabilities a request names

    /// Everything a provider must offer to serve this request: what the policy
    /// states, plus what the request's own content requires.
    ///
    /// BR-AIL-22: *"a request names what it needs"*. An `AIRequest` carrying an
    /// audio input needs ``AICapabilities/voiceToText`` whether or not the
    /// caller remembered to say so — and before this, a text-only provider
    /// could be selected for it and would refuse at dispatch, one round trip
    /// later, with the fan-out's budget already spent.
    ///
    /// The readings, each from a field of the request:
    ///   * an `.audio` input → `.voiceToText`
    ///   * an `.image` input → `.vision`
    ///   * a `.video` input → `.vision`, because serving it means seeing frames
    ///   * `tools` with at least one entry → `.toolUse`
    ///
    /// `.document` and `.text` inputs add nothing: they are carried in the
    /// prompt by every family, so requiring a flag for them would reject
    /// providers that can serve them. ``AIRequest/responseFormat`` adds nothing
    /// either — there is no capability flag for structured output, and a codec
    /// that cannot carry it refuses with
    /// ``AIKitError/unsupported(field:family:)`` (BR-AIL-17), which is the
    /// right place for a per-family limit.
    public func requiredCapabilities(for request: AIRequest) -> AICapabilities {
        var required = capabilities
        for input in request.inputs {
            switch input.kind {
            case .audio: required.insert(.voiceToText)
            case .image, .video: required.insert(.vision)
            case .document, .text: break
            }
        }
        if let tools = request.tools, !tools.isEmpty {
            required.insert(.toolUse)
        }
        return required
    }

    // MARK: - The device filter

    /// Why this device cannot serve `providerID`'s local model, or `nil` when
    /// it can — including when the provider is remote, has no placement, or no
    /// device was given to filter against.
    func deviceRejection(forProviderID providerID: String) -> AIRoutingRejection? {
        guard let placement = localModels[providerID] else { return nil }
        // The placement's own device first: a cluster's pooled memory is the
        // memory that would hold the weights, not this machine's.
        guard let device = placement.device ?? device, let shortfall = device.shortfall(for: placement.model) else { return nil }
        return .deviceCannotLoad(
            model: placement.model.name,
            requiredGB: shortfall.required,
            availableGB: shortfall.available
        )
    }

    // MARK: - The topic ranking

    /// Where `providerID` sits in the current topic's ranking, or `nil` when
    /// the topic does not name it (or there is no topic).
    ///
    /// A duplicate id keeps its FIRST position: a caller's list that repeats an
    /// entry meant it at its best rank, and reading the last one would demote a
    /// provider for being mentioned twice.
    func rank(ofProviderID providerID: String) -> Int? {
        guard let topic, let ranking = topicRanking[topic] else { return nil }
        return ranking.firstIndex(of: providerID)
    }

    /// Whether this policy ranks anything at all right now.
    var ranksByTopic: Bool {
        guard let topic, let ranking = topicRanking[topic] else { return false }
        return !ranking.isEmpty
    }
}
