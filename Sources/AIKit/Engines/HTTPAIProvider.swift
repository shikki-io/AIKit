import Foundation
import NetKit

/// The one entry point for talking to an AI server over HTTP, local or remote.
///
/// BR-AIL-15. A caller holds `any AIProvider` and calls `complete` or `stream`.
/// Which protocol the server speaks is the entry's ``AIProviderEntry/family``,
/// resolved ONCE here when the provider is built; no type above this one knows
/// a route, a body shape or a wire standard, and nothing switches on a standard
/// at call time.
///
/// Three collaborators, each with one job:
///
/// - the ``AIProviderEntry`` says who, where and which model (settings),
/// - the ``WireCodec`` translates request and reply (BR-AIL-17),
/// - the ``AIHTTPTransport`` sends bytes and reports the status (BR-AIL-19).
///
/// This replaces `OpenAIProvider` (one family, a 60-second ceiling and an SSE
/// delegate that turned a 429 into an empty success), `OpenAICompatibleEngine`'s
/// provider half, and `AIWireStandard.complete` (request, reply and routes in
/// one file, with three of four families having no caller at all).
///
/// A local server and a shared one are THE SAME TYPE with a different address
/// (BR-AIL-20): there is no "local provider" class here, and nothing decides
/// local-ness by looking for a substring in a host.
public struct HTTPAIProvider: AIProvider, Sendable {

    // MARK: - AIProvider

    public let id: String
    public let displayName: String
    public let capabilities: AICapabilities

    // MARK: - Collaborators

    /// Who, where, which model — as the operator configured it.
    public let entry: AIProviderEntry
    /// Request in, reply out, for this entry's family.
    public let codec: any WireCodec
    /// Bytes out, status and bytes in.
    public let network: any AIHTTPTransport
    /// The resolved secret, when the entry names one. Never stored in settings
    /// (BR-AIL-20): the entry carries an `apiKeyRef` and a broker resolves it
    /// before the provider is built.
    public let credential: String?

    // MARK: - Init

    /// Builds a provider for one settings entry, with its codec given
    /// explicitly.
    ///
    /// - Parameters:
    ///   - entry: The configured provider.
    ///   - codec: The wire translator. Must speak `entry.family`.
    ///   - network: The transport. Inject a stub in tests (BR-AIL-12).
    ///   - credential: The resolved secret, when there is one.
    ///   - displayName: Human-readable name. Defaults to the entry's id and
    ///     its family.
    ///   - capabilities: What the model can do. Defaults to what the CODEC can
    ///     carry, so a provider never advertises a capability its wire drops —
    ///     which is the inversion this wave removes.
    public init(
        entry: AIProviderEntry,
        codec: any WireCodec,
        network: any AIHTTPTransport,
        credential: String? = nil,
        displayName: String? = nil,
        capabilities: AICapabilities? = nil
    ) {
        self.id = entry.id
        self.displayName = displayName ?? "\(entry.id) (\(entry.family.displayName))"
        self.capabilities = capabilities ?? Self.capabilities(of: codec)
        self.entry = entry
        self.codec = codec
        self.network = network
        self.credential = credential
    }

    /// Builds a provider for one settings entry, resolving the codec from its
    /// family.
    ///
    /// This is the production caller every HTTP family has (T-AIL-42): any of
    /// the four named in settings builds a working provider here with no new
    /// code. W22 of spec ai-layer-aikit-core-2026-10-31 calls exactly this from
    /// its `AIProviderRepository`, once per `[[providers]]` row.
    ///
    /// - Throws: ``AIKitError/engineUnavailable(_:)`` when the entry names a
    ///   local family, which has no wire and is served by its own provider.
    public init(
        entry: AIProviderEntry,
        network: any AIHTTPTransport,
        credential: String? = nil,
        displayName: String? = nil,
        capabilities: AICapabilities? = nil
    ) throws(AIKitError) {
        self.init(
            entry: entry,
            codec: try entry.family.codec(),
            network: network,
            credential: credential,
            displayName: displayName,
            capabilities: capabilities
        )
    }

    /// Builds a provider for one settings entry on the production transport,
    /// carrying the entry's own timeout.
    ///
    /// - Throws: As above.
    public init(
        entry: AIProviderEntry,
        credential: String? = nil,
        displayName: String? = nil,
        capabilities: AICapabilities? = nil
    ) throws(AIKitError) {
        try self.init(
            entry: entry,
            network: URLSessionHTTPTransport(timeout: entry.requestTimeout),
            credential: credential,
            displayName: displayName,
            capabilities: capabilities
        )
    }

    /// What a codec's wire can actually serve, as `AICapabilities`.
    private static func capabilities(of codec: any WireCodec) -> AICapabilities {
        var capabilities: AICapabilities = .textGeneration
        if codec.capabilities.tools {
            capabilities.insert(.toolUse)
        }
        if codec.capabilities.inputKinds.contains(.image) {
            capabilities.insert(.vision)
        }
        return capabilities
    }

    public var status: AIProviderStatus {
        get async { .ready }
    }

    // MARK: - complete

    public func complete(request: AIRequest) async throws -> AIResponse {
        let start = ContinuousClock.now
        let endpoint = try codec.endpoint(for: request, entry: entry, credential: credential, stream: false)
        let target = AIRequestTarget(entry: entry, address: try entry.serverAddress(), route: endpoint.path)
        let urlRequest = try network.createRequest(endPoint: endpoint)

        let reply = try await network.send(urlRequest)
        // BR-AIL-19: the status decides, on both paths, before any body is
        // read as a reply. The error carries the status, the server's own text
        // and where the request went — and its fallback eligibility comes from
        // the status, never from a prefix of a sentence.
        guard 200..<300 ~= reply.status else {
            throw AIKitError.http(status: reply.status, body: reply.text, target: target)
        }

        return try codec.response(
            from: reply.body,
            context: WireReplyContext(
                target: target,
                requestedModel: entry.model,
                providerID: id,
                latencyMs: Self.milliseconds(since: start)
            )
        )
    }

    // MARK: - stream

    public func stream(request: AIRequest) async throws -> AsyncThrowingStream<AIChunk, Error> {
        let start = ContinuousClock.now
        let endpoint = try codec.endpoint(for: request, entry: entry, credential: credential, stream: true)
        let target = AIRequestTarget(entry: entry, address: try entry.serverAddress(), route: endpoint.path)
        let urlRequest = try network.createRequest(endPoint: endpoint)

        // The status is read here, before this function returns a stream at
        // all, so a 401 / 429 / 500 THROWS to the caller instead of arriving
        // as a stream that finishes with no content and no error. That silent
        // success is what stopped a fallback chain from ever firing on a
        // streamed call.
        let lines: AsyncThrowingStream<String, any Error>
        switch try await network.open(urlRequest, lineByteCap: entry.streamLineByteCap) {
        case .open(_, let opened):
            lines = opened
        case .rejected(let status, let body):
            throw AIKitError.http(status: status, body: body, target: target)
        }

        let context = WireReplyContext(
            target: target,
            requestedModel: entry.model,
            providerID: id,
            latencyMs: Self.milliseconds(since: start)
        )
        let codec = self.codec

        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    for try await line in lines {
                        if let chunk = try codec.chunk(fromLine: line, context: context) {
                            continuation.yield(chunk)
                        }
                    }
                    continuation.finish()
                } catch AIHTTPTransportError.lineExceededCap(let cap) {
                    // Re-thrown with the target the transport does not know.
                    continuation.finish(throwing: AIKitError.streamLineTooLong(cap: cap, target: target))
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }

    // MARK: - Private

    private static func milliseconds(since start: ContinuousClock.Instant) -> Int {
        let elapsed = start.duration(to: .now)
        return Int(elapsed.components.seconds * 1000 + elapsed.components.attoseconds / 1_000_000_000_000_000)
    }
}
