import Foundation
import NetKit

/// The embeddings route for a settings row, over HTTP.
///
/// BR-AIL-24, the embeddings half of BR-AIL-15's one entry point. The division
/// is the one ``HTTPAIProvider`` already follows, with the same three
/// collaborators:
///
/// - the ``AIProviderEntry`` says who, where and which model (settings),
/// - the API's `EndPoint` enum owns the ROUTE (BR-AIL-16),
/// - the ``AIHTTPTransport`` sends bytes and reports the status (BR-AIL-19).
///
/// WHAT THIS REPLACES, and why the shape it replaces was wrong (AIKit#24,
/// operator review r4504... on `AIEmbeddingWireShape.swift:25` — *"EndPoint is
/// the SSoT for end points and per topics, not mixing several systems"* — and
/// on `AIEmbeddingProvider.swift:58` — *"it's the endpoint protocol that has to
/// manage this, not a trick, because we can have 2 different api root paths
/// from one protocol"*):
///
/// 1. An `AIEmbeddingWireShape` enum carried `v1/embeddings` and `api/embed` as
///    path COMPONENTS and an `EmbeddingsEndPoint` concatenated them. That is a
///    SECOND route system beside the `EndPoint` enums, which are the SSoT: a
///    route lives in `Engines/Endpoints/` and the `AIKitNoLocalhostFallback`
///    scope fails any route literal outside it. The route used here is
///    ``OpenAIEndPoint/embeddings(address:credential:body:)``, a case that
///    already existed on the enum that already owns `/v1`.
/// 2. It took `baseURL: String` and STRIPPED a `/v1` suffix off it so the
///    shape could re-add its own root. ``AIServerAddress`` makes that
///    impossible and unnecessary: it refuses an address carrying a path, by
///    name, and the route owns the whole path (BR-AIL-09). Two roots on one
///    server are two cases on their APIs' enums, not two spellings of a base
///    that something trims.
/// 3. It raced the call in a throwing task group to honour
///    `request_timeout`. The seam for that is the transport:
///    ``URLSessionHTTPTransport`` is built with ``AIProviderEntry/requestTimeout``
///    and puts it on every request. The deadline is still honoured — by the
///    layer that owns it.
struct HTTPEmbeddingProvider: AIEmbeddingProviding {

    // MARK: - Collaborators

    /// Who, where, which embedding model — as the operator configured it.
    let entry: AIProviderEntry
    /// Bytes out, status and bytes in.
    let network: any AIHTTPTransport
    /// The resolved secret, when the entry names one. Never in settings.
    let credential: String?

    // MARK: - AIEmbeddingProviding

    var id: String { entry.id }

    /// The row's `embedding_model`.
    ///
    /// Non-throwing here because ``init(entry:network:credential:)`` has
    /// already refused a row without one: a provider that exists cannot be
    /// missing the model it embeds with.
    let embeddingModel: String

    // MARK: - Init

    /// Builds the provider for one settings row.
    ///
    /// - Throws: ``AIKitError/settings(file:entry:reason:)`` when the row
    ///   declares no `embedding_model` (BR-AIL-25), and
    ///   ``AIKitError/unsupported(field:family:)`` when its family serves no
    ///   embeddings route — refused HERE, when the provider is built, rather
    ///   than on the first call, so a misconfigured row is reported with the
    ///   rest of the settings.
    init(
        entry: AIProviderEntry,
        network: any AIHTTPTransport,
        credential: String? = nil
    ) throws(AIKitError) {
        self.embeddingModel = try entry.embeddingModelName()
        _ = try Self.endpointFamily(of: entry)
        self.entry = entry
        self.network = network
        self.credential = credential
    }

    /// The API whose `EndPoint` enum carries this row's embeddings route.
    ///
    /// Resolved from the family ONCE, not switched on at call time — the rule
    /// ``HTTPAIProvider`` follows for codecs (BR-AIL-15).
    ///
    /// Only the OpenAI-shaped surface documents an embeddings route among the
    /// families AIKit serves: ``AnthropicEndPoint`` has none, and
    /// ``LMStudioEndPoint``'s own REST API is model MANAGEMENT. Both of the
    /// operator's embedding servers — LM Studio and Ollama — are configured
    /// `openai-compatible`, which is this family (measured in
    /// `~/.shikki/settings/providers.toml`, 2026-11-15). Ollama's NATIVE
    /// `/api/embed` is a different API: when a row needs it, it arrives as an
    /// `OllamaEndPoint` enum with its own captured route table, not as a
    /// second root inside this one.
    private static func endpointFamily(of entry: AIProviderEntry) throws(AIKitError) -> AIProviderFamily {
        switch entry.family {
        case .openAIChat, .openAIResponses:
            return entry.family
        case .anthropicMessages, .lmStudioNative, .cliSubprocess, .mlxInProcess, .coreML:
            throw .unsupported(field: "embeddings", family: entry.family)
        }
    }

    // MARK: - embed

    func embed(texts: [String]) async throws -> [[Float]] {
        // No request at all for no input: a server asked to embed nothing
        // answers an error on some surfaces and an empty list on others, and
        // neither is worth a round trip.
        guard !texts.isEmpty else { return [] }

        let address = try entry.serverAddress()
        let endpoint = OpenAIEndPoint.embeddings(
            address: address,
            credential: credential,
            body: EmbeddingsRequest(model: embeddingModel, input: texts)
        )
        let target = AIRequestTarget(entry: entry, address: address, route: endpoint.path)

        // Throws before anything is sent when the body cannot be encoded,
        // rather than sending a request with no body.
        _ = try endpoint.bodyObject()
        let request = try network.createRequest(endPoint: endpoint)
        let reply = try await network.send(request)

        // BR-AIL-19: the status decides, before any body is read as a reply.
        guard 200..<300 ~= reply.status else {
            throw AIKitError.http(status: reply.status, body: reply.text, target: target)
        }

        let dto: EmbeddingsResponseDTO
        do {
            dto = try JSONDecoder().decode(EmbeddingsResponseDTO.self, from: reply.body)
        } catch {
            throw AIKitError.replyUndecodable(target: target, reason: "\(error)")
        }

        let vectors = dto.vectorsInInputOrder
        guard !vectors.isEmpty else {
            throw AIKitError.emptyReply(target: target)
        }
        guard vectors.count == texts.count else {
            throw AIKitError.embeddingCountMismatch(
                provider: id, expected: texts.count, received: vectors.count)
        }
        return vectors
    }
}
