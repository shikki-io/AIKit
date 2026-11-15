import Foundation

/// Turning text into vectors through a configured provider.
///
/// BR-AIL-24. Public, and part of DOOR 2 (TALK) rather than a fifth door
/// (BR-AIL-21): embedding is talking to a configured provider, the same
/// conversation as `complete`, so it belongs beside ``AIProvider`` and not in a
/// new kit. It is still a separate PROTOCOL, for the reason below, and its
/// arrival is visible as an edit to `AIKitPublicAPI.txt`.
///
/// WHY NOT A METHOD ON ``AIProvider``. `complete` and `stream` are chat
/// surfaces and an embedding is neither — it takes no conversation, carries no
/// tools and answers no text. Most configured providers serve no embedding
/// model at all, so a requirement on `AIProvider` would be one every CLI,
/// MLX-in-process and Core ML row had to fake. That is exactly how
/// ``AICapabilities/embedding`` came to be declared and never called.
///
/// WHY NOT A WIRE-SHAPE ENUM. The route is the API's
/// `EndPoint` enum's business and nothing else's (BR-AIL-16, AIKit#18
/// r4493304830 — *"This is the EndPoint protocol and system the SSoT, so
/// exchange that!"*). An implementation here picks the endpoint enum for its
/// entry's family ONCE, when it is built, the same division
/// ``HTTPAIProvider`` follows for codecs.
///
/// The model is NOT a parameter. It comes from the settings row's
/// `embedding_model` (BR-AIL-25), because the operator's own `providers.toml`
/// states why it has to: *"`embedding_model` is deliberately separate from
/// `default_model`/`model`, which are CHAT models: a provider serves both, and
/// embedding a query with a chat model yields a vector that is silently wrong
/// rather than an error."* A caller that could pass a model could pass the
/// chat one.
public protocol AIEmbeddingProviding: Sendable {

    /// The settings row this provider was built from — the id the operator
    /// wrote.
    var id: String { get }

    /// The model it embeds with, as the settings row declares it.
    ///
    /// Readable because a stored vector is only comparable to another vector
    /// from the SAME model on the SAME server: a caller that keeps vectors has
    /// to be able to record which model made them.
    var embeddingModel: String { get }

    /// Embeds every text, answering one vector per input IN INPUT ORDER.
    ///
    /// The call ANSWERS OR REFUSES. There is no `nil`, no empty array for a
    /// non-empty input and no substitute vector: a short or re-ordered reply
    /// would pair vectors with the wrong texts, which produces plausible
    /// similarity scores and no error at all.
    ///
    /// - Returns: `texts.count` vectors, aligned with `texts`. An empty input
    ///   answers an empty array without sending a request.
    /// - Throws: ``AIKitError`` — `.http` for a non-2xx,
    ///   `.embeddingCountMismatch` for a reply that does not align,
    ///   `.emptyReply` for a 2xx carrying no vector, `.settings` for a row
    ///   that declares no `embedding_model`, and
    ///   `.unsupported(field:family:)` for a family with no embeddings route.
    func embed(texts: [String]) async throws -> [[Float]]
}

// MARK: - One text

extension AIEmbeddingProviding {

    /// One text, one vector.
    ///
    /// - Throws: As ``embed(texts:)``, plus `.emptyReply` rather than an empty
    ///   array: a caller asking for one vector has no use for none.
    public func embed(text: String) async throws -> [Float] {
        let vectors = try await embed(texts: [text])
        guard let first = vectors.first else {
            throw AIKitError.requestFailed(
                "'\(id)' answered no vector for one text, which `embed(texts:)` should already have refused"
            )
        }
        return first
    }
}
