import AIKit
import Foundation

/// A double for ``AIEmbeddingProviding``: answers vectors a test configured,
/// and records what it was asked to embed.
///
/// It lives HERE and not in the product for the reason `MockAIProvider` does
/// (AIKit#18 r4500328752): an app that depends on `AIKit` must not be able to
/// build one.
///
/// Its default answer is NOT a zero vector. A zero vector has cosine
/// similarity undefined against everything and 0 against most things, so a
/// recall test reading one gets no rows and no error — the same silent-wrong
/// shape `AIEmbeddingProviding` exists to refuse. The default is a distinct
/// unit-ish vector per input index instead.
public final class MockEmbeddingProvider: AIEmbeddingProviding, @unchecked Sendable {

    // MARK: - Identity

    public let id: String
    public let embeddingModel: String

    /// How many components each answered vector has.
    public let dimensions: Int

    // MARK: - Configuration

    /// Vectors to answer with, in order, instead of the generated ones.
    public var answer: [[Float]]?
    /// Error to throw instead of answering.
    public var error: (any Error)?

    // MARK: - Recording

    /// Every batch it was asked to embed, in order.
    public private(set) var embedded: [[String]] = []

    // MARK: - Init

    public init(id: String, embeddingModel: String, dimensions: Int = 4) {
        self.id = id
        self.embeddingModel = embeddingModel
        self.dimensions = dimensions
    }

    // MARK: - AIEmbeddingProviding

    public func embed(texts: [String]) async throws -> [[Float]] {
        embedded.append(texts)
        if let error { throw error }
        if let answer { return answer }
        guard !texts.isEmpty else { return [] }
        // One distinguishable vector per input position, so a test that
        // asserts input ORDER survived can tell the vectors apart.
        return texts.indices.map { index in
            (0..<dimensions).map { component in
                Float(index + 1) + Float(component) / Float(dimensions)
            }
        }
    }
}
