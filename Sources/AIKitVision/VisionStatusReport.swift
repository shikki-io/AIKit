// MARK: - VisionStatusReport
// Spec §5 Phase 2 deliverable 7: back-end of `shi vision status`.
//
// The CLI surface is thin — render this report as text or JSON from a
// `shi vision status` command. We keep the data shape here so the CLI
// can be wired without dragging CLI dependencies into ShikkiKit.

#if canImport(SQLite3)
import Foundation

public struct VisionStatusReport: Codable, Sendable, Equatable {
    public let modelLoaded: Bool
    public let modelName: String
    public let modelVersion: String
    public let inputFrames: Int
    public let embeddingDim: Int
    public let latencyP50Ms: Int
    public let memoryPeakMB: Int
    public let embeddingsStorePath: String
    public let embeddingsCount: Int

    public init(
        modelLoaded: Bool,
        modelName: String,
        modelVersion: String,
        inputFrames: Int,
        embeddingDim: Int,
        latencyP50Ms: Int,
        memoryPeakMB: Int,
        embeddingsStorePath: String,
        embeddingsCount: Int
    ) {
        self.modelLoaded = modelLoaded
        self.modelName = modelName
        self.modelVersion = modelVersion
        self.inputFrames = inputFrames
        self.embeddingDim = embeddingDim
        self.latencyP50Ms = latencyP50Ms
        self.memoryPeakMB = memoryPeakMB
        self.embeddingsStorePath = embeddingsStorePath
        self.embeddingsCount = embeddingsCount
    }

    // MARK: - Rendering

    /// Plain text renderer for terminal output (CLI default).
    public func renderText() -> String {
        let loaded = modelLoaded ? "yes" : "no"
        return """
        V-JEPA Vision Node
          model loaded       : \(loaded)
          model name         : \(modelName)
          model version      : \(modelVersion)
          input frames       : \(inputFrames)
          embedding dim      : \(embeddingDim)
          latency p50 (ms)   : \(latencyP50Ms)
          memory peak (MB)   : \(memoryPeakMB)
          store path         : \(embeddingsStorePath)
          embeddings count   : \(embeddingsCount)
        """
    }

    // MARK: - Factory

    /// Compose a report from a live node + store.
    public static func make(node: VJEPANode, store: EmbeddingStore) async throws -> VisionStatusReport {
        let metadata = await node.makeMetadata()
        let count = try store.count()
        return VisionStatusReport(
            modelLoaded: true,
            modelName: metadata.modelName,
            modelVersion: metadata.modelVersion,
            inputFrames: metadata.inputFrames,
            embeddingDim: metadata.embeddingDim,
            latencyP50Ms: metadata.latencyP50Ms,
            memoryPeakMB: metadata.memoryPeakMB,
            embeddingsStorePath: store.path,
            embeddingsCount: count
        )
    }
}

#endif
