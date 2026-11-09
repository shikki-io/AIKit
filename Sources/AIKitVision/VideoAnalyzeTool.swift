// MARK: - VideoAnalyzeTool
// Spec §5 Phase 2 deliverable 6 + §7 Scenario 2.
//
// Agent-facing tool: "embed this video + return nearest neighbours from
// local history". The tool owns no state — it composes a
// `ScreenRecordingExtractor` (for the embed-and-store path) with an
// `EmbeddingRetriever` (for nearest-neighbour lookup), both of which share
// the same `EmbeddingStore`.
//
// Agents wire this into their tool registry via `VideoAnalyzeTool.name`
// and `VideoAnalyzeTool.execute(input:)`.

#if canImport(AVFoundation) && canImport(SQLite3)
import AVFoundation
import Foundation

// MARK: - Input / Output

public struct VideoAnalyzeInput: Codable, Sendable, Equatable {
    /// Absolute path to the video to analyze.
    public let videoPath: String
    /// Stable identifier; optional — derived from filename if omitted.
    public let videoId: String?
    /// How many nearest neighbours to return.
    public let topK: Int

    public init(videoPath: String, videoId: String? = nil, topK: Int = 3) {
        self.videoPath = videoPath
        self.videoId = videoId
        self.topK = topK
    }
}

public struct VideoAnalyzeNeighbor: Codable, Sendable, Equatable {
    public let videoId: String
    public let sourcePath: String
    public let similarity: Float
    public let modelVersion: String

    public init(videoId: String, sourcePath: String, similarity: Float, modelVersion: String) {
        self.videoId = videoId
        self.sourcePath = sourcePath
        self.similarity = similarity
        self.modelVersion = modelVersion
    }
}

public struct VideoAnalyzeOutput: Codable, Sendable, Equatable {
    public let storedRecordId: Int64
    public let videoId: String
    public let frameCount: Int
    public let embeddingDim: Int
    public let modelVersion: String
    public let neighbors: [VideoAnalyzeNeighbor]

    public init(
        storedRecordId: Int64,
        videoId: String,
        frameCount: Int,
        embeddingDim: Int,
        modelVersion: String,
        neighbors: [VideoAnalyzeNeighbor]
    ) {
        self.storedRecordId = storedRecordId
        self.videoId = videoId
        self.frameCount = frameCount
        self.embeddingDim = embeddingDim
        self.modelVersion = modelVersion
        self.neighbors = neighbors
    }
}

// MARK: - Tool

public struct VideoAnalyzeTool: Sendable {

    public static let name: String = "VideoAnalyze"
    public static let description: String =
        "Embed a video with V-JEPA 2 and return its nearest neighbours from local history."

    public let extractor: ScreenRecordingExtractor
    public let retriever: EmbeddingRetriever

    public init(extractor: ScreenRecordingExtractor, retriever: EmbeddingRetriever) {
        self.extractor = extractor
        self.retriever = retriever
    }

    public init(node: VJEPANode, store: EmbeddingStore) {
        self.extractor = ScreenRecordingExtractor(node: node, store: store)
        self.retriever = EmbeddingRetriever(store: store)
    }

    public func execute(input: VideoAnalyzeInput) async throws -> VideoAnalyzeOutput {
        let url = URL(fileURLWithPath: input.videoPath)
        let record = try await extractor.extractAndStore(
            videoAt: url,
            videoId: input.videoId
        )
        // Nearest neighbours, excluding the record we just inserted.
        let all = try retriever.topK(query: record.embedding, k: input.topK + 1)
        let neighbors = all
            .filter { $0.record.id != record.id }
            .prefix(input.topK)
            .map {
                VideoAnalyzeNeighbor(
                    videoId: $0.record.videoId,
                    sourcePath: $0.record.sourcePath,
                    similarity: $0.similarity,
                    modelVersion: $0.record.modelVersion
                )
            }
        return VideoAnalyzeOutput(
            storedRecordId: record.id,
            videoId: record.videoId,
            frameCount: record.embedding.frameCount,
            embeddingDim: record.embedding.featureDim,
            modelVersion: record.modelVersion,
            neighbors: Array(neighbors)
        )
    }
}

#endif
