import Foundation
import NetKit

/// Search filter for HuggingFace model queries.
public struct HFSearchFilter: Sendable {
    public var tags: [String]?
    public var pipelineTag: String?
    public var sort: String?
    public var direction: String?
    public var limit: Int?

    public init(
        tags: [String]? = nil,
        pipelineTag: String? = nil,
        sort: String? = nil,
        direction: String? = nil,
        limit: Int? = nil
    ) {
        self.tags = tags
        self.pipelineTag = pipelineTag
        self.sort = sort
        self.direction = direction
        self.limit = limit
    }
}

/// Model information returned by the HuggingFace API.
public struct HFModelInfo: Sendable, Codable, Equatable {
    public let modelId: String
    public let author: String?
    public let tags: [String]
    public let downloads: Int
    public let likes: Int
    public let lastModified: String?
    public let pipelineTag: String?
    public let library: String?

    public init(
        modelId: String,
        author: String? = nil,
        tags: [String] = [],
        downloads: Int = 0,
        likes: Int = 0,
        lastModified: String? = nil,
        pipelineTag: String? = nil,
        library: String? = nil
    ) {
        self.modelId = modelId
        self.author = author
        self.tags = tags
        self.downloads = downloads
        self.likes = likes
        self.lastModified = lastModified
        self.pipelineTag = pipelineTag
        self.library = library
    }

    enum CodingKeys: String, CodingKey {
        case modelId
        case author
        case tags
        case downloads
        case likes
        case lastModified
        case pipelineTag = "pipeline_tag"
        case library = "library_name"
    }

    /// Convert to a ModelDescriptor with the specified format.
    public func toDescriptor(format: ModelFormat) -> ModelDescriptor {
        let domain = Self.domainFromPipelineTag(pipelineTag)
        let capabilities = Self.capabilitiesFromPipelineTag(pipelineTag)

        return ModelDescriptor(
            id: ModelIdentifier(provider: "huggingface", modelId: modelId),
            name: modelId.components(separatedBy: "/").last ?? modelId,
            author: author ?? "unknown",
            description: "HuggingFace model: \(modelId)",
            capabilities: capabilities,
            format: format,
            parameters: "unknown",
            sizeBytes: 0,
            architecture: "unknown",
            domain: domain,
            isLocal: false,
            huggingFaceId: modelId,
            tags: tags,
            downloadCount: downloads
        )
    }

    // MARK: - Private helpers

    private static func domainFromPipelineTag(_ tag: String?) -> ModelDomain {
        switch tag {
        case "text-generation", "text2text-generation", "summarization", "translation":
            return .llm
        case "feature-extraction", "sentence-similarity":
            return .embedding
        case "automatic-speech-recognition", "text-to-speech", "audio-classification":
            return .voice
        case "image-classification", "object-detection", "image-segmentation",
             "visual-question-answering", "image-to-text", "document-question-answering":
            return .vision
        case "text-to-image", "image-to-image":
            return .inpainting
        default:
            return .llm
        }
    }

    private static func capabilitiesFromPipelineTag(_ tag: String?) -> AICapabilities {
        switch tag {
        case "text-generation", "text2text-generation", "summarization":
            return .textGeneration
        case "translation":
            return .translation
        case "feature-extraction", "sentence-similarity":
            return .embedding
        case "automatic-speech-recognition":
            return .voiceToText
        case "text-to-speech":
            return .textToVoice
        case "image-classification", "object-detection", "visual-question-answering",
             "image-to-text", "document-question-answering":
            return .vision
        case "text-to-image", "image-to-image":
            return .imageGeneration
        default:
            return .textGeneration
        }
    }
}

/// Client for HuggingFace Hub API.
public struct HuggingFaceClient: Sendable {
    private let networkService: any NetworkProtocol

    public init(networkService: (any NetworkProtocol)? = nil) {
        self.networkService = networkService ?? NetworkService()
    }

    /// Search models on HuggingFace.
    public func search(
        query: String,
        filter: HFSearchFilter? = nil
    ) async throws -> [HFModelInfo] {
        let endpoint = HuggingFaceEndPoint.searchModels(query: query, filter: filter)
        let results: [HFModelInfo] = try await networkService.sendRequest(endpoint: endpoint)
        return results
    }

    /// Get details for a single model.
    public func getModel(id: String) async throws -> HFModelInfo {
        let endpoint = HuggingFaceEndPoint.getModel(id: id)
        let result: HFModelInfo = try await networkService.sendRequest(endpoint: endpoint)
        return result
    }

    /// Build a download URL for a specific file in a model repo.
    public func downloadURL(repoId: String, filename: String) -> URL {
        URL(string: "https://huggingface.co/\(repoId)/resolve/main/\(filename)")!
    }
}
