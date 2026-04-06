import Foundation

/// Full description of an AI model.
public struct ModelDescriptor: Sendable, Codable, Identifiable, Equatable {
    public let id: ModelIdentifier
    public var name: String
    public var author: String
    public var description: String
    public var capabilities: AICapabilities
    public var format: ModelFormat
    public var parameters: String
    public var quantization: String?
    public var sizeBytes: Int64
    public var architecture: String
    public var domain: ModelDomain
    public var isLocal: Bool
    public var localPath: URL?
    public var huggingFaceId: String?
    public var tags: [String]
    public var downloadCount: Int?
    public var updatedAt: Date?

    public init(
        id: ModelIdentifier,
        name: String,
        author: String,
        description: String,
        capabilities: AICapabilities,
        format: ModelFormat,
        parameters: String,
        quantization: String? = nil,
        sizeBytes: Int64,
        architecture: String,
        domain: ModelDomain,
        isLocal: Bool = false,
        localPath: URL? = nil,
        huggingFaceId: String? = nil,
        tags: [String] = [],
        downloadCount: Int? = nil,
        updatedAt: Date? = nil
    ) {
        self.id = id
        self.name = name
        self.author = author
        self.description = description
        self.capabilities = capabilities
        self.format = format
        self.parameters = parameters
        self.quantization = quantization
        self.sizeBytes = sizeBytes
        self.architecture = architecture
        self.domain = domain
        self.isLocal = isLocal
        self.localPath = localPath
        self.huggingFaceId = huggingFaceId
        self.tags = tags
        self.downloadCount = downloadCount
        self.updatedAt = updatedAt
    }
}

/// Criteria for filtering models.
public struct ModelFilter: Sendable, Equatable {
    public var domain: ModelDomain?
    public var format: ModelFormat?
    public var isLocal: Bool?
    public var capabilities: AICapabilities?
    public var maxSizeBytes: Int64?

    public init(
        domain: ModelDomain? = nil,
        format: ModelFormat? = nil,
        isLocal: Bool? = nil,
        capabilities: AICapabilities? = nil,
        maxSizeBytes: Int64? = nil
    ) {
        self.domain = domain
        self.format = format
        self.isLocal = isLocal
        self.capabilities = capabilities
        self.maxSizeBytes = maxSizeBytes
    }

    /// Returns true if the descriptor matches all non-nil filter criteria.
    public func matches(_ descriptor: ModelDescriptor) -> Bool {
        if let domain, descriptor.domain != domain { return false }
        if let format, descriptor.format != format { return false }
        if let isLocal, descriptor.isLocal != isLocal { return false }
        if let capabilities, !descriptor.capabilities.contains(capabilities) { return false }
        if let maxSizeBytes, descriptor.sizeBytes > maxSizeBytes { return false }
        return true
    }
}
