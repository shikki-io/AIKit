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

    /// Minimum free RAM needed to load this model, in gigabytes.
    ///
    /// A number somebody measured on a real device, not a computed figure.
    /// `nil` means unmeasured, which `DeviceCapability.canLoad` reads as "do
    /// not refuse" — never as "needs nothing".
    ///
    /// Folded in from shikki-client's `CatalogEntry.minRAMGB` in W7 of spec
    /// ai-layer-aikit-core-2026-10-31: the client's `Registry/` defined a
    /// second `ModelDescriptor`, and this one is the survivor.
    public var minRAMGB: Double?

    /// Expected tokens-per-second per Apple silicon class.
    ///
    /// Lets a caller say "expected ~20 tok/s" before a user commits to a
    /// 2.4 GB download. An absent class falls back to
    /// `DeviceCapability.estimatedTokPerSecFloor`.
    ///
    /// Folded in from shikki-client's `CatalogEntry.estimatedTokPerSec`.
    public var estimatedTokPerSec: [DeviceClass: Double]

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
        updatedAt: Date? = nil,
        minRAMGB: Double? = nil,
        estimatedTokPerSec: [DeviceClass: Double] = [:]
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
        self.minRAMGB = minRAMGB
        self.estimatedTokPerSec = estimatedTokPerSec
    }

    /// Human-readable size (e.g. "1.2 GB").
    public var formattedSize: String {
        ByteCountFormatter.string(fromByteCount: sizeBytes, countStyle: .file)
    }

    enum CodingKeys: String, CodingKey {
        case id
        case name
        case author
        case description
        case capabilities
        case format
        case parameters
        case quantization
        case sizeBytes
        case architecture
        case domain
        case isLocal
        case localPath
        case huggingFaceId
        case tags
        case downloadCount
        case updatedAt
        case minRAMGB
        case estimatedTokPerSec
    }

    /// Decodes a descriptor, including one stored before the device fields
    /// existed.
    ///
    /// `LocalModelStore` writes descriptors to disk, so a store written by an
    /// earlier version must stay readable: a missing `estimatedTokPerSec` means
    /// no measurements, not a corrupt store. Without this the synthesized
    /// decoder would throw `keyNotFound` on every pre-W7 registry — the same
    /// failure `AIRequest.inputs` and `AIResponse.providerID` already guard.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(ModelIdentifier.self, forKey: .id)
        name = try container.decode(String.self, forKey: .name)
        author = try container.decode(String.self, forKey: .author)
        description = try container.decode(String.self, forKey: .description)
        capabilities = try container.decode(AICapabilities.self, forKey: .capabilities)
        format = try container.decode(ModelFormat.self, forKey: .format)
        parameters = try container.decode(String.self, forKey: .parameters)
        quantization = try container.decodeIfPresent(String.self, forKey: .quantization)
        sizeBytes = try container.decode(Int64.self, forKey: .sizeBytes)
        architecture = try container.decode(String.self, forKey: .architecture)
        domain = try container.decode(ModelDomain.self, forKey: .domain)
        isLocal = try container.decode(Bool.self, forKey: .isLocal)
        localPath = try container.decodeIfPresent(URL.self, forKey: .localPath)
        huggingFaceId = try container.decodeIfPresent(String.self, forKey: .huggingFaceId)
        tags = try container.decode([String].self, forKey: .tags)
        downloadCount = try container.decodeIfPresent(Int.self, forKey: .downloadCount)
        updatedAt = try container.decodeIfPresent(Date.self, forKey: .updatedAt)
        minRAMGB = try container.decodeIfPresent(Double.self, forKey: .minRAMGB)
        estimatedTokPerSec =
            try container.decodeIfPresent([DeviceClass: Double].self, forKey: .estimatedTokPerSec)
            ?? [:]
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
