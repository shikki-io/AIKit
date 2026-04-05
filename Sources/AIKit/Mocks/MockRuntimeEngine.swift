import Foundation

/// Mock runtime engine for testing.
public final class MockRuntimeEngine: RuntimeEngine, @unchecked Sendable {
    public let id: String
    public let displayName: String
    public let supportedFormats: [ModelFormat]
    public let isAvailable: Bool
    private var _loadedModels: [ModelIdentifier] = []
    public var shouldThrow: Error?

    public init(
        id: String = "mock-engine",
        displayName: String = "Mock Engine",
        supportedFormats: [ModelFormat] = [.gguf],
        isAvailable: Bool = true
    ) {
        self.id = id
        self.displayName = displayName
        self.supportedFormats = supportedFormats
        self.isAvailable = isAvailable
    }

    public func loadModel(_ descriptor: ModelDescriptor) async throws -> any AIProvider {
        if let error = shouldThrow { throw error }
        _loadedModels.append(descriptor.id)
        return MockAIProvider(
            id: descriptor.id.description,
            displayName: descriptor.name,
            capabilities: descriptor.capabilities
        )
    }

    public func unloadModel(_ id: ModelIdentifier) async throws {
        if let error = shouldThrow { throw error }
        _loadedModels.removeAll { $0 == id }
    }

    public func loadedModels() -> [ModelIdentifier] {
        _loadedModels
    }
}
