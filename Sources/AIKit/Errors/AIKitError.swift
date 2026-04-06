import Foundation

/// Errors thrown by AIKit operations.
public enum AIKitError: Error, LocalizedError, Equatable {
    case noProviderAvailable(AICapabilities)
    case modelNotFound(ModelIdentifier)
    case downloadFailed(String)
    case engineUnavailable(String)
    case requestFailed(String)
    case streamingNotSupported
    case allProvidersFailed

    public var errorDescription: String? {
        switch self {
        case let .noProviderAvailable(capabilities):
            "No provider available for capabilities: \(capabilities)"
        case let .modelNotFound(id):
            "Model not found: \(id)"
        case let .downloadFailed(reason):
            "Download failed: \(reason)"
        case let .engineUnavailable(engine):
            "Engine unavailable: \(engine)"
        case let .requestFailed(reason):
            "Request failed: \(reason)"
        case .streamingNotSupported:
            "Streaming is not supported by this provider"
        case .allProvidersFailed:
            "All providers failed to complete the request"
        }
    }
}
