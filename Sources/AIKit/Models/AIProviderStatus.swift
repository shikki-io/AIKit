/// Current status of an AI provider.
public enum AIProviderStatus: Sendable {
    /// Provider is ready to accept requests.
    case ready
    /// Model is loading into memory.
    case loading(progress: Double)
    /// Model is being downloaded.
    case downloading(progress: Double, totalBytes: Int64)
    /// Provider encountered an error.
    case error(String)
    /// Provider is not available on this platform.
    case unavailable
}

extension AIProviderStatus: Equatable {
    public static func == (lhs: AIProviderStatus, rhs: AIProviderStatus) -> Bool {
        switch (lhs, rhs) {
        case (.ready, .ready):
            return true
        case let (.loading(a), .loading(b)):
            return a == b
        case let (.downloading(p1, t1), .downloading(p2, t2)):
            return p1 == p2 && t1 == t2
        case let (.error(a), .error(b)):
            return a == b
        case (.unavailable, .unavailable):
            return true
        default:
            return false
        }
    }
}
