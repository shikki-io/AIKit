import Foundation

// MARK: - ProviderHealthCheck

/// Quick ping to verify a provider endpoint is alive and responsive.
/// Used to pre-check availability before routing prompts.
public struct ProviderHealthCheck: Sendable {

    // MARK: - Result

    /// Result of a health check ping.
    public struct HealthStatus: Sendable, Equatable {
        /// Whether the provider is reachable.
        public let available: Bool
        /// Round-trip latency in milliseconds (nil if unavailable).
        public let latencyMs: Double?
        /// Human-readable status message.
        public let message: String

        public init(available: Bool, latencyMs: Double?, message: String) {
            self.available = available
            self.latencyMs = latencyMs
            self.message = message
        }
    }

    // MARK: - Check

    /// Ping the provider's base URL to check availability.
    /// Uses GET /v1/models as a lightweight health endpoint (OpenAI-compatible).
    /// - Parameters:
    ///   - baseURL: The provider base URL (e.g. `http://127.0.0.1:1234`).
    ///   - timeout: Maximum wait time in seconds (default 5).
    /// - Returns: A `HealthStatus` with availability and latency.
    public static func check(baseURL: String, timeout: TimeInterval = 5) async -> HealthStatus {
        guard let url = URL(string: "\(baseURL)/v1/models") else {
            return HealthStatus(available: false, latencyMs: nil, message: "Invalid URL: \(baseURL)")
        }

        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.timeoutInterval = timeout

        let start = CFAbsoluteTimeGetCurrent()

        do {
            let (_, response) = try await URLSession.shared.data(for: request)
            let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000.0

            guard let httpResponse = response as? HTTPURLResponse else {
                return HealthStatus(available: false, latencyMs: nil, message: "Non-HTTP response")
            }

            if (200...299).contains(httpResponse.statusCode) {
                return HealthStatus(
                    available: true,
                    latencyMs: elapsed,
                    message: "OK (\(Int(elapsed))ms)"
                )
            } else {
                return HealthStatus(
                    available: false,
                    latencyMs: elapsed,
                    message: "HTTP \(httpResponse.statusCode)"
                )
            }
        } catch {
            return HealthStatus(
                available: false,
                latencyMs: nil,
                message: "Connection failed: \(error.localizedDescription)"
            )
        }
    }
}
