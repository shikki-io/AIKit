import Foundation
import NetKit

// MARK: - ProviderHealthCheck

/// Quick ping to verify a provider endpoint is alive and responsive.
/// Used to pre-check availability before routing prompts.
struct ProviderHealthCheck: Sendable {

    // MARK: - Result

    /// Result of a health check ping.
    struct HealthStatus: Sendable, Equatable {
        /// Whether the provider is reachable.
        let available: Bool
        /// Round-trip latency in milliseconds (nil if unavailable).
        let latencyMs: Double?
        /// Human-readable status message.
        let message: String
    }

    // MARK: - Check

    /// Ping a provider's server root to check availability.
    ///
    /// Calls ``OpenAIEndPoint/models`` — `GET /v1/models` — as a lightweight
    /// health route. The URL is BUILT by that endpoint, not concatenated: a
    /// `"\(baseURL)/v1/models"` interpolation produced
    /// `http://host:1234/v1/v1/models` for any address that still carried
    /// `/v1`, and silently reported the server as down (AIKit#18, panel finding
    /// on `ProviderHealthCheck.swift:37`).
    ///
    /// - Parameters:
    ///   - baseURL: The provider's server ROOT (e.g. `http://127.0.0.1:1234`),
    ///     with no path (BR-AIL-09).
    ///   - timeout: Maximum wait time in seconds (default 5).
    /// - Returns: A `HealthStatus` with availability and latency. An address
    ///   that is not a root is reported unavailable, naming why.
    static func check(baseURL: String, timeout: TimeInterval = 5) async -> HealthStatus {
        let request: URLRequest
        do {
            let address = try AIServerAddress(validating: baseURL)
            request = try URLSessionHTTPTransport(timeout: timeout)
                .createRequest(endPoint: OpenAIEndPoint.models(address: address, credential: nil))
        } catch {
            return HealthStatus(
                available: false,
                latencyMs: nil,
                message: AIProviderAttempt.describe(error)
            )
        }

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
