import Foundation
#if canImport(FoundationNetworking)
    // Spec c8fa126c, the one conditional import the spec declares CORRECT: off
    // Apple, Foundation's URL loading lives in its own module. This is upstream
    // Foundation's split, not ours — without it `URLSession` is `AnyObject` and
    // every call through it fails to type-check for a Linux target.
    import FoundationNetworking
#endif

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

        // `ContinuousClock`, not `CFAbsoluteTimeGetCurrent`: CoreFoundation does
        // not exist off Apple, and a latency measurement must not be the reason
        // a health check cannot be compiled for Linux (spec c8fa126c). It is
        // also the better clock for the job — `CFAbsoluteTimeGetCurrent` reads
        // wall time, so an NTP step or a DST change during the request lands
        // directly in the reported latency, while `ContinuousClock` is
        // monotonic and keeps counting across sleep.
        let start = ContinuousClock.now

        do {
            let (_, response) = try await URLSession.shared.data(for: request)
            let elapsed = Self.milliseconds(start.duration(to: .now))

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

    /// A `Duration` in milliseconds.
    ///
    /// `Duration.components` is `(seconds, attoseconds)`; an attosecond is
    /// 1e-18 s, so a millisecond is 1e15 attoseconds. Reading the clock once
    /// and converting here is deliberate — calling `.now` twice in one
    /// expression measures two different instants.
    static func milliseconds(_ duration: Duration) -> Double {
        let parts = duration.components
        return Double(parts.seconds) * 1000.0 + Double(parts.attoseconds) / 1_000_000_000_000_000.0
    }

}
