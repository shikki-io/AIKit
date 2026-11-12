import Foundation
import NetKit

// `URLSession` and `URLRequest` live in FoundationNetworking on a
// swift-corelibs platform, and in Foundation on an Apple one (W30 of spec
// ai-layer-aikit-core-2026-10-31).
#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

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

        // `ContinuousClock` and not `CFAbsoluteTimeGetCurrent()`, for two
        // reasons (W30). CoreFoundation is Apple-only, so the old call did not
        // compile on Linux at all; and it reads the WALL clock, so an NTP step
        // or a daylight-saving jump landing between the two reads could report
        // a negative or hour-long latency for a server that answered in 3 ms.
        // A continuous clock cannot move backwards.
        let start = ContinuousClock.now

        do {
            let (_, response) = try await URLSession.shared.data(for: request)
            let elapsed = Self.milliseconds(since: start)

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

    // MARK: - Timing

    /// Milliseconds elapsed since `start` on the continuous clock.
    ///
    /// One `Duration`, read once, then split: assembling the figure from two
    /// separate `.now` reads — one for the seconds, one for the attoseconds —
    /// would measure two different instants.
    private static func milliseconds(since start: ContinuousClock.Instant) -> Double {
        let components = (ContinuousClock.now - start).components
        return Double(components.seconds) * 1000.0
            + Double(components.attoseconds) / 1_000_000_000_000_000.0
    }
}
