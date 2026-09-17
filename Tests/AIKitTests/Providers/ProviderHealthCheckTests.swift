import Foundation
import Testing
@testable import AIKit

// MARK: - ProviderHealthCheckTests
//
// Moved with the provider family (relocation S2): the health status shape and a
// probe against a port nothing listens on.

@Suite("ProviderHealthCheck — availability and latency")
struct ProviderHealthCheckTests {

    // MARK: - Test 9: HealthCheck returns available with latency

    @Test("HealthCheck available result has latency")
    func healthCheckAvailableHasLatency() {
        let status = ProviderHealthCheck.HealthStatus(
            available: true,
            latencyMs: 42.5,
            message: "OK (42ms)"
        )

        #expect(status.available == true)
        #expect(status.latencyMs == 42.5)
        #expect(status.message.contains("OK"))
    }

    // MARK: - Test 10: HealthCheck returns unavailable when offline

    @Test("HealthCheck unavailable result has nil latency")
    func healthCheckUnavailableNilLatency() {
        let status = ProviderHealthCheck.HealthStatus(
            available: false,
            latencyMs: nil,
            message: "Connection failed: could not connect"
        )

        #expect(status.available == false)
        #expect(status.latencyMs == nil)
        #expect(status.message.contains("Connection failed"))
    }

    // MARK: - Additional: HealthCheck against unreachable host

    @Test("HealthCheck against unreachable host returns unavailable")
    func healthCheckUnreachableHost() async {
        // Use a port that is almost certainly not listening
        let status = await ProviderHealthCheck.check(
            baseURL: "http://127.0.0.1:59999",
            timeout: 2
        )

        #expect(status.available == false)
        #expect(status.latencyMs == nil)
    }
}
