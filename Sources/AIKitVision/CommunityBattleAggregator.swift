// MARK: - CommunityBattleAggregator
// Spec §5 Phase 3 deliverable 8 — anonymised battle scores publishable to
// `shikki.community.vision_battles`. Opt-in. No raw embeddings leave the
// machine (BR-20 privacy posture preserved).
//
// The aggregator rolls per-category `CategoryWinRate` snapshots into a
// community payload. The payload carries:
//
//   - corpus version (from `VisionBenchmark.corpusVersion`)
//   - an anonymous `nodeId` hash (SHA-256 of the real node id, truncated)
//   - per-category counts + mean cost-adjusted scores
//   - aggregate cost telemetry totals (energy Wh, token USD) so the
//     community can compare hardware/price regimes
//
// What is explicitly stripped:
//   - individual `BattleMetrics` records (no task-level leakage)
//   - provider free-form answers (no prompt leakage)
//   - host name and per-call memory footprints
//
// Publishing is delegated to the same `VisionNATSPublisher` interface
// `VJEPANode` uses, so existing NATS adapters wire in unchanged.

import Foundation

// MARK: - Payload

public struct CommunityBattleReport: Codable, Sendable, Equatable {

    public static let subject: String = "shikki.community.vision_battles"

    public let corpusVersion: String
    public let anonymousNodeId: String
    public let totalBattles: Int
    public let categories: [CategoryWinRate]
    public let totalEnergyWh: Double
    public let totalTokenUSD: Double
    public let generatedAt: Date

    public init(
        corpusVersion: String,
        anonymousNodeId: String,
        totalBattles: Int,
        categories: [CategoryWinRate],
        totalEnergyWh: Double,
        totalTokenUSD: Double,
        generatedAt: Date
    ) {
        self.corpusVersion = corpusVersion
        self.anonymousNodeId = anonymousNodeId
        self.totalBattles = totalBattles
        self.categories = categories
        self.totalEnergyWh = totalEnergyWh
        self.totalTokenUSD = totalTokenUSD
        self.generatedAt = generatedAt
    }
}

// MARK: - Aggregator

public struct CommunityBattleAggregator: Sendable {

    public let corpusVersion: String

    public init(corpusVersion: String = VisionBenchmark.corpusVersion) {
        self.corpusVersion = corpusVersion
    }

    /// Roll a batch of `BattleMetrics` into an anonymised community report.
    /// `nodeId` is hashed (SHA-256 prefix) so the publisher cannot be traced
    /// back to a specific installation.
    public func aggregate(
        battles: [BattleMetrics],
        nodeId: String,
        now: Date = Date()
    ) -> CommunityBattleReport {
        // Per-category roll-up via a private router. Cheaper than reaching
        // into `AdaptiveVisionRouter`'s internals.
        let router = AdaptiveVisionRouter(minBattlesForDecision: 1, decisionThreshold: 0)
        router.recordAll(battles)
        let categories = router.winRates()

        let energy = battles.reduce(0.0) { $0 + $1.providerA.energyWh + $1.providerB.energyWh }
        let tokens = battles.reduce(0.0) { $0 + $1.providerA.tokenCostUSD + $1.providerB.tokenCostUSD }

        return CommunityBattleReport(
            corpusVersion: corpusVersion,
            anonymousNodeId: Self.anonymise(nodeId: nodeId),
            totalBattles: battles.count,
            categories: categories,
            totalEnergyWh: energy,
            totalTokenUSD: tokens,
            generatedAt: now
        )
    }

    /// Encode the report for NATS publishing. Stable, sorted JSON so two
    /// identical aggregations produce byte-identical payloads.
    public func encode(_ report: CommunityBattleReport) throws -> Data {
        let enc = JSONEncoder()
        enc.dateEncodingStrategy = .iso8601
        enc.outputFormatting = [.sortedKeys, .prettyPrinted]
        return try enc.encode(report)
    }

    // MARK: Publishing (opt-in)

    /// Publish an aggregated report to the community NATS subject. Caller
    /// must have already confirmed user opt-in — this method does NOT
    /// second-guess consent.
    public func publish(
        _ report: CommunityBattleReport,
        via publisher: VisionNATSPublisher
    ) async throws {
        let data = try encode(report)
        try await publisher.publish(data, subject: CommunityBattleReport.subject)
    }

    // MARK: Anonymisation

    /// 16-hex-char prefix of the SHA-256 of the node id. 64 bits is enough
    /// to deduplicate installations while making lookup infeasible.
    static func anonymise(nodeId: String) -> String {
        let digest = sha256(Array(nodeId.utf8))
        let prefix = digest.prefix(8)
        return prefix.map { String(format: "%02x", $0) }.joined()
    }

    // MARK: Minimal SHA-256 (no Crypto dep)
    //
    // Keep the Vision module self-contained — CryptoKit is available on
    // Darwin but pulling it in here widens the dependency surface for a
    // trivial digest. The implementation below is the textbook FIPS 180-4
    // reference, exercised by the unit tests.

    static func sha256(_ input: [UInt8]) -> [UInt8] {
        let k: [UInt32] = [
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
            0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
            0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
            0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
            0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
            0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
            0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
            0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
        ]
        var h: [UInt32] = [
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
            0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
        ]
        var msg = input
        let bitLen = UInt64(input.count) * 8
        msg.append(0x80)
        while msg.count % 64 != 56 { msg.append(0) }
        for i in (0..<8).reversed() {
            msg.append(UInt8((bitLen >> (UInt64(i) * 8)) & 0xff))
        }

        for chunkStart in stride(from: 0, to: msg.count, by: 64) {
            var w = [UInt32](repeating: 0, count: 64)
            for i in 0..<16 {
                let j = chunkStart + i * 4
                w[i] = (UInt32(msg[j]) << 24) |
                       (UInt32(msg[j + 1]) << 16) |
                       (UInt32(msg[j + 2]) << 8) |
                       UInt32(msg[j + 3])
            }
            for i in 16..<64 {
                let s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3)
                let s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10)
                w[i] = w[i - 16] &+ s0 &+ w[i - 7] &+ s1
            }
            var a = h[0], b = h[1], c = h[2], d = h[3]
            var e = h[4], f = h[5], g = h[6], hh = h[7]
            for i in 0..<64 {
                let S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)
                let ch = (e & f) ^ (~e & g)
                let t1 = hh &+ S1 &+ ch &+ k[i] &+ w[i]
                let S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)
                let mj = (a & b) ^ (a & c) ^ (b & c)
                let t2 = S0 &+ mj
                hh = g; g = f; f = e; e = d &+ t1
                d = c; c = b; b = a; a = t1 &+ t2
            }
            h[0] = h[0] &+ a; h[1] = h[1] &+ b; h[2] = h[2] &+ c; h[3] = h[3] &+ d
            h[4] = h[4] &+ e; h[5] = h[5] &+ f; h[6] = h[6] &+ g; h[7] = h[7] &+ hh
        }

        var out = [UInt8]()
        out.reserveCapacity(32)
        for value in h {
            out.append(UInt8((value >> 24) & 0xff))
            out.append(UInt8((value >> 16) & 0xff))
            out.append(UInt8((value >> 8) & 0xff))
            out.append(UInt8(value & 0xff))
        }
        return out
    }

    private static func rotr(_ x: UInt32, _ n: UInt32) -> UInt32 {
        (x >> n) | (x << (32 - n))
    }
}
