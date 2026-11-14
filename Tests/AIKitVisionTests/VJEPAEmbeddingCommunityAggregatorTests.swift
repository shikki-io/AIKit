// MARK: - VJEPAEmbeddingCommunityAggregatorTests
// Spec §5 Phase 3 deliverable 8 — anonymised community aggregation.

import Foundation
import Testing
@testable import AIKitVision

// MARK: - Recording publisher double

private actor RecordingPublisher: VisionNATSPublisher {
    private(set) var published: [(Data, String)] = []

    func publish(_ data: Data, subject: String) async throws {
        published.append((data, subject))
    }

    var isConnected: Bool { true }

    func snapshot() -> [(Data, String)] { published }
}

@Suite("VJEPAEmbeddingCommunityAggregator")
struct VJEPAEmbeddingCommunityAggregatorTests {

    // MARK: Helpers

    private static func sampleBatch() -> [BattleMetrics] {
        (0..<6).map { i in
            let category = i < 3 ? "ui-identification" : "error-detection"
            let a = ProviderMeasurement(
                providerName: "vjepa",
                relevance: 1.0,
                latencyMs: 100,
                memoryMB: 180,
                tokenCostUSD: 0,
                energyWh: 0.002,
                answer: "ok"
            )
            let b = ProviderMeasurement(
                providerName: "llm",
                relevance: 0.5,
                latencyMs: 300,
                memoryMB: 8,
                tokenCostUSD: 0.003,
                energyWh: 0,
                answer: "ok-ish"
            )
            return BattleMetrics(
                battleId: "b-\(i)",
                taskId: "t-\(i)",
                taskCategory: category,
                providerA: a,
                providerB: b,
                winner: .providerA,
                scoreA: 1.0,
                scoreB: 0.485,
                policy: .default,
                overheadMs: 1,
                timestamp: Date(timeIntervalSince1970: 1_700_000_000 + Double(i))
            )
        }
    }

    // MARK: Anonymisation (SHA-256 prefix)

    @Test("anonymise produces a stable 16-hex SHA-256 prefix")
    func anonymiseStable() {
        let a = CommunityBattleAggregator.anonymise(nodeId: "node-xyz")
        let b = CommunityBattleAggregator.anonymise(nodeId: "node-xyz")
        #expect(a == b)
        #expect(a.count == 16)
        let other = CommunityBattleAggregator.anonymise(nodeId: "node-abc")
        #expect(a != other)
    }

    @Test("SHA-256 of empty input matches FIPS 180-4 test vector")
    func sha256EmptyVector() {
        let digest = CommunityBattleAggregator.sha256([])
        let hex = digest.map { String(format: "%02x", $0) }.joined()
        #expect(hex == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
    }

    @Test("SHA-256 of 'abc' matches FIPS 180-4 test vector")
    func sha256AbcVector() {
        let digest = CommunityBattleAggregator.sha256(Array("abc".utf8))
        let hex = digest.map { String(format: "%02x", $0) }.joined()
        #expect(hex == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
    }

    // MARK: Aggregation shape

    @Test("aggregation rolls per-category win rates + totals")
    func aggregationRollsUp() {
        let agg = CommunityBattleAggregator()
        let report = agg.aggregate(
            battles: Self.sampleBatch(),
            nodeId: "node-xyz",
            now: Date(timeIntervalSince1970: 1_700_000_100)
        )
        #expect(report.totalBattles == 6)
        #expect(report.categories.count == 2)
        // Energy total = 6 battles * 0.002 Wh
        #expect(abs(report.totalEnergyWh - 0.012) < 1e-9)
        // Token total = 6 battles * 0.003 USD
        #expect(abs(report.totalTokenUSD - 0.018) < 1e-9)
        #expect(report.anonymousNodeId.count == 16)
        #expect(report.corpusVersion == VisionBenchmark.corpusVersion)
    }

    // MARK: Payload strips raw embeddings + per-task answers

    @Test("community payload does not leak per-battle metrics or answers")
    func payloadStripsRawBattles() throws {
        let agg = CommunityBattleAggregator()
        let report = agg.aggregate(battles: Self.sampleBatch(), nodeId: "node-xyz")
        let data = try agg.encode(report)
        let json = String(data: data, encoding: .utf8) ?? ""
        // No battle_id or taskId leakage — raw BattleMetrics records must not
        // appear in the community payload.
        #expect(!json.contains("b-0"))
        #expect(!json.contains("t-0"))
        #expect(!json.contains("ok-ish"))
        #expect(!json.contains("node-xyz"))
        // Schema we DO publish.
        #expect(json.contains("\"corpusVersion\""))
        #expect(json.contains("\"anonymousNodeId\""))
        #expect(json.contains("\"totalEnergyWh\""))
        #expect(json.contains("\"totalTokenUSD\""))
    }

    // MARK: Publish via NATS publisher double

    @Test("publish emits bytes on the shikki.community.vision_battles subject")
    func publishEmitsOnSubject() async throws {
        let agg = CommunityBattleAggregator()
        let report = agg.aggregate(battles: Self.sampleBatch(), nodeId: "node-xyz")
        let publisher = RecordingPublisher()
        try await agg.publish(report, via: publisher)
        let published = await publisher.snapshot()
        #expect(published.count == 1)
        #expect(published[0].1 == "shikki.community.vision_battles")
        #expect(published[0].1 == CommunityBattleReport.subject)
        let text = String(data: published[0].0, encoding: .utf8) ?? ""
        #expect(text.contains("\"totalBattles\""))
    }

    // MARK: Determinism

    @Test("two aggregations over the same batch encode to identical bytes")
    func encodeDeterministic() throws {
        let agg = CommunityBattleAggregator()
        let now = Date(timeIntervalSince1970: 1_700_000_100)
        let a = agg.aggregate(battles: Self.sampleBatch(), nodeId: "node-xyz", now: now)
        let b = agg.aggregate(battles: Self.sampleBatch(), nodeId: "node-xyz", now: now)
        let da = try agg.encode(a)
        let db = try agg.encode(b)
        #expect(da == db)
    }
}
