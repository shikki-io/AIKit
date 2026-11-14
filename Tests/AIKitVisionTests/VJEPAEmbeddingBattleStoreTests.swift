// MARK: - VJEPAEmbeddingBattleStoreTests
// Phase 3 extension — SQLite `vision_battles` table (BR-18 + BR-20).
// Verifies schema matches spec and metrics round-trip end-to-end.

#if canImport(SQLite3)
import Foundation
import Testing
@testable import AIKitVision

@Suite("VJEPAEmbeddingBattleStore")
struct VJEPAEmbeddingBattleStoreTests {

    // MARK: Helpers

    private static func sampleMetrics(
        battleId: String = UUID().uuidString,
        category: String = "error-detection",
        winner: BattleWinner = .providerA
    ) -> BattleMetrics {
        let a = ProviderMeasurement(
            providerName: "vjepa-vitb16",
            relevance: 1.0,
            latencyMs: 120,
            memoryMB: 180,
            tokenCostUSD: 0,
            energyWh: 0.0021,
            answer: "submit"
        )
        let b = ProviderMeasurement(
            providerName: "mock-llm-haiku",
            relevance: 0.5,
            latencyMs: 350,
            memoryMB: 8,
            tokenCostUSD: 0.0031,
            energyWh: 0,
            answer: "submit-button"
        )
        return BattleMetrics(
            battleId: battleId,
            taskId: "task-\(UUID().uuidString.prefix(6))",
            taskCategory: category,
            providerA: a,
            providerB: b,
            winner: winner,
            scoreA: 1.0,
            scoreB: 0.485,
            policy: .default,
            overheadMs: 3,
            timestamp: Date(timeIntervalSince1970: 1_700_000_000)
        )
    }

    // MARK: Schema gate (BR-18)

    @Test("BattleStore creates vision_battles table with BR-18 schema")
    func schemaMatchesBR18() throws {
        let store = try BattleStore(path: ":memory:")
        let cols = try store.dumpSchema()
        let expected = [
            "battle_id", "task_id", "task_category",
            "provider_a", "provider_b", "winner",
            "metrics_json", "timestamp",
        ]
        #expect(cols == expected)
    }

    // MARK: Round-trip

    @Test("insert + fetch round-trips the full BattleMetrics payload")
    func roundTripFetch() throws {
        let store = try BattleStore(path: ":memory:")
        let metrics = Self.sampleMetrics()
        try store.insert(metrics)
        let fetched = try #require(try store.fetch(battleId: metrics.battleId))
        #expect(fetched == metrics)
    }

    @Test("all returns every inserted row in timestamp order")
    func allReturnsInserted() throws {
        let store = try BattleStore(path: ":memory:")
        let m1 = Self.sampleMetrics(battleId: "b-1")
        let m2 = Self.sampleMetrics(battleId: "b-2", category: "ui", winner: .providerB)
        try store.insertAll([m1, m2])
        let rows = try store.all()
        #expect(rows.count == 2)
        #expect(rows.map(\.battleId).sorted() == ["b-1", "b-2"])
    }

    @Test("count reflects row count")
    func countMatchesInserts() throws {
        let store = try BattleStore(path: ":memory:")
        try store.insert(Self.sampleMetrics(battleId: "b-1"))
        try store.insert(Self.sampleMetrics(battleId: "b-2"))
        #expect(try store.count() == 2)
    }

    // MARK: BR-20 shared enclave — embeddings + battles coexist in one file

    @Test("BattleStore shares the enclave file with EmbeddingStore (BR-20)")
    func sharedEnclave() throws {
        let tmp = NSTemporaryDirectory() + "shikki-vision-enclave-\(UUID().uuidString).sqlite"
        defer { try? FileManager.default.removeItem(atPath: tmp) }
        // Open the embedding store first, then the battle store against
        // the same file — both schemas should coexist cleanly.
        let emb = try EmbeddingStore(path: tmp)
        let battles = try BattleStore(path: tmp)
        let m = Self.sampleMetrics()
        try battles.insert(m)
        let fetched = try #require(try battles.fetch(battleId: m.battleId))
        #expect(fetched.battleId == m.battleId)
        // Embedding store is still usable.
        #expect(try emb.count() == 0)
    }

    // MARK: Default path (BR-20 — same file as embeddings)

    @Test("BattleStore.defaultPath aligns with the embedding enclave (BR-20)")
    func defaultPathMatchesEnclave() {
        #expect(BattleStore.defaultPath == EmbeddingStore.defaultPath)
        #expect(BattleStore.defaultPath.hasSuffix(".shikki/vision/embeddings.sqlite"))
    }

    // MARK: Cost telemetry survives the round-trip (BR-22)

    @Test("cost telemetry (token USD + energy Wh) round-trips via SQLite")
    func costTelemetryRoundTrip() throws {
        let store = try BattleStore(path: ":memory:")
        let m = Self.sampleMetrics()
        try store.insert(m)
        let fetched = try #require(try store.fetch(battleId: m.battleId))
        #expect(fetched.providerA.energyWh == 0.0021)
        #expect(fetched.providerB.tokenCostUSD == 0.0031)
    }
}

#endif
