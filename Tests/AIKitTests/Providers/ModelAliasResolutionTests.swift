import Foundation
@testable import AIKit
import Testing

// MARK: - ModelAliasMap Tests
//
// Verifies short-name → canonical-model resolution used by CLI-subprocess providers.

@Suite("ModelAliasMap resolution")
struct ModelAliasResolutionTests {

    // MARK: - Basic resolution

    @Test("Alias resolves to the mapped canonical model name")
    func resolvesKnownAlias() {
        let map = ModelAliasMap([
            "sonnet": "claude-sonnet-4-6",
            "opus": "claude-opus-4-7",
        ])
        #expect(map.resolve("sonnet") == "claude-sonnet-4-6")
        #expect(map.resolve("opus") == "claude-opus-4-7")
    }

    // MARK: - Passthrough for unknown

    @Test("Unknown alias returns the input verbatim (passthrough)")
    func passesThroughUnknownAlias() {
        let map = ModelAliasMap(["sonnet": "claude-sonnet-4-6"])
        #expect(map.resolve("claude-opus-4-7") == "claude-opus-4-7")
        #expect(map.resolve("mystery-model") == "mystery-model")
    }

    // MARK: - Empty map

    @Test("Empty map returns the input verbatim")
    func emptyMapPassesThrough() {
        let map = ModelAliasMap([:])
        #expect(map.resolve("anything") == "anything")
        #expect(map.resolve("claude-sonnet-4-6") == "claude-sonnet-4-6")
    }

    // MARK: - Case sensitivity

    @Test("Resolution is case-sensitive")
    func caseSensitiveResolution() {
        let map = ModelAliasMap(["sonnet": "claude-sonnet-4-6"])
        #expect(map.resolve("Sonnet") == "Sonnet")
        #expect(map.resolve("SONNET") == "SONNET")
    }

    // MARK: - Equality + Sendable

    @Test("Two maps with equal contents compare equal")
    func equality() {
        let a = ModelAliasMap(["sonnet": "claude-sonnet-4-6"])
        let b = ModelAliasMap(["sonnet": "claude-sonnet-4-6"])
        let c = ModelAliasMap(["sonnet": "claude-sonnet-3-5"])
        #expect(a == b)
        #expect(a != c)
    }

    // MARK: - Non-transitive resolution

    @Test("Resolution is single-hop only (aliases pointing to other aliases are NOT chained)")
    func resolutionIsSingleHop() {
        // The right-hand side is treated as the canonical model, even if it
        // happens to also appear as an alias key. This keeps the resolution
        // deterministic and avoids cycle-detection complexity.
        let map = ModelAliasMap([
            "quick": "sonnet",
            "sonnet": "claude-sonnet-4-6",
        ])
        #expect(map.resolve("quick") == "sonnet")
        #expect(map.resolve("sonnet") == "claude-sonnet-4-6")
    }
}
