import Foundation
import Testing
@testable import AIKit

// MARK: - ProviderModelCatalog tests (SSoT contract)

@Suite("ProviderModelCatalog — canonical ids and alias resolution")
struct ProviderModelCatalogTests {

    // MARK: - Canonical ids

    @Test("Every AnthropicFamily case round-trips through the catalog")
    func canonicalRoundTrip() {
        let cat = ProviderModelCatalog.default
        for family in AnthropicFamily.allCases {
            #expect(cat.canonical(for: family) == family.rawValue)
        }
    }

    @Test("Catalog exposes exactly the AnthropicFamily ids")
    func allCanonicalIdsMatchFamily() {
        let cat = ProviderModelCatalog.default
        let familyIds = AnthropicFamily.allCases.map { $0.rawValue }
        #expect(cat.allCanonicalIds == familyIds)
    }

    // MARK: - Alias resolution (single-hop, deterministic)

    @Test("Short aliases (opus, sonnet, haiku, fable) resolve to the current family")
    func shortAliasesResolveToCurrentFamily() {
        let cat = ProviderModelCatalog.default
        #expect(cat.resolve("opus")   == cat.canonical(for: .opus5))
        #expect(cat.resolve("sonnet") == cat.canonical(for: .sonnet5))
        #expect(cat.resolve("haiku")  == cat.canonical(for: .haiku45))
        #expect(cat.resolve("fable")  == cat.canonical(for: .fable51))
    }

    @Test("`claude-<tier>` aliases (as used in provider config) resolve to the current family")
    func providerConfigAliasesResolveToCurrentFamily() {
        let cat = ProviderModelCatalog.default
        #expect(cat.resolve("claude-opus")   == cat.canonical(for: .opus5))
        #expect(cat.resolve("claude-sonnet") == cat.canonical(for: .sonnet5))
        #expect(cat.resolve("claude-haiku")  == cat.canonical(for: .haiku45))
        #expect(cat.resolve("claude-fable")  == cat.canonical(for: .fable51))
    }

    @Test("A raw canonical id resolves to itself")
    func canonicalIdIsIdentity() {
        let cat = ProviderModelCatalog.default
        for family in AnthropicFamily.allCases {
            #expect(cat.resolve(family.rawValue) == family.rawValue)
        }
    }

    @Test("Unknown alias returns nil (no silent passthrough)")
    func unknownAliasReturnsNil() {
        let cat = ProviderModelCatalog.default
        #expect(cat.resolve("gpt-something") == nil)
        #expect(cat.resolve("mistral-medium-3.1") == nil)
        #expect(cat.resolve("") == nil)
    }

    // MARK: - Operator merging

    @Test("merging(operatorAliases:) lets operators rebind an alias")
    func operatorOverrideRebindsAlias() {
        let base = ProviderModelCatalog.default
        let overridden = base.merging(operatorAliases: [
            "opus": AnthropicFamily.sonnet5.rawValue,   // pin opus to sonnet
        ])
        #expect(overridden.resolve("opus") == AnthropicFamily.sonnet5.rawValue)
        #expect(overridden.canonical(for: .opus5) == AnthropicFamily.opus5.rawValue)
    }

    @Test("Every current-family canonical id is code-capable")
    func currentFamilyIsCodeCapable() {
        for id in ProviderModelCatalog.default.allCanonicalIds {
            #expect(ProviderModelCapability.codeCapablePrefixes.contains { id.contains($0) },
                    "\(id) should match a code-capable prefix")
        }
    }

    @Test("The frontier tier is a subset of the code-capable tier")
    func frontierImpliesCodeCapable() {
        for prefix in ProviderModelCapability.frontierPrefixes {
            #expect(ProviderModelCapability.codeCapablePrefixes.contains { prefix.contains($0) },
                    "frontier prefix \(prefix) should also be code-capable")
        }
    }
}
