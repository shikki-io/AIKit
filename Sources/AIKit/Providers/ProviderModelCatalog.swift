import Foundation

// MARK: - ProviderModelCatalog
//
// SSoT for provider model identity.
//
// **The rule** — every canonical model id lives in exactly one place:
// this file. Aliases (`opus`, `sonnet`, `haiku`, `fable`, `claude-opus`,
// `claude-sonnet`, `claude-haiku`, `claude-fable`) also resolve here, in
// exactly one place — a config stage that names `model = "claude-opus"`
// gets `canonical(for: .opus5)` back, and nothing else needs to interpret
// the alias.

// MARK: - Historical Anthropic ids (frozen — kept for cost telemetry)

/// Historical Anthropic canonical ids shipped through in the past. A priced
/// model row lasts forever, so their raw strings still appear once in the
/// tree — here, and here only. Rebuild-only: never mutate a value, only
/// append. Excluded from the current-family alias table on purpose: aliases
/// always point at the CURRENT family.
public enum HistoricalAnthropicId {
    public static let opus46   = "claude-opus-4-6"
    public static let opus48   = "claude-opus-4-8"
    public static let sonnet46 = "claude-sonnet-4-6"
    public static let haiku45Short = "claude-haiku-4-5"
    /// Every historical id in one list — cost telemetry consumers walk this
    /// collection instead of hand-listing.
    public static let all: [String] = [opus46, opus48, sonnet46, haiku45Short]
}

// MARK: - Model family (the canonical ids)

/// The Anthropic Claude family targeted by default.
///
/// The right-hand string is the ONLY place a `claude-*` literal appears in
/// this module. Historical rows stay for telemetry lookups; they are never
/// deleted (a priced model row lasts forever).
public enum AnthropicFamily: String, CaseIterable, Sendable, Codable, Hashable {
    /// Frontier tier. Alias `opus` / `claude-opus`.
    case opus5   = "claude-opus-5"
    /// Balanced tier. Alias `sonnet` / `claude-sonnet`.
    case sonnet5 = "claude-sonnet-5"
    /// High-throughput tier. Alias `haiku` / `claude-haiku`.
    case haiku45 = "claude-haiku-4-5-20251001"
    /// Above-frontier tier (Claude 5.1 family). Alias `fable` / `claude-fable`.
    case fable51 = "claude-fable-5-1"
}

// MARK: - Model capability tiers (cost-router matching)

/// Model-id prefixes a cost router matches against, per capability tier.
/// Cross-provider on purpose (Mistral, Anthropic, OpenAI): a router asks
/// "is this model code-capable / frontier?", and the answer lives with the
/// rest of model identity instead of inline in the router.
public enum ProviderModelCapability {
    /// A model id containing one of these (lowercased) qualifies for code tasks.
    public static let codeCapablePrefixes: [String] = ["devstral", "claude-", "gpt-4", "o3"]
    /// A model id containing one of these (lowercased) qualifies for high-stakes tasks.
    public static let frontierPrefixes: [String] = ["claude-opus", "gpt-4o", "o3"]
}

// MARK: - ProviderModelCatalog

/// Typed catalog of provider model identity.
///
/// Load order (SSoT-first):
///   1. `ProviderModelCatalog.default` — the ONLY hardcoded ids in the tree.
///   2. Operator overrides merge in through ``merging(operatorAliases:)``.
///      The typed wire shape for those overrides is owned by the caller;
///      this catalog is only the runtime lookup.
public struct ProviderModelCatalog: Sendable, Equatable {

    // MARK: State

    /// Canonical id per family member.
    public let canonical: [AnthropicFamily: String]

    /// Alias → canonical id. Populated with both the short (`opus`) and the
    /// operator-friendly `claude-<tier>` names used in provider config today.
    public let aliases: [String: String]

    // MARK: Init

    public init(
        canonical: [AnthropicFamily: String],
        aliases: [String: String]
    ) {
        self.canonical = canonical
        self.aliases = aliases
    }

    // MARK: Default catalog

    /// The shipped catalog. Every canonical id comes from ``AnthropicFamily``
    /// — no literal is written anywhere in this module outside that enum's
    /// raw values above.
    public static let `default`: ProviderModelCatalog = {
        let canonical: [AnthropicFamily: String] = {
            var map: [AnthropicFamily: String] = [:]
            for family in AnthropicFamily.allCases { map[family] = family.rawValue }
            return map
        }()
        let aliases: [String: String] = [
            "opus": AnthropicFamily.opus5.rawValue,
            "claude-opus": AnthropicFamily.opus5.rawValue,
            "sonnet": AnthropicFamily.sonnet5.rawValue,
            "claude-sonnet": AnthropicFamily.sonnet5.rawValue,
            "haiku": AnthropicFamily.haiku45.rawValue,
            "claude-haiku": AnthropicFamily.haiku45.rawValue,
            "fable": AnthropicFamily.fable51.rawValue,
            "claude-fable": AnthropicFamily.fable51.rawValue,
        ]
        return ProviderModelCatalog(canonical: canonical, aliases: aliases)
    }()

    // MARK: Lookup

    /// The canonical model id for a family member.
    public func canonical(for family: AnthropicFamily) -> String {
        canonical[family] ?? family.rawValue
    }

    /// Resolve an alias to a canonical id.
    ///
    /// - A registered alias (`opus`, `claude-opus`, …) → its canonical id.
    /// - A raw canonical id already in the catalog (`claude-opus-5`) → itself.
    /// - Anything else → `nil` (callers decide whether that is a passthrough
    ///   or a failure; ``ModelAliasMap`` still handles per-provider passthrough).
    public func resolve(_ alias: String) -> String? {
        if let hit = aliases[alias] { return hit }
        if canonical.values.contains(alias) { return alias }
        return nil
    }

    /// Every canonical id in the catalog. Order is `AnthropicFamily.allCases`.
    public var allCanonicalIds: [String] {
        AnthropicFamily.allCases.map { canonical(for: $0) }
    }

    // MARK: Merging (operator overrides)

    /// Merge `operatorAliases` on top of the catalog's aliases (right wins).
    /// Canonical ids stay untouched — an operator can only rebind an alias,
    /// never invent a canonical id that has no `AnthropicFamily` behind it.
    public func merging(operatorAliases: [String: String]) -> ProviderModelCatalog {
        var merged = aliases
        for (key, value) in operatorAliases { merged[key] = value }
        return ProviderModelCatalog(canonical: canonical, aliases: merged)
    }
}
