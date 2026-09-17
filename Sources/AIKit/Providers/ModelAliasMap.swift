import Foundation

// MARK: - ModelAliasMap

/// Value-type registry that maps short human names to canonical model ids.
///
/// Used by CLI-subprocess providers so a config can expose friendly names
/// like `sonnet` or `opus` while the concrete `--model` flag receives the
/// full canonical id (resolved via ``ProviderModelCatalog``).
///
/// Resolution is deliberately **single-hop**: the right-hand side of an
/// entry is treated as the canonical value even if it happens to appear
/// as another alias key. This keeps resolution deterministic (no cycle
/// detection needed) and matches how operators actually reason about
/// their model list.
public struct ModelAliasMap: Sendable, Equatable {
    private let aliases: [String: String]

    /// Create a map from the given `alias → canonical` pairs.
    /// An empty map is a valid passthrough (every input resolves to itself).
    public init(_ aliases: [String: String]) {
        self.aliases = aliases
    }

    /// Resolve `name` through the map.
    /// - Returns: the canonical model id when `name` matches an alias key;
    ///            otherwise `name` itself (passthrough).
    public func resolve(_ name: String) -> String {
        aliases[name] ?? name
    }
}
