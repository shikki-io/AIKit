// AIKitAssembly+Mock.swift — the `.mock` wiring, kept with the doubles.
//
// `AIKitAssembly` lives in the product and cannot name `MockAIProvider`: the
// doubles moved here so an app that depends on `AIKit` cannot build one
// (AIKit#18 r4500328752). The `.mock` environment therefore receives its
// provider by injection, and this file is the one place that injects it.

import AIKit

extension AIKitAssembly {

    /// The provider an entry becomes in the `.mock` environment: a
    /// `MockAIProvider` carrying the row's own id, so a mocked answer still
    /// names who answered (BR-AIL-06).
    public static let mockProvider: @Sendable (AIProviderEntry) -> any AIProvider = { entry in
        MockAIProvider(
            id: entry.id,
            displayName: "\(entry.id) (mock \(entry.family.displayName))",
            capabilities: .textGeneration
        )
    }

    /// An assembly whose `.mock` environment serves a `MockAIProvider` per
    /// settings row. Production behaviour is unchanged.
    public static func mock(
        settings: AIProviderSettings,
        factory: AIProviderFactory = AIProviderFactory(),
        source: String = "the provider settings"
    ) -> AIKitAssembly {
        AIKitAssembly(settings: settings, factory: factory, source: source, mockProvider: mockProvider)
    }
}
