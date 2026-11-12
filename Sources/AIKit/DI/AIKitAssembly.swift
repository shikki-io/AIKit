import CoreKit
import Foundation

// MARK: - AIKitAssembly

/// Registers AIKit's providers into a CoreKit container, from settings.
///
/// BR-AIL-20: providers come from settings through a `Repository` and are
/// INJECTED. Nothing below resolves a provider by reading an environment
/// variable, a disk path or a default host — the three reads that used to
/// configure providers (`LMSTUDIO_URL`, `LMSTUDIO_MODEL`, `MLX_LM_BASE_URL`)
/// are gone from Sources, and this type is where their job went.
///
/// What it registers, in both environments:
///
/// | Type | Name | What it is |
/// |---|---|---|
/// | ``AIProviderSettings`` | — | the document, as read |
/// | ``AIProviderRepository`` | — | the rows, keyed and ordered |
/// | ``AIProviderSet`` | — | the built providers AND the refused rows |
/// | `[any AIProvider]` | — | the built providers, in settings order |
/// | `any AIProvider` | each entry's `id` | one provider per row |
/// | ``ProviderRouter`` | — | the ordered chain over those providers |
///
/// The two environments differ ONLY in what a row becomes:
/// `.production` builds the real provider through ``AIProviderFactory``,
/// `.mock` builds a ``MockAIProvider`` carrying the row's id. Same rows, same
/// names, same order, same count — so a test resolves exactly what production
/// resolves and no mock-only wiring can hide a production gap (T-AIL-46).
///
/// ```swift
/// DI.configure(for: .production, assemblies: [
///     AIKitAssembly(
///         settings: try AIProviderSettings.load(from: hostChosenURL),
///         factory: AIProviderFactory(credentials: broker.secret(for:))
///     )
/// ])
/// let providers = DI.require([any AIProvider].self)   // settings order
/// ```
public struct AIKitAssembly: DIAssembly {

    // MARK: - Properties

    /// The document, as the host read it.
    public let settings: AIProviderSettings
    /// How a row becomes a provider in `.production`.
    public let factory: AIProviderFactory
    /// How a row becomes a provider in `.mock`.
    /// Builds the provider an entry gets in the `.mock` environment.
    ///
    /// Injected, never defaulted: the doubles live in `AIKitTestSupport`, out
    /// of the product (AIKit#18 r4500328752), so this module cannot name one.
    /// A `.mock` container built without it reports every entry unavailable,
    /// naming what to add, instead of serving anything (see `providerSet`).
    private let mockProvider: (@Sendable (AIProviderEntry) -> any AIProvider)?
    /// The document's path, for an error that must name it.
    private let source: String

    // MARK: - The one transport factory

    /// How a settings row becomes an HTTP transport in production: `URLSession`
    /// with the row's own timeout.
    ///
    /// ONE place, and internal (W27, AIKit#18 r4500408737). Six initializers
    /// used to carry `transport: (any AIHTTPTransport)? = nil` and build this
    /// for themselves when nothing was passed, which is how a test that forgot
    /// the argument talked to the operator's live LM Studio. Now every one of
    /// them requires a transport, ``AIProviderFactory`` takes this as its
    /// default, and a test hands a stub instead (BR-AIL-12).
    static let urlSessionTransport: @Sendable (AIProviderEntry) -> any AIHTTPTransport = { entry in
        URLSessionHTTPTransport(timeout: entry.requestTimeout)
    }

    // MARK: - Init

    /// - Parameters:
    ///   - settings: the provider document. ALREADY loaded: the host owns the
    ///     path, so this assembly never touches a filesystem.
    ///   - factory: how a row becomes a provider. Inject one carrying a
    ///     credential resolver, and one carrying a stub transport in tests.
    ///   - source: the document's path, named in errors.
    ///   - mockProvider: what a row becomes under `.mock`. The default carries
    ///     the row's id and model, so a mock answer still names who answered
    ///     (BR-AIL-06).
    public init(
        settings: AIProviderSettings,
        factory: AIProviderFactory = AIProviderFactory(),
        source: String = "the provider settings",
        mockProvider: (@Sendable (AIProviderEntry) -> any AIProvider)? = nil
    ) {
        self.settings = settings
        self.factory = factory
        self.source = source
        self.mockProvider = mockProvider
    }

    // MARK: - DIAssembly

    public func assemble(container: Container, environment: DIEnvironment) {
        let settings = self.settings
        let source = self.source

        container.register(AIProviderSettings.self) { _ in settings }
        container.register(AIProviderRepository.self) { _ in
            AIProviderRepository(settings: settings, source: source)
        }

        let set = providerSet(for: environment)
        container.register(AIProviderSet.self) { _ in set }
        container.register([any AIProvider].self) { _ in set.providers }
        container.register(ProviderRouter.self) { _ in ProviderRouter(providers: set.providers) }

        // One registration per row, under the row's own id, so a caller that
        // wants a NAMED provider ("the local one") resolves it by the name the
        // operator wrote rather than by an index into a list.
        for entry in settings.providers {
            let id = entry.id
            container.register((any AIProvider).self, name: id) { resolver in
                guard let provider = try resolver.resolve(AIProviderSet.self).provider(id: id) else {
                    // Reached when THIS row could not be built while others
                    // could. The cause is in `AIProviderSet.unavailable`, so
                    // it is reported rather than reduced to "not registered".
                    throw Self.refusal(of: id, in: try resolver.resolve(AIProviderSet.self), file: source)
                }
                return provider
            }
        }
    }

    // MARK: - Private

    /// The providers for an environment.
    ///
    /// Built ONCE, here, rather than inside each registration closure: the
    /// list and the per-id registrations must agree, and a factory call per
    /// closure would build a different provider instance for each.
    ///
    /// A `.production` row that cannot be built does NOT take the others down
    /// (W25): its refusal is recorded in ``AIProviderSet/unavailable`` and
    /// surfaces when that row is resolved by name. A document where NOTHING
    /// could be built is a configuration error, and the set is then empty with
    /// every cause kept.
    private func providerSet(for environment: DIEnvironment) -> AIProviderSet {
        switch environment {
        case .mock:
            guard let mockProvider else {
                let reason = "the .mock environment needs a mock provider: build the assembly with AIKitAssembly.mock(settings:) from AIKitTestSupport"
                return AIProviderSet(
                    providers: [],
                    unavailable: settings.providers.map {
                        AIProviderAttempt(providerID: $0.id, error: AIKitError.settings(file: source, entry: $0.id, reason: reason))
                    }
                )
            }
            return AIProviderSet(providers: settings.providers.map(mockProvider), unavailable: [])
        case .production:
            var providers: [any AIProvider] = []
            var unavailable: [AIProviderAttempt] = []
            for entry in settings.providers {
                do {
                    providers.append(try factory.make(entry))
                } catch {
                    unavailable.append(AIProviderAttempt(providerID: entry.id, error: error))
                }
            }
            return AIProviderSet(providers: providers, unavailable: unavailable)
        }
    }

    /// Why a named row has no provider.
    private static func refusal(of id: AIProviderEntry.EntryID, in set: AIProviderSet, file: String) -> AIKitError {
        if let attempt = set.unavailable.first(where: { $0.providerID == id }) {
            return .settings(file: file, entry: id, reason: attempt.cause)
        }
        return .settings(file: file, entry: id, reason: "no [[providers]] row carries this id")
    }
}
