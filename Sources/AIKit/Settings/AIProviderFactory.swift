import Foundation

// MARK: - AIProviderSet

/// Every provider a settings document could be turned into, and every row it
/// could not.
///
/// Both halves, because dropping the second is the silent failure: an operator
/// whose `cli-subprocess` row cannot run on this phone must still get their
/// three HTTP providers, AND must be able to find out that the fourth row was
/// refused and why. A plain `[any AIProvider]` can only do the first.
public struct AIProviderSet: Sendable {

    /// The providers that were built, IN SETTINGS ORDER (BR-AIL-11).
    public let providers: [any AIProvider]
    /// One record per row that could not be built: its id and the cause.
    public let unavailable: [AIProviderAttempt]

    public init(providers: [any AIProvider], unavailable: [AIProviderAttempt]) {
        self.providers = providers
        self.unavailable = unavailable
    }

    /// The provider with this entry id, or nil.
    public func provider(id: AIProviderEntry.EntryID) -> (any AIProvider)? {
        providers.first { $0.id == id }
    }
}

// MARK: - AIProviderFactory

/// Turns a settings entry into a provider.
///
/// BR-AIL-15 and BR-AIL-20: the family is read ONCE, here, and what comes back
/// is `any AIProvider`. A local server and a shared one are the SAME TYPE with
/// a different address — there is no local-provider class and nothing decides
/// local-ness from a substring in a host.
///
/// Everything it needs from outside arrives as an injected closure, so the
/// factory itself reads no environment, no disk and no clock:
///
/// - `credentials` resolves an ``AIProviderEntry/apiKeyRef`` to a secret. The
///   default resolves nothing, which is correct for a package that must not
///   know a broker: a host without one gets providers with no credential
///   rather than a provider holding a key AIKit invented.
/// - `transport` supplies the ``AIHTTPTransport``. Tests inject a stub, so no
///   test reaches a real provider (BR-AIL-12).
/// - `builders` carries a family AIKit cannot serve on its own. `AIKitMLX`
///   registers `mlx-in-process` through it (W27), which is why on-device
///   inference stays an optional target (BR-AIL-14) instead of a dependency
///   AIKit links to satisfy a settings case.
public struct AIProviderFactory: Sendable {

    /// Builds a provider for a row of a family this factory does not serve
    /// itself. Takes the entry and the resolved credential.
    ///
    /// Untyped `throws` rather than `throws(AIKitError)`: typed throws in a
    /// STORED function type needs a macOS 15 runtime, and this package's floor
    /// is macOS 14. ``make(_:)`` converts whatever a builder throws into an
    /// `AIKitError`, so the error a caller sees is typed either way.
    public typealias Builder = @Sendable (AIProviderEntry, String?) throws -> any AIProvider

    private let credentials: @Sendable (String) -> String?
    private let transport: @Sendable (AIProviderEntry) -> any AIHTTPTransport
    private let builders: [AIProviderFamily: Builder]

    // MARK: - Init

    /// - Parameters:
    ///   - credentials: resolves an ``AIProviderEntry/apiKeyRef`` to a secret.
    ///   - builders: a family this factory cannot serve on its own.
    ///     `AIKitMLX.register(into:)` adds `mlx-in-process` through it.
    ///
    /// The transport is NOT a parameter here, and that is the point of the
    /// thin surface (W27, BR-AIL-21): ``AIHTTPTransport`` is a provider API,
    /// so it is internal, and the one production transport factory is
    /// ``AIKitAssembly/urlSessionTransport``. A test reaches the init below
    /// through `@testable` to hand a stub (BR-AIL-12).
    public init(
        credentials: @escaping @Sendable (String) -> String? = { _ in nil },
        builders: [AIProviderFamily: Builder] = [:]
    ) {
        self.init(
            credentials: credentials,
            transport: AIKitAssembly.urlSessionTransport,
            builders: builders
        )
    }

    /// The full init, including the transport factory.
    ///
    /// Internal because its transport parameter is: a host picks providers, not
    /// the socket they are spoken over.
    init(
        credentials: @escaping @Sendable (String) -> String? = { _ in nil },
        transport: @escaping @Sendable (AIProviderEntry) -> any AIHTTPTransport,
        builders: [AIProviderFamily: Builder] = [:]
    ) {
        self.credentials = credentials
        self.transport = transport
        self.builders = builders
    }

    /// The same factory with one more family it can serve.
    ///
    /// This is how an optional target joins without AIKit linking it
    /// (BR-AIL-14): `AIKitMLX.register(into:)` is one call to this.
    public func registering(_ family: AIProviderFamily, builder: @escaping Builder) -> AIProviderFactory {
        var builders = self.builders
        builders[family] = builder
        return AIProviderFactory(credentials: credentials, transport: transport, builders: builders)
    }

    // MARK: - Making one

    /// Builds the provider one row describes.
    ///
    /// - Throws: ``AIKitError/engineUnavailable(_:)`` when this host or this
    ///   build cannot serve the row's family, and
    ///   ``AIKitError/settings(file:entry:reason:)`` when the row itself does
    ///   not describe a runnable provider.
    public func make(_ entry: AIProviderEntry) throws(AIKitError) -> any AIProvider {
        let credential = entry.apiKeyRef.flatMap(credentials)

        if let builder = builders[entry.family] {
            do {
                return try builder(entry, credential)
            } catch let error as AIKitError {
                throw error
            } catch {
                throw .engineUnavailable(
                    "the \(entry.family.rawValue) builder refused '\(entry.id)': \(error.localizedDescription)"
                )
            }
        }

        if entry.family.isHTTP {
            // One provider, four families: the codec comes from the family and
            // nothing switches on a wire standard at call time (BR-AIL-15).
            return try HTTPAIProvider(
                entry: entry,
                network: transport(entry),
                credential: credential
            )
        }

        switch entry.family {
        case .cliSubprocess:
            return try makeCLISubprocess(entry)
        case .mlxInProcess:
            throw .engineUnavailable(
                "'\(entry.id)' is an mlx-in-process provider, and this build links no MLX. "
                    + "Depend on the AIKitMLX product and register its family with the factory."
            )
        case .coreML:
            throw .engineUnavailable(
                "'\(entry.id)' is a coreml provider, and no Core ML provider is registered with the factory."
            )
        case .openAIChat, .openAIResponses, .anthropicMessages, .lmStudioNative:
            // Unreachable: `isHTTP` is exactly these four and is handled above.
            throw .engineUnavailable("'\(entry.id)' names the HTTP family \(entry.family.rawValue)")
        }
    }

    /// Builds every row, keeping the refusals.
    ///
    /// - Throws: ``AIKitError/allProvidersFailed(_:)`` when NOTHING could be
    ///   built from a non-empty document — a settings file that yields no
    ///   provider at all is a configuration error, not an empty fleet.
    public func makeAll(_ settings: AIProviderSettings) throws(AIKitError) -> AIProviderSet {
        var providers: [any AIProvider] = []
        var unavailable: [AIProviderAttempt] = []

        for entry in settings.providers {
            do {
                providers.append(try make(entry))
            } catch {
                unavailable.append(AIProviderAttempt(providerID: entry.id, error: error))
            }
        }

        guard !providers.isEmpty || unavailable.isEmpty else {
            throw .allProvidersFailed(unavailable)
        }
        return AIProviderSet(providers: providers, unavailable: unavailable)
    }

    // MARK: - CLI

    /// The `cli-subprocess` provider for one row, presented as an
    /// `AIProvider` through ``AgentProviderAdapter``.
    private func makeCLISubprocess(_ entry: AIProviderEntry) throws(AIKitError) -> any AIProvider {
        guard let options = entry.options.cliSubprocess else {
            throw .settings(
                file: "the provider settings",
                entry: entry.id,
                reason: "a cli-subprocess row names no binary to spawn"
            )
        }

        #if canImport(ShellKit)
            // `CLISubprocessProvider` accounts against the CLOSED routing
            // vocabulary, so the row's id has to be one of its values. Named
            // rather than defaulted: a budget and a concurrency cap attributed
            // to the wrong provider is worse than a refusal the operator can
            // read.
            guard let routingID = AIProviderID(rawValue: entry.id) else {
                throw .settings(
                    file: "the provider settings",
                    entry: entry.id,
                    reason: "a cli-subprocess id must be one of the routing ids, because budget and "
                        + "concurrency are accounted against it. Write one of: "
                        + AIProviderID.allCases.map(\.rawValue).joined(separator: ", ")
                )
            }

            let model = options.modelAliases[entry.model] ?? entry.model
            let rendered: [String]
            do {
                // `{model}` only. `CLISubprocessProvider` pipes the prompt on
                // STDIN, so a `{prompt}` placeholder has no value to take and
                // `ArgsTemplate` refuses it — which is the right answer: a
                // template rendered with an empty prompt would spawn the CLI
                // with an empty argument and the model would answer nothing.
                rendered = try ArgsTemplate.render(
                    template: options.argsTemplate,
                    substitutions: ["model": model]
                )
            } catch ArgsTemplate.RenderError.unresolvedPlaceholder(let key) {
                throw .settings(
                    file: "the provider settings",
                    entry: entry.id,
                    reason: "args_template names '{\(key)}', which a cli-subprocess row cannot fill. "
                        + "Only '{model}' is substituted; the prompt is piped on stdin."
                )
            } catch {
                throw .settings(file: "the provider settings", entry: entry.id, reason: "\(error)")
            }

            return AgentProviderAdapter(
                id: entry.id,
                displayName: "\(entry.id) (\(entry.family.displayName))",
                modelName: model,
                timeout: entry.requestTimeout,
                agent: CLISubprocessProvider(
                    providerID: routingID,
                    arguments: [options.binary] + rendered,
                    workingDirectory: options.workingDirectory,
                    environment: options.environment
                )
            )
        #else
            // W25: a typed refusal naming the platform, never a crash. Settings
            // written on a Mac and synced to a phone keep every other row.
            throw .engineUnavailable(
                "'\(entry.id)' spawns '\(options.binary)', and \(HostPlatform.current.rawValue) "
                    + "has no subprocess API. Route this stage to an HTTP or on-device provider."
            )
        #endif
    }
}
