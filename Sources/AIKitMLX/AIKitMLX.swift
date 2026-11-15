import AIKit
import Foundation

// MARK: - AIKitMLX

/// What this optional target adds to AIKit: the `mlx-in-process` family.
///
/// W27 of spec ai-layer-aikit-core-2026-10-31, BR-AIL-21 item 3: *"`AIKitMLX`:
/// one public `AIKitMLX.register(into:)` that adds the `mlx-in-process` family
/// to the factory. `MLXInProcessProvider` becomes internal."*
///
/// WHY A REGISTRATION AND NOT A TYPE. Before this wave, a host wanting
/// on-device inference imported `AIKitMLX` and CONSTRUCTED
/// `MLXInProcessProvider` itself — which is how shikki-client ended up naming
/// an engine type instead of writing a settings row (the W29 census). A
/// provider the host builds by hand is a provider outside settings: nothing
/// ranks it, nothing records why it was unavailable, and the operator cannot
/// move it without a recompile. So the only thing this target exports is the
/// sentence *"this build can serve `mlx-in-process` rows"*, said to the one
/// factory that reads settings (BR-AIL-15, BR-AIL-20).
///
/// This is also what keeps on-device inference optional (BR-AIL-14): `AIKit`
/// links no MLX and refuses an `mlx-in-process` row by name, and a host that
/// wants one adds this product and this call.
///
/// ```swift
/// let factory = AIKitMLX.register(into: AIProviderFactory(credentials: broker.secret(for:)),
///                                 store: hostStore)
/// DI.configure(for: .production, assemblies: [AIKitAssembly(settings: settings, factory: factory)])
/// ```
public enum AIKitMLX {

    /// The family this target serves. One value, so a caller never spells the
    /// string.
    public static let family: AIProviderFamily = .mlxInProcess

    /// The same factory, able to build `mlx-in-process` rows.
    ///
    /// - Parameters:
    ///   - factory: the factory settings are read through.
    ///   - store: where weights live and are verified. Injected because a
    ///     `ModelStore` carries the HOST's path layer (W23): a default here
    ///     would be a cache root invented by a kit, which is the duplicate-4 GB
    ///     defect W13 measured. With no store, a row whose model is not already
    ///     on disk is refused by name rather than downloaded somewhere chosen
    ///     here.
    ///   - device: what to measure a load against. Defaults to this device; a
    ///     test describes one.
    ///   - downloadPlanner: the download policy a missing model is fetched
    ///     under (W33, BR-AIL-23). Defaults to the shipped one — a snapshot
    ///     that does not fit is refused, a metered path needs an allow. This is
    ///     a parameter so the operator's `[models.download]` settings reach the
    ///     on-device path: without it the default would be the only policy MLX
    ///     downloads could ever run under.
    ///   - downloadConsent: an answer the operator already gave for this
    ///     download.
    /// - Returns: a factory with one more family, leaving the original alone.
    public static func register(
        into factory: AIProviderFactory,
        store: ModelStore? = nil,
        device: DeviceCapability = .current(),
        downloadPlanner: ModelDownloadPlanner = ModelDownloadPlanner(),
        downloadConsent: ModelDownloadConsent = .none
    ) -> AIProviderFactory {
        factory.registering(family) { entry, _ in
            let descriptor = descriptor(for: entry)
            guard device.canLoad(descriptor) else {
                // The LIMIT, named, rather than a bare refusal: an operator
                // reading this learns whether to pick a smaller model or
                // another device (BR-AIL-22's device filter, S3.7).
                throw AIKitError.engineUnavailable(
                    "'\(entry.id)' asks for \(entry.model) on device class "
                        + "\(device.deviceClass.rawValue) with \(device.availableRAMGB) GB free, "
                        + "and this device cannot load it"
                )
            }
            return MLXInProcessProvider(
                model: descriptor,
                device: device,
                id: entry.id,
                downloader: store.map {
                    MLXHubDownloader(
                        store: $0, planner: downloadPlanner, consent: downloadConsent)
                }
            )
        }
    }

    // MARK: - Private

    /// The model a settings row names.
    ///
    /// A row carries a model IDENTIFIER, not a catalogue entry, so what is
    /// known here is the id and nothing measured: `minRAMGB` stays nil, which
    /// ``DeviceCapability/canLoad(_:)`` reads as "do not refuse" and never as
    /// "needs nothing". A host that has the measured figures resolves the
    /// descriptor through door 3's catalogue and keeps them.
    private static func descriptor(for entry: AIProviderEntry) -> ModelDescriptor {
        ModelDescriptor(
            id: ModelIdentifier(provider: "mlx", modelId: entry.model),
            name: entry.model,
            author: entry.model.split(separator: "/").first.map(String.init) ?? "unknown",
            description: "On-device MLX model \(entry.model), from settings row '\(entry.id)'",
            capabilities: [.textGeneration, .synthesis],
            format: .mlx,
            parameters: "unknown",
            sizeBytes: 0,
            architecture: "unknown",
            domain: .llm,
            huggingFaceId: entry.model
        )
    }
}
