import AIKit
import Foundation
import MLXLMCommon

/// What a snapshot on disk will cost in memory, read from the files themselves:
/// the weights are the sum of its `.safetensors`, the KV geometry comes from its
/// `config.json`. A settings row names only a model id, so nothing about its
/// size is known before the snapshot exists; this is where it becomes known,
/// and the last point before the weights are mapped into memory.
struct MLXSnapshotFootprint: Sendable {
    let weightsBytes: Int64
    let geometry: KVCacheGeometry?

    init(directory: URL) {
        let files =
            (try? FileManager.default.contentsOfDirectory(
                at: directory, includingPropertiesForKeys: [.fileSizeKey])) ?? []
        weightsBytes = files.filter { $0.pathExtension == "safetensors" }
            .reduce(Int64(0)) { sum, url in
                sum + Int64((try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
            }
        geometry = (try? Data(contentsOf: directory.appendingPathComponent("config.json")))
            .flatMap { KVCacheGeometry(configJSON: $0) }
    }

    /// The context cap this device allows, or a refusal naming the limit. Nil
    /// cap means the geometry is unknown: the weights alone were checked.
    func contextTokenLimit(on device: DeviceCapability, model: String) throws -> Int? {
        guard let geometry else {
            let floor = DeviceCapability.footprintGB(weightsBytes: weightsBytes, kv: nil, contextTokens: 0)
            if device.availableRAMGB < floor {
                throw MLXProviderError.insufficientRAM(
                    model: model, required: floor, available: device.availableRAMGB)
            }
            return nil
        }
        if let cap = device.contextTokenCap(weightsBytes: weightsBytes, kv: geometry) { return cap }
        throw MLXProviderError.insufficientRAM(
            model: model,
            required: DeviceCapability.footprintGB(
                weightsBytes: weightsBytes, kv: geometry,
                contextTokens: DeviceCapability.minimumContextTokens),
            available: device.availableRAMGB)
    }
}

/// A downloader that refuses a snapshot that cannot fit before MLX maps it.
struct FootprintGatedDownloader: Downloader {
    let inner: any Downloader
    let device: DeviceCapability
    let model: String

    func download(
        id: String, revision: String?, matching patterns: [String], useLatest: Bool,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL {
        let directory = try await inner.download(
            id: id, revision: revision, matching: patterns, useLatest: useLatest,
            progressHandler: progressHandler)
        _ = try MLXSnapshotFootprint(directory: directory).contextTokenLimit(on: device, model: model)
        return directory
    }
}
