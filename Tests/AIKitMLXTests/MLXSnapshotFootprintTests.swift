import Foundation
import Testing

@testable import AIKit
@testable import AIKitMLX

@Suite("MLXSnapshotFootprint")
struct MLXSnapshotFootprintTests {
    private func snapshot(weightsBytes: Int, config: [String: Any]) throws -> URL {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try Data(count: weightsBytes).write(to: dir.appendingPathComponent("model.safetensors"))
        try JSONSerialization.data(withJSONObject: config).write(to: dir.appendingPathComponent("config.json"))
        return dir
    }

    private let ipad8 = DeviceCapability(deviceClass: .iPadM2, totalRAMGB: 8, availableRAMGB: 4.8)
    private let cfg: [String: Any] = [
        "num_hidden_layers": 28, "num_attention_heads": 24, "num_key_value_heads": 8, "hidden_size": 3072,
    ]

    @Test("A snapshot that fits yields a context cap")
    func fitsYieldsCap() throws {
        let dir = try snapshot(weightsBytes: 1_900_000, config: cfg)  // small stand-in for the weights
        let cap = try MLXSnapshotFootprint(directory: dir).contextTokenLimit(on: ipad8, model: "m")
        #expect(cap == DeviceCapability.maximumContextTokens)
    }

    @Test("A snapshot with no room for the minimum context is refused by name")
    func noRoomRefuses() throws {
        let dir = try snapshot(weightsBytes: 4_500_000, config: cfg)
        let tight = DeviceCapability(deviceClass: .iPadM2, totalRAMGB: 8, availableRAMGB: 0.85)
        #expect(throws: MLXProviderError.self) {
            try MLXSnapshotFootprint(directory: dir).contextTokenLimit(on: tight, model: "m")
        }
    }
}
