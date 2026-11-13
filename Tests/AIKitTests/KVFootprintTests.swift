import Foundation
import Testing

@testable import AIKit

@Suite("KV cache footprint and the context cap")
struct KVFootprintTests {
    /// Minted, never a raw literal: the keys are the ones a Hugging Face config carries.
    static func config(_ fields: [String: Any]) -> Data {
        (try? JSONSerialization.data(withJSONObject: fields)) ?? Data()
    }

    private let llama8B = config(
        ["num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8, "hidden_size": 4096])
    private let llama3B = config(
        ["num_hidden_layers": 28, "num_attention_heads": 24, "num_key_value_heads": 8, "hidden_size": 3072])
    private let eightGBiPad: DeviceCapability = {
        let available = DeviceCapability.systemSafeAvailableGB(
            processHeadroomGB: 8.0, residentGB: 0.1, physicalGB: 8)
        return DeviceCapability(deviceClass: .iPadM2, totalRAMGB: 8, availableRAMGB: available)
    }()

    private func model(weightsBytes: Int64) -> ModelDescriptor {
        ModelDescriptor(
            id: ModelIdentifier(provider: "test", modelId: "m"), name: "M", author: "T",
            description: "fixture", capabilities: .textGeneration, format: .mlx,
            parameters: "x", quantization: "Q4", sizeBytes: weightsBytes,
            architecture: "llama", domain: .llm)
    }

    @Test("KV bytes per token come from config.json, not the model name")
    func geometryFromConfig() throws {
        let g8 = try #require(KVCacheGeometry(configJSON: llama8B))
        let g3 = try #require(KVCacheGeometry(configJSON: llama3B))
        #expect(g8.bytesPerToken() == 128 * 1024)
        #expect(g3.bytesPerToken() == 112 * 1024)
    }

    @Test("A nested text_config and an explicit head_dim are read; a missing field is nil")
    func geometryNestedAndMissing() throws {
        let nested = Self.config(
            ["text_config": ["num_hidden_layers": 2, "num_attention_heads": 4, "head_dim": 64]])
        let g = try #require(KVCacheGeometry(configJSON: nested))
        #expect(g == KVCacheGeometry(layers: 2, kvHeads: 4, headDim: 64))
        #expect(KVCacheGeometry(configJSON: Self.config(["hidden_size": 8])) == nil)
    }

    @Test("8 GB iPad: the 3B fits with the default context")
    func threeBFitsDefaultContext() throws {
        let kv = try #require(KVCacheGeometry(configJSON: llama3B))
        #expect(eightGBiPad.footprintShortfall(for: model(weightsBytes: 1_900_000_000), kv: kv) == nil)
    }

    @Test("8 GB iPad: the 8B does not fit even at a small context")
    func eightBRefusedEvenSmall() throws {
        let kv = try #require(KVCacheGeometry(configJSON: llama8B))
        let eightB = model(weightsBytes: 4_500_000_000)
        #expect(eightGBiPad.footprintShortfall(for: eightB, kv: kv, contextTokens: 512) != nil)
        #expect(eightGBiPad.contextTokenCap(weightsBytes: 4_500_000_000, kv: kv) == nil)
    }

    @Test("The cap is what is left after weights and overhead, within bounds")
    func contextCap() throws {
        let kv = try #require(KVCacheGeometry(configJSON: llama3B))
        let cap = try #require(eightGBiPad.contextTokenCap(weightsBytes: 1_900_000_000, kv: kv))
        #expect(cap >= DeviceCapability.defaultContextBudgetTokens)
        #expect(cap <= DeviceCapability.maximumContextTokens)
        #expect(
            eightGBiPad.footprintShortfall(for: model(weightsBytes: 1_900_000_000), kv: kv, contextTokens: cap) == nil)
    }

    @Test("A 16 GB device caps the 8B at the ceiling")
    func sixteenGBCap() throws {
        let kv = try #require(KVCacheGeometry(configJSON: llama8B))
        let device = DeviceCapability(deviceClass: .iPadM4, totalRAMGB: 16, availableRAMGB: 9.5)
        #expect(device.contextTokenCap(weightsBytes: 4_500_000_000, kv: kv) == DeviceCapability.maximumContextTokens)
    }
}
