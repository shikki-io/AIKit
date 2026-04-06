import Foundation
import Testing
@testable import AIKit

@Suite("ModelDescriptor")
struct ModelDescriptorTests {

    static let sampleDescriptor = ModelDescriptor(
        id: ModelIdentifier(provider: "lmstudio", modelId: "nvidia/nemotron-3-nano-4b"),
        name: "Nemotron 3 Nano 4B",
        author: "NVIDIA",
        description: "Small efficient model",
        capabilities: [.textGeneration, .toolUse],
        format: .gguf,
        parameters: "4B",
        quantization: "Q8_0",
        sizeBytes: 4_200_000_000,
        architecture: "llama",
        domain: .llm,
        isLocal: true,
        tags: ["staff-pick"]
    )

    @Test("Codable round-trip preserves all fields")
    func codableRoundTrip() throws {
        let encoder = JSONEncoder()
        let decoder = JSONDecoder()
        let data = try encoder.encode(Self.sampleDescriptor)
        let decoded = try decoder.decode(ModelDescriptor.self, from: data)
        #expect(decoded == Self.sampleDescriptor)
    }

    @Test("ModelIdentifier equality")
    func identifierEquality() {
        let a = ModelIdentifier(provider: "openai", modelId: "gpt-5")
        let b = ModelIdentifier(provider: "openai", modelId: "gpt-5")
        let c = ModelIdentifier(provider: "openai", modelId: "gpt-4")
        #expect(a == b)
        #expect(a != c)
    }

    @Test("ModelIdentifier hash consistency")
    func identifierHash() {
        let a = ModelIdentifier(provider: "ollama", modelId: "llama3")
        let b = ModelIdentifier(provider: "ollama", modelId: "llama3")
        #expect(a.hashValue == b.hashValue)
    }

    @Test("ModelFilter matches by domain")
    func filterByDomain() {
        let filter = ModelFilter(domain: .llm)
        #expect(filter.matches(Self.sampleDescriptor))
        let voiceFilter = ModelFilter(domain: .voice)
        #expect(!voiceFilter.matches(Self.sampleDescriptor))
    }

    @Test("ModelFilter matches by format")
    func filterByFormat() {
        let filter = ModelFilter(format: .gguf)
        #expect(filter.matches(Self.sampleDescriptor))
        let mlxFilter = ModelFilter(format: .mlx)
        #expect(!mlxFilter.matches(Self.sampleDescriptor))
    }

    @Test("ModelFilter matches by isLocal")
    func filterByLocal() {
        let localFilter = ModelFilter(isLocal: true)
        #expect(localFilter.matches(Self.sampleDescriptor))
        let remoteFilter = ModelFilter(isLocal: false)
        #expect(!remoteFilter.matches(Self.sampleDescriptor))
    }

    @Test("ModelFilter matches by capabilities")
    func filterByCapabilities() {
        let filter = ModelFilter(capabilities: .textGeneration)
        #expect(filter.matches(Self.sampleDescriptor))
        let visionFilter = ModelFilter(capabilities: .vision)
        #expect(!visionFilter.matches(Self.sampleDescriptor))
    }

    @Test("ModelFilter matches by maxSizeBytes")
    func filterBySize() {
        let bigEnough = ModelFilter(maxSizeBytes: 5_000_000_000)
        #expect(bigEnough.matches(Self.sampleDescriptor))
        let tooSmall = ModelFilter(maxSizeBytes: 1_000_000_000)
        #expect(!tooSmall.matches(Self.sampleDescriptor))
    }

    @Test("Nil filter matches everything")
    func nilFilterMatchesAll() {
        let filter = ModelFilter()
        #expect(filter.matches(Self.sampleDescriptor))
    }
}
