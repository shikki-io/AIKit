import Foundation
import Testing
@testable import AIKit
@testable import NetKit

@Suite("HuggingFaceClient")
struct HuggingFaceClientTests {

    // MARK: - HFModelInfo decoding

    private func sampleJSON() -> Data {
        """
        [
            {
                "modelId": "nvidia/nemotron-3-nano-4b",
                "author": "nvidia",
                "tags": ["gguf", "text-generation"],
                "downloads": 50000,
                "likes": 1200,
                "lastModified": "2025-11-20T10:30:00.000Z",
                "pipeline_tag": "text-generation",
                "library_name": "transformers"
            },
            {
                "modelId": "meta-llama/Llama-3-8B",
                "author": "meta-llama",
                "tags": ["safetensors", "text-generation"],
                "downloads": 200000,
                "likes": 5000,
                "lastModified": "2025-10-15T08:00:00.000Z",
                "pipeline_tag": "text-generation",
                "library_name": "transformers"
            }
        ]
        """.data(using: .utf8)!
    }

    private func singleModelJSON() -> Data {
        """
        {
            "modelId": "nvidia/nemotron-3-nano-4b",
            "author": "nvidia",
            "tags": ["gguf", "text-generation"],
            "downloads": 50000,
            "likes": 1200,
            "lastModified": "2025-11-20T10:30:00.000Z",
            "pipeline_tag": "text-generation",
            "library_name": "transformers"
        }
        """.data(using: .utf8)!
    }

    @Test("Search parses response from mock network")
    func searchParsesResponse() async throws {
        let mock = MockNetworkService()
        mock.resultData = sampleJSON()

        let client = HuggingFaceClient(networkService: mock)
        let results = try await client.search(query: "nemotron")

        #expect(results.count == 2)
        #expect(results[0].modelId == "nvidia/nemotron-3-nano-4b")
        #expect(results[0].author == "nvidia")
        #expect(results[0].downloads == 50000)
        #expect(results[0].pipelineTag == "text-generation")
        #expect(results[0].library == "transformers")
        #expect(results[1].modelId == "meta-llama/Llama-3-8B")
    }

    @Test("GetModel parses single model from mock network")
    func getModelParsesResponse() async throws {
        let mock = MockNetworkService()
        mock.resultData = singleModelJSON()

        let client = HuggingFaceClient(networkService: mock)
        let model = try await client.getModel(id: "nvidia/nemotron-3-nano-4b")

        #expect(model.modelId == "nvidia/nemotron-3-nano-4b")
        #expect(model.author == "nvidia")
        #expect(model.tags.contains("gguf"))
        #expect(model.likes == 1200)
    }

    @Test("toDescriptor converts HFModelInfo to ModelDescriptor")
    func toDescriptorConversion() {
        let info = HFModelInfo(
            modelId: "nvidia/nemotron-3-nano-4b",
            author: "nvidia",
            tags: ["gguf", "text-generation"],
            downloads: 50000,
            likes: 1200,
            pipelineTag: "text-generation",
            library: "transformers"
        )

        let descriptor = info.toDescriptor(format: .gguf)

        #expect(descriptor.id.provider == "huggingface")
        #expect(descriptor.id.modelId == "nvidia/nemotron-3-nano-4b")
        #expect(descriptor.name == "nemotron-3-nano-4b")
        #expect(descriptor.author == "nvidia")
        #expect(descriptor.format == .gguf)
        #expect(descriptor.domain == .llm)
        #expect(descriptor.capabilities.contains(.textGeneration))
        #expect(descriptor.huggingFaceId == "nvidia/nemotron-3-nano-4b")
        #expect(descriptor.downloadCount == 50000)
        #expect(!descriptor.isLocal)
    }

    @Test("toDescriptor maps embedding pipeline correctly")
    func toDescriptorEmbedding() {
        let info = HFModelInfo(
            modelId: "sentence-transformers/all-MiniLM-L6-v2",
            pipelineTag: "feature-extraction"
        )

        let descriptor = info.toDescriptor(format: .safetensors)

        #expect(descriptor.domain == .embedding)
        #expect(descriptor.capabilities.contains(.embedding))
        #expect(descriptor.format == .safetensors)
    }

    @Test("Download URL format is correct")
    func downloadURLFormat() {
        let client = HuggingFaceClient(networkService: MockNetworkService())
        let url = client.downloadURL(repoId: "nvidia/nemotron-3-nano-4b", filename: "model.gguf")

        #expect(url.absoluteString == "https://huggingface.co/nvidia/nemotron-3-nano-4b/resolve/main/model.gguf")
    }

    @Test("Search with filter builds request")
    func searchWithFilter() async throws {
        let mock = MockNetworkService()
        mock.resultData = "[]".data(using: .utf8)!

        let client = HuggingFaceClient(networkService: mock)
        let filter = HFSearchFilter(
            tags: ["gguf"],
            pipelineTag: "text-generation",
            sort: "downloads",
            direction: "-1",
            limit: 10
        )
        let results = try await client.search(query: "llama", filter: filter)
        #expect(results.isEmpty)

        // Verify the request was captured
        #expect(mock.capturedRequests.count == 1)
        let url = mock.capturedRequests[0].url!
        #expect(url.host == "huggingface.co")
        #expect(url.path.contains("/api/models"))
    }
}
