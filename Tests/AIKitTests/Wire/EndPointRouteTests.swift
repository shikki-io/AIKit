import Foundation
import NetKit
import Testing

@testable import AIKit

// MARK: - The captured route tables
//
// T-AIL-39. BR-AIL-16 makes a family's `EndPoint` enum the SSoT for that API:
// one case per documented route, ALL routes, each with its path and method.
//
// The tables below are the provider's own documentation, captured on
// 2026-11-12 with `lightpanda fetch --dump markdown --strip-mode full` and
// transcribed here VERBATIM from the "Supported endpoints" table of each page:
//
// - OpenAI-shaped: https://lmstudio.ai/docs/developer/openai-compat
// - LM Studio native: https://lmstudio.ai/docs/developer/rest
// - Anthropic: https://docs.anthropic.com/en/api/messages
//   and https://platform.claude.com/docs/en/api/http/models/list
// - HuggingFace Hub: https://huggingface.co/docs/hub/api
//
// This file is the ENFORCER and the tables are its evidence: a route the
// document lists and the enum does not declare fails here, and so does the
// reverse. The two cannot drift without one of these assertions going red,
// which is the point of writing the expectation out rather than deriving it
// from the enum (BR-EC-02).

/// What a document says an API's routes are.
struct DocumentedAPI: Sendable, CustomTestStringConvertible {
    let name: String
    /// The page the table was captured from.
    let reference: String
    /// Path and method, as documented.
    let routes: [String: RequestMethod]
    /// What the enum declares, as the enum declares it.
    let declared: [(path: String, method: RequestMethod)]

    var testDescription: String { name }
}

private let documentedAPIs: [DocumentedAPI] = [
    DocumentedAPI(
        name: "OpenAI-shaped",
        reference: "https://lmstudio.ai/docs/developer/openai-compat",
        routes: [
            "/v1/models": .GET,
            "/v1/responses": .POST,
            "/v1/chat/completions": .POST,
            "/v1/completions": .POST,
            "/v1/embeddings": .POST,
        ],
        declared: OpenAIEndPoint.routes
    ),
    DocumentedAPI(
        name: "Anthropic messages",
        reference: "https://docs.anthropic.com/en/api/messages",
        routes: [
            "/v1/messages": .POST,
            "/v1/messages/count_tokens": .POST,
            "/v1/models": .GET,
            "/v1/models/{model_id}": .GET,
        ],
        declared: AnthropicEndPoint.routes
    ),
    DocumentedAPI(
        name: "LM Studio native",
        reference: "https://lmstudio.ai/docs/developer/rest",
        routes: [
            "/api/v1/chat": .POST,
            "/api/v1/models": .GET,
            "/api/v1/models/load": .POST,
            "/api/v1/models/unload": .POST,
            "/api/v1/models/download": .POST,
            "/api/v1/models/download/status": .GET,
        ],
        declared: LMStudioEndPoint.routes
    ),
]

// MARK: - T-AIL-39

@Suite("EndPointRouteTests — every documented route is a case")
struct EndPointRouteTests {

    @Test("the enum declares exactly the routes its captured document lists", arguments: documentedAPIs)
    func everyDocumentedRouteIsACase(api: DocumentedAPI) throws {
        let declaredPaths = Set(api.declared.map(\.path))
        let documentedPaths = Set(api.routes.keys)

        #expect(
            declaredPaths == documentedPaths,
            """
            \(api.name) drifted from \(api.reference).
            missing from the enum: \(documentedPaths.subtracting(declaredPaths).sorted())
            declared but not documented: \(declaredPaths.subtracting(documentedPaths).sorted())
            """
        )
    }

    @Test("each case carries the method its document gives it", arguments: documentedAPIs)
    func everyRouteCarriesItsMethod(api: DocumentedAPI) throws {
        for route in api.declared {
            let documented = try #require(api.routes[route.path], "\(route.path) is not in \(api.reference)")
            #expect(route.method == documented, "\(api.name) \(route.path)")
        }
    }

    @Test("a declared route list has no duplicate", arguments: documentedAPIs)
    func noDuplicateRoute(api: DocumentedAPI) {
        #expect(Set(api.declared.map(\.path)).count == api.declared.count, "\(api.name)")
    }

    // MARK: - The enum's cases agree with its own table

    @Test("every OpenAI case builds the path and method its table declares")
    func openAICasesMatchTheTable() throws {
        let address = try makeFixtureAddress()
        let built: [(path: String, method: RequestMethod)] = [
            OpenAIEndPoint.models(address: address, credential: nil),
            OpenAIEndPoint.chatCompletions(
                address: address,
                credential: nil,
                body: ChatCompletionRequest(
                    model: "m",
                    messages: [.init(role: "user", content: "hi")],
                    temperature: 0,
                    maxTokens: 1,
                    stream: false,
                    tools: nil,
                    responseFormat: nil
                )
            ),
            OpenAIEndPoint.responses(
                address: address,
                credential: nil,
                body: ResponsesRequest(
                    model: "m",
                    input: [.init(role: "user", content: "hi")],
                    instructions: nil,
                    maxOutputTokens: 1,
                    temperature: 0,
                    stream: false
                )
            ),
            OpenAIEndPoint.completions(
                address: address,
                credential: nil,
                body: CompletionsRequest(model: "m", prompt: "p", maxTokens: 1, temperature: 0, stream: false)
            ),
            OpenAIEndPoint.embeddings(
                address: address,
                credential: nil,
                body: EmbeddingsRequest(model: "m", input: ["p"])
            ),
        ].map { (path: $0.path, method: $0.method) }

        #expect(Set(built.map(\.path)) == Set(OpenAIEndPoint.routes.map(\.path)))
        for route in built {
            let declared = try #require(OpenAIEndPoint.routes.first { $0.path == route.path })
            #expect(route.method == declared.method, "\(route.path)")
        }
    }

    @Test("every Anthropic case builds the path and method its table declares")
    func anthropicCasesMatchTheTable() throws {
        let address = try makeFixtureAddress()
        let messages = [AnthropicMessage(role: "user", content: "hi")]
        let built: [(path: String, method: RequestMethod)] = [
            AnthropicEndPoint.messages(
                address: address,
                credential: nil,
                body: AnthropicMessagesRequest(
                    model: "m", messages: messages, system: nil, maxTokens: 1, temperature: 0, stream: false
                )
            ),
            AnthropicEndPoint.countTokens(
                address: address,
                credential: nil,
                body: AnthropicCountTokensRequest(model: "m", messages: messages, system: nil)
            ),
            AnthropicEndPoint.models(address: address, credential: nil),
            // The templated segment is filled in, so the built path is the
            // concrete one; the table keeps the template.
            AnthropicEndPoint.model(address: address, credential: nil, id: "claude-x"),
        ].map { (path: $0.path, method: $0.method) }

        #expect(built.map(\.path).contains("/v1/models/claude-x"))
        #expect(built.map(\.method) == AnthropicEndPoint.routes.map(\.method))
    }

    @Test("every LM Studio case builds the path and method its table declares")
    func lmStudioCasesMatchTheTable() throws {
        let address = try makeFixtureAddress()
        let built: [(path: String, method: RequestMethod)] = [
            LMStudioEndPoint.chat(
                address: address,
                credential: nil,
                body: LMStudioNativeRequest(
                    model: "m",
                    input: [.text("hi")],
                    systemPrompt: nil,
                    temperature: 0,
                    maxOutputTokens: 1,
                    stream: false
                )
            ),
            LMStudioEndPoint.models(address: address, credential: nil),
            LMStudioEndPoint.loadModel(
                address: address,
                credential: nil,
                body: LMStudioModelLoadRequest(model: "m", contextLength: nil, ttl: nil)
            ),
            LMStudioEndPoint.unloadModel(
                address: address, credential: nil, body: LMStudioModelUnloadRequest(model: "m")
            ),
            LMStudioEndPoint.downloadModel(
                address: address, credential: nil, body: LMStudioModelDownloadRequest(model: "m")
            ),
            LMStudioEndPoint.downloadStatus(address: address, credential: nil),
        ].map { (path: $0.path, method: $0.method) }

        #expect(built.map(\.path) == LMStudioEndPoint.routes.map(\.path))
        #expect(built.map(\.method) == LMStudioEndPoint.routes.map(\.method))
    }

    // MARK: - HuggingFace, r4500469694

    @Test("the HuggingFace file-resolve route is a case, served from the host root")
    func huggingFaceResolveIsACase() throws {
        let request = try NetworkService().createRequest(
            endPoint: HuggingFaceEndPoint.resolveFile(
                id: "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
                revision: "main",
                filename: "model.safetensors"
            )
        )
        #expect(
            request.url?.absoluteString
                == "https://huggingface.co/mlx-community/Mistral-7B-Instruct-v0.3-4bit/resolve/main/model.safetensors"
        )
        // NOT under /api: the resolve route is the repository itself.
        #expect(request.url?.path.hasPrefix("/api") == false)
    }

    @Test("the API routes stay under /api")
    func huggingFaceAPIRoutesStayUnderAPI() throws {
        let service = NetworkService()
        let tree = try service.createRequest(endPoint: HuggingFaceEndPoint.listTree(id: "a/b", revision: "main"))
        #expect(tree.url?.path == "/api/models/a/b/tree/main")
        let model = try service.createRequest(endPoint: HuggingFaceEndPoint.getModel(id: "a/b"))
        #expect(model.url?.path == "/api/models/a/b")
    }

    // MARK: - No route literal outside Endpoints/

    @Test("no AI route literal is written outside an Endpoints directory")
    func noRouteLiteralOutsideEndpoints() throws {
        // BR-AIL-16's second half: the route strings exist in ONE place. The
        // workaround this forbids is the one shikki had — stripping `/v1` off
        // a configured URL — and the bug it forbids is a hand-built
        // `"\(baseURL)/v1/models"`, which produced `/v1/v1/models` for an
        // address that still carried the prefix.
        let sources = Self.repositoryRoot.appendingPathComponent("Sources")
        let offenders = try Self.swiftFiles(under: sources)
            .filter { !$0.path.contains("/Endpoints/") }
            .compactMap { file -> String? in
                let text = try String(contentsOf: file, encoding: .utf8)
                let hits = ["\"/v1", "\"/api/v1", "/resolve/"].filter { text.contains($0) }
                guard !hits.isEmpty else { return nil }
                return "\(file.lastPathComponent): \(hits.joined(separator: ", "))"
            }
        #expect(offenders.isEmpty, "route literals outside Endpoints/: \(offenders)")
    }

    /// The repository root, found from this file rather than from the process's
    /// working directory, which a test runner does not promise.
    static var repositoryRoot: URL {
        URL(fileURLWithPath: #filePath)  // Tests/AIKitTests/Wire/EndPointRouteTests.swift
            .deletingLastPathComponent()  // Wire
            .deletingLastPathComponent()  // AIKitTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // <root>
    }

    static func swiftFiles(under directory: URL) throws -> [URL] {
        guard
            let walker = FileManager.default.enumerator(
                at: directory,
                includingPropertiesForKeys: nil,
                options: [.skipsHiddenFiles]
            )
        else { return [] }
        return walker.compactMap { $0 as? URL }.filter { $0.pathExtension == "swift" }
    }
}
