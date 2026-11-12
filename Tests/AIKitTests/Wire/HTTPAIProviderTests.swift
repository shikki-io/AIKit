import Foundation
import NetKit
import Testing

@testable import AIKit

// MARK: - HTTPAIProviderTests
//
// T-AIL-41 and T-AIL-42. BR-AIL-15: one entry point. BR-AIL-19: a non-2xx is an
// error on BOTH paths, carrying the status, the server's text and the target,
// and its fallback eligibility comes from its case — never from a prefix of a
// message.
//
// The defect this replaces is specific: `OpenAIProvider.stream` handed its
// `URLRequest` to a `URLSessionDataDelegate` that was only ever given BYTES, so
// a 401, a 429 or a 500 arrived as a body the SSE parser found no `data:` lines
// in, and the stream finished as an EMPTY SUCCESS. A caller saw no text and no
// error, and a fallback chain never fired.

/// The statuses the panel named, with whether a chain may fall back on them.
struct StatusCase: Sendable, CustomTestStringConvertible {
    let status: Int
    let body: String
    let fallbackEligible: Bool

    var testDescription: String { "\(status)" }
}

private let statusCases: [StatusCase] = [
    StatusCase(status: 401, body: #"{"error":{"message":"Invalid API key"}}"#, fallbackEligible: false),
    StatusCase(status: 429, body: #"{"error":{"message":"Rate limit reached"}}"#, fallbackEligible: true),
    StatusCase(status: 500, body: #"{"error":{"message":"Internal error"}}"#, fallbackEligible: true),
]

@Suite("HTTPAIProviderTests — a non-success is an error on both paths")
struct HTTPAIProviderStatusTests {

    @Test("complete throws .http with the status, the body and the target", arguments: statusCases)
    func completeThrowsOnNonSuccess(statusCase: StatusCase) async throws {
        let network = MockHTTPTransport()
        network.answer(json: statusCase.body, status: statusCase.status)
        let provider = try makeProvider(network: network)

        let error = await #expect(throws: AIKitError.self) {
            _ = try await provider.complete(request: makeFixtureRequest())
        }
        guard case .http(let status, let body, let target) = try #require(error) else {
            Issue.record("expected .http, got \(String(describing: error))")
            return
        }
        #expect(status == statusCase.status)
        #expect(body == statusCase.body)
        #expect(target.entryID == "lmstudio")
        #expect(target.route == "/v1/chat/completions")
        #expect(target.address.description == "http://127.0.0.1:1234")
    }

    @Test("stream throws the same .http, before any chunk", arguments: statusCases)
    func streamThrowsOnNonSuccess(statusCase: StatusCase) async throws {
        let network = MockHTTPTransport()
        network.answer(json: statusCase.body, status: statusCase.status)
        // Lines are configured and must NOT be reached: the status decides
        // first.
        network.lines = capturedChatSSELines
        let provider = try makeProvider(network: network)

        let error = await #expect(throws: AIKitError.self) {
            _ = try await provider.stream(request: makeFixtureRequest())
        }
        guard case .http(let status, let body, _) = try #require(error) else {
            Issue.record("expected .http, got \(String(describing: error))")
            return
        }
        #expect(status == statusCase.status)
        #expect(body == statusCase.body)
    }

    @Test("a streamed non-success is never an empty success", arguments: statusCases)
    func aStreamedNonSuccessIsNeverAnEmptySuccess(statusCase: StatusCase) async throws {
        let network = MockHTTPTransport()
        network.answer(json: statusCase.body, status: statusCase.status)
        let provider = try makeProvider(network: network)

        var chunks: [AIChunk] = []
        var thrown: (any Error)?
        do {
            for try await chunk in try await provider.stream(request: makeFixtureRequest()) {
                chunks.append(chunk)
            }
        } catch {
            thrown = error
        }
        #expect(chunks.isEmpty)
        #expect(thrown != nil, "a \(statusCase.status) finished the stream with no error")
    }

    @Test("eligibility comes from the status, not from a message", arguments: statusCases)
    func eligibilityComesFromTheStatus(statusCase: StatusCase) throws {
        let error = AIKitError.http(
            status: statusCase.status,
            body: statusCase.body,
            target: try makeFixtureContext(family: .openAIChat, route: "/v1/chat/completions").target
        )
        #expect(FallbackEligibility.allows(error) == statusCase.fallbackEligible)
    }

    @Test("the same status is the same verdict whatever the body says", arguments: statusCases)
    func theBodyDoesNotChangeTheVerdict(statusCase: StatusCase) throws {
        // The failure mode this pins: `LMStudioProvider` used to decide
        // fallback with `message.hasPrefix(…)`, so one wire event got two
        // verdicts depending on how its sentence had been composed.
        let target = try makeFixtureContext(family: .openAIChat, route: "/v1/chat/completions").target
        for body in ["", "anything at all", "LM Studio answered with no content"] {
            let error = AIKitError.http(status: statusCase.status, body: body, target: target)
            #expect(FallbackEligibility.allows(error) == statusCase.fallbackEligible, "body: \(body)")
        }
    }

    @Test("an error with no body still names the status and where it came from")
    func anEmptyErrorBodyStillNamesTheStatus() async throws {
        let network = MockHTTPTransport()
        network.status = 503
        network.body = Data()
        let provider = try makeProvider(network: network)

        let error = await #expect(throws: AIKitError.self) {
            _ = try await provider.complete(request: makeFixtureRequest())
        }
        let described = try #require(error?.errorDescription)
        #expect(described.contains("503"))
        #expect(described.contains("127.0.0.1:1234"))
        #expect(described.contains("the server sent no body"))
    }

    @Test("a 2xx outside 200 is still a success")
    func anAcceptedStatusIsASuccess() async throws {
        let network = MockHTTPTransport()
        network.answer(json: capturedChatCompletionsReply, status: 201)
        let provider = try makeProvider(network: network)
        #expect(try await provider.complete(request: makeFixtureRequest()).content == "pong")
    }

    // MARK: - The bounded buffer

    @Test("a newline-free body does not grow the stream buffer past the cap")
    func aNewlineFreeBodyIsCapped() async throws {
        // The reader this replaces appended every chunk to an unbounded `Data`
        // and re-decoded the WHOLE buffer per chunk, so a 200 carrying a
        // newline-free error body grew without limit.
        let cap = 512
        let network = MockHTTPTransport()
        network.status = 200
        network.lines = [String(repeating: "x", count: cap + 1)]
        let provider = try makeProvider(network: network, streamLineByteCap: cap)

        var thrown: (any Error)?
        do {
            for try await _ in try await provider.stream(request: makeFixtureRequest()) {}
        } catch {
            thrown = error
        }
        #expect(network.observedLineByteCap == cap)
        guard case .streamLineTooLong(let reportedCap, let target) = thrown as? AIKitError else {
            Issue.record("expected .streamLineTooLong, got \(String(describing: thrown))")
            return
        }
        #expect(reportedCap == cap)
        #expect(target.entryID == "lmstudio")
    }

    @Test("a line exactly at the cap is still read")
    func aLineAtTheCapIsRead() async throws {
        let line = "data: [DONE]"
        let network = MockHTTPTransport()
        network.status = 200
        network.lines = [line]
        let provider = try makeProvider(network: network, streamLineByteCap: line.utf8.count)

        var chunks: [AIChunk] = []
        for try await chunk in try await provider.stream(request: makeFixtureRequest()) {
            chunks.append(chunk)
        }
        #expect(chunks.map(\.isComplete) == [true])
    }

    @Test("the default cap is stated once and is what an entry carries")
    func theDefaultCapIsTheEntrysCap() async throws {
        let network = MockHTTPTransport()
        network.status = 200
        network.lines = ["data: [DONE]"]
        let provider = try makeProvider(network: network)
        for try await _ in try await provider.stream(request: makeFixtureRequest()) {}
        #expect(network.observedLineByteCap == AIProviderEntry.defaultStreamLineByteCap)
    }

    @Test("a stream that fails mid-flight reports it rather than finishing clean")
    func aMidStreamFailureIsReported() async throws {
        let network = MockHTTPTransport()
        network.status = 200
        network.lines = Array(capturedChatSSELines.prefix(2))
        network.lineStreamError = URLError(.networkConnectionLost)
        let provider = try makeProvider(network: network)

        var chunks: [AIChunk] = []
        var thrown: (any Error)?
        do {
            for try await chunk in try await provider.stream(request: makeFixtureRequest()) {
                chunks.append(chunk)
            }
        } catch {
            thrown = error
        }
        #expect(chunks.map(\.delta) == ["p"])
        #expect(thrown is URLError)
    }

    // MARK: - Helpers

    private func makeProvider(
        network: MockHTTPTransport,
        streamLineByteCap: Int = AIProviderEntry.defaultStreamLineByteCap
    ) throws -> HTTPAIProvider {
        HTTPAIProvider(
            entry: try makeFixtureEntry(family: .openAIChat, streamLineByteCap: streamLineByteCap),
            codec: OpenAIChatCodec(),
            network: network
        )
    }
}

// MARK: - T-AIL-42

@Suite("HTTPAIProviderTests — every family has a production caller")
struct HTTPAIProviderFamilyTests {

    @Test("an entry alone builds a working provider, for every HTTP family", arguments: wireFamilyFixtures)
    func anEntryAloneBuildsAWorkingProvider(fixture: WireFamilyFixture) async throws {
        // This is the init W22's repository calls, once per `[[providers]]`
        // row: the family is resolved to its codec HERE, and nothing above
        // this line knows a wire standard (BR-AIL-15). Three of the four
        // families had no production caller at all before this wave.
        let network = MockHTTPTransport()
        network.answer(json: fixture.reply)
        let provider = try HTTPAIProvider(
            entry: try makeFixtureEntry(family: fixture.family),
            network: network
        )

        let response = try await provider.complete(request: makeFixtureRequest())
        #expect(response.content == "pong")
        #expect(response.providerID == "lmstudio")
        #expect(provider.codec.family == fixture.family)
        #expect(network.sentRequest?.url?.path == fixture.route)
    }

    @Test("an entry alone streams, for every HTTP family", arguments: wireFamilyFixtures)
    func anEntryAloneStreams(fixture: WireFamilyFixture) async throws {
        let network = MockHTTPTransport()
        network.status = 200
        network.lines = fixture.sseLines
        let provider = try HTTPAIProvider(
            entry: try makeFixtureEntry(family: fixture.family),
            network: network
        )

        var deltas: [String] = []
        var complete = false
        for try await chunk in try await provider.stream(request: makeFixtureRequest()) {
            deltas.append(chunk.delta)
            complete = complete || chunk.isComplete
        }
        #expect(deltas.joined() == "pong")
        #expect(complete)
    }

    @Test("every HTTP family is reachable from a kind string a settings file carries")
    func everyFamilyIsReachableFromAKind() throws {
        for family in AIProviderFamily.httpFamilies {
            #expect(AIProviderFamily(kind: family.rawValue) == family)
        }
        // The spellings already written in operators' `providers.toml`.
        #expect(AIProviderFamily(kind: "openai-compatible") == .openAIChat)
        #expect(AIProviderFamily(kind: "openai") == .openAIChat)
        #expect(AIProviderFamily(kind: "lmstudio") == .openAIChat)
        #expect(AIProviderFamily(kind: "ollama") == .openAIChat)
        #expect(AIProviderFamily(kind: "mistral") == .openAIChat)
        #expect(AIProviderFamily(kind: "anthropic") == .anthropicMessages)
        #expect(AIProviderFamily(kind: " OpenAI-Compatible ") == .openAIChat)
        #expect(AIProviderFamily(kind: "no-such-family") == nil)
    }

    @Test("an entry naming a local family refuses to build an HTTP provider", arguments: [
        AIProviderFamily.cliSubprocess, .mlxInProcess, .coreML,
    ])
    func aLocalFamilyRefusesAnHTTPProvider(family: AIProviderFamily) throws {
        let entry = try makeFixtureEntry(family: family)
        #expect(throws: AIKitError.self) {
            _ = try HTTPAIProvider(entry: entry, network: MockHTTPTransport())
        }
    }

    @Test("one server backs two families, and the entry decides which route each takes")
    func oneServerBacksTwoFamilies() async throws {
        // The whole argument for declaring a family separately from a server:
        // the operator's LM Studio answers all four on one port, measured.
        let chatNetwork = MockHTTPTransport()
        chatNetwork.answer(json: capturedChatCompletionsReply)
        let chat = try HTTPAIProvider(
            entry: try makeFixtureEntry(family: .openAIChat, id: "lms-chat"),
            network: chatNetwork
        )

        let nativeNetwork = MockHTTPTransport()
        nativeNetwork.answer(json: capturedLMStudioNativeReply)
        let native = try HTTPAIProvider(
            entry: try makeFixtureEntry(family: .lmStudioNative, id: "lms-native"),
            network: nativeNetwork
        )

        _ = try await chat.complete(request: makeFixtureRequest())
        _ = try await native.complete(request: makeFixtureRequest())

        #expect(chatNetwork.sentRequest?.url?.absoluteString == "http://127.0.0.1:1234/v1/chat/completions")
        #expect(nativeNetwork.sentRequest?.url?.absoluteString == "http://127.0.0.1:1234/api/v1/chat")
    }

    @Test("a provider names itself from its entry, and its answers carry that id")
    func aProviderNamesItselfFromItsEntry() async throws {
        let network = MockHTTPTransport()
        network.answer(json: capturedChatCompletionsReply)
        let provider = try HTTPAIProvider(
            entry: try makeFixtureEntry(family: .openAIChat, id: "devstral-local"),
            network: network
        )
        #expect(provider.id == "devstral-local")
        #expect(provider.displayName == "devstral-local (OpenAI chat completions)")
        #expect(try await provider.complete(request: makeFixtureRequest()).providerID == "devstral-local")
        #expect(await provider.status == .ready)
    }

    @Test("the model asked for is the entry's, and the model reported is the reply's")
    func theEntrysModelIsWhatIsAsked() async throws {
        let network = MockHTTPTransport()
        network.answer(json: capturedChatCompletionsReply)
        let provider = try HTTPAIProvider(
            entry: try makeFixtureEntry(family: .openAIChat, model: "asked-for-this"),
            network: network
        )
        let response = try await provider.complete(request: makeFixtureRequest())
        let body = try fixtureBodyObject(try #require(network.sentRequest))
        #expect(body["model"] as? String == "asked-for-this")
        // Measured: this surface ignores an unknown model and answers with
        // whatever is loaded, so the reply's name is the honest one.
        #expect(response.model == capturedModel)
    }

    @Test("an unencodable request body throws and sends nothing", arguments: wireFamilyFixtures)
    func anUnencodableBodyThrowsAndSendsNothing(fixture: WireFamilyFixture) async throws {
        // A tool schema that is not JSON. The body cannot be built, so nothing
        // may be sent — a bodyless request would read as a server fault.
        let network = MockHTTPTransport()
        network.answer(json: fixture.reply)
        let provider = HTTPAIProvider(
            entry: try makeFixtureEntry(family: fixture.family),
            codec: fixture.codec,
            network: network
        )
        var request = makeFixtureRequest()
        request.tools = [AITool(name: "broken", description: "d", parametersSchema: "{not json")]

        await #expect(throws: (any Error).self) {
            _ = try await provider.complete(request: request)
        }
        #expect(network.sentRequests.isEmpty)
    }

    @Test("the entry's timeout rides on the request the transport builds")
    func theEntrysTimeoutRidesOnTheRequest() async throws {
        let network = MockHTTPTransport()
        network.answer(json: capturedChatCompletionsReply)
        network.timeout = 123
        let provider = try HTTPAIProvider(entry: try makeFixtureEntry(family: .openAIChat), network: network)
        _ = try await provider.complete(request: makeFixtureRequest())
        #expect(network.sentRequest?.timeoutInterval == 123)
    }

    @Test("the production transport carries the entry's own timeout")
    func theProductionTransportCarriesTheEntrysTimeout() throws {
        let entry = try AIProviderEntry(
            id: "slow-local",
            family: .openAIChat,
            address: fixtureServerRoot,
            model: "m",
            requestTimeout: 420
        )
        // 300 rather than URLSession's 60 is the default a cold local model
        // needs; an entry may raise it.
        #expect(AIProviderEntry.defaultRequestTimeout == 300)
        let transport = URLSessionHTTPTransport(timeout: entry.requestTimeout)
        let request = try transport.createRequest(
            endPoint: OpenAIEndPoint.models(address: entry.address, credential: nil)
        )
        #expect(request.timeoutInterval == 420)
    }
}
