import Foundation
import Testing

@testable import AIKit

// MARK: - FixturePinTests
//
// T-AIL-49, W24 of spec ai-layer-aikit-core-2026-10-31.
//
// TWO RULES, one file, because they are halves of one claim: that every byte a
// test feeds a decoder is either MINTED from the type that reads it, or is
// declared evidence that something outside this repo really sent.
//
//   1. A captured server reply is PINNED to its model both ways — decode,
//      encode, decode, and the two models are equal.
//   2. No other raw JSON literal exists in `Tests/` without a declaration
//      naming its kind and saying why it cannot be minted.
//
// WHY. The operator asked the same question on nine separate payloads in one
// review (AIKit#18 r4500530906, r4500580810, r4500583527, r4500589635,
// r4500592040, r4500592399, r4500592834, r4500593486, r4500599462): *"it's
// normal it doesn't use typed model with mock data for this?"* The answer was
// no, and the cost is specific rather than aesthetic. A hand-written payload
// and the DTO that decodes it are two independent statements of one wire
// shape. When the shape moved, only the DTO was edited: the payload kept its
// old key, `JSONDecoder` ignored it, the field decoded to nil, the optional
// swallowed it, and the test went on passing. A MINTED payload has one
// statement — the DTO — so the same move is a compile error in the fixture.
//
// WHY A CAPTURED REPLY STAYS RAW. It is the only thing in the suite that is
// evidence rather than construction: it is what the operator's LM Studio
// actually sent, including the keys AIKit does not read. Minting it would make
// it a restatement of AIKit's beliefs about the server, which is precisely what
// it exists to check. So it stays verbatim, and the pin below is what keeps it
// honest: if a DTO stops reading a field the capture carries, the round trip
// stops being equal.
//
// WHY A DECLARATION AND NOT A FILE-LEVEL OPT-OUT. A declaration carries a KIND
// and a REASON and a COUNT, so widening the rule is an edit a reviewer sees in
// the diff; a bare file-level exemption is one nobody reads again.
//
// The scanner itself is `NoLocalhostFallbackRatchetTests`'s, reused rather than
// copied: W21's rule and W24's rule disagreeing about what a comment or a
// literal is would be two enforcers of one idea, which is the defect BR-EC-02
// names.

@Suite("FixturePinTests — a captured reply is pinned to its typed model")
struct FixturePinTests {

    // MARK: - T-AIL-49, first half: the pin

    @Test("every captured reply round trips through its typed model", arguments: wireFamilyFixtures)
    func everyCapturedReplyRoundTrips(fixture: WireFamilyFixture) throws {
        // Decode → encode → decode. The second model must equal the first.
        //
        // What equality alone does NOT catch, and why each arm below also
        // asserts VALUES: a DTO that declares a field under the WRONG wire key
        // decodes to nil from the capture, re-encodes as absent, and decodes
        // to nil again — equal, and so invisible to the round trip. The text,
        // the model that answered and both token counts are read from the
        // capture, so a mis-keyed `prompt_tokens` fails here.
        switch fixture.family {
        case .openAIChat:
            let model = try pinned(ChatCompletionResponseDTO.self, fixture.reply)
            #expect(model.choices.first?.message.content == "pong")
            #expect(model.model == capturedModel)
            #expect(model.usage?.promptTokens == fixture.promptTokens)
            #expect(model.usage?.completionTokens == fixture.completionTokens)
            #expect(model.choices.first?.finishReason == "stop")

        case .openAIResponses:
            let model = try pinned(ResponsesResponseDTO.self, fixture.reply)
            #expect(model.text == "pong")
            #expect(model.model == capturedModel)
            #expect(model.status == "completed")
            #expect(model.usage?.inputTokens == fixture.promptTokens)
            #expect(model.usage?.outputTokens == fixture.completionTokens)

        case .anthropicMessages:
            let model = try pinned(AnthropicMessagesResponseDTO.self, fixture.reply)
            #expect(model.text == "pong")
            #expect(model.model == capturedModel)
            #expect(model.stopReason == "end_turn")
            #expect(model.usage?.inputTokens == fixture.promptTokens)
            #expect(model.usage?.outputTokens == fixture.completionTokens)

        case .lmStudioNative:
            let model = try pinned(LMStudioNativeResponseDTO.self, fixture.reply)
            #expect(model.text == "pong")
            #expect(model.modelInstanceId == capturedModel)
            #expect(model.stats?.inputTokens == fixture.promptTokens)
            #expect(model.stats?.totalOutputTokens == fixture.completionTokens)

        case .cliSubprocess, .mlxInProcess, .coreML:
            Issue.record("\(fixture.family.rawValue) is in the HTTP fixture table and has no wire reply")
        }
    }

    @Test("a captured stream's events pin to their models too", arguments: wireFamilyFixtures)
    func everyCapturedStreamEventRoundTrips(fixture: WireFamilyFixture) throws {
        // The same rule for the streams. Every `data:` payload in a captured
        // stream is decoded, re-encoded and decoded again, and the two event
        // models must agree. A chunk DTO that loses a key loses the delta with
        // it, and a lost delta reads as a short answer rather than as a defect.
        var pinnedEvents = 0
        for line in fixture.sseLines {
            guard let payload = Self.ssePayload(of: line), payload != "[DONE]" else { continue }
            switch fixture.family {
            case .openAIChat: _ = try pinned(ChatCompletionChunkDTO.self, payload)
            case .openAIResponses: _ = try pinned(ResponsesStreamEvent.self, payload)
            case .anthropicMessages: _ = try pinned(AnthropicStreamEvent.self, payload)
            case .lmStudioNative: _ = try pinned(LMStudioStreamEvent.self, payload)
            case .cliSubprocess, .mlxInProcess, .coreML:
                Issue.record("\(fixture.family.rawValue) has no stream to pin")
            }
            pinnedEvents += 1
        }
        // The silent fail this closes: a `data:` prefix that stopped matching
        // would pin NOTHING and pass. Every captured stream carries at least
        // the two text deltas plus a terminating event.
        #expect(pinnedEvents >= 3, "only \(pinnedEvents) events pinned for \(fixture.family.rawValue)")
    }

    // MARK: - The minting layer is verified against the production decoder

    @Test("a minted reply reads back as the answer it was minted with", arguments: wireFamilyFixtures)
    func aMintedReplyIsReadByTheRealCodec(fixture: WireFamilyFixture) throws {
        // BR-EC-02: the generator and the enforcer share one code path, and
        // that agreement is ASSERTED rather than assumed because both were
        // "written from the same spec".
        //
        // Without this test, a `mintReply` that wrote a key no codec reads
        // would leave every minted-fixture test asserting against a payload
        // the production path cannot parse — and they would still fail in the
        // same way for every family, which reads like one bug in the codecs
        // rather than one bug in the fixtures.
        let context = try makeFixtureContext(family: fixture.family, route: fixture.route)
        let minted = try mintReply(family: fixture.family)

        let response = try fixture.codec.response(from: Data(minted.utf8), context: context)

        #expect(response.content == fakeReplyText)
        #expect(response.model == fakeReplyModel)
        #expect(response.tokensUsed.prompt == fakeReplyPromptTokens)
        #expect(response.tokensUsed.completion == fakeReplyCompletionTokens)
    }

    @Test("a minted error envelope is read by the real error decoder", arguments: wireFamilyFixtures)
    func aMintedErrorEnvelopeIsReadByTheRealDecoder(fixture: WireFamilyFixture) throws {
        // `FixtureWireError` is the one minting helper with no production
        // counterpart — AIKit reads an error body and never writes one — so it
        // is proved the only way that settles it: the production decoder reads
        // what it writes, in both forms the servers send.
        let context = try makeFixtureContext(family: fixture.family, route: fixture.route)

        for body in [try mintErrorReply("minted refusal"), try mintBareErrorReply("minted refusal")] {
            let error = #expect(throws: AIKitError.self) {
                _ = try fixture.codec.response(from: Data(body.utf8), context: context)
            }
            #expect(error?.errorDescription?.contains("minted refusal") == true)
        }
    }

    @Test("a minted empty reply is the typed empty-reply error, not a decode failure", arguments: wireFamilyFixtures)
    func aMintedEmptyReplyIsTyped(fixture: WireFamilyFixture) throws {
        // The distinction that matters to a fallback chain: a 2xx with no text
        // is `.emptyReply` (the server answered), where a payload the codec
        // cannot read at all is `.replyUndecodable`. A minted empty reply that
        // came out malformed would land in the second and the difference would
        // go untested.
        let context = try makeFixtureContext(family: fixture.family, route: fixture.route)
        let empty = try mintEmptyReply(family: fixture.family)

        #expect(throws: AIKitError.emptyReply(target: context.target)) {
            _ = try fixture.codec.response(from: Data(empty.utf8), context: context)
        }
    }

    @Test("a minted reply is not mistakeable for a captured one")
    func mintedAndCapturedAreToldApart() throws {
        // The fixtures answer different words on purpose. When an assertion
        // fails, which KIND of fixture it was reading is the first thing worth
        // knowing, and "pong" vs "minted pong" says it without a stack trace.
        #expect(fakeReplyText != "pong")
        #expect(fakeReplyModel != capturedModel)
        for fixture in wireFamilyFixtures {
            #expect(!fixture.reply.contains(fakeReplyText))
            #expect(try mintReply(family: fixture.family).contains(fakeReplyText))
        }
    }

    @Test(
        "a local family has no reply to mint, and says so",
        arguments: [
            AIProviderFamily.cliSubprocess, .mlxInProcess, .coreML,
        ])
    func aLocalFamilyCannotBeMinted(family: AIProviderFamily) {
        // Minting is a WIRE concern. A local family has no reply envelope, so
        // asking for one is a typed refusal rather than an empty string that
        // would reach a decoder and fail there instead.
        #expect(throws: AIKitError.self) { _ = try mintReply(family: family) }
        #expect(throws: AIKitError.self) { _ = try mintEmptyReply(family: family) }
    }

    // MARK: - Pinning

    /// Decodes `json`, encodes the result, decodes that, and asserts the two
    /// models are equal — returning the first.
    ///
    /// `Equatable` is the whole requirement on `T`: the pin is that a model
    /// survives its own wire form. The DTOs gained `Encodable` in W24 for this
    /// and for minting; AIKit itself never encodes a reply.
    private func pinned<T: Codable & Equatable>(_ type: T.Type, _ json: String) throws -> T {
        let decoder = JSONDecoder()
        let first = try decoder.decode(T.self, from: Data(json.utf8))
        let reEncoded = try JSONEncoder().encode(first)
        let second = try decoder.decode(T.self, from: reEncoded)
        #expect(
            first == second,
            """
            \(T.self) did not survive its own wire form. Re-encoded: \
            \(String(bytes: reEncoded, encoding: .utf8) ?? "<not utf8>")
            """
        )
        return first
    }

    /// The payload of an SSE line, or nil for a label, a comment or a blank.
    static func ssePayload(of line: String) -> String? {
        guard line.hasPrefix("data:") else { return nil }
        let payload = line.dropFirst("data:".count).trimmingCharacters(in: .whitespaces)
        return payload.isEmpty ? nil : payload
    }
}

// MARK: - The raw-literal ratchet

@Suite("FixturePinTests — no raw JSON payload outside what is declared")
struct FixtureLiteralRatchetTests {

    // MARK: - T-AIL-49, second half

    @Test("every raw JSON literal in Tests is declared evidence")
    func noUndeclaredJSONLiteral() throws {
        var offenders: [String] = []
        var counted: [String: Int] = [:]

        for file in try NoLocalhostFallbackRatchetTests.swiftFiles(
            under: NoLocalhostFallbackRatchetTests.testsRoot
        ) {
            let path = NoLocalhostFallbackRatchetTests.relativeToRepo(file)
            let literals = Self.jsonLiterals(in: try Self.code(of: file))
            guard !literals.isEmpty else { continue }
            counted[path] = literals.count

            guard DeclaredEvidence.byPath[path] != nil else {
                offenders.append(
                    "\(path) — \(literals.count) raw JSON literal(s), first at line \(literals[0].line)"
                )
                continue
            }
        }

        #expect(
            offenders.isEmpty,
            """
            W24: a payload a decoder reads is MINTED from the model that reads \
            it (`mintReply`, `mintEmptyReply`, `mintJSONText` and the rest in \
            WireFixtures.swift), not hand-written. If this literal really is \
            evidence — a captured server reply, a payload from an older shape \
            of a type, a deliberately malformed one, an opaque JSON value a \
            type carries as a String, or another tool's protocol — declare it \
            in `DeclaredEvidence.files` with its kind and reason. \
            Undeclared: \(offenders.joined(separator: " | "))
            """
        )

        // The counts, each direction. Growing one needs a deliberate edit a
        // reviewer sees; shrinking one without lowering it leaves a baseline
        // that no longer describes the tree.
        for declared in DeclaredEvidence.files {
            let found = counted[declared.path] ?? 0
            #expect(
                found == declared.count,
                """
                \(declared.path) declares \(declared.count) raw JSON \
                literal(s) and holds \(found). \
                \(found > declared.count
                    ? "A new raw literal was added — mint it instead, or raise this number and say why."
                    : "One was removed — lower this number so the baseline still describes the tree.")
                """
            )
        }
    }

    // MARK: - The detector is verified, not assumed (BR-EC-02)

    @Test("the JSON detector recognises a payload, and leaves ordinary text alone")
    func theJSONDetectorStillWorks() {
        // A ratchet that has stopped matching passes forever. These are the
        // shapes W24 removed, and the shapes it must not flag.
        let payloads = [
            #"{"id":"x","choices":[]}"#,
            #"{ "kind": "audio", "source": { } }"#,
            #"[{"modelId":"a"}]"#,
            "{}",
            "[]",
        ]
        for payload in payloads {
            #expect(
                !Self.jsonLiterals(in: "let a = \"\"\"\n\(payload)\n\"\"\"").isEmpty,
                "the JSON rule no longer recognises: \(payload)"
            )
        }

        let notPayloads = [
            #"let a = "hello world""#,
            #"let a = "/v1/chat/completions""#,
            #"let a = "data: [DONE]""#,
            // A brace that is not JSON: `JSONFragment` must refuse this, and a
            // rule that demanded a typed model for it would be demanding an
            // encoder for something no encoder can produce.
            #"let a = "{not json""#,
        ]
        for line in notPayloads {
            #expect(
                Self.jsonLiterals(in: line).isEmpty,
                "the JSON rule flags something it should not: \(line)"
            )
        }
    }

    @Test("the detector reads raw and interpolated literals, which is how a payload hides")
    func theDetectorReadsEveryLiteralSpelling() {
        // Three spellings, each of which used to escape a naive scan:
        //   * a raw literal, where nothing is escaped;
        //   * an escaped literal, where `\"` must become `"` before a JSON
        //     parser will take it;
        //   * an INTERPOLATED payload, which carries `\(` — not a legal JSON
        //     escape — so a scanner that did not substitute it would hand
        //     `JSONSerialization` something invalid, get nil, and let a
        //     hand-written payload through.
        #expect(!Self.jsonLiterals(in: ##"let a = #"{"id":"x"}"#"##).isEmpty)
        #expect(!Self.jsonLiterals(in: #"let a = "{\"id\":\"x\"}""#).isEmpty)
        #expect(!Self.jsonLiterals(in: #"let a = "{\"model\":\"\(name)\"}""#).isEmpty)

        // And a comment is not code: a documented payload must not be counted
        // as a use of one, or the explanation would have to leave the repo.
        let commented = NoLocalhostFallbackRatchetTests.codeOnly(
            "// the old fixture was \"\"\"{\"id\":\"x\"}\"\"\"\nlet a = 1")
        #expect(Self.jsonLiterals(in: commented).isEmpty)
    }

    @Test("every declared file exists, so a rename cannot silently widen the rule")
    func everyDeclaredFileExists() throws {
        let present = Set(
            try NoLocalhostFallbackRatchetTests.swiftFiles(
                under: NoLocalhostFallbackRatchetTests.testsRoot
            ).map(NoLocalhostFallbackRatchetTests.relativeToRepo)
        )
        for path in DeclaredEvidence.files.map(\.path) {
            #expect(present.contains(path), "declared but absent: \(path) — a stale exemption")
        }
    }

    @Test("every declared kind is used, so the taxonomy describes the tree")
    func everyKindIsUsed() {
        // A kind nobody claims is a category that was argued for and never
        // needed. Either a file uses it or it should not be in the enum.
        let used = Set(DeclaredEvidence.files.map(\.kind))
        for kind in DeclaredEvidence.Kind.allCases {
            #expect(used.contains(kind), "no file declares \(kind.rawValue) — drop the case or use it")
        }
    }

    @Test("the scan actually read the Tests tree")
    func theScanFoundFiles() throws {
        // The silent fail this closes: a wrong root makes the rule pass on an
        // empty file list, and the scope reports green having inspected
        // nothing. The floor is well under the real count at W24 so a deleted
        // file never turns this red for the wrong reason.
        let files = try NoLocalhostFallbackRatchetTests.swiftFiles(
            under: NoLocalhostFallbackRatchetTests.testsRoot)
        #expect(files.count >= 40, "only \(files.count) Swift files found under Tests/")
        #expect(files.contains { $0.lastPathComponent == "WireFixtures.swift" })
    }

    // MARK: - The JSON rule

    /// `file` with every comment replaced by spaces, so a rule reads code only
    /// and a documented pattern is not counted as a use of it.
    static func code(of file: URL) throws -> String {
        NoLocalhostFallbackRatchetTests.codeOnly(try String(contentsOf: file, encoding: .utf8))
    }

    /// Every string literal in `code` whose content is a JSON object or array.
    ///
    /// "Parses as JSON" rather than "looks like JSON". A brace is not a
    /// payload, and `"{not json"` — the fixture for a decoder's refusal, which
    /// is NOT valid JSON — must not be counted, or the rule would demand a
    /// typed model for something no encoder can emit.
    static func jsonLiterals(in code: String) -> [NoLocalhostFallbackRatchetTests.Hit] {
        NoLocalhostFallbackRatchetTests.stringLiterals(in: code).filter { hit in
            let text = hit.text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard text.hasPrefix("{") || text.hasPrefix("["),
                let data = text.data(using: .utf8),
                let value = try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed])
            else { return false }
            return value is [String: Any] || value is [Any]
        }
    }
}

// MARK: - DeclaredEvidence

/// The raw JSON literals `Tests/` is allowed to hold, and why each one cannot
/// be minted.
///
/// Five kinds, and the reason they are a closed set: a payload stays raw only
/// when the CURRENT typed model cannot produce it. Anything else is a payload
/// today's model can mint, and must.
enum DeclaredEvidence {

    enum Kind: String, CaseIterable {
        /// Bytes a real server sent, kept verbatim and pinned by
        /// `FixturePinTests.everyCapturedReplyRoundTrips`.
        case capturedReply
        /// Bytes an older shape of the type wrote, kept so a stored payload
        /// still decodes. Minting it with today's encoder would write today's
        /// shape and test nothing.
        case legacyPayload
        /// Bytes that are deliberately wrong, so a decoder's refusal can be
        /// asserted. A typed encoder cannot produce them.
        case malformedPayload
        /// A JSON *value* that a typed model carries as a `String` by design —
        /// a tool's JSON Schema, a tool call's `arguments`, an accepted call's
        /// empty body. AIKit does not model these on purpose, so there is
        /// nothing to mint them from.
        case opaqueValue
        /// Another tool's wire or file format, which AIKit reads with a
        /// scanner or a `JSONSerialization` walk rather than with a `Codable`
        /// DTO. There is no typed model, and inventing one is its own wave.
        case foreignProtocol
    }

    struct File {
        let path: String
        let kind: Kind
        let count: Int
        let reason: String
    }

    /// Counts measured at the W24 commit by the detector above, not estimated.
    static let files: [File] = [
        File(
            path: "Tests/AIKitTests/Support/WireFixtures.swift",
            kind: .capturedReply,
            count: 9,
            reason: """
                Four complete replies captured from the operator's LM Studio, one per HTTP \
                family, kept with the fields AIKit does not read and pinned both ways by \
                FixturePinTests. The other five are the suite's opaque JSON VALUES, spelled once \
                here so they stop being spelled in six suites: two tool schemas, two forms of \
                tool-call arguments, and the empty object an accepted call answers with.
                """
        ),
        File(
            path: "Tests/AIKitTests/Wire/FixturePinTests.swift",
            kind: .malformedPayload,
            count: 5,
            reason: """
                This file, for the same reason the sibling ratchet quotes the expressions W21 \
                deleted: its detector is fed the five shapes it must recognise, so a rule that \
                has rotted goes red here instead of the gate going quietly green (BR-EC-02). \
                None of the five is a payload a decoder in this repo reads.
                """
        ),
        File(
            path: "Tests/AIKitTests/AIRequestInputTests.swift",
            kind: .legacyPayload,
            count: 4,
            reason: """
                Three payloads the PRE-W2 encoder wrote — no `inputs` on a request, no \
                `providerID` on a response — and one media source carrying neither a url nor \
                data. Today's encoder cannot write any of them: it always writes the new keys, \
                and `AIMediaSource` cannot be built sourceless. Minting these would delete the \
                stored-payload guarantee (T-AIL-03) instead of proving it.
                """
        ),
        File(
            path: "Tests/AIKitTests/MLXVideoPresetTests.swift",
            kind: .legacyPayload,
            count: 1,
            reason: """
                A `VideoGenerationOptions` as it was stored before W25 added `preset`. The key is \
                simply absent, which today's encoder cannot reproduce.
                """
        ),
        File(
            path: "Tests/AIKitMLXTests/MLXInProcessProviderTests.swift",
            kind: .legacyPayload,
            count: 1,
            reason: """
                A `ModelDescriptor` written by LocalModelStore before W7 added the device fields. \
                The synthesized decoder would have thrown `keyNotFound` on every row of such a \
                registry, which is the regression this payload exists to pin.
                """
        ),
        File(
            path: "Tests/AIKitTests/Wire/WireCodecTests.swift",
            kind: .malformedPayload,
            count: 2,
            reason: """
                An `error` whose value is a NUMBER, and a reply belonging to no family's shape. \
                Both are what they are because no typed encoder in this repo can produce them: \
                they prove a 2xx that cannot be read is reported as such rather than reading as \
                a success.
                """
        ),
        File(
            path: "Tests/AIKitTests/Wire/WireCodecTests+Usage.swift",
            kind: .malformedPayload,
            count: 1,
            reason: """
                An error envelope that is present, non-null and undecodable. `FixtureWireError` \
                writes a well-formed envelope by construction, so the only way to state "an \
                envelope the decoder cannot read" is to write one by hand.
                """
        ),
        File(
            path: "Tests/AIKitTests/Providers/CursorTextParserTests.swift",
            kind: .opaqueValue,
            count: 1,
            reason: """
                The tool INPUT blob inside Cursor's plain-text output. The parser carries it \
                through as a `String` — it is the tool's own argument object, not Cursor's \
                protocol — so this is the assertion that it arrived unchanged.
                """
        ),
        File(
            path: "Tests/AIKitTests/Providers/ClaudeCodeStreamJSONParserTests.swift",
            kind: .foreignProtocol,
            count: 9,
            reason: """
                Claude Code's own `--output-format stream-json` frames. `ClaudeCodeStreamJSONParser` \
                reads them with a `JSONSerialization` walk and a private envelope that decodes \
                only `type`, so there is no `Codable` DTO to mint from. Giving this protocol one \
                is its own wave, and W27 is about to make the CLI parsers internal.
                """
        ),
        File(
            path: "Tests/AIKitTests/AgentSessionIndexTests.swift",
            kind: .foreignProtocol,
            count: 4,
            reason: """
                Claude Code's `.jsonl` transcript lines. The function under test — \
                `ClaudeSessionIndex.extractStringField(json:key:)` — is a hand-rolled string \
                scanner over those lines, so its INPUT is raw JSON text by definition: two of \
                these four exist to prove it reads standard escapes. Minting is not just \
                unavailable here, it would remove the subject.
                """
        ),
        File(
            path: "Tests/AIKitTests/ModelStoreTests.swift",
            kind: .foreignProtocol,
            count: 2,
            reason: """
                The body of a Hugging Face `config.json`, written into a sandbox so the store \
                can find and HASH a real file on disk. AIKit never decodes it — the bytes matter, \
                not the shape — and a DTO for transformers' config format is not AIKit's to own.
                """
        ),
        File(
            path: "Tests/AIKitMLXTests/MLXHubDownloaderTests.swift",
            kind: .foreignProtocol,
            count: 1,
            reason: "The same `config.json` body, for the same reason, in the MLX target."
        ),
    ]

    static let byPath: [String: File] = Dictionary(uniqueKeysWithValues: files.map { ($0.path, $0) })
}
