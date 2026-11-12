import Foundation
import Testing

@testable import AIKit

// MARK: - AIServerAddressTests
//
// BR-AIL-18: an address is validated ONCE, and its only initializer throws.
// W21 of spec ai-layer-aikit-core-2026-10-31 completes the rule with `reach`
// and the ratchet scope (T-AIL-43, T-AIL-44); this suite covers the type this
// wave introduces.
//
// What it replaces, and why the refusals matter: `AIWireServer` defaulted a
// missing scheme to `http` and a missing host to `127.0.0.1`, so SIX of the
// panel's ten inputs were EXECUTED against the operator's own machine instead
// of being refused — one of them carrying a bearer key (AIKit#18 r4500281951,
// r4500300887).

@Suite("AIServerAddress — a root, validated once, with nothing defaulted")
struct AIServerAddressTests {

    @Test("a server root parses into scheme, host and port")
    func aRootParses() throws {
        let address = try AIServerAddress(validating: "http://127.0.0.1:1234")
        #expect(address.scheme == "http")
        #expect(address.host == "127.0.0.1")
        #expect(address.port == 1234)
        #expect(address.description == "http://127.0.0.1:1234")
    }

    @Test("a root with no port keeps none, so the scheme's default applies")
    func aRootWithNoPortKeepsNone() throws {
        let address = try AIServerAddress(validating: "https://api.openai.com")
        #expect(address.port == nil)
        #expect(address.description == "https://api.openai.com")
    }

    @Test("a trailing slash is a root written with a slash, not a path")
    func aTrailingSlashIsNotAPath() throws {
        #expect(try AIServerAddress(validating: "http://127.0.0.1:1234/").description == "http://127.0.0.1:1234")
    }

    @Test("surrounding whitespace is trimmed, not refused")
    func whitespaceIsTrimmed() throws {
        #expect(try AIServerAddress(validating: "  http://host:8080\n").description == "http://host:8080")
    }

    @Test("a scheme is read case-insensitively")
    func aSchemeIsCaseInsensitive() throws {
        #expect(try AIServerAddress(validating: "HTTPS://api.openai.com").scheme == "https")
    }

    @Test(
        "a private-network host is a perfectly good address",
        arguments: [
            "http://192.168.1.42:1234", "http://studio.local:1234", "http://10.0.0.7:11434",
        ])
    func aLANHostIsFine(root: String) throws {
        // The LAN Mac running LM Studio. W21 classifies it as
        // `.privateNetwork`; here it simply must not be refused.
        #expect(try AIServerAddress(validating: root).description == root)
    }

    // MARK: - Refusals

    @Test("a root carrying a path is refused, and the path is named")
    func aPathIsRefused() throws {
        let error = #expect(throws: AIKitError.self) {
            _ = try AIServerAddress(validating: "http://127.0.0.1:1234/v1")
        }
        guard case .invalidServerAddress(let value, let reason) = try #require(error) else {
            Issue.record("expected .invalidServerAddress, got \(String(describing: error))")
            return
        }
        #expect(value == "http://127.0.0.1:1234/v1")
        #expect(reason.contains("/v1"))
    }

    @Test("a proxy path is refused too, rather than silently stripped")
    func aProxyPathIsRefused() throws {
        // The failure the silent-fail table calls "looks like a bad reply":
        // this used to become `/openai/v1/v1/chat/completions`, answered 200
        // with an error body, and read as the model misbehaving.
        let error = #expect(throws: AIKitError.self) {
            _ = try AIServerAddress(validating: "https://proxy.corp/openai/v1")
        }
        #expect(error?.errorDescription?.contains("openai/v1") == true)
    }

    @Test(
        "something that is not an address is refused by name",
        arguments: [
            "", "   ", "not a url at all", "127.0.0.1:1234", "http:///v1", "://host", "ftp://host",
            "http://:1234", "localhost",
        ])
    func aNonAddressIsRefused(input: String) throws {
        let error = #expect(throws: AIKitError.self) {
            _ = try AIServerAddress(validating: input)
        }
        guard case .invalidServerAddress(let value, let reason) = try #require(error) else {
            Issue.record("expected .invalidServerAddress for '\(input)'")
            return
        }
        // The value is named, so the operator can find it in their file.
        #expect(value == input.trimmingCharacters(in: .whitespacesAndNewlines) || value == input)
        #expect(!reason.isEmpty)
    }

    @Test("a bare host:port is refused rather than assumed to be http")
    func aBareHostPortIsRefused() throws {
        // `URLComponents` reads `127.0.0.1:1234` as scheme `127.0.0.1`, which
        // is exactly how the old `?? "http"` defaulting hid the mistake.
        let error = #expect(throws: AIKitError.self) {
            _ = try AIServerAddress(validating: "127.0.0.1:1234")
        }
        #expect(error?.errorDescription?.contains("127.0.0.1:1234") == true)
    }

    @Test("a query or a fragment is refused")
    func aQueryOrFragmentIsRefused() throws {
        #expect(throws: AIKitError.self) { _ = try AIServerAddress(validating: "http://host:1234?key=abc") }
        #expect(throws: AIKitError.self) { _ = try AIServerAddress(validating: "http://host:1234#frag") }
    }

    @Test("credentials embedded in the address are refused")
    func embeddedCredentialsAreRefused() throws {
        let error = #expect(throws: AIKitError.self) {
            _ = try AIServerAddress(validating: "https://user:secret@api.openai.com")
        }
        #expect(error?.errorDescription?.contains("credentials") == true)
    }

    @Test("an invalid address never reaches a provider, and never falls back")
    func anInvalidAddressNeverFallsBack() throws {
        let error = AIKitError.invalidServerAddress(value: "127.0.0.1:1234", reason: "it names no scheme")
        // Not eligible on purpose: falling back would hide a misconfigured
        // entry behind a working one, and the operator would keep an address
        // that never works.
        #expect(!FallbackEligibility.allows(error))
    }

    @Test("an entry built from a bad address refuses the entry, naming the value")
    func anEntryRefusesABadAddress() throws {
        let error = #expect(throws: AIKitError.self) {
            _ = try AIProviderEntry(id: "broken", family: .openAIChat, address: "http:///v1", model: "m")
        }
        #expect(error?.errorDescription?.contains("http:///v1") == true)
    }

    @Test("two addresses written the same way are the same address")
    func equatableByValue() throws {
        let plain = try AIServerAddress(validating: "http://host:1234")
        let withSlash = try AIServerAddress(validating: "http://host:1234/")
        let otherPort = try AIServerAddress(validating: "http://host:1235")
        #expect(plain == withSlash)
        #expect(plain != otherPort)
    }

    @Test("a target names the entry, the address, the route and the host it ran on")
    func aTargetNamesAllThree() throws {
        let target = AIRequestTarget(
            entryID: "devstral-local",
            address: try AIServerAddress(validating: "http://127.0.0.1:1234"),
            route: "/v1/chat/completions"
        )
        // AIKit#18 r4493433754: an error names the injected target, never a
        // host written into the message.
        //
        // And W24 / r4500322517: it names the PLATFORM the request was sent
        // FROM, because "the server is local" says nothing about which device
        // asked — *"you can have several mac on the same network or the lms
        // server running on a gaming computer you gonna use even if shikki is
        // running on a MacBook Air"*. Without it, two devices failing against
        // one server produce identical log lines.
        #expect(
            target.description
                == "devstral-local at http://127.0.0.1:1234/v1/chat/completions, from "
                + HostPlatform.current.rawValue
        )
        #expect(target.hostPlatform == .current)
    }

    @Test("the host platform is a fact about the client, not about the target's identity")
    func theHostPlatformIsNotPartOfIdentity() throws {
        // Two targets naming the same entry, address and route ARE the same
        // target. Folding the platform into equality would make a stored
        // target from one device unequal to the same target on another, and
        // `AIKitError` is `Equatable` precisely so a test can state which
        // target an error carries.
        let address = try AIServerAddress(validating: fixtureServerRoot)
        let one = AIRequestTarget(entryID: "e", address: address, route: "/v1/chat/completions")
        let two = AIRequestTarget(entryID: "e", address: address, route: "/v1/chat/completions")
        #expect(one == two)
        #expect(one.hashValue == two.hashValue)
    }
}
