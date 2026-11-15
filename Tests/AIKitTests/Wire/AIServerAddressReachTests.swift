import Foundation
import Testing

@testable import AIKit

// MARK: - AIServerAddressReachTests
//
// T-AIL-43's second half. `reach` replaced a substring test:
// `host == "127.0.0.1" || host == "localhost"` decided whether a model was
// local, so the Mac on the desk running LM Studio at `192.168.1.42` — or at
// `studio.local`, which is how Bonjour spells the same machine — was reported
// as a CLOUD model in the catalogue the operator picks from (AIKit#18, panel
// finding on `OpenAICompatibleEngine.swift:62`).
//
// Three cases, not a `Bool`, because two different questions get asked: does
// this request stay on this machine, and does it leave this network. The LAN
// Mac answers them differently.

@Suite("AIServerAddress.reach — on this machine, on this network, or out there")
struct AIServerAddressReachTests {

    // MARK: - Loopback

    @Test(
        "this machine is loopback",
        arguments: [
            "http://127.0.0.1:1234",
            "http://127.0.0.1",
            "http://127.1.2.3:8080",  // all of 127.0.0.0/8, not just .0.1
            "http://localhost:1234",
            "http://LOCALHOST:1234",
            "http://api.localhost:3000",
            "http://[::1]:1234",
        ])
    func loopbackHosts(root: String) throws {
        #expect(try AIServerAddress(validating: root).reach == .loopback)
    }

    // MARK: - This network

    @Test(
        "this network is privateNetwork",
        arguments: [
            "http://10.0.0.7:11434",  // RFC 1918 10/8
            "http://192.168.1.42:1234",  // RFC 1918 192.168/16
            "http://172.16.0.1:1234",  // RFC 1918 172.16/12, low edge
            "http://172.31.255.254:1234",  // RFC 1918 172.16/12, high edge
            "http://169.254.10.20:1234",  // RFC 3927 link-local
            "http://studio.local:1234",  // mDNS — the LM Studio Mac
            "http://jeo.local",
            "http://[fd00::1]:1234",  // IPv6 unique-local
            "http://[fe80::1]:1234",  // IPv6 link-local
        ])
    func privateNetworkHosts(root: String) throws {
        #expect(try AIServerAddress(validating: root).reach == .privateNetwork)
    }

    @Test("the LAN Mac running LM Studio is not a cloud model")
    func theLANMacIsNotCloud() throws {
        // The regression in one line. `isLocal` in the catalogue reads
        // `reach != .remote`, so this host now answers the question the old
        // substring test answered wrongly.
        let lan = try AIServerAddress(validating: "http://192.168.1.42:1234")
        #expect(lan.reach != .remote)
        #expect(lan.reach != .loopback, "it is not this machine either — that distinction is the point")
    }

    // MARK: - Out there

    @Test(
        "anything else is remote",
        arguments: [
            "https://api.openai.com",
            "https://api.anthropic.com",
            "https://openrouter.ai",
            "http://172.32.0.1:1234",  // just outside 172.16/12
            "http://172.15.255.254:1234",  // just below it
            "http://11.0.0.1:1234",  // just outside 10/8
            "http://192.169.1.1:1234",  // just outside 192.168/16
            "http://169.255.0.1:1234",  // just outside 169.254/16
            "http://8.8.8.8",
            "http://localhost.example.com",  // a NAME that merely starts with it
            "http://notlocalhost",
            "http://my-local:1234",  // ".local" is a suffix rule, not a substring
            "http://[2001:4860:4860::8888]:443",
        ])
    func remoteHosts(root: String) throws {
        #expect(try AIServerAddress(validating: root).reach == .remote)
    }

    @Test("a host that only looks like an IPv4 literal is read as a name")
    func aNearMissIsAName() throws {
        // `10.0.0.300` has no 300th octet, so it is a hostname, and a hostname
        // that is not `.local` is remote until DNS says otherwise — which this
        // type never asks. The safe direction: it never calls an internet host
        // local.
        #expect(try AIServerAddress(validating: "http://10.0.0.300").reach == .remote)
        #expect(try AIServerAddress(validating: "http://10.0.0").reach == .remote)
        #expect(try AIServerAddress(validating: "http://10.0.0.1.2").reach == .remote)
    }

    @Test("reach is read from the host alone, so two schemes agree")
    func reachIgnoresTheScheme() throws {
        #expect(try AIServerAddress(validating: "https://127.0.0.1:8443").reach == .loopback)
        #expect(try AIServerAddress(validating: "http://127.0.0.1:8443").reach == .loopback)
    }

    @Test("an IPv6 zone identifier does not hide a link-local address")
    func anIPv6ZoneIsStripped() throws {
        #expect(AIServerAddress.reach(ofHost: "fe80::1%en0") == .privateNetwork)
    }

    @Test("every case is reachable from some host, so none is dead")
    func everyCaseIsReachable() throws {
        let classified = Set(
            [
                "http://127.0.0.1:1", "http://192.168.0.1:1", "https://api.openai.com",
            ].map { try! AIServerAddress(validating: $0).reach })
        #expect(classified == Set(AIServerAddress.Reach.allCases))
    }

    // MARK: - The port

    @Test("a configured port is kept, and an absent one resolves to the scheme's default")
    func theResolvedPort() throws {
        // BR-AIL-18: "the port explicit or the scheme's default". `port` stays
        // optional so `description` prints the address the operator wrote;
        // `resolvedPort` is the number a reader needs.
        #expect(try AIServerAddress(validating: "http://host:1234").resolvedPort == 1234)
        #expect(try AIServerAddress(validating: "http://host").resolvedPort == 80)
        #expect(try AIServerAddress(validating: "https://host").resolvedPort == 443)
        #expect(try AIServerAddress(validating: "https://host:8443").resolvedPort == 8443)
        // And the written form is untouched by the resolution.
        #expect(try AIServerAddress(validating: "https://host").description == "https://host")
    }

    // MARK: - Nothing configured at all
    //
    // W21 made an absent address its own error (`serverAddressNotConfigured`).
    // W22 made `LMStudioProvider` and `MLXEngine` take a non-optional address,
    // so "nothing configured" is no longer a value a caller can pass: the
    // compiler refuses it, and the error case went with it. A settings file
    // missing the address is refused at load, naming the file and the entry
    // (`AIKitError.settings`, AIProviderRepositoryTests).

    @Test("a provider refuses an address that is not a server root")
    func aProviderRefusesANonRoot() throws {
        let error = #expect(throws: AIKitError.self) {
            _ = try LMStudioProvider(
                baseURL: "\(fixtureServerRoot)/v1",
                model: "qwen",
                transport: MockHTTPTransport.factory()
            )
        }
        // Named at construction, not at first request: validated once.
        #expect(error?.errorDescription?.contains("/v1") == true)
    }
}
