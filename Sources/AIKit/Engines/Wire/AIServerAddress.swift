import Foundation

/// Where an AI server lives: scheme, host, port, and nothing else.
///
/// BR-AIL-09 makes a configured address the server ROOT — the route is the
/// endpoint enum's business (``OpenAIEndPoint``, ``AnthropicEndPoint``,
/// ``LMStudioEndPoint``). BR-AIL-18 makes this type the only way to hold one,
/// and its only initializer throwing, so an address is validated once and a
/// bad one is named at the point it was configured rather than read back out
/// of a reply.
///
/// There is deliberately no memberwise initializer and no default for anything.
/// The previous shape (`AIWireServer`) had both, and the defaults were the
/// defect: `URLComponents(string:)?.scheme ?? "http"` paired with
/// `?.host ?? "127.0.0.1"` turned every unparseable string into a request
/// against the local machine — six of the panel's ten inputs were EXECUTED
/// against the wrong server that way, one of them carrying a bearer key
/// (AIKit#18 r4500281951, r4500300887).
///
/// W21 of spec ai-layer-aikit-core-2026-10-31 completed the rule: `reach`
/// (loopback / private network / remote) replaces the substring test that
/// called a LAN Mac "cloud", `resolvedPort` answers the port question the rule
/// leaves open, and the `AIKitNoLocalhostFallback` kagami scope keeps
/// `?? "http"` and `?? "127.0.0.1"` out of Sources.
public struct AIServerAddress: Sendable, Equatable, Hashable, CustomStringConvertible {

    /// `http` or `https`, as configured. Never defaulted.
    public let scheme: String
    /// Host or IP literal, as configured. Never defaulted.
    public let host: String
    /// The configured port, or nil when the scheme's default applies.
    public let port: Int?

    /// Reads a server root such as `http://127.0.0.1:1234` or
    /// `https://api.openai.com`.
    ///
    /// Refused, each naming the value: an empty string, something that is not
    /// a URL, a missing scheme, a scheme that is not HTTP, a missing host, a
    /// path (including the `/v1` that used to be carried in `base_url` and
    /// would have produced `/v1/v1/chat/completions`), a query, a fragment,
    /// and embedded credentials.
    ///
    /// - Throws: ``AIKitError/invalidServerAddress(value:reason:)``.
    public init(validating text: String) throws(AIKitError) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)

        guard !trimmed.isEmpty else {
            throw .invalidServerAddress(value: text, reason: "it is empty")
        }
        guard let components = URLComponents(string: trimmed) else {
            throw .invalidServerAddress(value: trimmed, reason: "it is not a URL")
        }
        guard let scheme = components.scheme?.lowercased(), !scheme.isEmpty else {
            throw .invalidServerAddress(
                value: trimmed,
                reason: "it names no scheme — write http:// or https:// in front of the host"
            )
        }
        guard scheme == "http" || scheme == "https" else {
            throw .invalidServerAddress(value: trimmed, reason: "scheme '\(scheme)' is not http or https")
        }
        guard let host = components.host, !host.isEmpty else {
            throw .invalidServerAddress(
                value: trimmed,
                reason: "it names no host — 'http://:1234' and 'http:///v1' both reach nothing"
            )
        }
        // A lone "/" is a root written with a trailing slash, not a path.
        let leftover = components.path.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        guard leftover.isEmpty else {
            throw .invalidServerAddress(
                value: trimmed,
                reason: """
                    it carries the path '/\(leftover)'. An address is the server root; the route belongs \
                    to the endpoint enum (BR-AIL-09), so drop '/\(leftover)'
                    """
            )
        }
        guard components.query == nil else {
            throw .invalidServerAddress(value: trimmed, reason: "it carries a query string")
        }
        guard components.fragment == nil else {
            throw .invalidServerAddress(value: trimmed, reason: "it carries a fragment")
        }
        guard components.user == nil, components.password == nil else {
            throw .invalidServerAddress(
                value: trimmed,
                reason: "it embeds credentials in the URL — an api_key_ref carries the secret instead"
            )
        }

        self.scheme = scheme
        self.host = host
        self.port = components.port
    }

    /// The root as configured, for an error that has to name where it pointed.
    public var description: String {
        var text = "\(scheme)://\(host)"
        if let port { text += ":\(port)" }
        return text
    }
}

// MARK: - The port

extension AIServerAddress {

    /// The port this address is reached on: the configured one, or the
    /// scheme's default.
    ///
    /// BR-AIL-18 allows a port to be left out, and means the scheme's default
    /// when it is — not "unknown", and never a guess at the local machine's.
    /// ``port`` stays optional so ``description`` can print the address the
    /// way the operator wrote it; this is the number a reader needs.
    public var resolvedPort: Int {
        if let port { return port }
        return scheme == "https" ? 443 : 80
    }
}

// MARK: - Reach

extension AIServerAddress {

    /// How far away the server is: on this machine, on this network, or out
    /// on the internet.
    ///
    /// It replaces a substring test. `OpenAICompatibleEngine` decided locality
    /// with `host == "127.0.0.1" || host == "localhost"`, so the Mac on the
    /// desk running LM Studio at `192.168.1.42` or `studio.local` was reported
    /// as a cloud model — wrong in the catalogue the operator picks from, and
    /// wrong in any decision that reads "local" as "costs nothing and sends
    /// nothing off this network" (AIKit#18, panel finding on
    /// `OpenAICompatibleEngine.swift:62`).
    ///
    /// Three cases rather than a `Bool`, because the two questions a caller
    /// asks are different ones: "does this request stay on this machine"
    /// (``loopback``) and "does this request leave this network"
    /// (``remote``). A LAN server answers them differently, and a `Bool`
    /// cannot.
    public enum Reach: String, Sendable, Equatable, Hashable, Codable, CaseIterable {

        /// This machine: `127.0.0.0/8`, `::1`, or a `localhost` name.
        case loopback

        /// This network and no further: RFC 1918 (`10/8`, `172.16/12`,
        /// `192.168/16`), RFC 3927 link-local (`169.254/16`), their IPv6
        /// equivalents (`fc00::/7` unique-local, `fe80::/10` link-local), and
        /// an mDNS `.local` name.
        case privateNetwork

        /// Anything else — a name or address that resolves off this network.
        case remote
    }

    /// Where this address sits, read from its host.
    ///
    /// Computed from the host STRING, with no DNS lookup: a lookup would make
    /// a property that reads as pure into a network call, would differ between
    /// two runs on the same settings file, and would stall whatever drew a
    /// model list. The cost is that a `.remote` name whose DNS happens to
    /// answer with a LAN address reads as remote, which is the safe direction:
    /// it never calls an internet host local.
    public var reach: Reach {
        Self.reach(ofHost: host)
    }

    /// The classification, as a function of the host text alone.
    ///
    /// Split into the three families a host can belong to rather than written
    /// as one chain, so each rule is readable beside its RFC.
    static func reach(ofHost host: String) -> Reach {
        let bare = normalised(host)
        if isLoopbackName(bare) { return .loopback }
        if bare.hasSuffix(".local") {
            // mDNS, which is this network by definition (`studio.local`).
            return .privateNetwork
        }
        if let octets = ipv4Octets(bare) { return reach(ofIPv4: octets) }
        if bare.contains(":") { return reach(ofIPv6: bare) }
        return .remote
    }

    /// The host as the rules below read it: lowercased, unbracketed, and with
    /// an IPv6 zone identifier (`fe80::1%en0`) dropped.
    ///
    /// `URLComponents` hands back an IPv6 literal without its brackets on some
    /// inputs and with them on others; both spellings are one host.
    private static func normalised(_ host: String) -> String {
        let bare =
            host
            .trimmingCharacters(in: CharacterSet(charactersIn: "[]"))
            .lowercased()
        return bare.split(separator: "%", maxSplits: 1).first.map(String.init) ?? bare
    }

    /// Whether a NAME (or an IPv6 loopback literal) means this machine.
    ///
    /// `localhost.example.com` is not one: the rule is the whole name or a
    /// `.localhost` suffix, never a substring — which is the class of mistake
    /// `reach` exists to replace.
    private static func isLoopbackName(_ host: String) -> Bool {
        host == "localhost"
            || host.hasSuffix(".localhost")
            || host == "::1"
            || host == "::ffff:127.0.0.1"
    }

    /// Where a dotted-quad address sits: `127/8` is this machine, RFC 1918 and
    /// RFC 3927 link-local are this network, everything else is out there.
    private static func reach(ofIPv4 octets: [Int]) -> Reach {
        switch (octets[0], octets[1]) {
        case (127, _):
            return .loopback
        case (10, _), (192, 168), (172, 16...31):
            return .privateNetwork
        case (169, 254):
            // RFC 3927 link-local, what a host self-assigns with no DHCP.
            return .privateNetwork
        default:
            return .remote
        }
    }

    /// IPv6 unique-local (`fc00::/7`) and link-local (`fe80::/10`) are this
    /// network; any other v6 address is out there.
    private static func reach(ofIPv6 host: String) -> Reach {
        let uniqueLocal = ["fc", "fd"]
        let linkLocal = ["fe8", "fe9", "fea", "feb"]
        if uniqueLocal.contains(String(host.prefix(2))) { return .privateNetwork }
        if linkLocal.contains(where: host.hasPrefix) { return .privateNetwork }
        return .remote
    }

    /// The four octets of a dotted-quad IPv4 literal, or nil for anything else
    /// — including a name that merely looks like one (`10.0.0.300`, `1.2.3`).
    private static func ipv4Octets(_ host: String) -> [Int]? {
        let parts = host.split(separator: ".", omittingEmptySubsequences: false)
        guard parts.count == 4 else { return nil }
        var octets: [Int] = []
        octets.reserveCapacity(4)
        for part in parts {
            guard !part.isEmpty, part.allSatisfy(\.isNumber), let value = Int(part), value <= 255 else {
                return nil
            }
            octets.append(value)
        }
        return octets
    }
}
