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
/// W21 of spec ai-layer-aikit-core-2026-10-31 completes the rule with `reach`
/// (loopback / private network / remote) and the ratchet scope that keeps
/// `?? "http"` and `?? "127.0.0.1"` out of Sources. This wave introduces the
/// type and the throwing init.
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
