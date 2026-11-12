import Foundation

/// Errors thrown by AIKit operations.
public enum AIKitError: Error, LocalizedError, Equatable {
    case noProviderAvailable(AICapabilities)
    case modelNotFound(ModelIdentifier)
    case downloadFailed(String)
    case engineUnavailable(String)
    case requestFailed(String)
    case streamingNotSupported
    /// Every provider in a fallback list was tried and none answered.
    ///
    /// Carries one `AIProviderAttempt` per provider that was asked, in the order
    /// they were tried. A bare case here used to discard every cause — the
    /// request simply "failed", and the reason a local server refused the
    /// connection or a cloud one rate-limited was gone (BR-AIL-07).
    case allProvidersFailed([AIProviderAttempt])

    // MARK: - Rev 23 — the typed cases a request produces

    /// A configured server address is not one (BR-AIL-18).
    ///
    /// Names the value and why, because the only useful answer to this is an
    /// edit to the settings file. Thrown by ``AIServerAddress/init(validating:)``,
    /// which is the single place an address is read.
    case invalidServerAddress(value: String, reason: String)

    /// No server address is configured at all (BR-AIL-18).
    ///
    /// The other half of the rule. ``invalidServerAddress(value:reason:)``
    /// covers an address that was written badly; this covers one that was
    /// never written, and it exists because the alternative is what W21
    /// deleted: `?? "http://127.0.0.1:1234"` and `?? "http://localhost:8080"`
    /// answered the absence by pointing at the local machine, so a host that
    /// had configured nothing got a connection-refused from its own loopback
    /// instead of being told what to set (AIKit#18 r4500281951, r4500300887).
    ///
    /// `setting` names what to fill in — an environment variable today, a
    /// settings-file key once W22 lands — so the message is actionable.
    case serverAddressNotConfigured(setting: String)

    /// The request carries something this family's wire cannot express
    /// (BR-AIL-17).
    ///
    /// Thrown BEFORE anything is sent, by the codec, naming the field and the
    /// family. What this replaces is the actual defect: tools, `responseFormat`
    /// and media inputs were dropped on the floor while `AICapabilities` went
    /// on advertising them, so a caller's tool definitions simply never reached
    /// the model and nothing said so (AIKit#18, panel findings on
    /// `AnthropicEndPoint.swift:68-81`, `LMStudioEndPoint.swift:71-77`,
    /// `OpenAIEndPoint.swift:129`).
    case unsupported(field: String, family: AIProviderFamily)

    /// The server answered with a non-2xx status (BR-AIL-19).
    ///
    /// Carries the status, the server's own text, and where the request went.
    /// Both paths raise it: `complete` after reading the status, and `stream`
    /// BEFORE the first byte of body. Streaming used to swallow these — a 401,
    /// a 429 or a 500 finished the stream as an empty success, so a caller saw
    /// no text and no error and a fallback chain never fired
    /// (`OpenAIProvider.SSEStreamDelegate`).
    ///
    /// Fallback eligibility is read from the STATUS here, never from a prefix
    /// of the message: `LMStudioProvider` used to decide by
    /// `message.hasPrefix(…)`, which gave one wire event two verdicts
    /// depending on how its sentence had been composed.
    case http(status: Int, body: String, target: AIRequestTarget)

    /// The server answered 2xx, and the reply carried no text at all.
    ///
    /// A case rather than a sentence, for the reason above: the one caller that
    /// has to recognise this (``LMStudioProvider``, which maps it to
    /// `.emptyContent`) matches the case.
    case emptyReply(target: AIRequestTarget)

    /// A 2xx reply did not decode into the shape its family declares.
    ///
    /// Distinct from ``requestFailed(_:)`` because the server DID answer: the
    /// next provider in a chain cannot improve on a reply we failed to read,
    /// and retrying would report its answer in place of this cause.
    case replyUndecodable(target: AIRequestTarget, reason: String)

    /// One line of a streamed reply passed the entry's byte cap.
    ///
    /// Reached when a server sends an unbounded newline-free body on a 200 —
    /// the shape that used to grow the SSE buffer without limit.
    case streamLineTooLong(cap: Int, target: AIRequestTarget)

    public var errorDescription: String? {
        switch self {
        case .noProviderAvailable(let capabilities):
            "No provider available for capabilities: \(capabilities)"
        case .modelNotFound(let id):
            "Model not found: \(id)"
        case .downloadFailed(let reason):
            "Download failed: \(reason)"
        case .engineUnavailable(let engine):
            "Engine unavailable: \(engine)"
        case .requestFailed(let reason):
            "Request failed: \(reason)"
        case .streamingNotSupported:
            "Streaming is not supported by this provider"
        case .allProvidersFailed(let attempts):
            AIProviderAttempt.summary(of: attempts)
        case .invalidServerAddress(let value, let reason):
            "Not a server address: '\(value)' — \(reason)"
        case .serverAddressNotConfigured(let setting):
            """
            No server address is configured: set '\(setting)' to a server root such as             http://host:1234. AIKit does not default to the local machine (BR-AIL-18)
            """
        case .unsupported(let field, let family):
            "\(family.displayName) cannot carry '\(field)', so nothing was sent"
        case .http(let status, let body, let target):
            "HTTP \(status) from \(target): \(body.isEmpty ? "<the server sent no body>" : body)"
        case .emptyReply(let target):
            "\(target) answered with no content"
        case .replyUndecodable(let target, let reason):
            "The reply from \(target) could not be read: \(reason)"
        case .streamLineTooLong(let cap, let target):
            "A streamed line from \(target) passed the \(cap)-byte cap without a newline"
        }
    }
}

// MARK: - AIProviderAttempt

/// One provider's failed turn in a fallback list: who was asked, and why it did
/// not answer.
///
/// The cause is kept as text rather than as the error value itself. An
/// `any Error` existential is neither `Equatable` nor `Sendable`, and
/// `AIKitError` is both — carrying the live error would cost the whole error
/// family conformances that callers and tests already rely on. The description
/// is what an operator reads in a log or a CLI line, so nothing diagnostic is
/// lost; a caller that must branch on a type catches it at the provider instead.
public struct AIProviderAttempt: Sendable, Equatable, CustomStringConvertible {
    /// The `AIProvider.id` that was asked.
    public let providerID: String
    /// Why it did not answer, in the error's own words.
    public let cause: String

    public init(providerID: String, cause: String) {
        self.providerID = providerID
        self.cause = cause
    }

    /// Records an attempt from the error a provider threw.
    public init(providerID: String, error: any Error) {
        self.init(providerID: providerID, cause: Self.describe(error))
    }

    public var description: String { "\(providerID) — \(cause)" }

    /// The error's own sentence when it writes one, its value otherwise.
    ///
    /// `localizedDescription` on a plain `struct` error yields Foundation's
    /// "The operation couldn't be completed" boilerplate, which names nothing.
    /// A `LocalizedError` — every `AIKitError` — does name itself.
    static func describe(_ error: any Error) -> String {
        if let described = (error as? any LocalizedError)?.errorDescription {
            return described
        }
        return String(describing: error)
    }

    /// One line naming every provider that was asked and what it said.
    static func summary(of attempts: [AIProviderAttempt]) -> String {
        guard !attempts.isEmpty else {
            // Not the same failure: nothing was asked, so no provider failed.
            // Reporting "all providers failed" for an empty list would send a
            // reader looking for a provider fault that never happened.
            return "No provider was tried: the fallback list was empty"
        }
        let causes = attempts.map(\.description).joined(separator: "; ")
        return "All \(attempts.count) provider(s) failed: \(causes)"
    }
}

// MARK: - FallbackEligibility

/// Whether a failed attempt may be retried on the next provider.
///
/// The one rule, consulted by both `ProviderRouter.routeWithFallback` and
/// `FallbackProviderChain`. The two were written apart and drifted: the chain
/// weighed each error, the router retried on anything with `catch { continue }`.
///
/// The question each case answers is the same one the chain always asked: could
/// the *next* provider plausibly serve this request? A server that is down,
/// rate-limiting, or missing the model is a fact about that provider, so the
/// next one gets a turn. A reply the server really did produce, or a refusal
/// that will repeat everywhere, is not worth asking again.
public enum FallbackEligibility {

    /// HTTP statuses that mean "this server, right now" rather than "this
    /// request, anywhere" (BR-AIL-19).
    ///
    /// 408 and 429 are the server saying to come back later; 5xx is the server
    /// saying it broke. Everything else in the 4xx range is a verdict on the
    /// request — a wrong key (401), a forbidden model (403), a route that does
    /// not exist (404), a body it refused (400) — and the next provider would
    /// return the same verdict while hiding this one.
    static func allowsStatus(_ status: Int) -> Bool {
        status == 408 || status == 429 || status >= 500
    }

    /// Whether `error` means "ask the next provider".
    ///
    /// Unknown error types are not eligible. A caller whose providers throw
    /// their own types widens the rule through
    /// `FallbackProviderChain.extraFallbackEligible` rather than by editing
    /// this one.
    public static func allows(_ error: any Error) -> Bool {
        if let aiKitError = error as? AIKitError {
            return allows(aiKitError)
        }

        // LM Studio, reached as an `AgentProviding` backend.
        if let lmError = error as? LMStudioProvider.LMStudioError {
            switch lmError {
            case .connectionRefused, .rateLimited:
                return true
            case .invalidResponse, .httpError, .emptyContent:
                // The server answered. Its answer is the news, not a reason to
                // ask elsewhere.
                return false
            }
        }

        if let urlError = error as? URLError {
            switch urlError.code {
            case .cannotConnectToHost, .networkConnectionLost, .timedOut, .cannotFindHost:
                return true
            default:
                return false
            }
        }

        return false
    }

    private static func allows(_ error: AIKitError) -> Bool {
        switch error {
        // MARK: Facts about THIS provider — the next one gets its turn

        case  // This engine cannot run here — an MLX provider on iOS, Core ML off
        // an Apple platform. Exactly what a fallback list exists for.
        .engineUnavailable,
            // The provider refuses the capability itself: `whisper.cpp` is not
            // installed, so the Core ML transcriber behind it gets the request.
            .noProviderAvailable,
            // Absent from this provider's registry; the next one may hold it.
            .modelNotFound,
            // THIS family's wire cannot carry the field — a fact about this
            // provider, not about the request. LM Studio's native API takes no
            // custom tools; the chat-completions entry behind it does. So the
            // next provider gets its turn, and the attempt list keeps the
            // refusal visible if none of them can carry it either.
            .unsupported,
            // The catch-all, and the honest reading is that it is two errors
            // under one name: a server that could not be reached or replied with
            // an error body, and a request that was malformed before any server
            // saw it ("No user message found", "carries 0 bytes of audio").
            // Eligible, because the unreachable-server reading is the one a
            // fallback list is built for, and because the alternative — refusing
            // to fall back when a local server is simply down — is the failure
            // the operator actually hits. The cost is that a malformed request
            // is now retried on every provider; the attempt list is what keeps
            // that visible, printing the same cause once per provider instead of
            // hiding it behind a bare "all providers failed".
            .requestFailed:
            return true

        // MARK: The status decides (BR-AIL-19)

        case .http(let status, _, _):
            return allowsStatus(status)

        // MARK: Not worth asking again

        case  // The server answered. Whatever it sent is the news; the next
        // provider's answer would replace this evidence rather than add to
        // it.
        .emptyReply, .replyUndecodable, .streamLineTooLong,
            // A settings mistake, and NOT eligible on purpose: trying the next
            // provider would hide a misconfigured address behind a working one,
            // and the operator would keep an entry that never works. An absent
            // address reads the same way — it is the operator's file that has
            // to change, not the provider that gets asked.
            .invalidServerAddress, .serverAddressNotConfigured,
            // A registry transfer, not a completion attempt. Retrying the
            // request elsewhere does not finish a download.
            .downloadFailed,
            // A refusal that repeats: the caller asked a provider to stream and
            // must see that, not a different provider's answer.
            .streamingNotSupported,
            // A nested fallback already exhausted its own list and is carrying
            // its own evidence. Spending another list on it would bury that.
            .allProvidersFailed:
            return false
        }
    }
}
