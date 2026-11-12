import Foundation

/// What an AI provider is, and for an HTTP one which protocol its server
/// speaks.
///
/// BR-AIL-15. A caller holds `any AIProvider` and calls `complete` or `stream`;
/// the family is read ONCE, when the provider is built from its settings entry,
/// and nothing downstream switches on a wire standard at call time. That is the
/// difference from the `AIWireStandard` this replaces: the old enum carried the
/// dispatch itself (`complete(request:target:networkService:)`), so every call
/// re-decided what it was talking to and the request, the reply and the route
/// all lived in one file (AIKit#18 r4493790798).
///
/// This is also the ONE `kind` vocabulary. Two parsers used to exist, unaware
/// of each other: `AIWireStandard.init?(kind:)` and the `kind` switch in
/// `ProviderConfig+CLISubprocess`. W22 of spec ai-layer-aikit-core-2026-10-31
/// finishes that consolidation by decoding settings through this type; this
/// wave makes it the type the codecs and the provider agree on.
public enum AIProviderFamily: String, Sendable, Codable, CaseIterable {

    // MARK: - HTTP families

    /// `POST /v1/chat/completions` — OpenAI, LM Studio, Ollama, any proxy.
    case openAIChat = "openai-chat"
    /// `POST /v1/responses` — OpenAI's newer surface.
    case openAIResponses = "openai-responses"
    /// `POST /v1/messages` — Anthropic, and LM Studio's Anthropic-compatible
    /// surface.
    case anthropicMessages = "anthropic-messages"
    /// `POST /api/v1/chat` — LM Studio's own API.
    case lmStudioNative = "lmstudio-native"

    // MARK: - Local families

    /// A CLI run as a subprocess (`CLISubprocessProvider`). macOS and Linux.
    case cliSubprocess = "cli-subprocess"
    /// MLX in-process on the device (`AIKitMLX`).
    case mlxInProcess = "mlx-in-process"
    /// Core ML on the device (`CoreMLEngine`).
    case coreML = "coreml"

    // MARK: - Properties

    /// The four families reached over HTTP, in the order they are declared.
    ///
    /// Every one of them has a production caller: ``HTTPAIProvider``'s
    /// `init(entry:credential:network:)` resolves this family to its codec, so
    /// a settings entry naming any of the four builds a working provider with
    /// no new code (T-AIL-42). Three of them had none before this wave — AIKit
    /// could not talk Anthropic over HTTP at all.
    public static let httpFamilies: [AIProviderFamily] = [
        .openAIChat, .openAIResponses, .anthropicMessages, .lmStudioNative,
    ]

    /// The three families served on the host itself rather than over a wire.
    public static let localFamilies: [AIProviderFamily] = [.cliSubprocess, .mlxInProcess, .coreML]

    /// Whether this family is served by ``HTTPAIProvider``.
    public var isHTTP: Bool { Self.httpFamilies.contains(self) }

    /// Whether a `[[providers]]` row of this family carries
    /// ``AICLISubprocessOptions`` beyond the common fields.
    ///
    /// Read by ``AIProviderEntry``'s decoder, so the family — the one `kind`
    /// vocabulary — is also what decides which options shape a row has. That
    /// is the second half of the consolidation this type began: the `kind`
    /// switch in `ProviderConfig+CLISubprocess` used to answer this question
    /// on its own, from a string literal, with no knowledge of this enum
    /// (AIKit#18, panel finding on two `kind` parsers unaware of each other).
    public var carriesCLIOptions: Bool { self == .cliSubprocess }

    /// Human-readable name, for `shi providers discover` output and errors.
    public var displayName: String {
        switch self {
        case .openAIChat: "OpenAI chat completions"
        case .openAIResponses: "OpenAI responses"
        case .anthropicMessages: "Anthropic messages"
        case .lmStudioNative: "LM Studio native"
        case .cliSubprocess: "CLI subprocess"
        case .mlxInProcess: "MLX in-process"
        case .coreML: "Core ML"
        }
    }

    /// The codec that translates ``AIRequest`` onto this family's wire, and the
    /// reply back.
    ///
    /// - Throws: ``AIKitError/engineUnavailable(_:)`` for a local family, which
    ///   has no wire and is served by its own provider type instead.
    public func codec() throws(AIKitError) -> any WireCodec {
        switch self {
        case .openAIChat: OpenAIChatCodec()
        case .openAIResponses: OpenAIResponsesCodec()
        case .anthropicMessages: AnthropicMessagesCodec()
        case .lmStudioNative: LMStudioNativeCodec()
        case .cliSubprocess, .mlxInProcess, .coreML:
            throw .engineUnavailable(
                "\(displayName) is a local family and has no wire codec: it is served by its own provider"
            )
        }
    }

    // MARK: - Reading a configured kind

    /// Reads the family a settings `kind` names, accepting the spellings
    /// already written in operators' `providers.toml`.
    ///
    /// `openai-compatible` is the value all 7 existing provider profiles carry,
    /// written when nothing read `kind` at all. It means the chat-completions
    /// family, which is what those profiles have always talked to, so it maps
    /// there rather than failing a configuration that was valid yesterday.
    ///
    /// Mistral (La Plateforme, and a local mistral.rs server) speaks the same
    /// chat-completions wire — `POST /v1/chat/completions`, bearer auth, the
    /// same request and reply shapes — so `mistral` and the two
    /// ``AIProviderID`` spellings map there instead of a fifth family that
    /// would duplicate it.
    public init?(kind: String) {
        let normalised = kind.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        switch normalised {
        case "openai-compatible", "openai", "lmstudio", "ollama",
            "mistral", AIProviderID.mistralLaPlateforme.rawValue, AIProviderID.mistralRsLocal.rawValue:
            self = .openAIChat
        case "anthropic":
            self = .anthropicMessages
        default:
            guard let parsed = AIProviderFamily(rawValue: normalised) else { return nil }
            self = parsed
        }
    }
}
