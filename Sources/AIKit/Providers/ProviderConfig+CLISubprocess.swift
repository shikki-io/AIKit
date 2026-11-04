// MARK: - Platform gate
//
// W1 of spec `ai-layer-aikit-core-2026-10-31` (BR-AIL-02). This config type
// describes a `kind = "cli-subprocess"` provider, and that provider only
// exists where ShellKit does (macOS). Gating the config with its provider
// keeps the feature family whole: iOS never offers a config for a backend it
// cannot run. The file imports no ShellKit symbol itself — the gate is about
// the feature, not the dependency.
#if canImport(ShellKit)

    import Foundation

    // MARK: - CLISubprocessProviderConfig

    /// Parsed configuration for a `kind = "cli-subprocess"` provider entry.
    ///
    /// Wire shape (TOML / JSON dictionary) — parsed via ``init(dictionary:)``:
    ///
    /// ```toml
    /// [providers.claude-code]
    /// kind          = "cli-subprocess"
    /// binary        = "claude"
    /// args_template = ["-p", "--model", "{model}", "{prompt}"]
    /// default_model = "sonnet"
    /// model_aliases = { sonnet = "<sonnet-canonical>", opus = "<opus-canonical>" }
    /// env           = { CLAUDE_HOME = "/tmp/claude" }   # optional
    /// cwd           = "/workspace"                       # optional
    /// ```
    ///
    /// `<sonnet-canonical>` / `<opus-canonical>` resolve via
    /// ``ProviderModelCatalog/default`` — do NOT paste a `claude-*` literal
    /// into this doc block.
    ///
    /// The parser only accepts `kind = "cli-subprocess"`; any other kind is a
    /// typed error so operators don't silently get a subprocess adapter when
    /// they wrote an HTTP one.
    public struct CLISubprocessProviderConfig: Sendable, Equatable {

        // MARK: - ConfigError

        public enum ConfigError: Error, Sendable, Equatable {
            /// The `kind` key was present but did not equal `cli-subprocess`.
            case wrongKind(String)
            /// A required field was missing (or, for `binary`, empty).
            case missingField(String)
            /// A field was present with the wrong shape (e.g. `args_template` not an array of strings).
            case invalidField(String)
        }

        // MARK: - Constants

        /// The single accepted value for the `kind` field.
        public static let kindLiteral: String = "cli-subprocess"

        // MARK: - Stored properties

        /// The CLI binary to spawn (e.g. `claude`, `codex`, `aider`).
        /// Looked up via `/usr/bin/env` by the shell executor.
        public let binary: String

        /// Positional arguments passed to `binary`, with `{prompt}` and `{model}`
        /// placeholders (see ``ArgsTemplate``). Rendered on each `run(...)`.
        public let argsTemplate: [String]

        /// Model name used when the caller does not override it. Resolved
        /// through ``modelAliases`` before rendering.
        public let defaultModel: String

        /// Short-name → canonical-id alias map (see ``ModelAliasMap``).
        public let modelAliases: ModelAliasMap

        /// Extra environment variables merged into the subprocess env.
        public let env: [String: String]

        /// Working directory for the subprocess; `nil` inherits the caller's.
        public let cwd: String?

        // MARK: - Init (explicit)

        public init(
            binary: String,
            argsTemplate: [String],
            defaultModel: String,
            modelAliases: ModelAliasMap = ModelAliasMap([:]),
            env: [String: String] = [:],
            cwd: String? = nil
        ) {
            self.binary = binary
            self.argsTemplate = argsTemplate
            self.defaultModel = defaultModel
            self.modelAliases = modelAliases
            self.env = env
            self.cwd = cwd
        }

        // MARK: - Init (dictionary — TOML/JSON shape)

        /// Parse a TOML-shaped dictionary (as produced by TOMLKit / JSONSerialization).
        ///
        /// - Throws:
        ///   - ``ConfigError/wrongKind(_:)`` if `kind` is present and is not `cli-subprocess`.
        ///   - ``ConfigError/missingField(_:)`` for missing `binary` / `default_model`
        ///     (or an empty `binary`).
        ///   - ``ConfigError/invalidField(_:)`` for shape mismatches.
        public init(dictionary: [String: Any]) throws {
            // kind: optional but if present must match.
            if let kind = dictionary["kind"] as? String, kind != Self.kindLiteral {
                throw ConfigError.wrongKind(kind)
            }

            guard let binary = dictionary["binary"] as? String, !binary.isEmpty else {
                throw ConfigError.missingField("binary")
            }

            let argsTemplate = try Self.optional([String].self, "args_template", in: dictionary) ?? []

            guard let defaultModel = dictionary["default_model"] as? String, !defaultModel.isEmpty else {
                throw ConfigError.missingField("default_model")
            }

            let aliases = try Self.optional([String: String].self, "model_aliases", in: dictionary) ?? [:]
            let env = try Self.optional([String: String].self, "env", in: dictionary) ?? [:]
            let cwd = try Self.optional(String.self, "cwd", in: dictionary)

            self.init(
                binary: binary,
                argsTemplate: argsTemplate,
                defaultModel: defaultModel,
                modelAliases: ModelAliasMap(aliases),
                env: env,
                cwd: cwd
            )
        }

        /// The value at `key`, nil when absent, and a typed error when it has another shape.
        private static func optional<T>(_ type: T.Type, _ key: String, in dictionary: [String: Any]) throws -> T? {
            guard let raw = dictionary[key] else { return nil }
            guard let value = raw as? T else { throw ConfigError.invalidField(key) }
            return value
        }
    }

#endif  // canImport(ShellKit)
