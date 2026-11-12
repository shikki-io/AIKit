import Foundation

// MARK: - The CLI family, on every platform
//
// W25 of spec `ai-layer-aikit-core-2026-10-31`. Everything down to the platform
// gate below compiles EVERYWHERE, including iOS, and that is the point.
//
// W1 made `CLISubprocessProvider` and `WhisperCLIProvider` compile out on iOS,
// which fixed the build and left a hole: on iOS the types do not exist, so a
// caller asking for a CLI-backed provider could not be told "not on this
// platform" — it could only fail to compile, or reach a `fatalError` in a
// hand-written stub. There was nothing to return and nothing to catch.
//
// So the FAMILY is named here, cross-platform, and answers for itself. On iOS
// it resolves to a typed `.unavailable(platform:)`; on macOS and Linux it
// resolves to available and the provider below exists to build.

// MARK: - SearchPathSeparator

/// The separator between entries of a `PATH`-style environment variable.
///
/// Distinct from a path-component separator (CoreKit's `PathSeparator`, which
/// is `/` or `\`): this is the separator BETWEEN paths in a list, and the two
/// differ independently — Windows uses `\` inside a path and `;` between them.
/// Splitting `PATH` on a hardcoded `":"` silently yields one bogus entry on
/// Windows, which reads as "the binary is not installed".
public enum SearchPathSeparator: String, Sendable, CaseIterable, Equatable {

    /// `:` — POSIX, so macOS and Linux.
    case posix = ":"

    /// `;` — Windows.
    case windows = ";"

    /// The separator for the OS this binary was compiled for.
    public static var native: SearchPathSeparator {
        #if os(Windows)
            .windows
        #else
            .posix
        #endif
    }

    /// The separator as a `Character`, for `split(separator:)`.
    public var character: Character { Character(rawValue) }
}

// MARK: - HostExecutable

/// Host binaries AIKit invokes, named once instead of spelled at a call site.
///
/// The fleet's rule is a typed table for host binaries; shikki-plugin-api
/// carries its own `HostExecutable` for the same reason and with the same
/// shape. That one is NOT reachable from here — it is a sibling package, not an
/// upstream of AIKit — and CoreKit ships a path vocabulary but no executable
/// table, so this is AIKit's. When one lands in CoreKit, this table is what
/// gets deleted in favour of it.
///
/// A name, never a path: where a binary lives is the host's business, which is
/// why resolution goes through ``url(in:)`` and ``resolve(in:)`` against
/// directories the caller supplies, and nothing here hardcodes `/usr/bin`.
public enum HostExecutable: String, Sendable, CaseIterable, Hashable {

    /// whisper.cpp's current CLI name.
    case whisperCLI = "whisper-cli"

    /// whisper.cpp's former CLI name, still what homebrew's formula installs.
    case whisperCPP = "whisper-cpp"

    /// CPython 3, which `mlx-video` and `mlx-audio` run under.
    ///
    /// Declared but not yet consumed: `MLXEngine.init` still picks between
    /// `/usr/bin/env` and `/usr/bin/python3` with those paths spelled inline.
    /// `MLXEngine.swift` belongs to W21 of this spec, which is removing the
    /// hardcoded locations in that file, so moving its python resolution here
    /// is W21's edit and not this wave's — two waves rewriting the same lines
    /// is a conflict, not a cleanup.
    case python3

    /// The binary's file name on this host.
    ///
    /// Windows needs the `.exe` suffix: `FileManager.isExecutableFile` on a
    /// suffix-less name finds nothing there, so a PATH scan would report every
    /// binary missing.
    public var fileName: String {
        #if os(Windows)
            "\(rawValue).exe"
        #else
            rawValue
        #endif
    }

    /// This binary's path inside `directory`, composed from components.
    ///
    /// Composed, never interpolated: a path built by `URL.appending(path:)`
    /// carries no separator literal and is therefore correct on every host.
    public func url(in directory: URL) -> URL {
        directory.appending(path: fileName)
    }

    /// The first of `candidates` that is executable in one of `directories`.
    ///
    /// - Returns: the resolved path, or `nil` when none of the candidates is
    ///   installed. Optional on purpose: a bare binary name handed on would
    ///   fail later as a launch error, where "not installed" is the thing the
    ///   caller actually needs to know.
    public static func resolve(
        _ candidates: [HostExecutable],
        in directories: [URL]
    ) -> String? {
        for directory in directories {
            for candidate in candidates {
                let url = candidate.url(in: directory)
                if FileManager.default.isExecutableFile(atPath: url.path) { return url.path }
            }
        }
        return nil
    }

    /// The directories of a `PATH`-style value, split on the host's separator.
    ///
    /// - Parameter value: the raw variable, or `nil` when it is unset — which
    ///   yields no directories rather than one empty one.
    public static func searchPaths(_ value: String?) -> [URL] {
        (value ?? "")
            .split(separator: SearchPathSeparator.native.character)
            .map(String.init)
            .filter { !$0.isEmpty }
            .map { URL(fileURLWithPath: $0) }
    }

    /// The host's `PATH`, as directories.
    public static var pathDirectories: [URL] {
        searchPaths(ProcessInfo.processInfo.environment["PATH"])
    }
}

// MARK: - CLIProviderFamily

/// Why a CLI-backed provider cannot be built on this host.
public enum CLIProviderError: Error, LocalizedError, Equatable {

    /// The host has no subprocess API, so no CLI provider can exist here.
    ///
    /// Carries the platform rather than a sentence, so a caller can branch on
    /// it — offer an on-device provider instead of a CLI one, say — without
    /// matching text.
    case unavailable(platform: HostPlatform)

    public var errorDescription: String? {
        switch self {
        case .unavailable(let platform):
            "A CLI-backed AI provider cannot run on \(platform.rawValue): "
                + "it has no subprocess API, so ShellKit is not linked. "
                + "Use an on-device or HTTP provider on this platform."
        }
    }
}

/// Whether a CLI-backed provider family can be built here.
public enum CLIProviderAvailability: Sendable, Equatable {

    /// The family can be built on this host.
    case available

    /// The family cannot be built, and this is the host that cannot.
    case unavailable(platform: HostPlatform)
}

/// The CLI-backed provider families AIKit ships.
///
/// Each case is a provider that reaches a model by running a binary, and every
/// one of them needs a subprocess API. The enum exists so that "is there a CLI
/// provider on this platform" is a value a caller can hold and test, rather
/// than an `#if` the caller has to repeat.
public enum CLIProviderFamily: String, Sendable, CaseIterable, Hashable {

    /// ``CLISubprocessProvider`` — any operator-defined CLI runner.
    case subprocess = "cli-subprocess"

    /// ``WhisperCLIProvider`` — whisper.cpp transcription.
    case whisperCLI = "whisper-cli"

    /// The executables this family may run, in preference order.
    public var executables: [HostExecutable] {
        switch self {
        // An operator-defined runner names its own binary in config; the table
        // has no entry to offer it, and inventing one would be a guess.
        case .subprocess: []
        case .whisperCLI: [.whisperCLI, .whisperCPP]
        }
    }

    /// Whether this family can be built on the host running now.
    ///
    /// Keyed on `canImport(ShellKit)`, which is THE fact — `Package.swift`
    /// conditions ShellKit on macOS and Linux, so the import and the
    /// availability cannot drift apart. A second platform list here would be a
    /// copy of the manifest that could disagree with it.
    public static var availability: CLIProviderAvailability {
        #if canImport(ShellKit)
            .available
        #else
            .unavailable(platform: .current)
        #endif
    }

    /// Whether this family can be built on the host running now.
    public static var isAvailable: Bool { availability == .available }

    /// Throws when this family cannot be built here.
    ///
    /// For a call site about to construct a provider: it turns the platform
    /// question into one `try` at the top instead of an `#if` around the whole
    /// construction, and on iOS it throws ``CLIProviderError/unavailable(platform:)``
    /// rather than trapping.
    public static func requireAvailable() throws {
        if case .unavailable(let platform) = availability {
            throw CLIProviderError.unavailable(platform: platform)
        }
    }
}

// MARK: - Platform gate
//
// W1 of spec `ai-layer-aikit-core-2026-10-31` (BR-AIL-02). ShellKit is a
// macOS-only dependency — it builds on `Foundation.Process`, absent from the
// iOS SDK — so the whole provider compiles only where ShellKit can be
// imported. On iOS the type is simply not part of the AIKit surface, and
// ``CLIProviderFamily/availability`` above is what says so at runtime.
#if canImport(ShellKit)

    import Foundation
    import Logging
    import ShellKit

    // MARK: - CLISubprocessProvider
    //
    // W4 of spec `provider-kind-cli-subprocess`.
    //
    // A provider that satisfies `AgentProviding` by shelling out to a CLI binary
    // (`claude -p`, `codex …`, or any operator-defined script). It is the
    // canonical way to plug a headless third-party AI runner into the Shikki
    // dispatch surface without importing that runner's SDK.
    //
    // The provider does two things beyond a raw shellout:
    //
    //   1. **Concurrency cap** — every `run(prompt:timeout:)` is gated by
    //      `BudgetLedger.acquireSlot(for:)`, so the maximum number of concurrent
    //      subprocesses per provider is bounded.
    //
    //   2. **Budget deduction** — after a successful run, the provider estimates
    //      the token cost (default: ceil(len/4) for input + output) and reports
    //      it to the ledger via `AIProviderDispatchRecord`. A ledger with no
    //      configured cap for this `providerID` treats the deduction as a no-op.
    //
    // If no `BudgetLedger` is supplied, the provider degrades gracefully to an
    // unlimited-concurrency plain shellout — useful for early bring-up and for
    // call sites that already own their own admission control.
    public struct CLISubprocessProvider: AgentProviding, Sendable {

        // MARK: - Errors

        public enum SubprocessError: Error, Sendable, Equatable {
            /// The CLI process exited with a non-zero status.
            case nonZeroExit(providerID: AIProviderID, exitCode: Int32, stderr: String)
            /// The CLI process exceeded `timeout` and was killed.
            case timeout(providerID: AIProviderID, limit: TimeInterval)
            /// `Process.run` failed (binary missing, permissions, etc.).
            case launchFailed(providerID: AIProviderID, reason: String)
            /// The ledger refused the run because the provider's monthly budget
            /// is already exhausted.
            case budgetExhausted(providerID: AIProviderID, spent: Int, cap: Int)
        }

        // MARK: - Configuration

        /// Stable id of the provider being fronted. Used for ledger accounting
        /// and error attribution.
        public let providerID: AIProviderID

        /// Binary + fixed arguments. The prompt is passed on stdin, so no
        /// prompt-shaped placeholder is required in `arguments`.
        ///
        /// Example: `["claude", "-p", "--dangerously-skip-permissions"]`.
        public let arguments: [String]

        /// Working directory for the child process (nil = inherit).
        public let workingDirectory: String?

        /// Extra environment variables merged into the child process env.
        public let environment: [String: String]

        // MARK: - Collaborators

        private let ledger: BudgetLedger?
        private let executor: any ShellExecutorProtocol
        private let tokenEstimator: @Sendable (_ prompt: String, _ output: String) -> (in: Int, out: Int)
        private let logger: Logger

        // MARK: - Init

        /// Create a provider.
        /// - Parameters:
        ///   - providerID: routing id used for ledger accounting.
        ///   - arguments: `[binary, arg1, arg2, …]`. The prompt is piped on stdin.
        ///   - executor: shell executor implementation (defaults to
        ///     `TimedShellExecutor()` from ShellKit).
        ///   - ledger: optional concurrency + budget gate. Pass `nil` to skip both.
        ///   - workingDirectory: inherit if nil.
        ///   - environment: extra env vars merged with the current process env.
        ///   - tokenEstimator: function mapping (prompt, output) → (in-tokens, out-tokens).
        ///     Defaults to `ceil(count / 4)` — a coarse but conservative approximation
        ///     that matches Anthropic's public rule of thumb.
        ///   - logger: diagnostic logger.
        public init(
            providerID: AIProviderID,
            arguments: [String],
            executor: any ShellExecutorProtocol = TimedShellExecutor(),
            ledger: BudgetLedger? = nil,
            workingDirectory: String? = nil,
            environment: [String: String] = [:],
            tokenEstimator: @escaping @Sendable (_ prompt: String, _ output: String) -> (in: Int, out: Int) = Self
                .defaultTokenEstimator,
            logger: Logger = Logger(label: "shikki.cli-subprocess-provider")
        ) {
            precondition(!arguments.isEmpty, "CLISubprocessProvider requires at least the binary name in `arguments`")
            self.providerID = providerID
            self.arguments = arguments
            self.executor = executor
            self.ledger = ledger
            self.workingDirectory = workingDirectory
            self.environment = environment
            self.tokenEstimator = tokenEstimator
            self.logger = logger
        }

        // MARK: - AgentProviding

        public func run(prompt: String, timeout: TimeInterval) async throws -> String {
            if let ledger {
                return try await ledger.withSlot(for: providerID) { _ in
                    try await runSubprocess(prompt: prompt, timeout: timeout, ledger: ledger)
                }
            }
            return try await runSubprocess(prompt: prompt, timeout: timeout, ledger: nil)
        }

        // MARK: - Private

        private func runSubprocess(
            prompt: String,
            timeout: TimeInterval,
            ledger: BudgetLedger?
        ) async throws -> String {
            let result: ShellCommandResult
            do {
                result = try await executor.run(
                    arguments,
                    cwd: workingDirectory,
                    env: environment.isEmpty ? nil : environment,
                    timeout: timeout,
                    stdin: Data(prompt.utf8)
                )
            } catch ShellError.timeout(_, let limit) {
                throw SubprocessError.timeout(providerID: providerID, limit: limit)
            } catch ShellError.launchFailed(_, let underlying) {
                throw SubprocessError.launchFailed(providerID: providerID, reason: underlying)
            } catch {
                throw SubprocessError.launchFailed(providerID: providerID, reason: String(describing: error))
            }

            guard result.exitCode == 0 else {
                let stderr = result.stderrString
                logger.error(
                    "CLI subprocess non-zero exit",
                    metadata: [
                        "provider": "\(providerID.rawValue)",
                        "exit_code": "\(result.exitCode)",
                        "stderr_preview": "\(String(stderr.prefix(200)))",
                    ]
                )
                throw SubprocessError.nonZeroExit(
                    providerID: providerID,
                    exitCode: result.exitCode,
                    stderr: stderr
                )
            }

            let output = result.stdoutString
            if let ledger {
                let (tokensIn, tokensOut) = tokenEstimator(prompt, output)
                await ledger.deduct(providerID: providerID, totalTokens: tokensIn + tokensOut)
            }
            return output
        }

        // MARK: - Token estimation

        /// Default estimator: `ceil(count / 4)` for prompt and output.
        /// Matches Anthropic's public "roughly 4 characters per token" heuristic.
        public static let defaultTokenEstimator: @Sendable (_ prompt: String, _ output: String) -> (in: Int, out: Int) = { prompt, output in
                (
                    in: max(1, (prompt.count + 3) / 4),
                    out: max(0, (output.count + 3) / 4)
                )
            }
    }

#endif  // canImport(ShellKit)
