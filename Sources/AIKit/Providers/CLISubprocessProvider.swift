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

    /// Task-kind label attached to the dispatch record deducted from the
    /// ledger. Defaults to `.codeGen` — pick whichever most closely matches
    /// the wrapped binary's typical workload for accurate accounting.

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
        let start = Date()
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
    public static let defaultTokenEstimator: @Sendable (_ prompt: String, _ output: String) -> (in: Int, out: Int) = {
        prompt, output in
        (
            in: max(1, (prompt.count + 3) / 4),
            out: max(0, (output.count + 3) / 4)
        )
    }
}
