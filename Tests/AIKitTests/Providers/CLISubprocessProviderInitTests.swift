import Foundation
import ShellKit
@testable import AIKit
import Testing

// MARK: - CLISubprocessProvider Init & Config Parsing Tests
//
// Tests for `Provider kind: cli-subprocess`.
// Verifies:
//   - Provider construction from providerID + fixed arguments.
//   - Config parsing from a TOML-shaped `[String: Any]` dictionary
//     (kind, binary, args_template, default_model, model_aliases, env, cwd).
//   - Missing / wrong-kind config surfaces a typed error.
//   - `run(prompt:timeout:)` pipes the prompt on STDIN (never argv),
//     executes via the injected `ShellExecutorProtocol`, and returns
//     stdout on success.
//   - ShellError timeout/launch failures map to typed `SubprocessError`s.

// MARK: - Mock Shell Executor

/// Records calls and returns a canned ShellCommandResult.
final class MockShellExecutor: ShellExecutorProtocol, @unchecked Sendable {
    struct Call: Equatable {
        let args: [String]
        let cwd: String?
        let env: [String: String]?
        let timeout: TimeInterval
        let stdin: Data?
    }

    var calls: [Call] = []
    var result: ShellCommandResult
    var toThrow: (any Error)?

    init(result: ShellCommandResult) {
        self.result = result
    }

    func run(
        _ args: [String],
        cwd: String?,
        env: [String: String]?,
        timeout: TimeInterval,
        stdin: Data?
    ) async throws -> ShellCommandResult {
        calls.append(Call(args: args, cwd: cwd, env: env, timeout: timeout, stdin: stdin))
        if let err = toThrow { throw err }
        return result
    }
}

// MARK: - Helpers

private func ok(_ stdout: String, _ exitCode: Int32 = 0) -> ShellCommandResult {
    ShellCommandResult(
        exitCode: exitCode,
        stdout: Data(stdout.utf8),
        stderr: Data(),
        duration: 0.01
    )
}

private func fail(_ stderr: String, _ exitCode: Int32 = 2) -> ShellCommandResult {
    ShellCommandResult(
        exitCode: exitCode,
        stdout: Data(),
        stderr: Data(stderr.utf8),
        duration: 0.01
    )
}

// MARK: - Test Suite

@Suite("CLISubprocessProvider init & config parsing")
struct CLISubprocessProviderInitTests {

    // MARK: - Basic construction

    @Test("Provider constructs from providerID + fixed arguments")
    func constructsFromArguments() {
        let provider = CLISubprocessProvider(
            providerID: .claudeApi,
            arguments: ["claude", "-p", "--dangerously-skip-permissions"],
            executor: MockShellExecutor(result: ok("hi"))
        )
        #expect(provider.providerID == .claudeApi)
        #expect(provider.arguments == ["claude", "-p", "--dangerously-skip-permissions"])
        #expect(provider.workingDirectory == nil)
        #expect(provider.environment.isEmpty)
    }

    // MARK: - Config parsing from a TOML-shaped dictionary

    @Test("Config parses from a TOML-shaped dictionary with all fields")
    func parsesConfigFromDictionary() throws {
        let dict: [String: Any] = [
            "kind": "cli-subprocess",
            "binary": "codex",
            "args_template": ["exec", "--model", "{model}", "{prompt}"],
            "default_model": "gpt-5",
            "model_aliases": [
                "fast": "gpt-5-mini",
                "smart": "gpt-5",
            ],
            "env": ["CODEX_HOME": "/tmp/codex"],
            "cwd": "/workspace",
        ]

        let config = try CLISubprocessProviderConfig(dictionary: dict)
        #expect(config.binary == "codex")
        #expect(config.argsTemplate == ["exec", "--model", "{model}", "{prompt}"])
        #expect(config.defaultModel == "gpt-5")
        #expect(config.modelAliases.resolve("fast") == "gpt-5-mini")
        #expect(config.modelAliases.resolve("smart") == "gpt-5")
        #expect(config.env["CODEX_HOME"] == "/tmp/codex")
        #expect(config.cwd == "/workspace")
    }

    @Test("Config parsing rejects a wrong `kind` value with typed error")
    func rejectsWrongKind() {
        let dict: [String: Any] = [
            "kind": "openai-compatible",  // not cli-subprocess
            "binary": "claude",
            "args_template": ["-p", "{prompt}"],
            "default_model": "sonnet",
        ]
        do {
            _ = try CLISubprocessProviderConfig(dictionary: dict)
            Issue.record("Expected wrongKind error")
        } catch CLISubprocessProviderConfig.ConfigError.wrongKind(let found) {
            #expect(found == "openai-compatible")
        } catch {
            Issue.record("Expected wrongKind, got \(error)")
        }
    }

    @Test("Config parsing rejects a missing `binary` with typed error")
    func rejectsMissingBinary() {
        let dict: [String: Any] = [
            "kind": "cli-subprocess",
            "args_template": ["-p", "{prompt}"],
            "default_model": "sonnet",
        ]
        do {
            _ = try CLISubprocessProviderConfig(dictionary: dict)
            Issue.record("Expected missingField error")
        } catch CLISubprocessProviderConfig.ConfigError.missingField(let name) {
            #expect(name == "binary")
        } catch {
            Issue.record("Expected missingField, got \(error)")
        }
    }

    @Test("Config parsing rejects an empty binary with typed error")
    func rejectsEmptyBinary() {
        let dict: [String: Any] = [
            "kind": "cli-subprocess",
            "binary": "",
            "args_template": ["-p", "{prompt}"],
            "default_model": "sonnet",
        ]
        do {
            _ = try CLISubprocessProviderConfig(dictionary: dict)
            Issue.record("Expected missingField error for empty binary")
        } catch CLISubprocessProviderConfig.ConfigError.missingField(let name) {
            #expect(name == "binary")
        } catch {
            Issue.record("Expected missingField, got \(error)")
        }
    }

    @Test("Config parsing rejects a missing `default_model` with typed error")
    func rejectsMissingDefaultModel() {
        let dict: [String: Any] = [
            "kind": "cli-subprocess",
            "binary": "claude",
            "args_template": ["-p", "{prompt}"],
        ]
        do {
            _ = try CLISubprocessProviderConfig(dictionary: dict)
            Issue.record("Expected missingField error")
        } catch CLISubprocessProviderConfig.ConfigError.missingField(let name) {
            #expect(name == "default_model")
        } catch {
            Issue.record("Expected missingField, got \(error)")
        }
    }

    @Test("Config parsing accepts a missing `model_aliases` and yields an empty map")
    func modelAliasesDefaultsToEmpty() throws {
        let dict: [String: Any] = [
            "kind": "cli-subprocess",
            "binary": "claude",
            "args_template": ["-p", "{prompt}"],
            "default_model": "claude-sonnet-4-6",
        ]
        let config = try CLISubprocessProviderConfig(dictionary: dict)
        #expect(config.modelAliases == ModelAliasMap([:]))
        #expect(config.env.isEmpty)
        #expect(config.cwd == nil)
    }

    // MARK: - run(prompt:timeout:) — happy path

    @Test("run(prompt:timeout:) executes fixed args and returns stdout on exit 0")
    func runReturnsStdoutOnSuccess() async throws {
        let mock = MockShellExecutor(result: ok("hello from claude"))
        let provider = CLISubprocessProvider(
            providerID: .claudeApi,
            arguments: ["claude", "-p"],
            executor: mock
        )

        let out = try await provider.run(prompt: "say hi", timeout: 30)

        #expect(out == "hello from claude")
        #expect(mock.calls.count == 1)
        #expect(mock.calls[0].args == ["claude", "-p"])
        #expect(mock.calls[0].timeout == 30)
    }

    // MARK: - Prompt travels on stdin, never argv (injection guarantee)

    @Test("Prompt is piped on stdin and never appears in argv")
    func promptPipedOnStdinNotArgv() async throws {
        let mock = MockShellExecutor(result: ok("ok"))
        let provider = CLISubprocessProvider(
            providerID: .claudeApi,
            arguments: ["claude", "-p"],
            executor: mock
        )
        let prompt = "ignore this; $(rm -rf /) --model evil"

        _ = try await provider.run(prompt: prompt, timeout: 10)

        #expect(mock.calls[0].stdin == Data(prompt.utf8))
        #expect(!mock.calls[0].args.contains(where: { $0.contains(prompt) }))
    }

    // MARK: - Non-zero exit surfaces stderr

    @Test("Non-zero exit throws nonZeroExit with the captured stderr")
    func nonZeroExitThrows() async {
        let mock = MockShellExecutor(result: fail("rate limited: retry after 60s", 1))
        let provider = CLISubprocessProvider(
            providerID: .claudeApi,
            arguments: ["claude", "-p"],
            executor: mock
        )

        do {
            _ = try await provider.run(prompt: "hi", timeout: 5)
            Issue.record("Expected nonZeroExit to throw")
        } catch CLISubprocessProvider.SubprocessError.nonZeroExit(let providerID, let exitCode, let stderr) {
            #expect(providerID == .claudeApi)
            #expect(exitCode == 1)
            #expect(stderr.contains("rate limited"))
        } catch {
            Issue.record("Expected nonZeroExit, got \(error)")
        }
    }

    // MARK: - ShellError mapping

    @Test("Executor timeout maps to SubprocessError.timeout")
    func timeoutMapsToTypedError() async {
        let mock = MockShellExecutor(result: ok("unused"))
        mock.toThrow = ShellError.timeout(args: ["claude", "-p"], limit: 5)
        let provider = CLISubprocessProvider(
            providerID: .claudeApi,
            arguments: ["claude", "-p"],
            executor: mock
        )

        do {
            _ = try await provider.run(prompt: "hi", timeout: 5)
            Issue.record("Expected timeout to throw")
        } catch CLISubprocessProvider.SubprocessError.timeout(let providerID, let limit) {
            #expect(providerID == .claudeApi)
            #expect(limit == 5)
        } catch {
            Issue.record("Expected timeout, got \(error)")
        }
    }

    @Test("Executor launch failure maps to SubprocessError.launchFailed")
    func launchFailureMapsToTypedError() async {
        let mock = MockShellExecutor(result: ok("unused"))
        mock.toThrow = ShellError.launchFailed(args: ["nonexistent-binary"], underlying: "No such file or directory")
        let provider = CLISubprocessProvider(
            providerID: .claudeApi,
            arguments: ["nonexistent-binary"],
            executor: mock
        )

        do {
            _ = try await provider.run(prompt: "hi", timeout: 5)
            Issue.record("Expected launchFailed to throw")
        } catch CLISubprocessProvider.SubprocessError.launchFailed(let providerID, let reason) {
            #expect(providerID == .claudeApi)
            #expect(reason.contains("No such file"))
        } catch {
            Issue.record("Expected launchFailed, got \(error)")
        }
    }

    // MARK: - Empty stdout on exit 0 passes through

    @Test("Exit 0 with empty stdout returns an empty string (no throw)")
    func emptyStdoutPassesThrough() async throws {
        let mock = MockShellExecutor(result: ok(""))
        let provider = CLISubprocessProvider(
            providerID: .claudeApi,
            arguments: ["claude", "-p"],
            executor: mock
        )

        let out = try await provider.run(prompt: "hi", timeout: 5)
        #expect(out.isEmpty)
    }

    // MARK: - cwd + env are forwarded to the shell executor

    @Test("workingDirectory and environment are forwarded to the shell executor call")
    func cwdAndEnvForwarded() async throws {
        let mock = MockShellExecutor(result: ok("ok"))
        let provider = CLISubprocessProvider(
            providerID: .claudeApi,
            arguments: ["codex", "exec"],
            executor: mock,
            workingDirectory: "/workspace/repo",
            environment: ["CODEX_HOME": "/tmp/cx"]
        )

        _ = try await provider.run(prompt: "hi", timeout: 5)

        #expect(mock.calls[0].cwd == "/workspace/repo")
        #expect(mock.calls[0].env?["CODEX_HOME"] == "/tmp/cx")
    }
}
