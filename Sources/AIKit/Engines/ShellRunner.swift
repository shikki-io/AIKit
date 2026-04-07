import Foundation

#if os(macOS)
/// Runs a shell command asynchronously with stdout/stderr capture.
/// Used by local engine wrappers (MLX, whisper, etc.) to shell-out to Python/CLI tools.
public enum ShellRunner {

    /// Result of a shell command execution.
    public struct Result: Sendable {
        public let exitCode: Int32
        public let stdout: String
        public let stderr: String

        public init(exitCode: Int32, stdout: String, stderr: String) {
            self.exitCode = exitCode
            self.stdout = stdout
            self.stderr = stderr
        }
    }

    /// Run a command and return the result when it completes.
    public static func run(_ command: String, arguments: [String] = []) async throws -> Result {
        try await withCheckedThrowingContinuation { continuation in
            let process = Process()
            process.executableURL = URL(fileURLWithPath: command)
            process.arguments = arguments

            let stdoutPipe = Pipe()
            let stderrPipe = Pipe()
            process.standardOutput = stdoutPipe
            process.standardError = stderrPipe

            do {
                try process.run()
            } catch {
                continuation.resume(throwing: AIKitError.requestFailed("Failed to launch \(command): \(error.localizedDescription)"))
                return
            }

            process.waitUntilExit()

            let stdoutData = stdoutPipe.fileHandleForReading.readDataToEndOfFile()
            let stderrData = stderrPipe.fileHandleForReading.readDataToEndOfFile()

            let result = Result(
                exitCode: process.terminationStatus,
                stdout: String(data: stdoutData, encoding: .utf8) ?? "",
                stderr: String(data: stderrData, encoding: .utf8) ?? ""
            )
            continuation.resume(returning: result)
        }
    }

    /// Run a command and stream stdout lines via callback.
    public static func stream(
        _ command: String,
        arguments: [String] = [],
        onLine: @Sendable @escaping (String) -> Void
    ) async throws -> Result {
        try await withCheckedThrowingContinuation { continuation in
            let process = Process()
            process.executableURL = URL(fileURLWithPath: command)
            process.arguments = arguments

            let stdoutPipe = Pipe()
            let stderrPipe = Pipe()
            process.standardOutput = stdoutPipe
            process.standardError = stderrPipe

            // Stream stdout lines as they arrive.
            stdoutPipe.fileHandleForReading.readabilityHandler = { handle in
                let data = handle.availableData
                guard !data.isEmpty,
                      let text = String(data: data, encoding: .utf8) else { return }
                let lines = text.components(separatedBy: "\n")
                for line in lines where !line.isEmpty {
                    onLine(line)
                }
            }

            do {
                try process.run()
            } catch {
                stdoutPipe.fileHandleForReading.readabilityHandler = nil
                continuation.resume(throwing: AIKitError.requestFailed("Failed to launch \(command): \(error.localizedDescription)"))
                return
            }

            process.waitUntilExit()
            stdoutPipe.fileHandleForReading.readabilityHandler = nil

            let stderrData = stderrPipe.fileHandleForReading.readDataToEndOfFile()

            let result = Result(
                exitCode: process.terminationStatus,
                stdout: "",
                stderr: String(data: stderrData, encoding: .utf8) ?? ""
            )
            continuation.resume(returning: result)
        }
    }

    /// Check if a command exists in PATH.
    public static func commandExists(_ name: String) -> Bool {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/which")
        process.arguments = [name]
        process.standardOutput = Pipe()
        process.standardError = Pipe()

        do {
            try process.run()
            process.waitUntilExit()
            return process.terminationStatus == 0
        } catch {
            return false
        }
    }
}

#endif // os(macOS)
