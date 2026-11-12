import Foundation
import Testing

@testable import AIKit

// MARK: - CLIProviderPlatformTests
//
// W25 of spec `ai-layer-aikit-core-2026-10-31`, first bullet: "CLISubprocessProvider
// and WhisperCLIProvider compile out cleanly on iOS; on iOS the family resolves
// to a typed `unavailable(platform:)`, never a crash. Their binary paths come
// from `HostExecutable` / settings, cross-platform."
//
// This whole suite is UNGATED on purpose. It is the regression test for the
// hole W1 left: the providers compiled out on iOS, and with them went any way
// to ASK whether they exist. If `CLIProviderFamily`, `HostExecutable` or
// `SearchPathSeparator` ever slip back behind `#if canImport(ShellKit)`, this
// file stops compiling for iOS and the gate's `xcodebuild` catches it.

@Suite("The CLI provider family, on every platform")
struct CLIProviderPlatformTests {

    // MARK: - Availability is typed, never a trap

    @Test("The family resolves to available or unavailable(platform:), and nothing else")
    func availabilityIsTyped() {
        switch CLIProviderFamily.availability {
        case .available:
            // ShellKit is linked, so this host builds CLI providers.
            #expect(CLIProviderFamily.isAvailable)
            #expect([HostPlatform.macOS, .linux].contains(HostPlatform.current))
        case .unavailable(let platform):
            #expect(CLIProviderFamily.isAvailable == false)
            #expect(platform == HostPlatform.current)
            // iOS is the platform this case exists for.
            #expect([HostPlatform.iOS, .windows, .other].contains(platform))
        }
    }

    @Test("requireAvailable either returns or throws the typed case")
    func requireAvailableMatchesAvailability() {
        do {
            try CLIProviderFamily.requireAvailable()
            #expect(CLIProviderFamily.isAvailable, "requireAvailable returned on a host with no ShellKit")
        } catch let error as CLIProviderError {
            // Never a `fatalError` and never an untyped string: a caller can
            // branch on the platform and offer an on-device provider instead.
            #expect(error == .unavailable(platform: HostPlatform.current))
            #expect(CLIProviderFamily.isAvailable == false)
        } catch {
            Issue.record("requireAvailable threw an untyped error: \(error)")
        }
    }

    @Test("The error names the platform and says what to do instead")
    func errorDescriptionNamesThePlatform() throws {
        let error = CLIProviderError.unavailable(platform: .iOS)
        let description = try #require(error.errorDescription)
        #expect(description.contains("iOS"))
        // An error that only says "unavailable" sends the reader to the source.
        #expect(description.contains("on-device"))
    }

    @Test("Every family names the executables it may run")
    func familiesNameTheirExecutables() {
        #expect(CLIProviderFamily.allCases.count == 2)
        #expect(CLIProviderFamily.whisperCLI.executables == [.whisperCLI, .whisperCPP])
        // An operator-defined runner names its own binary in config, so the
        // table has nothing to offer it and does not invent one (BR-AIL-08).
        #expect(CLIProviderFamily.subprocess.executables.isEmpty)
    }

    // MARK: - SearchPathSeparator

    @Test("The PATH list separator is the host's, not a hardcoded colon")
    func searchPathSeparatorIsNative() {
        #if os(Windows)
            #expect(SearchPathSeparator.native == .windows)
            #expect(SearchPathSeparator.native.rawValue == ";")
        #else
            #expect(SearchPathSeparator.native == .posix)
            #expect(SearchPathSeparator.native.rawValue == ":")
        #endif
        // Each raw value is one grapheme, so `character` cannot trap.
        for separator in SearchPathSeparator.allCases {
            #expect(separator.rawValue.count == 1)
            #expect(String(separator.character) == separator.rawValue)
        }
        // The list separator is NOT the path-component separator: on Windows a
        // path is joined with `\` and a list with `;`, and conflating them was
        // the bug.
        #expect(SearchPathSeparator.allCases.map(\.rawValue) == [":", ";"])
    }

    // MARK: - HostExecutable

    @Test("An executable is a name, and its file name follows the host")
    func executableFileName() {
        #if os(Windows)
            #expect(HostExecutable.whisperCLI.fileName == "whisper-cli.exe")
            #expect(HostExecutable.python3.fileName == "python3.exe")
        #else
            #expect(HostExecutable.whisperCLI.fileName == "whisper-cli")
            #expect(HostExecutable.python3.fileName == "python3")
        #endif
        // The raw value stays the bare name everywhere: it is the identity, and
        // the suffix is a host detail.
        #expect(HostExecutable.whisperCLI.rawValue == "whisper-cli")
        #expect(HostExecutable.whisperCPP.rawValue == "whisper-cpp")
    }

    @Test("A path is composed from components, never interpolated")
    func urlIsComposed() {
        let directory = URL(fileURLWithPath: "/opt/homebrew/bin")
        let url = HostExecutable.whisperCLI.url(in: directory)

        #expect(url.lastPathComponent == HostExecutable.whisperCLI.fileName)
        #expect(url.deletingLastPathComponent().path == directory.path)
        #expect(url.isFileURL)
        // Composed means the separator never appears in our source: the path
        // components are the API.
        #expect(url.pathComponents.contains("homebrew"))
        #expect(url.pathComponents.last == HostExecutable.whisperCLI.fileName)
    }

    @Test("An absent or empty PATH yields no directories, not one empty one")
    func searchPathsOnEmptyInput() {
        #expect(HostExecutable.searchPaths(nil).isEmpty)
        #expect(HostExecutable.searchPaths("").isEmpty)
    }

    @Test("PATH splits on the host separator and drops empty entries")
    func searchPathsSplits() {
        let separator = SearchPathSeparator.native.rawValue
        // A doubled separator is what an unset element looks like in a real
        // PATH; it must not become a directory named "".
        let raw = ["/usr/bin", "", "/opt/homebrew/bin"].joined(separator: separator)
        let directories = HostExecutable.searchPaths(raw)

        #expect(directories.map(\.path) == ["/usr/bin", "/opt/homebrew/bin"])
    }

    @Test("resolve finds an executable, prefers the first name, and skips a plain file")
    func resolvePrefersAndSkips() throws {
        let directory = try TempDirectory()

        // Nothing installed.
        #expect(HostExecutable.resolve([.whisperCLI, .whisperCPP], in: [directory.url]) == nil)

        // The legacy name alone is found...
        let legacy = try directory.executable(named: HostExecutable.whisperCPP.fileName)
        #expect(HostExecutable.resolve([.whisperCLI, .whisperCPP], in: [directory.url]) == legacy.path)

        // ...and loses to the preferred name once that exists.
        let current = try directory.executable(named: HostExecutable.whisperCLI.fileName)
        #expect(HostExecutable.resolve([.whisperCLI, .whisperCPP], in: [directory.url]) == current.path)
    }

    @Test("A non-executable file of the right name is not the binary")
    func resolveSkipsNonExecutable() throws {
        let directory = try TempDirectory()
        _ = try directory.file(named: HostExecutable.whisperCLI.fileName, contents: "just text")
        #expect(HostExecutable.resolve([.whisperCLI], in: [directory.url]) == nil)
    }

    @Test("An earlier directory wins over a better name in a later one")
    func earlierDirectoryWins() throws {
        let first = try TempDirectory()
        let second = try TempDirectory()
        // The LESS preferred name, in the EARLIER directory.
        let legacyFirst = try first.executable(named: HostExecutable.whisperCPP.fileName)
        _ = try second.executable(named: HostExecutable.whisperCLI.fileName)

        // A PATH is a precedence list: the directory decides before the name.
        #expect(
            HostExecutable.resolve([.whisperCLI, .whisperCPP], in: [first.url, second.url])
                == legacyFirst.path
        )
    }

    @Test("resolve tolerates a directory that does not exist")
    func resolveToleratesMissingDirectory() {
        let absent = URL(fileURLWithPath: "/nonexistent-aikit-w25-\(UUID().uuidString)")
        #expect(HostExecutable.resolve([.whisperCLI], in: [absent]) == nil)
    }

    // MARK: - The whisper provider reads its names from the table

    #if canImport(ShellKit)

        @Test("WhisperCLIProvider's binary names come from HostExecutable")
        func whisperNamesComeFromTheTable() {
            // Not a second list: the provider's names ARE the table's, so a new
            // whisper.cpp binary name is added in one place.
            #expect(WhisperCLIProvider.executables == CLIProviderFamily.whisperCLI.executables)
            #expect(WhisperCLIProvider.binaryNames == CLIProviderFamily.whisperCLI.executables.map(\.fileName))
        }

        @Test("The default weights directories are composed, not spelled")
        func weightsDirectoriesAreComposed() throws {
            let directories = WhisperCLIProvider.defaultWeightsDirectories
            #expect(directories.count == 2)

            let components = try #require(directories.first).pathComponents
            // The whole point: the path arrives as components, so no element is
            // a slash-bearing literal that would be wrong on another host.
            #expect(components.contains(".local"))
            #expect(components.contains("share"))
            #expect(components.contains("whisper-cpp"))
            #expect(components.last == "models")
            // No element below the root carries a separator. `pathComponents`
            // reports the filesystem root itself as "/", so that first element
            // is the one component allowed to be one.
            #expect(components.first == "/")
            #expect(components.dropFirst().contains(where: { $0.contains("/") }) == false)

            let fallback = try #require(directories.last).pathComponents
            #expect(fallback.contains(".aikit"))
            #expect(fallback.last == "whisper-models")
        }

    #endif

    // MARK: - Fixtures

    /// A directory that lives for one test and takes its files with it.
    final class TempDirectory {
        let url: URL

        init() throws {
            url = URL(fileURLWithPath: NSTemporaryDirectory())
                .appending(path: "aikit-w25-\(UUID().uuidString)", directoryHint: .isDirectory)
            try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        }

        deinit { try? FileManager.default.removeItem(at: url) }

        @discardableResult
        func file(named name: String, contents: String) throws -> URL {
            let fileURL = url.appending(path: name)
            try Data(contents.utf8).write(to: fileURL)
            return fileURL
        }

        @discardableResult
        func executable(named name: String) throws -> URL {
            let fileURL = try file(named: name, contents: "#!/bin/sh\nexit 0\n")
            try FileManager.default.setAttributes(
                [.posixPermissions: 0o755],
                ofItemAtPath: fileURL.path
            )
            return fileURL
        }
    }
}
