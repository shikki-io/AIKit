import Foundation
import Testing
@testable import AIKit

/// Spec c8fa126c W5, and the test the spec calls **T-L-11**: *"AIKit without a
/// ShellKit-backed provider builds and runs."*
///
/// The spec expected one failure here — "`ShellKit` being *called* where the
/// condition only excluded it at *link* time". The real state of `develop` was
/// worse and in a different place, and all of it was invisible because no build
/// of AIKit had ever targeted a non-Apple triple:
///
/// 1. `FJ-Studios/NetKit` 2.x was pinned, which transitively resolved **CoreKit
///    0.2.1** — fifteen minors behind, with an unguarded `import os`. Every
///    Linux build of AIKit failed inside a dependency nobody had looked at.
/// 2. ShellKit is linked **unconditionally** on `develop` (the
///    `.when(platforms: [.macOS])` condition the spec quotes lives on an
///    unmerged epic branch), so its two `Darwin.kill(...)` calls were AIKit's
///    blocker too.
/// 3. Four AIKit files used `URLSession` with no `canImport(FoundationNetworking)`
///    guard, so off Apple `URLSession` resolved to `AnyObject`.
/// 4. `ProviderHealthCheck` measured latency with `CFAbsoluteTimeGetCurrent`.
///
/// This suite pins 3 and 4, which are AIKit's own, by source scan — 1 and 2 are
/// manifest facts the cross build itself measures.
@Suite("AIKit portability ratchet")
struct AIKitPortabilityRatchetTests {
    private static func sourcesRoot() -> URL {
        var dir = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        while dir.path != "/" {
            if FileManager.default.fileExists(atPath: dir.appending(path: "Package.swift").path) {
                return dir.appending(path: "Sources/AIKit", directoryHint: .isDirectory)
            }
            dir = dir.deletingLastPathComponent()
        }
        fatalError("no Package.swift above \(#filePath)")
    }

    /// Every file that names a URL-loading type must guard for
    /// `FoundationNetworking`. This is the one conditional import spec c8fa126c
    /// declares CORRECT — it is upstream Foundation's own split — so the ratchet
    /// REQUIRES it rather than banning it, which is the opposite polarity from
    /// the Apple-framework scans and the reason it needs its own test.
    @Test("every URL-loading file guards for FoundationNetworking (T-L-11)")
    func urlLoadingFilesGuardForFoundationNetworking() throws {
        let urlTypes = ["URLSession", "URLSessionDownloadTask", "URLSessionDownloadDelegate", "HTTPURLResponse"]
        var offenders: [String] = []
        for file in try Self.swiftFiles(under: Self.sourcesRoot()) {
            let lines = try Self.lines(of: file)
            let usesURLLoading = lines.contains { line in
                let trimmed = line.trimmingCharacters(in: .whitespaces)
                guard !trimmed.hasPrefix("//") else { return false }
                return urlTypes.contains { trimmed.contains($0) }
            }
            guard usesURLLoading else { continue }
            let guarded = lines.contains { $0.contains("canImport(FoundationNetworking)") }
            if !guarded { offenders.append(file.lastPathComponent) }
        }
        #expect(
            offenders.isEmpty,
            """
            URL-loading files with no `#if canImport(FoundationNetworking)`: \(offenders).
            Off Apple, URLSession lives in its own module and resolves to AnyObject without it.
            """
        )
    }

    @Test("no CoreFoundation and no Apple-only framework in AIKit")
    func noCoreFoundationOrAppleFramework() throws {
        // `CFAbsoluteTimeGetCurrent` was the one hit, in a latency measurement.
        // It is matched as a CALL, not as a bare name, so a comment explaining
        // the replacement is not a violation.
        let bannedCalls = ["CFAbsoluteTimeGetCurrent(", "CFAbsoluteTime(", "CFRunLoop"]
        let bannedImports = ["CoreFoundation", "SwiftUI", "UIKit", "AppKit", "os", "Combine", "Network", "CryptoKit", "Security"]

        var offenders: [String] = []
        for file in try Self.swiftFiles(under: Self.sourcesRoot()) {
            for (index, raw) in try Self.lines(of: file).enumerated() {
                let line = raw.trimmingCharacters(in: .whitespaces)
                guard !line.hasPrefix("//") else { continue }
                if line.hasPrefix("import ") {
                    let module = line.dropFirst("import ".count).trimmingCharacters(in: .whitespaces)
                    if bannedImports.contains(module) {
                        offenders.append("\(file.lastPathComponent):\(index + 1) import \(module)")
                    }
                    continue
                }
                for call in bannedCalls where line.contains(call) {
                    offenders.append("\(file.lastPathComponent):\(index + 1) \(call)")
                }
            }
        }
        #expect(offenders.isEmpty, "non-portable API in AIKit: \(offenders)")
    }

    /// The replacement is monotonic, and that is a behaviour change worth a test
    /// of its own: `CFAbsoluteTimeGetCurrent` reads WALL time, so an NTP step
    /// during a request landed straight in the reported latency.
    @Test("latency is measured in milliseconds from a Duration, correctly")
    func latencyConversionIsCorrect() {
        #expect(ProviderHealthCheck.milliseconds(.seconds(1)) == 1000.0)
        #expect(ProviderHealthCheck.milliseconds(.milliseconds(250)) == 250.0)
        #expect(ProviderHealthCheck.milliseconds(.zero) == 0.0)
        // Sub-millisecond resolution survives the attosecond conversion.
        #expect(abs(ProviderHealthCheck.milliseconds(.microseconds(500)) - 0.5) < 1e-9)
        // And it is monotonic in its argument.
        #expect(ProviderHealthCheck.milliseconds(.seconds(2)) > ProviderHealthCheck.milliseconds(.seconds(1)))
    }

    /// T-L-11's literal reading: AIKit's surface is reachable without a
    /// ShellKit-backed provider ever being constructed.
    @Test("AIKit's own types construct with no CLI provider in play (T-L-11)")
    func aiKitWorksWithoutAShellBackedProvider() async throws {
        let mock = MockAIProvider()
        let request = AIRequest(messages: [AIMessage(role: .user, content: "ping")])
        let response = try await mock.complete(request: request)
        #expect(!response.content.isEmpty)
    }

    // MARK: - Scanner

    private static func lines(of file: URL) throws -> [String] {
        try String(contentsOf: file, encoding: .utf8).components(separatedBy: .newlines)
    }

    private static func swiftFiles(under root: URL) throws -> [URL] {
        guard let walker = FileManager.default.enumerator(at: root, includingPropertiesForKeys: nil) else { return [] }
        return walker.compactMap { $0 as? URL }.filter { $0.pathExtension == "swift" }
    }
}
