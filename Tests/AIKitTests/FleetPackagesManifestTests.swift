import Foundation
import Testing

/// T-AIL-38, W19 of spec ai-layer-aikit-core-2026-10-31: AIKit builds on the
/// packages the fleet ships today, before the AIWire rework (W20-W26) builds on
/// them.
///
/// The manifest is the thing under test, so the suite reads `Package.swift` and
/// `Package.resolved` from the package root. The iOS half of T-AIL-38 (ShellKit
/// stays out of the iOS link, TOMLKit and CoreKit link for iOS) is the wave's
/// `xcodebuild -destination 'generic/platform=iOS'` gate; this suite pins the
/// declarations that gate depends on.
@Suite("Fleet packages in the manifest (W19)")
struct FleetPackagesManifestTests {

    /// One fleet dependency as the manifest must declare it: the canonical URL
    /// and the `from:` floor.
    struct Requirement: Sendable, CustomTestStringConvertible {
        let identity: String
        let url: String
        let floor: String

        var testDescription: String { "\(identity) \(url) from \(floor)" }
    }

    static let requirements: [Requirement] = [
        Requirement(identity: "netkit", url: "https://github.com/obyw-one/NetKit.git", floor: "3.0.0"),
        Requirement(identity: "corekit", url: "https://github.com/obyw-one/CoreKit.git", floor: "0.16.0"),
        Requirement(identity: "shellkit", url: "https://github.com/shikki-io/ShellKit", floor: "0.5.1"),
        Requirement(identity: "tomlkit", url: "https://github.com/LebJe/TOMLKit.git", floor: "0.5.5"),
    ]

    private static let packageRoot = URL(filePath: #filePath)
        .deletingLastPathComponent()  // AIKitTests
        .deletingLastPathComponent()  // Tests
        .deletingLastPathComponent()  // package root

    private static func manifest() throws -> String {
        try String(contentsOf: packageRoot.appending(path: "Package.swift"), encoding: .utf8)
    }

    private static func resolved() throws -> String {
        try String(contentsOf: packageRoot.appending(path: "Package.resolved"), encoding: .utf8)
    }

    @Test("Each fleet package is declared at its canonical URL with a from: floor", arguments: requirements)
    func declaresTheFleetPackage(_ requirement: Requirement) throws {
        let manifest = try Self.manifest()
        let declaration = ".package(url: \"\(requirement.url)\", from: \"\(requirement.floor)\")"
        #expect(manifest.contains(declaration), "Package.swift must declare \(declaration)")
    }

    @Test("Package.resolved pins each fleet package at its floor or above", arguments: requirements)
    func resolvesTheFleetPackage(_ requirement: Requirement) throws {
        let data = try Data(contentsOf: Self.packageRoot.appending(path: "Package.resolved"))
        let lock = try JSONDecoder().decode(ResolvedFile.self, from: data)
        let pin = try #require(
            lock.pins.first { $0.identity == requirement.identity },
            "Package.resolved has no pin for \(requirement.identity)"
        )
        #expect(pin.location == requirement.url, "\(requirement.identity) resolves from \(pin.location)")
        let version = try #require(pin.state.version, "\(requirement.identity) is not pinned to a version")
        #expect(
            Self.isAtLeast(version, requirement.floor),
            "\(requirement.identity) resolved \(version), below \(requirement.floor)"
        )
    }

    @Test("No FJ-Studios URL is left in the package")
    func noFJStudios() throws {
        #expect(!(try Self.manifest()).contains("FJ-Studios"))
        #expect(!(try Self.resolved()).contains("FJ-Studios"))
    }

    @Test("ShellKit is linked only where Process exists, never on iOS (BR-AIL-02)")
    func shellKitIsPlatformConditioned() throws {
        let manifest = try Self.manifest()
        let uses = manifest.components(separatedBy: "\n")
            .filter { $0.contains(".product(name: \"ShellKit\"") }
        #expect(!uses.isEmpty, "the AIKit target still links ShellKit on macOS")
        for use in uses {
            #expect(use.contains("condition: .when(platforms: [.macOS, .linux])"), "unconditioned: \(use)")
        }
    }

    @Test("The AIKit target links CoreKit and TOMLKit, so the iOS gate proves they build")
    func aiKitTargetLinksTheNewPackages() throws {
        let manifest = try Self.manifest()
        #expect(manifest.contains(".product(name: \"CoreKit\", package: \"CoreKit\")"))
        #expect(manifest.contains(".product(name: \"TOMLKit\", package: \"TOMLKit\")"))
    }

    // MARK: - Package.resolved (format v2 and v3)

    private struct ResolvedFile: Decodable {
        let pins: [Pin]
    }

    private struct Pin: Decodable {
        let identity: String
        let location: String
        let state: State
    }

    private struct State: Decodable {
        let version: String?
    }

    private static func isAtLeast(_ version: String, _ floor: String) -> Bool {
        let parts = { (value: String) in
            value.split(separator: ".").map { Int($0.prefix { $0.isNumber }) ?? 0 }
        }
        let lhs = parts(version)
        let rhs = parts(floor)
        for index in 0..<max(lhs.count, rhs.count) {
            let left = index < lhs.count ? lhs[index] : 0
            let right = index < rhs.count ? rhs[index] : 0
            if left != right { return left > right }
        }
        return true
    }
}
