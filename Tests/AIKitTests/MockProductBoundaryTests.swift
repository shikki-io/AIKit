import AIKitTestSupport
import Foundation
import Testing
@testable import AIKit

/// T-AIL-51, W26 of spec ai-layer-aikit-core-2026-10-31: the test doubles are
/// not in the shipping product, and the lint suppression list only shrinks.
///
/// Operator, AIKit#18 r4500328752, on `MockAIProvider`: "Can you explicit with
/// a log you are using a mock and it's not supposed to be used for running ai
/// provider when we are in prod". The log is in ``MockAIProvider`` itself; this
/// suite pins the half a log cannot give, which is that a shipping target
/// CANNOT REACH THE TYPE AT ALL. A log fires after the mistake; a product
/// boundary prevents it, and only the boundary can be asserted.
///
/// Why these assertions and not a build of a downstream app: the claim is
/// "`import AIKit` does not bring a mock", and the three things that make it
/// true are the module a mock is declared in, the absence of any mock in the
/// `AIKit` target's sources, and the manifest that keeps them apart. Each is
/// read here from the artifact, not restated from the spec.
@Suite("Mocks are outside the AIKit product (W26)")
struct MockProductBoundaryTests {

    private static let packageRoot = URL(filePath: #filePath)
        .deletingLastPathComponent()  // AIKitTests
        .deletingLastPathComponent()  // Tests
        .deletingLastPathComponent()  // package root

    // MARK: - Where each double is declared

    /// `String(reflecting:)` yields a type's fully qualified name, so it names
    /// the module that actually declared it in THIS build — not the module the
    /// import list suggests.
    @Test(
        "Every test double is declared in AIKitTestSupport, not AIKit",
        arguments: [
            (String(reflecting: MockAIProvider.self), "MockAIProvider"),
            (String(reflecting: MockRuntimeEngine.self), "MockRuntimeEngine"),
            (String(reflecting: MockModelRegistry.self), "MockModelRegistry"),
        ]
    )
    func doubleLivesInTestSupport(_ qualifiedName: String, _ simpleName: String) {
        #expect(
            qualifiedName == "AIKitTestSupport.\(simpleName)",
            "\(simpleName) must be declared in AIKitTestSupport; it reports \(qualifiedName)"
        )
        #expect(
            qualifiedName != "AIKit.\(simpleName)",
            "\(simpleName) is back inside the AIKit product — an app linking AIKit can construct it again"
        )
    }

    // MARK: - The AIKit target's own sources

    /// A ratchet, not a restatement: it walks the target's real source tree.
    /// Moving a mock back under `Sources/AIKit/` fails here even if the
    /// manifest still looks right.
    @Test("No file in the AIKit target declares a mock type")
    func aiKitTargetDeclaresNoMock() throws {
        let sources = Self.packageRoot.appending(path: "Sources/AIKit")
        let enumerator = try #require(
            FileManager.default.enumerator(at: sources, includingPropertiesForKeys: nil),
            "Sources/AIKit must exist"
        )

        var offenders: [String] = []
        for case let url as URL in enumerator where url.pathExtension == "swift" {
            let text = try String(contentsOf: url, encoding: .utf8)
            let relative = url.path.replacingOccurrences(of: Self.packageRoot.path + "/", with: "")
            for keyword in ["class Mock", "struct Mock", "actor Mock", "enum Mock"]
            where text.contains(keyword) {
                offenders.append("\(relative) declares a `\(keyword)…` type")
            }
        }

        #expect(
            offenders.isEmpty,
            """
            The AIKit target must declare no test double — they belong to AIKitTestSupport:
            \(offenders.joined(separator: "\n"))
            """
        )
    }

    /// Emptiness rather than absence: git tracks files, not directories, so a
    /// `git mv` out of `Sources/AIKit/Mocks/` leaves the folder behind in the
    /// working tree that performed the move while a fresh clone never has it.
    /// Asserting it does not exist would therefore go red on the one machine
    /// that did the work and green everywhere else — a test about the
    /// developer's filesystem, not about the package.
    @Test("The Mocks directory holds no source in the AIKit target")
    func mocksDirectoryHoldsNothing() throws {
        let mocks = Self.packageRoot.appending(path: "Sources/AIKit/Mocks")
        guard FileManager.default.fileExists(atPath: mocks.path) else { return }
        let contents = try FileManager.default.contentsOfDirectory(atPath: mocks.path)
        let sources = contents.filter { $0.hasSuffix(".swift") }
        #expect(
            sources.isEmpty,
            """
            Sources/AIKit/Mocks still holds \(sources.joined(separator: ", ")) — \
            the doubles belong to Sources/AIKitTestSupport
            """
        )
    }

    // MARK: - The manifest

    /// Read with whitespace collapsed: the claim is about the declarations,
    /// and a formatter is free to move the line breaks.
    private static func manifest() throws -> String {
        let text = try String(
            contentsOf: packageRoot.appending(path: "Package.swift"), encoding: .utf8)
        return text.split(whereSeparator: \.isWhitespace).joined(separator: " ")
    }

    @Test("The manifest declares AIKitTestSupport as its own product")
    func manifestDeclaresTestSupportProduct() throws {
        let manifest = try Self.manifest()
        #expect(
            manifest.contains(#".library( name: "AIKitTestSupport", targets: ["AIKitTestSupport"] )"#),
            "Package.swift must declare an AIKitTestSupport library product"
        )
    }

    /// The point of the separation: a consumer of `AIKit` must not acquire the
    /// doubles transitively, so the `AIKit` target cannot depend on them.
    @Test("The AIKit target does not depend on AIKitTestSupport")
    func aiKitDoesNotDependOnTestSupport() throws {
        let manifest = try Self.manifest()
        let declaration = try #require(
            manifest.range(of: #".target( name: "AIKit", dependencies: ["#),
            "Package.swift must declare a target named AIKit"
        )
        // The target's declaration ends where the next target's begins.
        let rest = manifest[declaration.upperBound...]
        let end =
            rest.range(of: ".target(")?.lowerBound
            ?? rest.range(of: ".testTarget(")?.lowerBound
            ?? rest.endIndex
        #expect(
            !rest[..<end].contains("AIKitTestSupport"),
            "The AIKit target must not name AIKitTestSupport: a consumer would link the mocks"
        )
    }

    // MARK: - The SwiftLint suppression list

    /// Operator, AIKit#18 r4500611478, on `.swiftlint-baseline.json`: "Did we
    /// cannot reduce the linter warnings ?"
    ///
    /// Yes: W26 fixed all 10 violations the config still reported and the
    /// baseline is gone, so there is no suppression list left to grow. The
    /// rule is that it only goes down, and zero is the floor — a file
    /// reappearing means a warning was parked instead of fixed.
    @Test("No SwiftLint baseline file: the suppression list went to zero")
    func noSwiftLintBaselineRemains() {
        let baseline = Self.packageRoot.appending(path: ".swiftlint-baseline.json")
        #expect(
            !FileManager.default.fileExists(atPath: baseline.path),
            """
            .swiftlint-baseline.json is back. The baseline only goes DOWN (W26, r4500611478): \
            fix the violation instead of parking it.
            """
        )
    }

    @Test("The SwiftLint config declares no baseline")
    func configDeclaresNoBaseline() throws {
        let config = try String(
            contentsOf: Self.packageRoot.appending(path: ".swiftlint.yml"), encoding: .utf8)
        let declaresBaseline = config
            .split(separator: "\n")
            .contains { $0.hasPrefix("baseline:") }
        #expect(
            !declaresBaseline,
            ".swiftlint.yml must declare no `baseline:` key — there are no violations left to suppress"
        )
    }
}
