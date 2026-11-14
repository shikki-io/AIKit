// CIScopeCoverageTests — the macOS pipeline gates every scope this repository
// declares, except the ones that are opt-in by declaration.
//
// Operator on PR #18, against `.woodpecker/macos-build.yml`: *"I don't see
// kagami cli in the next command you use like in other project where I have
// something like this: `kagami test --all --strict`"*.
//
// Answering that uncovered the real defect: the pipeline named `AIKitAll` and
// `AIKitPublicSurface` and nothing else, so five declared scopes —
// `AgentSessionIndex`, `AIKitProviders`, `AIKitNoLocalhostFallback`,
// `AIKitServerAddress` and `AIKitMLX` — were gated by no pipeline at all. Four
// of those five are ratchets this epic added, which means the ratchets were
// enforced on the operator's laptop and nowhere else.
//
// `kagami test --all --strict` is the form that cannot drift, and it is what
// the operator asked for, but it is not available here: `--all` iterates every
// DECLARED scope and does not read `pre_pr_default` (kagami 1.10.0,
// `runAllScopes`). This repository declares two scopes that need a running LM
// Studio server and a real audio loopback, so `--all` on a CI agent is red
// forever for a reason that is not the code.
//
// So the pipeline names its scopes, and this test is what makes a hand-written
// list safe: a scope added to `.kagami/scopes.override.toml` without
// `pre_pr_default = false` and without a line in the pipeline fails here, in
// `AIKitAll`, with both names printed. It is the ratchet that stops item 3 of
// that file's header happening a second time.
//
// It reads the registry with TOMLKit — the same decoder the product uses for
// the providers document — rather than grepping it, because a scope name is a
// TOML key and a grep for `name = ` would also match the description strings
// and the `[[scopes]]` blocks' other fields.
//
// The pipeline side IS a text scan, deliberately: the thing being measured is
// the `--scope` ARGUMENT of a `kagami test` command line, which is what the
// agent will actually run. Parsing the YAML into a document and walking it
// would measure the file's shape; reading the command line measures the
// command.

import Foundation
import Testing
import TOMLKit

/// One `[[scopes]]` entry, reduced to the two fields this rule needs.
///
/// At file scope rather than nested in the suite: SwiftLint's `nesting` rule
/// caps a type at one level deep, and the registry is a document of documents.
private struct RegistryScope: Decodable {
    let name: String
    let prePrDefault: Bool?

    enum CodingKeys: String, CodingKey {
        case name
        case prePrDefault = "pre_pr_default"
    }
}

/// The `.kagami/scopes.override.toml` document.
private struct Registry: Decodable {
    let scopes: [RegistryScope]
}

@Suite("The CI gate runs every scope the registry declares (PR #18 review)")
struct CIScopeCoverageTests {

    /// The package root, from this file's own path — never a CWD, which under
    /// `swift test` is the caller's and not the package's.
    private static let packageRoot = URL(filePath: #filePath)
        .deletingLastPathComponent()  // AIKitTests
        .deletingLastPathComponent()  // Tests
        .deletingLastPathComponent()  // package root

    /// Scope names the registry declares, split into the ones a gate must run
    /// and the ones that are opt-in by their own declaration.
    private static func declaredScopes() throws -> (gated: Set<String>, optIn: Set<String>) {
        let url = packageRoot.appending(path: ".kagami/scopes.override.toml")
        let text = try String(contentsOf: url, encoding: .utf8)
        let registry = try TOMLDecoder().decode(Registry.self, from: text)
        // A registry that decodes to nothing would pass every assertion below
        // vacuously, which is the 0|0|0|0 shape in test form. Fail instead.
        #expect(registry.scopes.count >= 5, "the registry decoded \(registry.scopes.count) scopes — the file or the decode is wrong, not the gate")
        var gated: Set<String> = []
        var optIn: Set<String> = []
        for scope in registry.scopes {
            // Absent means "on": that is `pre_pr_default`'s own default, so a
            // new scope is gated unless it says otherwise.
            if scope.prePrDefault == false { optIn.insert(scope.name) } else { gated.insert(scope.name) }
        }
        return (gated, optIn)
    }

    /// Every scope named in a `kagami test --scope <list>` command in the
    /// macOS pipeline.
    private static func scopesGatedByCI() throws -> Set<String> {
        let url = packageRoot.appending(path: ".woodpecker/macos-build.yml")
        let text = try String(contentsOf: url, encoding: .utf8)
        var found: Set<String> = []
        for rawLine in text.split(separator: "\n", omittingEmptySubsequences: true) {
            let line = rawLine.trimmingCharacters(in: .whitespaces)
            // A command, not a comment about one: the header of that file
            // quotes `kagami test --all --strict` while explaining why it is
            // not used, and a scan that counted comments would read the
            // explanation as the gate.
            guard !line.hasPrefix("#"), line.contains("kagami test"), line.contains("--scope") else { continue }
            let fields = line.split(separator: " ").map(String.init)
            guard let index = fields.firstIndex(of: "--scope"), index + 1 < fields.count else { continue }
            for name in fields[index + 1].split(separator: ",") {
                found.insert(String(name).trimmingCharacters(in: .whitespaces))
            }
        }
        return found
    }

    @Test("every declared scope without pre_pr_default = false is named in the macOS pipeline")
    func theGateCoversTheRegistry() throws {
        let (gated, optIn) = try Self.declaredScopes()
        let inCI = try Self.scopesGatedByCI()

        // The scan read the file: without this, a rename of the pipeline or of
        // `kagami test` would make the rule vacuous and it would pass.
        #expect(!inCI.isEmpty, "no `kagami test --scope` command found in .woodpecker/macos-build.yml — the scan is reading the wrong file or the wrong command")

        let missing = gated.subtracting(inCI).sorted()
        #expect(
            missing.isEmpty,
            """
            these scopes are declared in .kagami/scopes.override.toml but no \
            `kagami test --scope` line in .woodpecker/macos-build.yml runs them: \
            \(missing.joined(separator: ", ")). Add them to a step's --scope list, \
            or declare `pre_pr_default = false` on the scope with the reason it \
            cannot run on a CI agent.
            """
        )

        // The other direction: a scope the pipeline names must exist. A typo
        // in the list is a scope kagami resolves to nothing, and `--strict`
        // catches that at run time — but only once the pipeline is registered,
        // which it is not yet. This catches it here.
        let unknown = inCI.subtracting(gated).subtracting(optIn).sorted()
        #expect(
            unknown.isEmpty,
            "the macOS pipeline names scopes the registry does not declare: \(unknown.joined(separator: ", "))"
        )
    }

    @Test("the two live scopes are the only ones the gate leaves out, and they say why")
    func onlyTheLiveScopesAreOptIn() throws {
        let (_, optIn) = try Self.declaredScopes()
        // Named rather than counted: a third opt-in scope should be a decision
        // someone makes here, not a number that quietly grows.
        #expect(
            optIn == ["AIKitLive", "AIKitVoiceLive", "AIKitMLX", "AIKitVision"],
            """
            the set of opt-in (`pre_pr_default = false`) scopes changed to \
            \(optIn.sorted().joined(separator: ", ")). AIKitLive needs a running LM \
            Studio server and AIKitVoiceLive needs an audio loopback, so neither can \
            run on a CI agent; AIKitMLX and AIKitVision need the selected Xcode's \
            MetalToolchain, so they are opt-in for a laptop but ARE gated in CI beside \
            AIKitPublicSurface, which builds AIKitMLX too. A new entry here needs the \
            same kind of reason written down.
            """
        )
    }
}
