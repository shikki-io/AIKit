import Foundation
@testable import AIKit
import Testing

// MARK: - ArgsTemplate Rendering Tests
//
// Verifies pure template rendering (placeholder substitution, error surface).

@Suite("ArgsTemplate rendering")
struct ArgsTemplateRenderingTests {

    // MARK: - Basic substitution

    @Test("Single {prompt} placeholder is replaced with the prompt value")
    func rendersPromptPlaceholder() throws {
        let out = try ArgsTemplate.render(
            template: ["-p", "{prompt}"],
            substitutions: ["prompt": "hello world"]
        )
        #expect(out == ["-p", "hello world"])
    }

    @Test("Single {model} placeholder is replaced with the model value")
    func rendersModelPlaceholder() throws {
        let out = try ArgsTemplate.render(
            template: ["--model", "{model}"],
            substitutions: ["model": "claude-sonnet-4-6"]
        )
        #expect(out == ["--model", "claude-sonnet-4-6"])
    }

    // MARK: - Multiple placeholders

    @Test("Multiple placeholders across positional args are all substituted")
    func rendersMultiplePlaceholders() throws {
        let out = try ArgsTemplate.render(
            template: ["--model", "{model}", "-p", "{prompt}"],
            substitutions: ["model": "opus", "prompt": "hi"]
        )
        #expect(out == ["--model", "opus", "-p", "hi"])
    }

    // MARK: - Embedded placeholder

    @Test("Placeholder embedded in a larger arg is substituted in place")
    func rendersEmbeddedPlaceholder() throws {
        let out = try ArgsTemplate.render(
            template: ["--prompt={prompt}", "--model={model}"],
            substitutions: ["prompt": "say hi", "model": "gemma"]
        )
        #expect(out == ["--prompt=say hi", "--model=gemma"])
    }

    // MARK: - No placeholders

    @Test("Template with no placeholders passes through unchanged")
    func passesThroughLiteralArgs() throws {
        let out = try ArgsTemplate.render(
            template: ["-p", "--json", "--strict"],
            substitutions: ["prompt": "unused"]
        )
        #expect(out == ["-p", "--json", "--strict"])
    }

    // MARK: - Unresolved placeholder

    @Test("Unresolved placeholder throws unresolvedPlaceholder with the missing key")
    func throwsOnUnresolvedPlaceholder() {
        do {
            _ = try ArgsTemplate.render(
                template: ["--model", "{model}", "-p", "{prompt}"],
                substitutions: ["prompt": "hi"]  // missing "model"
            )
            Issue.record("Expected unresolvedPlaceholder to throw")
        } catch ArgsTemplate.RenderError.unresolvedPlaceholder(let key) {
            #expect(key == "model")
        } catch {
            Issue.record("Expected unresolvedPlaceholder, got \(error)")
        }
    }

    // MARK: - Empty template

    @Test("Empty template renders to an empty argument list")
    func emptyTemplateRendersEmpty() throws {
        let out = try ArgsTemplate.render(template: [], substitutions: [:])
        #expect(out.isEmpty)
    }

    // MARK: - Prompt containing spaces / quotes stays a single arg

    @Test("Prompt with whitespace stays a single argument element (no shell splitting)")
    func promptStaysSingleArg() throws {
        let prompt = "explain this: 'hi there' and \"bye\""
        let out = try ArgsTemplate.render(
            template: ["-p", "{prompt}"],
            substitutions: ["prompt": prompt]
        )
        #expect(out == ["-p", prompt])
    }

    // MARK: - Unknown placeholders in substitutions dict are ignored

    @Test("Extra substitution keys not in template are silently ignored")
    func extraSubstitutionsIgnored() throws {
        let out = try ArgsTemplate.render(
            template: ["-p", "{prompt}"],
            substitutions: ["prompt": "hi", "unused": "x"]
        )
        #expect(out == ["-p", "hi"])
    }

    // MARK: - Same placeholder used twice

    @Test("Same placeholder appearing multiple times is substituted every occurrence")
    func repeatsSubstitution() throws {
        let out = try ArgsTemplate.render(
            template: ["--in", "{prompt}", "--echo", "{prompt}"],
            substitutions: ["prompt": "hi"]
        )
        #expect(out == ["--in", "hi", "--echo", "hi"])
    }
}
