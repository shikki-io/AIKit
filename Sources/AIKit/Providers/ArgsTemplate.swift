import Foundation

// MARK: - ArgsTemplate

/// Pure renderer for CLI argument templates.
///
/// A template is a `[String]` where any element may contain one or more
/// `{key}` placeholders. Rendering substitutes each placeholder from a
/// `[String: String]` dictionary; unresolved placeholders throw
/// ``RenderError/unresolvedPlaceholder(_:)``.
///
/// Placeholder syntax is intentionally trivial (`{key}` — no escaping,
/// no format specifiers). CLI templates are static and operator-authored;
/// a heavier syntax would only invite mistakes.
public enum ArgsTemplate {

    // MARK: - RenderError

    public enum RenderError: Error, Sendable, Equatable {
        /// A `{key}` placeholder appeared in the template but no value was
        /// supplied in `substitutions`.
        case unresolvedPlaceholder(String)
    }

    // MARK: - render

    /// Render `template` by substituting `{key}` occurrences from
    /// `substitutions`.
    ///
    /// - Parameters:
    ///   - template: The literal argument list, possibly containing
    ///     `{key}` placeholders (either as an entire element or embedded).
    ///   - substitutions: `key → value` map. Extra keys are ignored.
    /// - Returns: The rendered argument list, preserving element boundaries
    ///   (no shell-splitting of substituted values).
    /// - Throws: ``RenderError/unresolvedPlaceholder(_:)`` on the first
    ///   `{key}` whose key is not present in `substitutions`.
    public static func render(
        template: [String],
        substitutions: [String: String]
    ) throws -> [String] {
        try template.map { try renderElement($0, substitutions: substitutions) }
    }

    // MARK: - Private

    /// Render a single template element by walking `{key}` occurrences.
    /// Elements without placeholders are returned unchanged.
    private static func renderElement(
        _ element: String,
        substitutions: [String: String]
    ) throws -> String {
        // Fast path: no '{' at all → pass through.
        guard element.contains("{") else { return element }

        var result = ""
        var remainder = Substring(element)

        while let open = remainder.firstIndex(of: "{") {
            // Append the literal chunk before '{'.
            result.append(contentsOf: remainder[..<open])

            let afterOpen = remainder.index(after: open)
            guard let close = remainder[afterOpen...].firstIndex(of: "}") else {
                // Dangling '{' with no matching '}' — treat the rest as a literal.
                result.append(contentsOf: remainder[open...])
                return result
            }

            let key = String(remainder[afterOpen..<close])
            guard let value = substitutions[key] else {
                throw RenderError.unresolvedPlaceholder(key)
            }
            result.append(value)

            remainder = remainder[remainder.index(after: close)...]
        }

        result.append(contentsOf: remainder)
        return result
    }
}
