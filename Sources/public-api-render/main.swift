// public-api-render — render SwiftPM symbol graphs as one sorted line per
// public symbol.
//
// W27 of spec `ai-layer-aikit-core-2026-10-31` put this rendering in a 49-line
// Python script. Operator on PR #18, against `scripts/public-api-render.py`:
// *"Why it's python here? What is the value to have py script for that instead
// of swift? We cannot use our SoT already write in swift to give public api,
// so explain me or fix that if it's make more sense"*.
//
// It made more sense, so this is the fix. The rendering had no reason to be in
// another language: the thing it describes is a Swift package's own exported
// surface, the gate that reads it is a Swift test, and every machine that can
// run this gate already has a Swift toolchain by construction — it is building
// AIKit. What it did NOT have by construction was `python3`, which the script
// named in its own header as a requirement. A gate whose dependency list is
// shorter than the thing it gates is one fewer way for CI to be red for a
// reason that is not the code.
//
// This is an executable TARGET of this package, not a product: nothing links
// it and `swift build` for a consumer never reaches it. It takes the symbol
// graph directory as its one argument and prints the rendering on stdout,
// which is exactly the contract `scripts/public-api.sh` had with the Python
// script — so `public-api.sh` is still the ONE code path a human regenerating
// the allowlist and `PublicSurfaceTests` checking it both go through
// (BR-EC-02).
//
// Line format, unchanged, because `AIKitPublicAPI.txt` is a measurement and a
// reformatting would read as a hundred-line regression:
//
//     <module>  <kind>  <dotted path>
//
// `kind` keeps swift's own vocabulary with the `swift.` prefix dropped, so an
// `enum.case` reads as one. A rendering that collides — a protocol requirement
// and its default implementation, or two overloads — gets its mangled USR
// appended in brackets, which is the only thing that always separates them.

import Foundation

/// The modules whose graphs make up the surface. `AIKit` is the package's one
/// library door; `AIKitMLX` is the optional on-device half, whose three
/// symbols BR-AIL-21 pins separately.
let modules = ["AIKit", "AIKitMLX"]

/// The `kind.identifier` of one symbol, e.g. `swift.enum.case`.
///
/// Its own type rather than a nested one: SwiftLint's `nesting` rule caps a
/// type at one level deep, and the graph's shape is three levels of object.
struct SymbolKind: Decodable {
    let identifier: String
}

/// The mangled USR of one symbol — the only thing that always separates two
/// renderings that collide.
struct SymbolIdentifier: Decodable {
    let precise: String
}

/// One symbol, decoded from a SwiftPM symbol graph.
struct GraphSymbol: Decodable {
    let kind: SymbolKind
    let identifier: SymbolIdentifier
    let pathComponents: [String]
}

/// A SwiftPM symbol graph, reduced to what a surface rendering needs.
///
/// A `Decodable` model rather than `JSONSerialization` and a cast ladder: the
/// graph is a wire format we do not own, so a shape change has to surface as a
/// decode error naming the key, not as an empty rendering that passes the diff
/// by rendering nothing.
struct SymbolGraph: Decodable {
    let symbols: [GraphSymbol]
}

/// One rendered symbol, before the collision suffix is decided.
///
/// A struct and not a tuple: three members is one past SwiftLint's
/// `large_tuple` cap, and these three want names at the use site anyway.
struct RenderedSymbol {
    let kind: String
    let path: String
    let usr: String

    /// The identity a collision is measured on: two symbols that render to the
    /// same kind and path need their USRs printed to be told apart.
    var collisionKey: String { "\(kind)\u{0}\(path)" }
}

/// A rendering failure that names the file it could not read, rather than
/// exiting on a bare `nil`.
struct RenderFailure: Error, CustomStringConvertible {
    let description: String
}

/// Render every module's graph in `directory`, sorted and de-duplicated.
///
/// Sorted, so the file is a SET and a diff names the symbol that moved rather
/// than renumbering the whole file.
func render(directory: URL) throws -> [String] {
    var lines: [String] = []
    for module in modules {
        let path = directory.appending(path: "\(module).symbols.json")
        guard FileManager.default.fileExists(atPath: path.path) else {
            throw RenderFailure(description: "no symbol graph for \(module) at \(path.path)")
        }
        let data = try Data(contentsOf: path)
        let graph: SymbolGraph
        do {
            graph = try JSONDecoder().decode(SymbolGraph.self, from: data)
        } catch {
            // Named, not swallowed: an undecodable graph is a toolchain
            // change, and the surface it would render is unknown — not empty.
            throw RenderFailure(
                description:
                    "could not decode the symbol graph for \(module) at \(path.path): \(error)"
            )
        }

        let rows: [RenderedSymbol] = graph.symbols.map { symbol in
            let raw = symbol.kind.identifier
            let kind = raw.hasPrefix("swift.") ? String(raw.dropFirst("swift.".count)) : raw
            return RenderedSymbol(
                kind: kind,
                path: symbol.pathComponents.joined(separator: "."),
                usr: symbol.identifier.precise
            )
        }

        // Count the identities first: only one that occurs more than once
        // needs its USR to tell the two apart.
        var counts: [String: Int] = [:]
        for row in rows { counts[row.collisionKey, default: 0] += 1 }

        for row in rows {
            let suffix = (counts[row.collisionKey] ?? 0) > 1 ? "  [\(row.usr)]" : ""
            lines.append("\(module)  \(row.kind)  \(row.path)\(suffix)")
        }
    }
    return Array(Set(lines)).sorted()
}

let arguments = Array(CommandLine.arguments.dropFirst())
guard arguments.count == 1 else {
    FileHandle.standardError.write(Data("usage: public-api-render <symbol-graph-dir>\n".utf8))
    exit(2)
}

do {
    let rendered = try render(directory: URL(filePath: arguments[0]))
    print(rendered.joined(separator: "\n"))
} catch {
    FileHandle.standardError.write(Data("public-api-render: \(error)\n".utf8))
    exit(1)
}
