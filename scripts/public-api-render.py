"""Render a SwiftPM symbol graph directory as one sorted line per public symbol.

W27 of spec ai-layer-aikit-core-2026-10-31. Called only by public-api.sh, which
is called both by a human regenerating AIKitPublicAPI.txt and by
PublicSurfaceTests checking it (BR-EC-02: one code path).

Line format:

    <module>  <kind>  <dotted path>

`kind` keeps swift's own vocabulary with the `swift.` prefix dropped, so an
`enum.case` reads as one. A rendering that collides — a protocol requirement
and its default implementation, or two overloads — gets its mangled USR
appended in brackets, which is the only thing that always separates them.
"""

import json
import pathlib
import sys
from collections import Counter

MODULES = ("AIKit", "AIKitMLX")


def render(directory: pathlib.Path) -> list[str]:
    lines: list[str] = []
    for module in MODULES:
        path = directory / f"{module}.symbols.json"
        if not path.exists():
            raise SystemExit(f"no symbol graph for {module} at {path}")
        graph = json.loads(path.read_text())
        rows = []
        for symbol in graph["symbols"]:
            kind = symbol["kind"]["identifier"]
            kind = kind[len("swift."):] if kind.startswith("swift.") else kind
            rows.append((kind, ".".join(symbol["pathComponents"]),
                         symbol["identifier"]["precise"]))
        counts = Counter((kind, path_) for kind, path_, _ in rows)
        for kind, path_, usr in rows:
            suffix = f"  [{usr}]" if counts[(kind, path_)] > 1 else ""
            lines.append(f"{module}  {kind}  {path_}{suffix}")
    # Sorted, so the file is a set and a diff names the symbol that moved.
    return sorted(set(lines))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: public-api-render.py <symbol-graph-dir>")
    print("\n".join(render(pathlib.Path(sys.argv[1]))))
