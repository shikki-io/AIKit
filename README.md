# AIKit

AI provider abstraction layer.

## Installation

Add to your `Package.swift`:

```swift
.package(path: "../../packages/AIKit")
```

Depends on NetKit.

## Providers

`Sources/AIKit/Providers/` holds the dependency-free provider-parsing family:

- `ProviderModelCatalog` / `AnthropicFamily` — SSoT for canonical model ids and alias resolution.
- `ModelAliasMap` — short-name → canonical-id lookup used by CLI-subprocess providers.
- `ArgsTemplate` — placeholder substitution for CLI argument templates.
- `CLISubprocessProviderConfig` — parses a `kind = "cli-subprocess"` provider config entry.
- `StreamParsing` / `StreamEvent` — the normalized event vocabulary CLI stream parsers emit into.
- `ClaudeCodeStreamJSONParser`, `CursorTextParser`, `AiderMixedParser` — per-vendor stdout parsers.
- `ProviderHealthCheck` — lightweight reachability ping for HTTP-based providers.

## License

AGPL-3.0-or-later — see [LICENSE](LICENSE).
