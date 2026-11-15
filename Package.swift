// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "AIKit",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
    ],
    products: [
        .library(
            name: "AIKit",
            targets: ["AIKit"]
        ),
    ],
    dependencies: [
        // NetKit 3.1.0 from obyw-one — NOT FJ-Studios/NetKit 2.x, which this
        // package pinned until 2026-11-15. That URL transitively resolved
        // CoreKit 0.2.1, fifteen minors behind, whose `CacheRepository.swift`
        // carries an unguarded `import os`: every Linux build of AIKit failed
        // inside a dependency nobody had looked at. The graph moves forward
        // together (spec c8fa126c W5).
        .package(url: "https://github.com/obyw-one/NetKit.git", from: "3.1.0"),
        // swift-log — the three CLI providers log their attempts and fallbacks.
        .package(url: "https://github.com/apple/swift-log.git", from: "1.0.0"),
        // ShellKit — async subprocess primitives with a hard timeout
        // (TimedShellExecutor); CLISubprocessProvider runs every CLI through it.
        //
        // The product is linked UNCONDITIONALLY below, which is what the 1:1
        // rule wants and what makes `CLISubprocessProvider` exist on Linux —
        // `Foundation.Process` is there, unlike on iOS. Until 2026-11-15 that
        // unconditional link is precisely what made every Linux build of AIKit
        // fail inside ShellKit: two `Darwin.kill(...)` calls through a correctly
        // guarded `import Darwin`. Fixed in shikki-io/ShellKit#21.
        //
        // Floor 0.6.0 is the current release, so this manifest resolves today.
        // A LINUX build additionally needs ShellKit 0.7.0, the release carrying
        // that fix; the floor moves to 0.7.0 the moment it is tagged, and the
        // musl evidence in AIKit's PR was taken with `swift package edit
        // ShellKit --path` against the fixed checkout (spec c8fa126c W5).
        //
        // URL: `shikki-io/ShellKit`, the canonical repository. This line said
        // `obyw-one/ShellKit` until 2026-11-15 — a spelling inherited from
        // develop, where it has been since the floor was 0.1.0. GitHub redirects
        // it, so each branch resolved fine ALONE, which is why nothing caught it.
        // But AIKit#24 declares `shikki-io/ShellKit`, and SwiftPM keys a package
        // by URL: two spellings of one identity make the graph refuse as soon as
        // both land on develop. Whichever merged second would have broken, and
        // the failure would have pointed at the merge rather than at this line.
        .package(url: "https://github.com/shikki-io/ShellKit", from: "0.6.0"),
    ],
    targets: [
        .target(
            name: "AIKit",
            dependencies: [
                .product(name: "NetKit", package: "NetKit"),
                .product(name: "Logging", package: "swift-log"),
                .product(name: "ShellKit", package: "ShellKit"),
            ]
        ),
        .testTarget(
            name: "AIKitTests",
            dependencies: [
                "AIKit",
                .product(name: "NetKit", package: "NetKit"),
                .product(name: "Logging", package: "swift-log"),
                .product(name: "ShellKit", package: "ShellKit"),
            ]
        ),
    ]
)
