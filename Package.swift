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
        // NetKit v2.1.0 — generic name restored (name-restoration epic 2026-07-20); fuzzy fork consolidated so no collision.
        .package(url: "https://github.com/FJ-Studios/NetKit.git", from: "2.1.0"),
        // swift-log — the three CLI providers log their attempts and fallbacks.
        .package(url: "https://github.com/apple/swift-log.git", from: "1.0.0"),
        // ShellKit v0.1.0 — async subprocess primitives with a hard timeout
        // (TimedShellExecutor); CLISubprocessProvider runs every CLI through it.
        .package(url: "https://github.com/obyw-one/ShellKit", from: "0.1.0"),
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
