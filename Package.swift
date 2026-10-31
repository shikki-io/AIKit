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
        // A SEPARATE product on purpose (BR-AIL-14, W7 of spec
        // ai-layer-aikit-core-2026-10-31): on-device inference is optional, so
        // an app that wants only remote providers depends on `AIKit` alone and
        // links no MLX, no Metal shaders and no swift-transformers.
        .library(
            name: "AIKitMLX",
            targets: ["AIKitMLX"]
        ),
    ],
    dependencies: [
        // NetKit v2.1.0 — generic name restored (name-restoration epic 2026-07-20); fuzzy fork consolidated so no collision.
        .package(url: "https://github.com/FJ-Studios/NetKit.git", from: "2.1.0"),
        // swift-log — the three CLI providers log their attempts and fallbacks.
        .package(url: "https://github.com/apple/swift-log.git", from: "1.0.0"),
        // ShellKit v0.1.0 — async subprocess primitives with a hard timeout
        // (TimedShellExecutor); CLISubprocessProvider runs every CLI through it.
        //
        // macOS ONLY (W1 of spec ai-layer-aikit-core-2026-10-31, BR-AIL-02).
        // ShellKit declares `.macOS(.v14)` and builds on `Foundation.Process`,
        // which does not exist in the iOS SDK: linking it unconditionally made
        // `xcodebuild -scheme AIKit -destination 'generic/platform=iOS'` fail with
        // `cannot find 'Process' in scope` inside ShellKit's TimedShellExecutor.
        // The dependency is therefore conditioned per platform below, and the two
        // source files that need it sit behind `#if canImport(ShellKit)`.
        .package(url: "https://github.com/obyw-one/ShellKit", from: "0.1.0"),
        // mlx-swift-lm — in-process on-device inference, reached only by the
        // AIKitMLX target (W7). `from: "2.31.3"` is the constraint
        // shikki-client already resolves with, and it requires mlx-swift
        // >=0.31.3 <0.32.0: W11 has the client depend on AIKit *and* Irori,
        // and Irori pins mlx-swift from "0.31.3", so a wider range here would
        // make that graph unresolvable.
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", from: "2.31.3"),
    ],
    targets: [
        .target(
            name: "AIKit",
            dependencies: [
                .product(name: "NetKit", package: "NetKit"),
                .product(name: "Logging", package: "swift-log"),
                .product(name: "ShellKit", package: "ShellKit", condition: .when(platforms: [.macOS])),
            ]
        ),
        // On-device MLX, extracted from shikki-client's
        // Sources/ShikkiAppKit/Engine/MLXEngine.swift (W7). Depends on AIKit,
        // never the reverse: AIKit stays MLX-free so BR-AIL-14 holds.
        .target(
            name: "AIKitMLX",
            dependencies: [
                "AIKit",
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
            ]
        ),
        .testTarget(
            name: "AIKitLiveTests",
            dependencies: ["AIKit"]
        ),
        .testTarget(
            name: "AIKitTests",
            dependencies: [
                "AIKit",
                .product(name: "NetKit", package: "NetKit"),
                .product(name: "Logging", package: "swift-log"),
                .product(name: "ShellKit", package: "ShellKit", condition: .when(platforms: [.macOS])),
            ]
        ),
        // The MLX provider's own suite. It reaches MLX through the
        // `MLXModelSession` seam and stubs it, so `swift test` never asks for a
        // Metal device: SPM does not bundle MLX's metallib, and a test that
        // touched the real runtime would crash rather than fail.
        .testTarget(
            name: "AIKitMLXTests",
            dependencies: [
                "AIKitMLX",
                "AIKit",
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            ]
        ),
    ]
)
