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
        // AIKitMLX target. `from: "3.32.3"` and never a pin: W13 of spec
        // ai-layer-aikit-core-2026-10-31 moved this off `from: "2.31.3"`, which
        // admits no 3.x at all.
        //
        // 3.x is a breaking release: Downloader, Tokenizer and TokenizerLoader
        // became protocols with no shipped conformance, so AIKitMLX supplies
        // both (MLXHubDownloader, TransformersTokenizerLoader). In exchange the
        // package no longer depends on a Hub client at all, which is what lets
        // AIKit own the one cache root.
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", from: "3.32.3"),
        // mlx-swift, declared here rather than taken transitively, because
        // AIKitMLX imports `MLX` itself (MLX.Memory.cacheLimit). `from:` and
        // never `exact:` or `.upToNextMinor`.
        //
        // mlx-swift-lm 3.32.3 caps this at `.upToNextMinor(from: "0.32.3")`, so
        // 0.33 stays out of the graph until upstream moves. That cap is
        // upstream's shared constraint and is not ours to widen here: the
        // intersection with this range resolves to 0.32.3.
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.32.3"),
        // swift-transformers — `Tokenizers.AutoTokenizer`, the concrete
        // tokenizer mlx-swift-lm 3.x no longer carries. AIKitMLX only: AIKit
        // itself tokenizes nothing.
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.5"),
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
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"),
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
