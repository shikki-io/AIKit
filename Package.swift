// swift-tools-version: 6.0

import PackageDescription

// MARK: - The MLX half of the package exists only on an Apple host

// W30 of spec ai-layer-aikit-core-2026-10-31. BR-AIL-14 already said on-device
// inference is optional — a separate product, so an app wanting only remote
// providers links no MLX. Proving AIKit on Linux turns that from a linking
// choice into a manifest one, because `swift test` has no per-target build:
// it builds EVERY test target in the graph, so leaving `AIKitMLXTests`
// declared on Linux means compiling mlx-swift there, and mlx-swift is Metal
// and Accelerate — it does not build on Linux at all.
//
// This `#if` is therefore a HOST condition, not a target-platform one: a
// package manifest is compiled and run by the toolchain of the machine doing
// the resolving. On macOS (including when that macOS is building FOR iOS) the
// MLX product, target, suite and three dependencies are all declared exactly
// as before, so nothing about an Apple build changes. On Linux they are absent
// from the graph: a cold `swift build --target AIKit` there materialises
// exactly CoreKit, NetKit, ShellKit, TOMLKit, swift-asn1, swift-crypto and
// swift-log under `.build/checkouts`, and compiles no MLX module (measured,
// 718 tasks, 50s). SwiftPM does still reach the network for the pins listed in
// Package.resolved before pruning them, so this saves the Linux agent a
// compile, not a fetch.
//
// `.when(platforms:)` cannot do this job: a platform condition on a dependency
// still declares the dependency, and a target cannot be conditioned away at
// all. The alternative — declaring MLX everywhere and asking CI to build only
// some targets — is not available from `swift test`.
//
// KNOWN LIMIT, stated rather than discovered later: because the condition is
// on the HOST, cross-compiling to Linux FROM a Mac (`--swift-sdk
// x86_64-swift-linux-musl`) keeps MLX in the graph and would fail to build it.
// The Linux pipeline compiles natively on a Linux agent, so nothing in CI
// takes that path. Making this a target condition instead would mean guarding
// every file of `AIKitMLX` with `#if canImport(MLXLMCommon)`, which is a
// change to W7/W13's code and not this wave's.
#if canImport(Darwin)
    let mlxIsAvailableOnThisHost = true
#else
    let mlxIsAvailableOnThisHost = false
#endif

let mlxDependencies: [Package.Dependency] =
    mlxIsAvailableOnThisHost
    ? [
        // mlx-swift-lm — in-process on-device inference, reached only by the
        // AIKitMLX target. `from: "3.32.3"` and never a pin: W13 of spec
        // ai-layer-aikit-core-2026-10-31 moved this off `from: "2.31.3"`,
        // which admits no 3.x at all.
        //
        // 3.x is a breaking release: Downloader, Tokenizer and TokenizerLoader
        // became protocols with no shipped conformance, so AIKitMLX supplies
        // both (MLXHubDownloader, TransformersTokenizerLoader). In exchange
        // the package no longer depends on a Hub client at all, which is what
        // lets AIKit own the one cache root.
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", from: "3.32.3"),
        // mlx-swift, declared here rather than taken transitively, because
        // AIKitMLX imports `MLX` itself (MLX.Memory.cacheLimit). `from:` and
        // never `exact:` or `.upToNextMinor`.
        //
        // mlx-swift-lm 3.32.3 caps this at `.upToNextMinor(from: "0.32.3")`,
        // so 0.33 stays out of the graph until upstream moves. That cap is
        // upstream's shared constraint and is not ours to widen here: the
        // intersection with this range resolves to 0.32.3.
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.32.3"),
        // swift-transformers — `Tokenizers.AutoTokenizer`, the concrete
        // tokenizer mlx-swift-lm 3.x no longer carries. AIKitMLX only: AIKit
        // itself tokenizes nothing.
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.5"),
    ]
    : []

let mlxProducts: [Product] =
    mlxIsAvailableOnThisHost
    ? [
        // A SEPARATE product on purpose (BR-AIL-14, W7 of spec
        // ai-layer-aikit-core-2026-10-31): on-device inference is optional, so
        // an app that wants only remote providers depends on `AIKit` alone and
        // links no MLX, no Metal shaders and no swift-transformers.
        .library(
            name: "AIKitMLX",
            targets: ["AIKitMLX"]
        ),
    ]
    : []

let mlxTargets: [Target] =
    mlxIsAvailableOnThisHost
    ? [
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
        // The MLX provider's own suite. It reaches MLX through the
        // `MLXModelSession` seam and stubs it, so `swift test` never asks for
        // a Metal device: SPM does not bundle MLX's metallib, and a test that
        // touched the real runtime would crash rather than fail.
        .testTarget(
            name: "AIKitMLXTests",
            dependencies: [
                "AIKitMLX",
                "AIKit",
                .product(name: "CoreKit", package: "CoreKit"),
                .product(name: "CoreKitTestSupport", package: "CoreKit"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            ]
        ),
    ]
    : []

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
        // The test doubles, in their OWN product (W26 of spec
        // ai-layer-aikit-core-2026-10-31, T-AIL-51, thread AIKit#18
        // r4500328752). MockAIProvider, MockModelRegistry and
        // MockRuntimeEngine shipped inside `AIKit` itself, so every app linking
        // the kit could construct a provider that answers canned text and
        // reaches no model. Moving them here makes that a dependency a reviewer
        // can see: a shipping target depends on `AIKit` and cannot name a mock,
        // and a target that CAN name one says so in its own manifest.
        .library(
            name: "AIKitTestSupport",
            targets: ["AIKitTestSupport"]
        ),
    ] + mlxProducts,
    dependencies: [
        // NetKit 3.0.0 — the fleet's current transport (W19 of spec
        // ai-layer-aikit-core-2026-10-31). Two things moved at once:
        //
        //  * The canonical org is `obyw-one`. The previous owner's URL only
        //    kept resolving through GitHub's transfer redirect, and a redirect
        //    is not a declaration: SwiftPM records the location it was given,
        //    so `Package.resolved` named a repository that no longer owns the
        //    code. (The retired org name is spelled out nowhere in this file
        //    on purpose — T-AIL-38 asserts the manifest no longer carries it.)
        //  * 3.0.0 is a deliberate source break. `createRequest(endPoint:)` is
        //    now `throws(NetworkError)` instead of calling
        //    `preconditionFailure` on an endpoint that forms no URL and
        //    swallowing a body it could not encode with `try?`. Both of those
        //    are exactly the silent failures W20/W21 exist to remove, so the
        //    call sites here adopt the throwing form rather than pin back.
        //    3.0.0 also drops the Combine `sendRequest(endpoint:type:)`
        //    requirement (it lives in the Apple-only `NetKitApple` product
        //    now); AIKit never used it.
        //  * 3.1.0 is additive on top of that: `NetworkService(decoder:)` /
        //    `(decoderFactory:)` are back, and `NetworkProtocol` gains
        //    `injectedDecoder() -> JSONDecoder?` as a requirement WITH a
        //    `nil` default. AIKit's conformers compile and behave unchanged —
        //    this package decodes nothing snake_cased — so the bump is a
        //    floor move, not an adoption.
        .package(url: "https://github.com/obyw-one/NetKit.git", from: "3.1.0"),
        // CoreKit 0.16.0 — declared DIRECTLY, not taken transitively through
        // NetKit, because AIKit itself is about to import it: the DI container
        // and `RepositoryProtocol` (W22, BR-AIL-20) and the virtual path layer
        // (W23). A package whose types appear in our sources is our
        // dependency, whatever the graph would have supplied anyway.
        //
        // 0.16.0 is the floor NetKit 3.0.0 also takes, and the first CoreKit
        // without its own `URLSessionS3Transport` (obyw-one/CoreKit#34), which
        // 0.14/0.15 made ambiguous for type lookup against NetKit's copy.
        .package(url: "https://github.com/obyw-one/CoreKit.git", from: "0.16.0"),
        // swift-log — the three CLI providers log their attempts and fallbacks.
        .package(url: "https://github.com/apple/swift-log.git", from: "1.0.0"),
        // ShellKit 0.6.0 — async subprocess primitives with a hard timeout
        // (TimedShellExecutor); CLISubprocessProvider and WhisperCLIProvider
        // run every CLI through it. Now homed at `shikki-io`, the org that
        // ships it (W19).
        //
        // macOS and Linux ONLY (W1 of spec ai-layer-aikit-core-2026-10-31,
        // BR-AIL-02). ShellKit declares `.macOS(.v14)` and builds on
        // `Foundation.Process`, which does not exist in the iOS SDK: linking it
        // unconditionally made
        // `xcodebuild -scheme AIKit -destination 'generic/platform=iOS'` fail
        // with `cannot find 'Process' in scope` inside its TimedShellExecutor.
        // The dependency is therefore conditioned per platform below, and the
        // source files that need it sit behind `#if canImport(ShellKit)`.
        //
        // 0.5.1 adds `ShellExecutorProtocol.runStreaming` as a requirement WITH
        // a default implementation, so AIKit's conformers keep compiling
        // unchanged.
        //
        // 0.6.0 is additive — no existing public API changed — and carries one
        // fix this package wants: `DetachedProcess.exitCode()` could hang
        // forever on macOS, because it waited with `Process.waitUntilExit()`
        // on a GCD worker thread, and that call spins a run loop the worker
        // does not have. It now polls `isRunning` on Darwin as it already did
        // on Linux. Every CLI provider here reaches a subprocess from async
        // code, so that hang was reachable from AIKitTests.
        .package(url: "https://github.com/shikki-io/ShellKit", from: "0.6.0"),
        // TOMLKit 0.5.5 — the decoder for the `[[providers]]` settings document
        // (W22, BR-AIL-20). Declared and LINKED by the `AIKit` target in this
        // wave even though no source imports it yet, precisely so this wave's
        // `xcodebuild -destination 'generic/platform=iOS'` gate proves it
        // builds for iOS before W22 depends on that.
        //
        // The floor matters: TOMLKit 0.5.5's own manifest takes `Checkit` on
        // `.branch("master")`, and SwiftPM refuses a branch requirement inside
        // a versioned dependency graph. 0.6.0 moved it to `from: "0.0.2"`, and
        // `from: "0.5.5"` resolves there.
        .package(url: "https://github.com/LebJe/TOMLKit.git", from: "0.5.5"),
        // swift-crypto — SHA-256 and `Insecure.SHA1` on a platform with no
        // CryptoKit (W30). `ModelStore` verifies a downloaded weights file
        // against the digest Hugging Face publishes, which is the one piece of
        // AIKit that needs a hash function, and CryptoKit is Apple-only: a
        // Linux build of `--product AIKit` failed on
        // `no such module 'CryptoKit'` before this line existed.
        //
        // AIKit IMPORTS it only off Apple (see the two targets below), because
        // on an Apple platform CryptoKit is in the SDK; swift-crypto's
        // `Crypto` module is API-identical for both hashes used here, so the
        // source difference is one conditional `import`.
        //
        // That condition is about AIKit's own import and NOT about keeping the
        // module out of an Apple build: CoreKit links `Crypto` on every
        // platform, so a clean
        // `xcodebuild -scheme AIKit -destination generic/platform=iOS` built
        // `Crypto-product` before this declaration existed and still does —
        // measured, the explicit dependency is CoreKit's.
        //
        // Already in this graph transitively, through CoreKit (and through
        // swift-transformers on an Apple host): `swift package
        // show-dependencies` resolves 4.5.2 either way, so declaring it costs
        // no new fetch. It is declared anyway, for the same reason CoreKit
        // itself is declared rather than taken through NetKit — a package
        // whose types appear in our sources is our dependency.
        .package(url: "https://github.com/apple/swift-crypto.git", from: "4.0.0"),
    ] + mlxDependencies,
    targets: [
        .target(
            name: "AIKit",
            dependencies: [
                .product(name: "NetKit", package: "NetKit"),
                .product(name: "CoreKit", package: "CoreKit"),
                .product(name: "TOMLKit", package: "TOMLKit"),
                .product(name: "Logging", package: "swift-log"),
                .product(name: "ShellKit", package: "ShellKit", condition: .when(platforms: [.macOS, .linux])),
                // Off Apple only: CryptoKit ships in the Apple SDKs, and a
                // platform that has both would see an ambiguous `SHA256`.
                .product(
                    name: "Crypto",
                    package: "swift-crypto",
                    condition: .when(platforms: [.linux, .android, .windows, .wasi])
                ),
            ]
        ),
        // The test doubles (W26). A library target, not a test target, because
        // the suites of two different test targets use them — and because that
        // is what makes the product above possible.
        //
        // It depends on `AIKit` and never the reverse: a mock conforms to the
        // kit's protocols, so the arrow points this way, and `AIKit` keeps no
        // reference to anything in here.
        .target(
            name: "AIKitTestSupport",
            dependencies: [
                "AIKit",
                .product(name: "Logging", package: "swift-log"),
            ]
        ),
        // The public-surface renderer (W27, BR-AIL-21 item 5). An executable
        // TARGET and deliberately NOT a product: nothing links it, and a
        // consumer resolving AIKit never builds it.
        //
        // It was `scripts/public-api-render.py` until the operator asked, on
        // PR #18, what a Python script was doing rendering a Swift package's
        // own exported surface. Nothing — so it is Swift now. Every machine
        // that can run this gate has a Swift toolchain by construction (it is
        // building AIKit); `python3` it had only by assumption, and
        // `public-api.sh` named it as a requirement in its own header.
        //
        // Unconditional, unlike the MLX block above: the renderer reads JSON
        // and imports only Foundation, so the Linux job builds it too even
        // though the surface SUITE is Apple-only.
        .executableTarget(
            name: "public-api-render"
        ),
        .testTarget(
            name: "AIKitLiveTests",
            dependencies: ["AIKit"]
        ),
        .testTarget(
            name: "AIKitTests",
            dependencies: [
                "AIKit",
                "AIKitTestSupport",
                .product(name: "NetKit", package: "NetKit"),
                .product(name: "CoreKit", package: "CoreKit"),
                // W23: the suite builds its path layer over a run-scoped
                // scratch root (`TestScratch`) and holds every
                // `VirtualNodeNaming` enum to CoreKit's own
                // `VirtualNodeNamingRatchet`, instead of re-writing either.
                .product(name: "CoreKitTestSupport", package: "CoreKit"),
                .product(name: "TOMLKit", package: "TOMLKit"),
                .product(name: "Logging", package: "swift-log"),
                .product(name: "ShellKit", package: "ShellKit", condition: .when(platforms: [.macOS, .linux])),
                // `ModelStoreTests` computes the digests it asserts against,
                // so the suite needs the same hash module the source does.
                .product(
                    name: "Crypto",
                    package: "swift-crypto",
                    condition: .when(platforms: [.linux, .android, .windows, .wasi])
                ),
            ]
        ),
        // The public-surface ratchet (W27, T-AIL-52/T-AIL-53), in its OWN
        // target. Its one suite shells out to `scripts/public-api.sh`, which
        // runs `swift build --target AIKit` and `--target AIKitMLX` with
        // `-emit-symbol-graph` under a scratch path of its own — a cold run
        // compiles mlx-swift's Metal kernels. Folded into `AIKitTests` that
        // took the whole-target `AIKitAll` scope from 11s to 1m19s and put a
        // 7-core compile beside suites that assert a wall-clock budget.
        //
        // It imports `AIKit` WITHOUT `@testable`: the suite names only what a
        // consumer can reach, which is the thing being asserted.
        .testTarget(
            name: "AIKitPublicSurfaceTests",
            dependencies: [
                "AIKit",
                "AIKitTestSupport",
            ]
        ),
    ] + mlxTargets
)
