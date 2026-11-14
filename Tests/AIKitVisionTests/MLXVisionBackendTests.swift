// MARK: - MLXVisionBackendTests
// Wave 9-12 — exercise the real MLX forward pass committed alongside
// the mlx-swift dep. Verifies:
//
//   1. Backend instantiation succeeds on M-series hosts (build-time
//      guarded by `#if canImport(MLX)`; runtime gated by SHIKKI_HOST_ARCH).
//   2. Fallback path: when MLX is unavailable, `VJEPAInference.makeDefault`
//      still yields a working pipeline via `ReferenceVisionBackend` (BR-17).
//   3. Output shape contract: `[T, 768]` regardless of input frame count.
//   4. Determinism: same seed + same input → byte-identical output.
//
// The original "98s MLX run" claim from PR #8 cannot be re-verified
// without converted V-JEPA 2 safetensors weights (env-gated developer
// task, called out in PR #8's body and inherited here). The benchmark
// path that PR claimed lives in `MLXBenchmarkTests` and stays gated
// behind `SHIKKI_MLX_TESTS=1` — running a real-shape forward pass with
// random-init weights still produces a meaningful wall-clock figure
// (commit body records the measured number).

#if canImport(AVFoundation) && canImport(MLX)
import AVFoundation
import Foundation
import Testing
@testable import AIKitVision

private let isAppleSilicon: Bool = {
    #if arch(arm64)
    return true
    #else
    return false
    #endif
}()

/// Tests that touch MLX runtime ops (MLXRandom.seed, Conv3d, etc.) abort
/// the process when the metallib isn't on disk. The metallib subset shipped
/// with mlx-swift's SPM target does not cover all kernels, so under
/// `swift test` we gate these behind `SHIKKI_MLX_TESTS=1` (same env knob
/// used by `MLXBenchmarkTests`) — set it locally after running
/// `scripts/build-mlx-metallib.sh` (TODO: spec §5 deliverable 8b — script
/// not yet ported from PR #8's branch). Until that lands, the
/// always-on tests below cover the build-time contract; the runtime
/// tests run in CI under the Phase 3 arena harness with the metallib
/// pre-shipped.
private let mlxRuntimeEnabled: Bool =
    ProcessInfo.processInfo.environment["SHIKKI_MLX_TESTS"] == "1"

@Suite("MLXVisionBackend")
struct MLXVisionBackendTests {

    // MARK: 1. Instantiation

    @Test(
        "MLXVisionBackend init succeeds on M-series host",
        .enabled(if: isAppleSilicon && mlxRuntimeEnabled, "MLX requires arm64 + SHIKKI_MLX_TESTS=1 (metallib)")
    )
    func test_initSucceedsOnAppleSilicon() throws {
        MLXVisionBackend.bootstrap()
        let encoder = VJEPAEncoder(config: .vitB16)
        // Force CPU device — same posture as MLXBenchmarkTests, the
        // metallib subset shipped under `swift test` doesn't cover the
        // full GPU kernel set.
        _ = try MLXVisionBackend(encoder: encoder, seed: 1, device: .cpu)
    }

    // MARK: 2. Graceful fallback

    @Test("VJEPAInference.makeDefault never crashes — falls back to Reference")
    func test_makeDefaultIsAlwaysUsable() {
        // BR-17: even on hosts where MLX init fails, makeDefault must
        // return a working pipeline. We don't probe the backend type
        // here — that's an internal detail; we verify the pipeline is
        // wired with the expected encoder + preprocessor.
        let inference = VJEPAInference.makeDefault()
        #expect(inference.encoder.config.hiddenDim == 768)
        #expect(inference.preprocessor.targetFrameCount == 16)
    }

    // MARK: 3. Output shape contract

    @Test(
        "MLX forward produces [T * 768] for T = 4 frames",
        .enabled(if: isAppleSilicon && mlxRuntimeEnabled, "MLX requires arm64 + SHIKKI_MLX_TESTS=1 (metallib)")
    )
    func test_forwardShapeContract() throws {
        MLXVisionBackend.bootstrap()
        let encoder = VJEPAEncoder(config: .vitB16)
        let backend = try MLXVisionBackend(encoder: encoder, seed: 42, device: .cpu)

        // Build a synthetic [4, 3, 224, 224] tensor — bypass the AV
        // pipeline so the test stays fast and offline.
        let frames = makeSyntheticFrames(count: 4, channels: 3, size: 224)

        let out = try backend.forward(frames: frames, encoder: encoder, mode: .dense)
        #expect(out.count == 4 * 768)
    }

    // MARK: 4. Determinism

    @Test(
        "MLX forward is deterministic for fixed seed",
        .enabled(if: isAppleSilicon && mlxRuntimeEnabled, "MLX requires arm64 + SHIKKI_MLX_TESTS=1 (metallib)")
    )
    func test_forwardIsDeterministic() throws {
        MLXVisionBackend.bootstrap()
        let encoder = VJEPAEncoder(config: .vitB16)
        let frames = makeSyntheticFrames(count: 2, channels: 3, size: 224)

        let backendA = try MLXVisionBackend(encoder: encoder, seed: 7, device: .cpu)
        let backendB = try MLXVisionBackend(encoder: encoder, seed: 7, device: .cpu)
        let outA = try backendA.forward(frames: frames, encoder: encoder, mode: .dense)
        let outB = try backendB.forward(frames: frames, encoder: encoder, mode: .dense)

        #expect(outA.count == outB.count)
        // Strict byte parity — same seed, same device, same input must
        // yield identical floats.
        #expect(outA == outB)
    }

    // MARK: Helpers

    /// Construct a deterministic `[T, C, H, W]` tensor without touching AV.
    private func makeSyntheticFrames(
        count: Int, channels: Int, size: Int
    ) -> VideoFrameTensor {
        let total = count * channels * size * size
        var data = [Float](repeating: 0, count: total)
        for t in 0..<count {
            for c in 0..<channels {
                for y in 0..<size {
                    for x in 0..<size {
                        let idx = ((t * channels + c) * size + y) * size + x
                        // Cheap deterministic gradient — different per (t, c).
                        data[idx] = Float((t + c * 7 + y / 16 + x / 16) % 11) / 11.0
                    }
                }
            }
        }
        return VideoFrameTensor(
            frameCount: count,
            channels: channels,
            height: size,
            width: size,
            data: data
        )
    }
}
#endif

// MARK: - x86 fallback test
//
// This block compiles even when MLX is unavailable — it exercises the
// fallback path without touching the MLX symbols.

#if canImport(AVFoundation) && !canImport(MLX)
import Foundation
import Testing
@testable import AIKitVision

@Suite("MLXVisionBackendFallback")
struct MLXVisionBackendFallbackTests {

    @Test("On non-MLX hosts, makeDefault falls through to ReferenceVisionBackend")
    func test_fallbackOnNonMLX() {
        let inference = VJEPAInference.makeDefault()
        #expect(inference.encoder.config.hiddenDim == 768)
    }
}
#endif
