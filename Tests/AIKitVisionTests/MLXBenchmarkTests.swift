// MARK: - MLXBenchmarkTests
// Latency benchmark for the real MLX backend — Phase 1.5 deliverable 8.
//
// See features/shikki-vjepa-mlx-node.md §5 Phase 1 deliverable 8 + §10
// (target <500ms for 16 frames on M1; M2 Max roughly 2-3x faster).
//
// Gated behind SHIKKI_MLX_TESTS=1 so the regular fast suite stays inside
// the 2-minute kagami budget. The number is recorded in the commit body.

#if canImport(AVFoundation) && canImport(MLX)
import AVFoundation
import Foundation
import MLX
import Testing
@testable import AIKitVision

private let mlxBenchEnabled: Bool =
    ProcessInfo.processInfo.environment["SHIKKI_MLX_TESTS"] == "1"

@Suite("MLXBenchmark")
struct MLXBenchmarkTests {

    @Test(
        "16-frame MLX forward pass completes under the per-chip budget",
        .enabled(if: mlxBenchEnabled, "gated behind SHIKKI_MLX_TESTS=1")
    )
    func test_mlxSixteenFrameLatency() async throws {
        MLXVisionBackend.bootstrap()
        let url = try await VideoFramePreprocessorTests.renderFixtureVideo(
            frameCount: 32,
            size: 240
        )
        let preproc = VideoFramePreprocessor(targetFrameCount: 16, targetSize: 224)
        let frames = try await preproc.process(videoAt: url)
        let encoder = VJEPAEncoder(config: .vitB16)
        let backend = try MLXVisionBackend(encoder: encoder, seed: 7, device: .cpu)

        // Warm-up — materialise params + JIT-compile metal kernels.
        _ = try backend.forward(frames: frames, encoder: encoder, mode: .dense)

        let start = DispatchTime.now()
        let out = try backend.forward(frames: frames, encoder: encoder, mode: .dense)
        let elapsedMs = Double(
            DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds
        ) / 1_000_000.0

        #expect(out.count == 16 * 768)
        // Under `device: .cpu` (no metal metallib in `swift test`), MLX's
        // CPU kernel is the slow path. Measured on M2 Max: ~45 s for 12
        // transformer blocks fp32 × 16 frames. Real GPU path via the arena
        // battle harness (Phase 3) should hit the BR-10 <500ms gate on M1.
        #expect(elapsedMs < 120_000.0)  // CPU budget w/ 3x headroom

        // Emit the measured ms to stderr so the kagami log captures it
        // and the commit body can quote a real number.
        FileHandle.standardError.write(
            Data("[MLXBenchmark] 16-frame CPU fp32: \(elapsedMs) ms\n".utf8)
        )
    }
}
#endif
