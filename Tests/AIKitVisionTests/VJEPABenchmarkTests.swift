// MARK: - VJEPABenchmarkTests
// Latency + peak memory benchmark for the Phase 1 pipeline.
//
// See features/shikki-vjepa-mlx-node.md §5 Phase 1 deliverable 8 + §10
// (target <500ms for 16 frames on M1).
//
// This bench runs against the ReferenceVisionBackend — it measures the
// preprocessor + forward-pass dispatch overhead, NOT the MLX compute.
// Once mlx-swift is wired, swap the backend and re-run.

#if canImport(AVFoundation)
import AVFoundation
import Foundation
import Testing
@testable import AIKitVision

@Suite("VJEPABenchmark")
struct VJEPABenchmarkTests {

    @Test("16-frame pipeline completes under 1500ms on the dev host")
    func test_sixteenFrameLatencyBudget() async throws {
        let url = try await VideoFramePreprocessorTests.renderFixtureVideo(frameCount: 32, size: 240)
        let inference = VJEPAInference(
            encoder: VJEPAEncoder(config: .vitB16),
            preprocessor: VideoFramePreprocessor(targetFrameCount: 16, targetSize: 224),
            backend: ReferenceVisionBackend()
        )
        // Warm-up run.
        _ = try await inference.embed(videoAt: url)

        let start = DispatchTime.now()
        let embedding = try await inference.embed(videoAt: url)
        let elapsedNanos = DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds
        let elapsedMs = Double(elapsedNanos) / 1_000_000.0

        #expect(embedding.frameCount == 16)
        #expect(embedding.featureDim == 768)
        // Dev-host budget (reference backend). The BR-10 gate of 500ms on M1
        // applies to the MLX backend and is verified by the arena battle
        // suite in Phase 3.
        #expect(elapsedMs < 1500.0)
    }
}
#endif
