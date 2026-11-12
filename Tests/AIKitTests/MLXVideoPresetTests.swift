import Foundation
import Testing

@testable import AIKit

// MARK: - MLXVideoPresetTests
//
// W25 of spec `ai-layer-aikit-core-2026-10-31`, T-AIL-50: the resolution × fps
// grid, and an unsupported combination refused by the device.
//
// The grid itself is platform-free data, so these run on every host. The two
// tests that build a provider are macOS-gated, because `MLXVideoProvider` and
// `VideoGenerationOptions` live behind `#if os(macOS)` — they shell out, and
// iOS has no subprocess API (BR-AIL-02).

@Suite("MLX video presets")
struct MLXVideoPresetTests {

    // MARK: - The grid

    @Test("The grid is the full resolution × frame-rate cross product")
    func gridIsTheCrossProduct() {
        #expect(VideoResolution.allCases.count == 3)
        #expect(VideoFrameRate.allCases.count == 5)
        #expect(MLXVideoPreset.grid.count == 15)
        // Generated, not listed: the count must follow the enums, so adding a
        // case to either cannot leave a hole in the grid.
        #expect(MLXVideoPreset.grid.count == VideoResolution.allCases.count * VideoFrameRate.allCases.count)
        // No duplicates, and every combination present exactly once.
        #expect(Set(MLXVideoPreset.grid).count == MLXVideoPreset.grid.count)
        for resolution in VideoResolution.allCases {
            for rate in VideoFrameRate.allCases {
                #expect(MLXVideoPreset.grid.contains(MLXVideoPreset(resolution: resolution, frameRate: rate)))
            }
        }
    }

    @Test(
        "A resolution carries the geometry it is named for",
        arguments: [
            (VideoResolution.hd720, 1280, 720, "720p"),
            (.fullHD1080, 1920, 1080, "1080p"),
            (.uhd4K, 3840, 2160, "4K"),
        ]
    )
    func resolutionGeometry(resolution: VideoResolution, width: Int, height: Int, name: String) {
        #expect(resolution.width == width)
        #expect(resolution.height == height)
        #expect(resolution.pixelsPerFrame == width * height)
        #expect(resolution.displayName == name)
    }

    @Test("A frame rate's raw value IS its fps, so nothing converts")
    func frameRateRawValues() {
        #expect(VideoFrameRate.allCases.map(\.rawValue) == [24, 30, 60, 120, 240])
        for rate in VideoFrameRate.allCases {
            #expect(rate.framesPerSecond == rate.rawValue)
        }
    }

    @Test("pixelsPerSecond is frame pixels × fps, and orders the grid")
    func pixelsPerSecond() {
        let preset = MLXVideoPreset(resolution: .fullHD1080, frameRate: .fps60)
        #expect(preset.pixelsPerSecond == 1920 * 1080 * 60)

        // The reason throughput is the ordering number and not resolution
        // alone: 720p at 240 fps costs more than 1080p at 60.
        let fastSmall = MLXVideoPreset(resolution: .hd720, frameRate: .fps240)
        let slowLarge = MLXVideoPreset(resolution: .fullHD1080, frameRate: .fps60)
        #expect(fastSmall.pixelsPerSecond > slowLarge.pixelsPerSecond)
    }

    @Test("numFrames follows the rate and the duration, and is never zero")
    func numFrames() {
        #expect(MLXVideoPreset(resolution: .hd720, frameRate: .fps24).numFrames(seconds: 2) == 48)
        #expect(MLXVideoPreset(resolution: .hd720, frameRate: .fps60).numFrames(seconds: 4) == 240)
        // A duration shorter than one frame still produces one frame: a
        // zero-frame generation is not a video and the CLI rejects it.
        #expect(MLXVideoPreset(resolution: .hd720, frameRate: .fps24).numFrames(seconds: 0) == 1)
        #expect(MLXVideoPreset(resolution: .hd720, frameRate: .fps24).numFrames(seconds: 0.001) == 1)
    }

    @Test("displayName is short enough for a log line")
    func displayNames() {
        #expect(MLXVideoPreset(resolution: .hd720, frameRate: .fps30).displayName == "720p30")
        #expect(MLXVideoPreset(resolution: .uhd4K, frameRate: .fps120).displayName == "4K120")
    }

    @Test("The named points sit on the grid")
    func namedPointsAreOnTheGrid() {
        for preset in [MLXVideoPreset.compact, .standard, .cinema] {
            #expect(MLXVideoPreset.grid.contains(preset))
        }
        #expect(MLXVideoPreset.standard == MLXVideoPreset(resolution: .fullHD1080, frameRate: .fps30))
        #expect(MLXVideoPreset.compact.pixelsPerSecond < MLXVideoPreset.standard.pixelsPerSecond)
        #expect(MLXVideoPreset.standard.pixelsPerSecond < MLXVideoPreset.cinema.pixelsPerSecond)
    }

    @Test("A preset round-trips through Codable")
    func presetRoundTrips() throws {
        for preset in MLXVideoPreset.grid {
            let data = try JSONEncoder().encode(preset)
            #expect(try JSONDecoder().decode(MLXVideoPreset.self, from: data) == preset)
        }
    }

    // MARK: - The device refuses a combination (T-AIL-50)

    @Test("An unsupported resolution × fps is refused by the device")
    func unsupportedCombinationIsRefused() throws {
        let phone = DeviceCapability(deviceClass: .iPhoneA18, totalRAMGB: 8, availableRAMGB: 6)

        // 1080p30 is exactly what an A18 is budgeted.
        #expect(phone.supports(MLXVideoPreset(resolution: .fullHD1080, frameRate: .fps30)))

        // The SAME resolution at a higher rate is refused, and the SAME rate at
        // a higher resolution is refused. It is the combination that decides,
        // which is the whole point of refusing on the product.
        let fasterSameSize = MLXVideoPreset(resolution: .fullHD1080, frameRate: .fps120)
        let biggerSameRate = MLXVideoPreset(resolution: .uhd4K, frameRate: .fps30)
        #expect(phone.supports(fasterSameSize) == false)
        #expect(phone.supports(biggerSameRate) == false)

        let refusal = try #require(phone.refusal(for: fasterSameSize))
        guard case .throughput(let required, let ceiling, let deviceClass) = refusal else {
            Issue.record("expected a throughput refusal, got \(refusal)")
            return
        }
        #expect(deviceClass == .iPhoneA18)
        #expect(required == fasterSameSize.pixelsPerSecond)
        #expect(ceiling == MLXVideoPreset.standard.pixelsPerSecond)
    }

    @Test("Each tier of hardware serves a larger slice of the grid")
    func tiersAreOrdered() {
        func supported(_ deviceClass: DeviceClass) -> Set<MLXVideoPreset> {
            Set(
                DeviceCapability(deviceClass: deviceClass, totalRAMGB: 32, availableRAMGB: 24)
                    .supportedVideoPresets
            )
        }
        let a16 = supported(.iPhoneA16)
        let a18 = supported(.iPhoneA18)
        let m1 = supported(.macM1)
        let m4 = supported(.macM4)
        let m6 = supported(.macM6)

        #expect(a16.isSubset(of: a18))
        #expect(a18.isSubset(of: m1))
        #expect(m1.isSubset(of: m4))
        #expect(m4.isSubset(of: m6))
        #expect(a16.count < m6.count)
        // Nothing in the table serves the whole grid: 4K240 is budgeted
        // nowhere, so a full-grid claim would mean a ceiling went missing.
        #expect(m6.count < MLXVideoPreset.grid.count)
    }

    // MARK: - The preset reaches the provider

    #if os(macOS)

        @Test("Options built from a preset derive every number from it")
        func optionsFromPreset() {
            let preset = MLXVideoPreset(resolution: .fullHD1080, frameRate: .fps60)
            let options = VideoGenerationOptions(preset: preset, seconds: 3)

            #expect(options.width == 1920)
            #expect(options.height == 1080)
            #expect(options.fps == 60)
            #expect(options.numFrames == 180)
            #expect(options.preset == preset)
        }

        @Test("Loose options carry no preset, so nothing is inferred from geometry")
        func looseOptionsCarryNoPreset() {
            // 480x320 is on no grid point. Inferring a nearest preset would
            // refuse — or allow — a shape nobody asked for.
            #expect(VideoGenerationOptions.default.preset == nil)
            #expect(VideoGenerationOptions.default.width == 480)
            #expect(VideoGenerationOptions.default.height == 320)
        }

        @Test("A payload stored before W25 still decodes, with no preset")
        func legacyPayloadDecodes() throws {
            // The key is simply absent, which is what every stored
            // `VideoGenerationOptions` written before this wave looks like.
            let legacy = """
                {"width":480,"height":320,"numFrames":49,"fps":24,
                 "numInferenceSteps":30,"guidanceScale":5.0}
                """
            let options = try JSONDecoder().decode(
                VideoGenerationOptions.self,
                from: Data(legacy.utf8)
            )
            #expect(options.preset == nil)
            #expect(options.width == 480)
            #expect(options.numFrames == 49)
        }

        @Test("The provider refuses a preset the device cannot serve, before launching anything")
        func providerRefusesBeforeLaunching() async throws {
            // python3 is named but never reached: the refusal happens before
            // any subprocess, which is what makes this test safe to run
            // anywhere (BR-AIL-12).
            let provider = MLXVideoProvider(
                id: "mlx/test",
                displayName: "Test",
                pythonPath: "/nonexistent/python3",
                modelName: "wan2.1-t2v-1.3b",
                defaultOptions: VideoGenerationOptions(
                    preset: MLXVideoPreset(resolution: .uhd4K, frameRate: .fps240)
                ),
                device: DeviceCapability(deviceClass: .iPhoneA16, totalRAMGB: 6, availableRAMGB: 4)
            )

            let request = AIRequest(messages: [AIMessage(role: .user, content: "a cat")])

            await #expect(throws: MLXVideoError.self) {
                _ = try await provider.complete(request: request)
            }
            // `stream` refuses on the same path, so one of the two cannot be
            // the lenient one.
            await #expect(throws: MLXVideoError.self) {
                _ = try await provider.stream(request: request)
            }
        }

        @Test("The refusal names the limit, not just the fact")
        func refusalNamesTheLimit() async throws {
            let preset = MLXVideoPreset(resolution: .uhd4K, frameRate: .fps120)
            let provider = MLXVideoProvider(
                id: "mlx/test",
                displayName: "Test",
                pythonPath: "/nonexistent/python3",
                modelName: "wan2.1-t2v-1.3b",
                defaultOptions: VideoGenerationOptions(preset: preset),
                device: DeviceCapability(deviceClass: .macM1, totalRAMGB: 16, availableRAMGB: 12)
            )
            let request = AIRequest(messages: [AIMessage(role: .user, content: "a cat")])

            do {
                _ = try await provider.complete(request: request)
                Issue.record("expected the preset to be refused")
            } catch let error as MLXVideoError {
                guard case .presetRefused(let refusal, let refused) = error else {
                    Issue.record("expected .presetRefused, got \(error)")
                    return
                }
                #expect(refused == preset)
                guard case .throughput(let required, let ceiling, _) = refusal else {
                    Issue.record("expected a throughput refusal, got \(refusal)")
                    return
                }
                #expect(required == preset.pixelsPerSecond)
                #expect(ceiling == MLXVideoPreset(resolution: .fullHD1080, frameRate: .fps60).pixelsPerSecond)
                // The message carries both numbers, so a log line is actionable.
                let description = try #require(error.errorDescription)
                #expect(description.contains("\(required)"))
                #expect(description.contains("\(ceiling)"))
            }
        }

        @Test("A preset the device serves is not refused")
        func servedPresetIsNotRefused() async throws {
            // The device accepts 720p24, so the refusal does not fire and the
            // provider proceeds to the subprocess — which fails, because the
            // python path does not exist. Reaching a LAUNCH failure instead of
            // a refusal is the assertion: an over-eager guard would have
            // stopped earlier.
            let provider = MLXVideoProvider(
                id: "mlx/test",
                displayName: "Test",
                pythonPath: "/nonexistent/python3",
                modelName: "wan2.1-t2v-1.3b",
                defaultOptions: VideoGenerationOptions(
                    preset: MLXVideoPreset(resolution: .hd720, frameRate: .fps24)
                ),
                device: DeviceCapability(deviceClass: .macM6, totalRAMGB: 64, availableRAMGB: 48)
            )
            let request = AIRequest(messages: [AIMessage(role: .user, content: "a cat")])

            do {
                _ = try await provider.complete(request: request)
            } catch let error as MLXVideoError {
                if case .presetRefused = error {
                    Issue.record("a served preset must not be refused: \(error)")
                }
            } catch {
                // Any other error is the launch failing, which is expected.
            }
        }

        @Test("A request with no user message is a typed refusal, not a crash")
        func noPromptIsTyped() async throws {
            let provider = MLXVideoProvider(
                id: "mlx/test",
                displayName: "Test",
                pythonPath: "/nonexistent/python3",
                modelName: "wan2.1-t2v-1.3b",
                device: DeviceCapability(deviceClass: .macM6, totalRAMGB: 64, availableRAMGB: 48)
            )
            let request = AIRequest(messages: [AIMessage(role: .assistant, content: "not a prompt")])

            await #expect(throws: MLXVideoError.noPrompt) {
                _ = try await provider.complete(request: request)
            }
        }

    #endif
}
