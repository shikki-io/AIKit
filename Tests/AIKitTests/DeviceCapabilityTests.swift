import Foundation
import Testing

@testable import AIKit

// MARK: - DeviceCapabilityTests
//
// W25 of spec `ai-layer-aikit-core-2026-10-31`, T-AIL-50: "M5/M6, iPhone 18,
// DUO, iPad rows present; an unsupported resolution × fps is refused by
// device."
//
// Every assertion here runs on whatever machine the suite runs on, because
// `DeviceCapability.deviceClass(forMachineIdentifier:totalRAMGB:platform:)` is
// pure and takes all three inputs. Before W25 the mapping was two private
// `#if`-gated functions reading `uname`, so the only row a test could reach was
// the one this Mac happens to be, and the other twenty were unverifiable.

@Suite("DeviceCapability")
struct DeviceCapabilityTests {

    // MARK: - The table covers every class

    @Test("Every DeviceClass has exactly one row")
    func everyClassHasOneRow() {
        for deviceClass in DeviceClass.allCases {
            let rows = DeviceCapability.deviceTable.filter { $0.deviceClass == deviceClass }
            #expect(
                rows.count == 1,
                "\(deviceClass.rawValue) has \(rows.count) rows; expected exactly 1"
            )
        }
        // And no row describes a class twice over, so `row(for:)` cannot pick
        // between two answers.
        #expect(DeviceCapability.deviceTable.count == DeviceClass.allCases.count)
    }

    @Test("row(for:) answers for every class without falling through")
    func rowForEveryClass() {
        for deviceClass in DeviceClass.allCases {
            let row = DeviceCapability.row(for: deviceClass)
            #expect(row.deviceClass == deviceClass)
            // `.other` is the conservative answer the fallback invents when a
            // class has no row. Reaching it means the table lost a row.
            #expect(row.platform != .other, "\(deviceClass.rawValue) fell through to the default row")
        }
    }

    // MARK: - The rows W25 was asked for

    @Test("The M5 and M6 Mac rows are present")
    func macM5AndM6RowsPresent() {
        for deviceClass in [DeviceClass.macM5, .macM6] {
            let row = DeviceCapability.row(for: deviceClass)
            #expect(row.platform == .macOS)
            #expect(row.machinePrefixes.isEmpty == false)
            // Both are budgeted the grid's top shape.
            #expect(row.videoCeiling == MLXVideoPreset(resolution: .uhd4K, frameRate: .fps60))
        }
        #expect(DeviceCapability.row(for: .macM5).machinePrefixes == ["Mac17,"])
        #expect(DeviceCapability.row(for: .macM6).machinePrefixes == ["Mac18,"])
    }

    @Test("The iPhone 18 line and the DUO are present")
    func iPhone18AndDuoRowsPresent() {
        let a20 = DeviceCapability.row(for: .iPhoneA20)
        #expect(a20.platform == .iOS)
        #expect(a20.machinePrefixes == ["iPhone19,"])
        #expect(a20.videoCeiling == MLXVideoPreset(resolution: .fullHD1080, frameRate: .fps30))

        let duo = DeviceCapability.row(for: .iPhoneDuo)
        #expect(duo.platform == .iOS)
        #expect(duo.videoCeiling == MLXVideoPreset(resolution: .fullHD1080, frameRate: .fps30))
        // The DUO carries NO prefix on purpose: a foldable's model identifier
        // is not known here, and an invented one would match a real device
        // wrongly. The row exists to carry the platform and the ceiling.
        #expect(duo.machinePrefixes.isEmpty)
        #expect(duo.provenance == .projected)
    }

    @Test("The current iPad rows are present, M1 through M5")
    func iPadRowsPresent() {
        let iPads: [DeviceClass] = [.iPadM1, .iPadM2, .iPadM3, .iPadM4, .iPadM5]
        for deviceClass in iPads {
            let row = DeviceCapability.row(for: deviceClass)
            #expect(row.platform == .iOS, "\(deviceClass.rawValue) should be an iOS-family host")
            #expect(row.machinePrefixes.isEmpty == false, "\(deviceClass.rawValue) needs an identifier")
            #expect(row.videoCeiling != nil, "\(deviceClass.rawValue) runs MLX and needs a ceiling")
        }
        // M3 onward is budgeted 60 fps where M1/M2 stop at 30.
        #expect(DeviceCapability.row(for: .iPadM2).videoCeiling?.frameRate == .fps30)
        #expect(DeviceCapability.row(for: .iPadM3).videoCeiling?.frameRate == .fps60)
    }

    @Test("Linux and Windows hosts are rows, with no video ceiling")
    func nonAppleHostRowsPresent() {
        let linux = DeviceCapability.row(for: .linuxHost)
        #expect(linux.platform == .linux)
        #expect(linux.videoCeiling == nil)

        let windows = DeviceCapability.row(for: .windowsHost)
        #expect(windows.platform == .windows)
        #expect(windows.videoCeiling == nil)
    }

    // MARK: - Identifier mapping

    @Test(
        "A machine identifier maps to its class",
        arguments: [
            ("iPhone13,2", DeviceClass.iPhoneA14),
            ("iPhone14,3", .iPhoneA15),
            ("iPhone15,2", .iPhoneA16),
            ("iPhone16,1", .iPhoneA17),
            ("iPhone17,1", .iPhoneA18),
            ("iPhone18,1", .iPhoneA19),
            ("iPhone19,1", .iPhoneA20),
            ("iPad13,8", .iPadM1),
            ("iPad14,5", .iPadM2),
            ("iPad16,6", .iPadM4),
            ("iPad17,1", .iPadM5),
        ]
    )
    func identifierMapsToClass(identifier: String, expected: DeviceClass) {
        #expect(
            DeviceCapability.deviceClass(
                forMachineIdentifier: identifier,
                totalRAMGB: 8,
                platform: .iOS
            ) == expected
        )
    }

    // Apple's published identifier ↔ chip list for every iPad with an M chip.
    // The pre-fix table matched the "iPad13," and "iPad14," families and put
    // the operator's iPad Pro 12.9" 6th gen (iPad14,5, M2) on M1 (backlog
    // 65e5866d); every row here is one real model.
    @Test(
        "Every M-series iPad identifier maps to its chip",
        arguments: [
            // M1: iPad Pro 11" 3rd gen, iPad Pro 12.9" 5th gen, iPad Air 5th gen.
            ("iPad13,4", DeviceClass.iPadM1), ("iPad13,5", .iPadM1), ("iPad13,6", .iPadM1), ("iPad13,7", .iPadM1),
            ("iPad13,8", .iPadM1), ("iPad13,9", .iPadM1), ("iPad13,10", .iPadM1), ("iPad13,11", .iPadM1),
            ("iPad13,16", .iPadM1), ("iPad13,17", .iPadM1),
            // M2: iPad Pro 11" 4th gen, iPad Pro 12.9" 6th gen, iPad Air 11"/13" M2.
            ("iPad14,3", .iPadM2), ("iPad14,4", .iPadM2), ("iPad14,5", .iPadM2), ("iPad14,6", .iPadM2),
            ("iPad14,8", .iPadM2), ("iPad14,9", .iPadM2), ("iPad14,10", .iPadM2), ("iPad14,11", .iPadM2),
            // M3: iPad Air 11"/13" M3.
            ("iPad15,3", .iPadM3), ("iPad15,4", .iPadM3), ("iPad15,5", .iPadM3), ("iPad15,6", .iPadM3),
            // M4: iPad Pro 11"/13" M4.
            ("iPad16,3", .iPadM4), ("iPad16,4", .iPadM4), ("iPad16,5", .iPadM4), ("iPad16,6", .iPadM4),
        ]
    )
    func iPadIdentifierMapsToChip(identifier: String, expected: DeviceClass) {
        #expect(
            DeviceCapability.deviceClass(forMachineIdentifier: identifier, totalRAMGB: 8, platform: .iOS)
                == expected
        )
    }

    // A-series iPads share a family with M-series ones, so they must not be
    // caught by an M row: an exact identifier never matches a longer one.
    @Test(
        "An A-series iPad is never mistaken for an M-series one",
        arguments: ["iPad13,1", "iPad13,2", "iPad13,18", "iPad13,19", "iPad14,1", "iPad14,2", "iPad15,7", "iPad16,1"]
    )
    func aSeriesIPadIsNotAnMChip(identifier: String) {
        let mClasses: Set<DeviceClass> = [.iPadM1, .iPadM2, .iPadM3, .iPadM4, .iPadM5]
        let measured = DeviceCapability.deviceClass(forMachineIdentifier: identifier, totalRAMGB: 4, platform: .iOS)
        #expect(!mClasses.contains(measured), "\(identifier) read as \(measured)")
    }

    @Test(
        "A Mac identifier maps to its class",
        arguments: [
            ("Mac13,1", DeviceClass.macM1),
            ("Mac14,2", .macM2),
            ("Mac15,3", .macM3),
            ("Mac16,1", .macM4),
            ("Mac17,1", .macM5),
            ("Mac18,1", .macM6),
            ("MacBookAir10,1", .macM1),
        ]
    )
    func macIdentifierMapsToClass(identifier: String, expected: DeviceClass) {
        #expect(
            DeviceCapability.deviceClass(
                forMachineIdentifier: identifier,
                totalRAMGB: 16,
                platform: .macOS
            ) == expected
        )
    }

    @Test("A prefix cannot be shadowed by a shorter one")
    func longestPrefixWins() {
        // "MacBookAir10," and "Mac13," both begin with "Mac1". Matching the
        // longest prefix first is what keeps them apart, and the trailing comma
        // is what makes each one unambiguous.
        #expect(
            DeviceCapability.deviceClass(
                forMachineIdentifier: "MacBookAir10,1",
                totalRAMGB: 8,
                platform: .macOS
            ) == .macM1
        )
        // A near-miss identifier must NOT match: "iPhone1,5" is not an
        // iPhone15,x and must fall through to the RAM fallback.
        //
        // The RAM matters to this assertion. At 6 GB the fallback ITSELF
        // answers .iPhoneA16, so a test written at 6 GB passes whether the
        // prefix matched or not and proves nothing. At 4 GB the fallback
        // answers .iPhoneA15, so the two outcomes are distinguishable and a
        // wrongly-matched prefix shows up.
        #expect(
            DeviceCapability.deviceClass(
                forMachineIdentifier: "iPhone1,5",
                totalRAMGB: 4,
                platform: .iOS
            ) == .iPhoneA15
        )
        // Pinned, so the reasoning above cannot silently stop being true.
        #expect(DeviceCapability.fallbackDeviceClass(totalRAMGB: 4, platform: .iOS) == .iPhoneA15)
        #expect(DeviceCapability.fallbackDeviceClass(totalRAMGB: 6, platform: .iOS) == .iPhoneA16)
    }

    // MARK: - The fallback stays on its own host

    @Test("An unknown identifier falls back within its platform")
    func fallbackStaysOnPlatform() {
        // The pre-W25 code answered a Mac class for an unrecognised iOS device
        // and `.macM1` on Linux, which named hardware the process was
        // certainly not running on.
        let unknownPhone = DeviceCapability.deviceClass(
            forMachineIdentifier: "iPhone99,9",
            totalRAMGB: 6,
            platform: .iOS
        )
        #expect(DeviceCapability.row(for: unknownPhone).platform == .iOS)

        #expect(
            DeviceCapability.deviceClass(forMachineIdentifier: "x86_64", totalRAMGB: 64, platform: .linux)
                == .linuxHost
        )
        #expect(
            DeviceCapability.deviceClass(forMachineIdentifier: "AMD64", totalRAMGB: 32, platform: .windows)
                == .windowsHost
        )
        // A host nobody listed reports a class with no MLX rather than
        // inventing Apple hardware.
        #expect(
            DeviceCapability.row(
                for:
                    DeviceCapability.deviceClass(forMachineIdentifier: "", totalRAMGB: 8, platform: .other)
            ).videoCeiling == nil)
    }

    @Test("The RAM fallback separates a tablet from a phone")
    func fallbackUsesRAMBands() {
        let tablet = DeviceCapability.deviceClass(
            forMachineIdentifier: "iPad99,9",
            totalRAMGB: 16,
            platform: .iOS
        )
        let phone = DeviceCapability.deviceClass(
            forMachineIdentifier: "iPhone99,9",
            totalRAMGB: 6,
            platform: .iOS
        )
        #expect(tablet == .iPadM4)
        #expect(phone == .iPhoneA16)
    }

    // MARK: - HostPlatform

    @Test("HostPlatform.current names the platform this test is running on")
    func hostPlatformCurrent() {
        #if os(macOS)
            #expect(HostPlatform.current == .macOS)
        #elseif os(iOS) || targetEnvironment(macCatalyst)
            #expect(HostPlatform.current == .iOS)
        #elseif os(Linux)
            #expect(HostPlatform.current == .linux)
        #elseif os(Windows)
            #expect(HostPlatform.current == .windows)
        #endif
    }

    // MARK: - current() still measures the real machine

    @Test("current() returns a capability with positive RAM")
    func currentReturnsValidCapability() {
        let capability = DeviceCapability.current()
        #expect(capability.totalRAMGB > 0)
        #expect(capability.availableRAMGB > 0)
        #expect(capability.availableRAMGB <= capability.totalRAMGB + 0.01)
    }

    @Test("current() reports a class belonging to this host")
    func currentReportsThisHostsFamily() {
        let capability = DeviceCapability.current()
        #expect(
            DeviceCapability.row(for: capability.deviceClass).platform == HostPlatform.current,
            "got \(capability.deviceClass.rawValue), which does not belong to \(HostPlatform.current.rawValue)"
        )
    }

    // MARK: - Video preset refusal (T-AIL-50)

    @Test("A device refuses a preset above its budget, and names the limit")
    func refusesAboveBudget() throws {
        // An A16 phone is budgeted 720p30. 4K at 240 fps is four orders of
        // magnitude past it.
        let phone = DeviceCapability(deviceClass: .iPhoneA16, totalRAMGB: 6, availableRAMGB: 4)
        let tooMuch = MLXVideoPreset(resolution: .uhd4K, frameRate: .fps240)

        #expect(phone.supports(tooMuch) == false)
        let refusal = try #require(phone.refusal(for: tooMuch))
        guard case .throughput(let required, let ceiling, let deviceClass) = refusal else {
            Issue.record("expected a throughput refusal, got \(refusal)")
            return
        }
        #expect(required == tooMuch.pixelsPerSecond)
        #expect(ceiling == MLXVideoPreset(resolution: .hd720, frameRate: .fps30).pixelsPerSecond)
        #expect(deviceClass == .iPhoneA16)
        #expect(required > ceiling)
    }

    @Test("A device accepts the preset it is budgeted, and its own ceiling")
    func acceptsWithinBudget() throws {
        let phone = DeviceCapability(deviceClass: .iPhoneA16, totalRAMGB: 6, availableRAMGB: 4)
        #expect(phone.supports(MLXVideoPreset(resolution: .hd720, frameRate: .fps24)))
        // The ceiling itself is allowed: a budget is inclusive, or the row's
        // own declared maximum would be refused.
        let ceiling = try #require(phone.videoCeiling)
        #expect(phone.supports(ceiling))
    }

    @Test("Every class accepts its own ceiling and refuses one tier past it")
    func everyCeilingIsAttainable() throws {
        for deviceClass in DeviceClass.allCases {
            let device = DeviceCapability(deviceClass: deviceClass, totalRAMGB: 16, availableRAMGB: 12)
            guard let ceiling = device.videoCeiling else {
                // No runtime at all: every preset is refused, for a reason that
                // is not about speed.
                #expect(device.supportedVideoPresets.isEmpty)
                let refusal = try #require(device.refusal(for: .compact))
                guard case .unsupportedPlatform(let platform) = refusal else {
                    Issue.record("\(deviceClass.rawValue) should refuse on platform, got \(refusal)")
                    continue
                }
                #expect(platform == DeviceCapability.row(for: deviceClass).platform)
                continue
            }
            #expect(device.supports(ceiling), "\(deviceClass.rawValue) refuses its own ceiling")
            #expect(
                device.supportedVideoPresets.isEmpty == false,
                "\(deviceClass.rawValue) has a ceiling but supports nothing"
            )
            // 4K at 240 fps is the grid's most expensive point; nothing in the
            // table is budgeted for it.
            #expect(
                device.supports(MLXVideoPreset(resolution: .uhd4K, frameRate: .fps240)) == false,
                "\(deviceClass.rawValue) claims to serve 4K240"
            )
        }
    }

    @Test("A Mac serves strictly more of the grid than a phone")
    func aMacOutranksAPhone() {
        let phone = DeviceCapability(deviceClass: .iPhoneA16, totalRAMGB: 6, availableRAMGB: 4)
        let mac = DeviceCapability(deviceClass: .macM6, totalRAMGB: 64, availableRAMGB: 48)

        let phonePresets = Set(phone.supportedVideoPresets)
        let macPresets = Set(mac.supportedVideoPresets)
        #expect(phonePresets.isSubset(of: macPresets))
        #expect(macPresets.count > phonePresets.count)
    }

    @Test("A host with no MLX refuses every preset on the grid")
    func noRuntimeRefusesEverything() {
        for deviceClass in [DeviceClass.linuxHost, .windowsHost] {
            let host = DeviceCapability(deviceClass: deviceClass, totalRAMGB: 64, availableRAMGB: 60)
            #expect(host.videoCeiling == nil)
            for preset in MLXVideoPreset.grid {
                #expect(host.supports(preset) == false, "\(deviceClass.rawValue) claims \(preset.displayName)")
            }
        }
    }

    // MARK: - The RAM queries W7 shipped, unchanged by W25

    @Test("canLoad refuses a model needing more RAM than is free")
    func canLoadRefusesOversized() {
        let capability = DeviceCapability(deviceClass: .iPhoneA16, totalRAMGB: 6.0, availableRAMGB: 4.0)
        #expect(capability.canLoad(makeModel(minRAMGB: 12.0)) == false)
    }

    @Test("shortfall names both numbers, and is nil when the model fits")
    func shortfallNamesBothNumbers() {
        let capability = DeviceCapability(deviceClass: .iPhoneA16, totalRAMGB: 6.0, availableRAMGB: 4.0)

        let shortfall = capability.shortfall(for: makeModel(minRAMGB: 12.0))
        #expect(shortfall?.required == 12.0)
        #expect(shortfall?.available == 4.0)

        #expect(capability.shortfall(for: makeModel(minRAMGB: 3.2)) == nil)
        // An unmeasured minimum is not a refusal.
        #expect(capability.shortfall(for: makeModel(minRAMGB: nil)) == nil)
    }

    @Test("estimatedTokPerSec reads this class, or falls back to the floor")
    func estimatedTokPerSecByClass() {
        let model = makeModel(estimatedTokPerSec: [.iPhoneA16: 20, .macM6: 90])
        #expect(
            DeviceCapability(deviceClass: .iPhoneA16, totalRAMGB: 6, availableRAMGB: 4)
                .estimatedTokPerSec(model) == 20
        )
        #expect(
            DeviceCapability(deviceClass: .macM6, totalRAMGB: 64, availableRAMGB: 48)
                .estimatedTokPerSec(model) == 90
        )
        // A class the model never measured gets the floor, never zero.
        #expect(
            DeviceCapability(deviceClass: .iPhoneDuo, totalRAMGB: 12, availableRAMGB: 8)
                .estimatedTokPerSec(model) == DeviceCapability.estimatedTokPerSecFloor
        )
    }

    // MARK: - Fixture

    private func makeModel(
        minRAMGB: Double? = nil,
        estimatedTokPerSec: [DeviceClass: Double] = [:]
    ) -> ModelDescriptor {
        ModelDescriptor(
            id: ModelIdentifier(provider: "test", modelId: "fixture"),
            name: "Fixture",
            author: "Test",
            description: "A descriptor built for this suite only",
            capabilities: .textGeneration,
            format: .mlx,
            parameters: "1B",
            quantization: "Q4",
            sizeBytes: 1_000_000,
            architecture: "fixture",
            domain: .llm,
            minRAMGB: minRAMGB,
            estimatedTokPerSec: estimatedTokPerSec
        )
    }
}
