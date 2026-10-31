import Foundation

#if canImport(Darwin)
    import Darwin
#endif

// MARK: - Device Class

/// Apple silicon device classes used for inference performance estimation.
///
/// Used by `ModelDescriptor.estimatedTokPerSec` to tell the user what to
/// expect on their hardware before they download a 4 GB model.
public enum DeviceClass: String, Sendable, Codable, Hashable, CaseIterable {
    case iPhoneA15  // iPhone 13 Pro
    case iPhoneA16  // iPhone 14 Pro (floor target)
    case iPhoneA17  // iPhone 15 Pro
    case iPadM1
    case iPadM2
    case macM1
    case macM2
    case macM3
    case macM4
    case macM5
}

// MARK: - Device Capability

/// Runtime snapshot of the current device's capability to load local models.
///
/// Combines a `DeviceClass` (A16, M4, …) with RAM numbers so callers can:
///   - block a download with insufficient free RAM
///   - show "expected ~20 tok/s" before the user commits 2.4 GB
///
/// Extracted from shikki-client's `Sources/ShikkiAppKit/Engine/` in W7 of spec
/// ai-layer-aikit-core-2026-10-31. The thermal, battery and memory *monitors*
/// stay in the app: they watch a running device over time, which is a UI
/// concern, while this is a single measurement a provider can refuse on.
public struct DeviceCapability: Sendable, Equatable {

    /// Apple silicon class detected at runtime.
    public let deviceClass: DeviceClass

    /// Total physical RAM in gigabytes.
    public let totalRAMGB: Double

    /// Available (free) RAM in gigabytes, measured at instance-creation time.
    ///
    /// Note: "available" on iOS is conservative — Apple counts active,
    /// inactive, and free pages; we subtract only wired + active.
    public let availableRAMGB: Double

    /// Conservative tokens-per-second floor reported for a model that carries
    /// no estimate for this device class, so the UI never promises zero speed.
    public static let estimatedTokPerSecFloor: Double = 10.0

    /// Create a capability snapshot from known numbers.
    ///
    /// Tests build one directly to describe a device they do not run on; live
    /// code calls `current()`.
    public init(
        deviceClass: DeviceClass,
        totalRAMGB: Double,
        availableRAMGB: Double
    ) {
        self.deviceClass = deviceClass
        self.totalRAMGB = totalRAMGB
        self.availableRAMGB = availableRAMGB
    }

    // MARK: - Detection

    /// Snapshot the current device's capability.
    public static func current() -> DeviceCapability {
        let total = totalRAMGigabytes()
        let available = availableRAMGigabytes(total: total)
        let deviceClass = detectDeviceClass(totalRAMGB: total)
        return DeviceCapability(
            deviceClass: deviceClass,
            totalRAMGB: total,
            availableRAMGB: available
        )
    }

    // MARK: - Queries

    /// Whether the model's minimum RAM fits current availability.
    ///
    /// A descriptor that declares no `minRAMGB` is not refused: the number is
    /// a measurement somebody made on a real device, and its absence means
    /// "unmeasured", never "needs nothing". Refusing on a missing measurement
    /// would block every model that has not been through the catalog.
    public func canLoad(_ model: ModelDescriptor) -> Bool {
        shortfall(for: model) == nil
    }

    /// The memory limit that stops this model loading, or `nil` when it fits.
    ///
    /// `canLoad` answers whether, this answers by how much. A refusal has to
    /// name its limit to be actionable — "needs 12.0 GB, 4.0 GB free" tells a
    /// user to close something, where `false` tells them nothing — and a
    /// caller that reads both numbers from here cannot drift from the verdict
    /// by re-deriving the comparison itself.
    public func shortfall(for model: ModelDescriptor) -> (required: Double, available: Double)? {
        guard let required = model.minRAMGB, availableRAMGB < required else { return nil }
        return (required: required, available: availableRAMGB)
    }

    /// Expected tokens-per-second for the given model on this device.
    ///
    /// Falls back to `estimatedTokPerSecFloor` when the model carries no
    /// estimate for this specific device class.
    public func estimatedTokPerSec(_ model: ModelDescriptor) -> Double {
        model.estimatedTokPerSec[deviceClass] ?? Self.estimatedTokPerSecFloor
    }

    // MARK: - Private: RAM

    private static func totalRAMGigabytes() -> Double {
        Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824.0
    }

    private static func availableRAMGigabytes(total: Double) -> Double {
        #if canImport(Darwin)
            var info = mach_task_basic_info()
            var count = mach_msg_type_number_t(
                MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size
            )

            let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) { infoPtr in
                infoPtr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { reboundPtr in
                    task_info(
                        mach_task_self_,
                        task_flavor_t(MACH_TASK_BASIC_INFO),
                        reboundPtr,
                        &count
                    )
                }
            }

            if kerr == KERN_SUCCESS {
                let usedBytes = Double(info.resident_size)
                let usedGB = usedBytes / 1_073_741_824.0
                // Reserve 15% for system processes. iOS jetsam kills at ~65-70%
                // total usage, so 15% reserve leaves enough headroom while not
                // blocking models that actually fit (e.g. Gemma 4 4B on 6GB devices).
                let systemReserveGB = total * 0.15
                let available = max(0.0, total - usedGB - systemReserveGB)
                return available
            }
        #endif
        // Fallback: assume 75% of total is available
        return total * 0.75
    }

    // MARK: - Private: Device class detection

    private static func detectDeviceClass(totalRAMGB: Double) -> DeviceClass {
        let machine = machineIdentifier()

        #if os(iOS) || os(tvOS) || targetEnvironment(macCatalyst)
            return iOSDeviceClass(machine: machine, totalRAMGB: totalRAMGB)
        #elseif os(macOS)
            return macDeviceClass(machine: machine, totalRAMGB: totalRAMGB)
        #else
            return .macM1
        #endif
    }

    /// `uname -m`-style machine identifier (e.g. `"iPhone15,2"`, `"Mac14,3"`).
    private static func machineIdentifier() -> String {
        #if canImport(Darwin)
            var sysinfo = utsname()
            uname(&sysinfo)
            let machine = withUnsafePointer(to: &sysinfo.machine) { ptr in
                ptr.withMemoryRebound(to: CChar.self, capacity: Int(_SYS_NAMELEN)) { cstr in
                    String(cString: cstr)
                }
            }
            return machine
        #else
            return ""
        #endif
    }

    private static func iOSDeviceClass(machine: String, totalRAMGB: Double) -> DeviceClass {
        // iPhone model identifiers (see https://www.theiphonewiki.com/wiki/Models)
        // iPhone 13 Pro         → iPhone14,2 / iPhone14,3  (A15)
        // iPhone 14 / 14 Plus   → iPhone14,7 / iPhone14,8  (A15)
        // iPhone 14 Pro / Pro Max → iPhone15,2 / iPhone15,3 (A16)
        // iPhone 15 / 15 Plus   → iPhone15,4 / iPhone15,5  (A16)
        // iPhone 15 Pro / Pro Max → iPhone16,1 / iPhone16,2 (A17)
        if machine.hasPrefix("iPhone14,") || machine.hasPrefix("iPhone13,") {
            return .iPhoneA15
        }
        if machine.hasPrefix("iPhone15,") {
            return .iPhoneA16
        }
        if machine.hasPrefix("iPhone16,") {
            return .iPhoneA17
        }
        // iPad M-series (simplified)
        if machine.hasPrefix("iPad14,") || machine.hasPrefix("iPad13,") {
            return .iPadM1
        }
        if machine.hasPrefix("iPad16,") || machine.hasPrefix("iPad15,") {
            return .iPadM2
        }
        // Unknown iPhone/iPad — guess from RAM
        if totalRAMGB >= 7.5 { return .iPhoneA17 }
        if totalRAMGB >= 5.5 { return .iPhoneA16 }
        return .iPhoneA15
    }

    private static func macDeviceClass(machine: String, totalRAMGB: Double) -> DeviceClass {
        // Apple silicon Mac identifiers:
        //   Mac13,x → M1 Pro/Max/Ultra
        //   Mac14,x → M2 / M2 Pro/Max/Ultra
        //   Mac15,x → M3 / M3 Pro/Max
        //   Mac16,x → M4 / M4 Pro/Max
        //   Mac17,x → M5 (projected)
        if machine.hasPrefix("Mac17,") { return .macM5 }
        if machine.hasPrefix("Mac16,") { return .macM4 }
        if machine.hasPrefix("Mac15,") { return .macM3 }
        if machine.hasPrefix("Mac14,") { return .macM2 }
        if machine.hasPrefix("Mac13,") || machine.hasPrefix("MacBookAir10,") || machine.hasPrefix("MacBookPro17,")
            || machine.hasPrefix("MacBookPro18,") || machine.hasPrefix("Macmini9,") || machine.hasPrefix("iMac21,")
        {
            return .macM1
        }
        // Unknown — conservative default
        if totalRAMGB >= 32 { return .macM3 }
        return .macM1
    }
}
