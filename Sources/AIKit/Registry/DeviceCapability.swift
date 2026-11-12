import Foundation

#if canImport(Darwin)
    import Darwin
#endif

// MARK: - HostPlatform

/// The operating-system family this binary was compiled for.
///
/// W25 of spec `ai-layer-aikit-core-2026-10-31`. AIKit runs on four hosts now,
/// and two of its decisions turn on which one: whether a CLI-backed provider
/// can exist at all (``CLIProviderFamily``), and whether there is an MLX
/// runtime to generate video with. Before this type both were answered by an
/// `#if` at the point of use, so the answer could not be named, returned or
/// tested — only compiled away.
///
/// `other` exists because a platform nobody listed must not read as one of the
/// four. A new Apple OS is `other` until a row says otherwise, which refuses
/// safely rather than guessing.
public enum HostPlatform: String, Sendable, Codable, Hashable, CaseIterable {

    /// macOS, the only host with both a subprocess API and an MLX runtime.
    case macOS

    /// iOS, and the Apple platforms that share its SDK shape — tvOS, watchOS,
    /// visionOS, and Mac Catalyst.
    ///
    /// Catalyst sits here rather than with ``macOS`` on purpose: it runs on Mac
    /// hardware but builds against the iOS SDK, and `Package.swift` conditions
    /// ShellKit on `.macOS`/`.linux`, so a Catalyst build links no ShellKit and
    /// has no subprocess provider — the same situation as an iPhone.
    case iOS

    /// Linux. ShellKit builds here, MLX does not.
    case linux

    /// Windows. Neither ShellKit nor MLX; `PATH` is also separated differently
    /// (``SearchPathSeparator``).
    case windows

    /// A host no row describes.
    case other

    /// The platform this binary was compiled for.
    public static var current: HostPlatform {
        #if os(macOS)
            .macOS
        #elseif os(iOS) || os(tvOS) || os(watchOS) || os(visionOS) || targetEnvironment(macCatalyst)
            .iOS
        #elseif os(Linux)
            .linux
        #elseif os(Windows)
            .windows
        #else
            .other
        #endif
    }
}

// MARK: - Device Class

/// The hardware class inference runs on.
///
/// Named for the SoC rather than the marketing name, because the SoC is what
/// decides whether a model loads and how fast it answers: one entry covers
/// every device carrying that chip.
///
/// Used by `ModelDescriptor.estimatedTokPerSec` to tell the user what to expect
/// on their hardware before they download a 4 GB model, and by
/// ``DeviceCapability/refusal(for:)`` to refuse a video preset the class cannot
/// serve.
///
/// Raw values are the case names and are never renamed: a stored
/// `estimatedTokPerSec` map is keyed by them, so a rename would silently drop
/// a measurement on decode. Adding a case is safe; changing one is not.
public enum DeviceClass: String, Sendable, Codable, Hashable, CaseIterable {

    // MARK: iPhone

    /// A14 — iPhone 12 family.
    case iPhoneA14
    /// A15 — iPhone 13 family, iPhone 14 and 14 Plus.
    case iPhoneA15
    /// A16 — iPhone 14 Pro, iPhone 15 and 15 Plus. The fleet's floor target.
    case iPhoneA16
    /// A17 Pro — iPhone 15 Pro.
    case iPhoneA17
    /// A18 — iPhone 16 family.
    case iPhoneA18
    /// A19 — iPhone 17 family.
    case iPhoneA19
    /// A20 — the iPhone 18 line.
    case iPhoneA20
    /// The foldable iPhone, the DUO.
    ///
    /// A class of its own rather than an A20 alias: a fold has two display
    /// geometries and a thermal envelope that is not the bar's, so a video
    /// ceiling shared with the bar phone would be a guess dressed as a
    /// measurement.
    case iPhoneDuo

    // MARK: iPad

    /// M1 — iPad Pro and iPad Air of that generation.
    case iPadM1
    /// M2 — iPad Pro and iPad Air of that generation.
    case iPadM2
    /// M3 — iPad Air of that generation.
    case iPadM3
    /// M4 — iPad Pro of that generation.
    case iPadM4
    /// M5 — current iPad Pro.
    case iPadM5

    // MARK: Mac

    case macM1
    case macM2
    case macM3
    case macM4
    case macM5
    case macM6

    // MARK: Hosts with no Apple silicon

    /// A Linux host. ShellKit builds here, so a CLI provider works; MLX does
    /// not, so no local video or in-process inference does.
    case linuxHost

    /// A Windows host. No ShellKit in this package's dependency conditions and
    /// no MLX.
    case windowsHost
}

// MARK: - Provenance

/// Whether a table row's machine prefixes were read off a shipped device or
/// forecast from Apple's numbering.
///
/// Carried in the data instead of a comment because the two must never read
/// alike. A projected prefix is a guess about a model identifier that may not
/// exist: when it is wrong the identifier simply misses the table and
/// ``DeviceCapability`` falls back to RAM, which is why a wrong guess degrades
/// instead of lying.
public enum DeviceRowProvenance: String, Sendable, Codable, Hashable, CaseIterable {

    /// The prefixes were read from a device that shipped.
    case shipped

    /// The prefixes are a forecast of Apple's numbering and are unverified.
    case projected
}

// MARK: - Device table row

/// One row of the device table: a class, the host it belongs to, the machine
/// identifiers that mean it, and the most video it may attempt.
///
/// The table is the single place these four facts live. Before W25 they were
/// spread across two `if`-chains and a comment block, so adding a chip meant
/// editing control flow in three places and the video ceiling had nowhere to
/// live at all.
public struct DeviceRow: Sendable, Equatable {

    /// The class this row describes.
    public let deviceClass: DeviceClass

    /// The host family the class belongs to. Keeps a RAM fallback from
    /// answering `macM1` on an iPhone, which the pre-W25 `#else` branch did.
    public let platform: HostPlatform

    /// `uname -m` prefixes that identify this class, comma-terminated.
    ///
    /// The trailing comma is what makes a prefix unambiguous: `"iPhone15,"`
    /// cannot also match `iPhone1,5`, so plain prefix matching is enough and no
    /// row can shadow another.
    ///
    /// May be empty, which means "no identifier is known for this class". Such
    /// a row still carries the platform and the video ceiling, so a caller that
    /// learns the class some other way — from settings, or from a later
    /// identifier — gets a correct verdict out of it.
    public let machinePrefixes: [String]

    /// Whether ``machinePrefixes`` is measured or forecast.
    public let provenance: DeviceRowProvenance

    /// The largest video preset this class may attempt, or `nil` when it has no
    /// MLX video runtime at all.
    ///
    /// A DECLARED BUDGET, not a benchmark. Nothing here was timed: the number
    /// exists so that a request which could never finish is refused up front
    /// instead of appearing to hang, and the rule setting it is one line —
    /// a class is budgeted the preset one tier below the next class up, scaled
    /// by its memory-bandwidth tier. Replace a row with a measurement whenever
    /// one is taken; that is a strictly better basis than this.
    public let videoCeiling: MLXVideoPreset?

    public init(
        deviceClass: DeviceClass,
        platform: HostPlatform,
        machinePrefixes: [String],
        provenance: DeviceRowProvenance,
        videoCeiling: MLXVideoPreset?
    ) {
        self.deviceClass = deviceClass
        self.platform = platform
        self.machinePrefixes = machinePrefixes
        self.provenance = provenance
        self.videoCeiling = videoCeiling
    }
}

// MARK: - Video preset refusal

/// Why a device may not attempt a video preset.
///
/// Each case names its limit, for the same reason
/// ``DeviceCapability/shortfall(for:)`` returns both numbers: "4K at 120 fps
/// needs 995 Mpx/s, this M4 is budgeted 249" tells a caller what to ask for
/// instead, where `false` tells them nothing.
public enum VideoPresetRefusal: Sendable, Equatable {

    /// The host has no MLX video runtime, whatever the preset.
    case unsupportedPlatform(HostPlatform)

    /// The preset asks for more pixels per second than the class is budgeted.
    case throughput(required: Int, ceiling: Int, deviceClass: DeviceClass)
}

// MARK: - Device Capability

/// Runtime snapshot of the current device's capability to load local models.
///
/// Combines a `DeviceClass` (A16, M4, …) with RAM numbers so callers can:
///   - block a download with insufficient free RAM
///   - show "expected ~20 tok/s" before the user commits 2.4 GB
///   - refuse a video preset the hardware cannot serve (W25)
///
/// Extracted from shikki-client's `Sources/ShikkiAppKit/Engine/` in W7 of spec
/// ai-layer-aikit-core-2026-10-31. The thermal, battery and memory *monitors*
/// stay in the app: they watch a running device over time, which is a UI
/// concern, while this is a single measurement a provider can refuse on.
public struct DeviceCapability: Sendable, Equatable {

    /// Hardware class detected at runtime.
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

    // MARK: - The device table

    /// Every device class, with its host, its machine identifiers and its video
    /// budget. One table, in class order.
    ///
    /// Ceilings follow one rule: a phone is budgeted 720p30 until the A18 line,
    /// then 1080p30; a tablet 1080p30 on M1/M2 and 1080p60 from M3; a Mac
    /// 1080p60 on M1/M2, 4K30 on M3/M4 and 4K60 on M5/M6. Linux and Windows
    /// carry `nil`: MLX is Apple-silicon only, so every preset is refused there
    /// for a reason that has nothing to do with speed.
    public static let deviceTable: [DeviceRow] = [
        // iPhone. Identifiers are the model-identifier series, not the
        // marketing generation: iPhone15,2 is an iPhone 14 Pro.
        DeviceRow(
            deviceClass: .iPhoneA14,
            platform: .iOS,
            machinePrefixes: ["iPhone13,"],
            provenance: .shipped,
            videoCeiling: MLXVideoPreset(resolution: .hd720, frameRate: .fps30)
        ),
        DeviceRow(
            deviceClass: .iPhoneA15,
            platform: .iOS,
            machinePrefixes: ["iPhone14,"],
            provenance: .shipped,
            videoCeiling: MLXVideoPreset(resolution: .hd720, frameRate: .fps30)
        ),
        DeviceRow(
            deviceClass: .iPhoneA16,
            platform: .iOS,
            machinePrefixes: ["iPhone15,"],
            provenance: .shipped,
            videoCeiling: MLXVideoPreset(resolution: .hd720, frameRate: .fps30)
        ),
        DeviceRow(
            deviceClass: .iPhoneA17,
            platform: .iOS,
            machinePrefixes: ["iPhone16,"],
            provenance: .shipped,
            videoCeiling: MLXVideoPreset(resolution: .hd720, frameRate: .fps30)
        ),
        DeviceRow(
            deviceClass: .iPhoneA18,
            platform: .iOS,
            machinePrefixes: ["iPhone17,"],
            provenance: .shipped,
            videoCeiling: MLXVideoPreset(resolution: .fullHD1080, frameRate: .fps30)
        ),
        DeviceRow(
            deviceClass: .iPhoneA19,
            platform: .iOS,
            machinePrefixes: ["iPhone18,"],
            provenance: .projected,
            videoCeiling: MLXVideoPreset(resolution: .fullHD1080, frameRate: .fps30)
        ),
        DeviceRow(
            deviceClass: .iPhoneA20,
            platform: .iOS,
            machinePrefixes: ["iPhone19,"],
            provenance: .projected,
            videoCeiling: MLXVideoPreset(resolution: .fullHD1080, frameRate: .fps30)
        ),
        // The DUO carries no prefix: a foldable's model identifier is not
        // something this table can state without inventing it, and an invented
        // prefix would match a real device wrongly. Reached by explicit
        // construction until an identifier is measured, and refused by RAM
        // fallback meanwhile — never silently mapped.
        DeviceRow(
            deviceClass: .iPhoneDuo,
            platform: .iOS,
            machinePrefixes: [],
            provenance: .projected,
            videoCeiling: MLXVideoPreset(resolution: .fullHD1080, frameRate: .fps30)
        ),

        // iPad. The M1/M2 rows keep the pre-W25 mapping, which lumped the
        // iPad13/14 and iPad15/16 series; it is coarse, and it is what shipped.
        DeviceRow(
            deviceClass: .iPadM1,
            platform: .iOS,
            machinePrefixes: ["iPad13,", "iPad14,"],
            provenance: .shipped,
            videoCeiling: MLXVideoPreset(resolution: .fullHD1080, frameRate: .fps30)
        ),
        DeviceRow(
            deviceClass: .iPadM2,
            platform: .iOS,
            machinePrefixes: ["iPad15,", "iPad16,"],
            provenance: .shipped,
            videoCeiling: MLXVideoPreset(resolution: .fullHD1080, frameRate: .fps30)
        ),
        DeviceRow(
            deviceClass: .iPadM3,
            platform: .iOS,
            machinePrefixes: ["iPad17,"],
            provenance: .projected,
            videoCeiling: MLXVideoPreset(resolution: .fullHD1080, frameRate: .fps60)
        ),
        DeviceRow(
            deviceClass: .iPadM4,
            platform: .iOS,
            machinePrefixes: ["iPad18,"],
            provenance: .projected,
            videoCeiling: MLXVideoPreset(resolution: .fullHD1080, frameRate: .fps60)
        ),
        DeviceRow(
            deviceClass: .iPadM5,
            platform: .iOS,
            machinePrefixes: ["iPad19,"],
            provenance: .projected,
            videoCeiling: MLXVideoPreset(resolution: .fullHD1080, frameRate: .fps60)
        ),

        // Mac.
        DeviceRow(
            deviceClass: .macM1,
            platform: .macOS,
            machinePrefixes: [
                "Mac13,", "MacBookAir10,", "MacBookPro17,", "MacBookPro18,", "Macmini9,", "iMac21,",
            ],
            provenance: .shipped,
            videoCeiling: MLXVideoPreset(resolution: .fullHD1080, frameRate: .fps60)
        ),
        DeviceRow(
            deviceClass: .macM2,
            platform: .macOS,
            machinePrefixes: ["Mac14,"],
            provenance: .shipped,
            videoCeiling: MLXVideoPreset(resolution: .fullHD1080, frameRate: .fps60)
        ),
        DeviceRow(
            deviceClass: .macM3,
            platform: .macOS,
            machinePrefixes: ["Mac15,"],
            provenance: .shipped,
            videoCeiling: MLXVideoPreset(resolution: .uhd4K, frameRate: .fps30)
        ),
        DeviceRow(
            deviceClass: .macM4,
            platform: .macOS,
            machinePrefixes: ["Mac16,"],
            provenance: .shipped,
            videoCeiling: MLXVideoPreset(resolution: .uhd4K, frameRate: .fps30)
        ),
        DeviceRow(
            deviceClass: .macM5,
            platform: .macOS,
            machinePrefixes: ["Mac17,"],
            provenance: .projected,
            videoCeiling: MLXVideoPreset(resolution: .uhd4K, frameRate: .fps60)
        ),
        DeviceRow(
            deviceClass: .macM6,
            platform: .macOS,
            machinePrefixes: ["Mac18,"],
            provenance: .projected,
            videoCeiling: MLXVideoPreset(resolution: .uhd4K, frameRate: .fps60)
        ),

        // Hosts with no MLX. `nil` ceiling, so every preset is refused with
        // `.unsupportedPlatform` rather than a throughput number that would
        // imply a faster machine could do it.
        DeviceRow(
            deviceClass: .linuxHost,
            platform: .linux,
            machinePrefixes: [],
            provenance: .shipped,
            videoCeiling: nil
        ),
        DeviceRow(
            deviceClass: .windowsHost,
            platform: .windows,
            machinePrefixes: [],
            provenance: .shipped,
            videoCeiling: nil
        ),
    ]

    /// The row for a class. Total by construction: every ``DeviceClass`` has a
    /// row, which ``DeviceCapabilityTests`` pins, so this cannot be reached
    /// with a class the table does not describe.
    public static func row(for deviceClass: DeviceClass) -> DeviceRow {
        deviceTable.first { $0.deviceClass == deviceClass }
            // A class with no row would be a table edit that forgot one. The
            // test catches it; this keeps a release from trapping on it, and
            // reports the most conservative class rather than the nearest.
            ?? DeviceRow(
                deviceClass: deviceClass,
                platform: .other,
                machinePrefixes: [],
                provenance: .projected,
                videoCeiling: nil
            )
    }

    // MARK: - Detection

    /// Snapshot the current device's capability.
    public static func current() -> DeviceCapability {
        let total = totalRAMGigabytes()
        let available = availableRAMGigabytes(total: total)
        let deviceClass = self.deviceClass(
            forMachineIdentifier: machineIdentifier(),
            totalRAMGB: total,
            platform: .current
        )
        return DeviceCapability(
            deviceClass: deviceClass,
            totalRAMGB: total,
            availableRAMGB: available
        )
    }

    /// The class a machine identifier means on a given host.
    ///
    /// Pure, and takes all three inputs, so a test can assert the whole table
    /// from any machine instead of only the row this Mac happens to be. That is
    /// the point of extracting it: before W25 the mapping lived inside two
    /// private `#if`-gated functions and nothing could reach it.
    ///
    /// - Parameters:
    ///   - machineIdentifier: a `uname -m` value such as `"iPhone17,1"`.
    ///   - totalRAMGB: physical RAM, used only when no row matches.
    ///   - platform: the host, which bounds the fallback to classes that
    ///     belong to it.
    public static func deviceClass(
        forMachineIdentifier machineIdentifier: String,
        totalRAMGB: Double,
        platform: HostPlatform
    ) -> DeviceClass {
        // Longest prefix first, so a longer row can never be shadowed by a
        // shorter one that happens to prefix it.
        let matched =
            deviceTable
            .flatMap { row in row.machinePrefixes.map { (prefix: $0, deviceClass: row.deviceClass) } }
            .sorted { $0.prefix.count > $1.prefix.count }
            .first { machineIdentifier.hasPrefix($0.prefix) }

        if let matched { return matched.deviceClass }
        return fallbackDeviceClass(totalRAMGB: totalRAMGB, platform: platform)
    }

    /// The class to report when no identifier matched.
    ///
    /// Bounded by platform: an unknown iPhone can only come back as an iPhone
    /// class. The pre-W25 code answered `.macM1` on Linux and could answer a
    /// Mac class for an unrecognised iOS device, which made the result read as
    /// hardware the process was certainly not running on.
    static func fallbackDeviceClass(totalRAMGB: Double, platform: HostPlatform) -> DeviceClass {
        switch platform {
        case .iOS:
            // RAM is the only signal left. An iPad carries more than a phone of
            // the same era, so the top band is a tablet.
            if totalRAMGB >= 11.5 { return .iPadM4 }
            if totalRAMGB >= 7.5 { return .iPhoneA19 }
            if totalRAMGB >= 5.5 { return .iPhoneA16 }
            return .iPhoneA15
        case .macOS:
            if totalRAMGB >= 32 { return .macM3 }
            return .macM1
        case .linux:
            return .linuxHost
        case .windows:
            return .windowsHost
        case .other:
            // Nothing is known about this host, so report the class with no MLX
            // runtime and no subprocess promise rather than inventing hardware.
            return .linuxHost
        }
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

    // MARK: - Queries: video presets

    /// This device's row in ``deviceTable``.
    public var row: DeviceRow { Self.row(for: deviceClass) }

    /// The largest video preset this device may attempt, or `nil` on a host
    /// with no MLX video runtime.
    public var videoCeiling: MLXVideoPreset? { row.videoCeiling }

    /// Whether this device may attempt the preset.
    public func supports(_ preset: MLXVideoPreset) -> Bool {
        refusal(for: preset) == nil
    }

    /// Why this device may not attempt the preset, or `nil` when it may.
    ///
    /// The `supports` / `refusal` pair mirrors `canLoad` / `shortfall`
    /// deliberately: one of them answers whether, the other answers why, and
    /// the verdict is computed once here so the two can never disagree.
    public func refusal(for preset: MLXVideoPreset) -> VideoPresetRefusal? {
        let row = self.row
        guard let ceiling = row.videoCeiling else {
            return .unsupportedPlatform(row.platform)
        }
        guard preset.pixelsPerSecond <= ceiling.pixelsPerSecond else {
            return .throughput(
                required: preset.pixelsPerSecond,
                ceiling: ceiling.pixelsPerSecond,
                deviceClass: deviceClass
            )
        }
        return nil
    }

    /// Every preset of the resolution × frame-rate grid this device may
    /// attempt, in grid order.
    public var supportedVideoPresets: [MLXVideoPreset] {
        MLXVideoPreset.grid.filter(supports)
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

    // MARK: - Private: machine identifier

    /// `uname -m`-style machine identifier (e.g. `"iPhone15,2"`, `"Mac14,3"`).
    static func machineIdentifier() -> String {
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
}
