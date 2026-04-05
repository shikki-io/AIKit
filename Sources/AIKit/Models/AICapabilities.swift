import Foundation

/// Capability flags for AI providers and models.
public struct AICapabilities: OptionSet, Sendable, Codable, Hashable {
    public let rawValue: Int

    public init(rawValue: Int) {
        self.rawValue = rawValue
    }

    public static let textGeneration  = AICapabilities(rawValue: 1 << 0)
    public static let translation     = AICapabilities(rawValue: 1 << 1)
    public static let ocr             = AICapabilities(rawValue: 1 << 2)
    public static let imageGeneration = AICapabilities(rawValue: 1 << 3)
    public static let voiceToText     = AICapabilities(rawValue: 1 << 4)
    public static let textToVoice     = AICapabilities(rawValue: 1 << 5)
    public static let inpainting      = AICapabilities(rawValue: 1 << 6)
    public static let embedding       = AICapabilities(rawValue: 1 << 7)
    public static let vision          = AICapabilities(rawValue: 1 << 8)
    public static let toolUse         = AICapabilities(rawValue: 1 << 9)
    public static let videoGeneration = AICapabilities(rawValue: 1 << 10)
}
