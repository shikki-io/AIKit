import Foundation

/// Per-media-type language preferences.
/// "I read manga in FR, articles in EN, watch anime in JP+FR sub"
public struct MediaLanguagePreferences: Sendable, Codable, Equatable {
    public var profiles: [MediaProfile]
    public var defaultSourceBehavior: SourceBehavior

    public init(
        profiles: [MediaProfile] = [],
        defaultSourceBehavior: SourceBehavior = .keepOriginal
    ) {
        self.profiles = profiles
        self.defaultSourceBehavior = defaultSourceBehavior
    }

    /// Look up the profile for a given media type.
    public func profile(for mediaType: MediaType) -> MediaProfile? {
        profiles.first { $0.mediaType == mediaType }
    }
}

/// Language and translation preferences for a specific media type.
public struct MediaProfile: Sendable, Codable, Identifiable, Equatable {
    public let id: String
    public var mediaType: MediaType
    public var sourceBehavior: SourceBehavior
    /// Preferred consumption language (translation target).
    public var preferredLanguage: String
    /// Languages the user can consume without translation.
    public var acceptedSourceLanguages: [String]
    /// Subtitle language when source is foreign.
    public var subtitleLanguage: String?
    /// Whether auto-translate is enabled for this media type.
    public var autoTranslate: Bool
    /// Quality threshold: skip translation if below this quality score.
    public var qualityThreshold: Double?

    public init(
        id: String = UUID().uuidString,
        mediaType: MediaType,
        sourceBehavior: SourceBehavior,
        preferredLanguage: String,
        acceptedSourceLanguages: [String],
        subtitleLanguage: String? = nil,
        autoTranslate: Bool = false,
        qualityThreshold: Double? = nil
    ) {
        self.id = id
        self.mediaType = mediaType
        self.sourceBehavior = sourceBehavior
        self.preferredLanguage = preferredLanguage
        self.acceptedSourceLanguages = acceptedSourceLanguages
        self.subtitleLanguage = subtitleLanguage
        self.autoTranslate = autoTranslate
        self.qualityThreshold = qualityThreshold
    }
}

/// Media types supported by the language preference system.
public enum MediaType: String, Sendable, Codable, CaseIterable {
    case manga
    case manhwa
    /// Bande dessinee (French comics).
    case bd
    case article
    case book
    case anime
    case series
    case movie
    case youtube
    case podcast
}

/// How to handle the original language of content.
public enum SourceBehavior: String, Sendable, Codable {
    /// Always show original (VO). Translation is manual opt-in.
    case keepOriginal
    /// Auto-translate to preferred language.
    case autoTranslate
    /// Show original + translation side by side.
    case sideBySide
}
