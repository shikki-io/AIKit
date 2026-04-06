import Foundation

/// Determines translation behavior based on media type, source language, and user preferences.
public enum MediaLanguageResolver {

    /// Determine if translation is needed and what languages to use.
    public static func resolve(
        mediaType: MediaType,
        sourceLanguage: String,
        preferences: MediaLanguagePreferences
    ) -> TranslationDecision {
        guard let profile = preferences.profile(for: mediaType) else {
            // No profile for this media type — use default behavior.
            switch preferences.defaultSourceBehavior {
            case .keepOriginal:
                return .keepOriginal
            case .autoTranslate, .sideBySide:
                return .askUser
            }
        }

        let sourceLower = sourceLanguage.lowercased()
        let accepted = profile.acceptedSourceLanguages.map { $0.lowercased() }

        // Source language is accepted — no translation needed.
        if accepted.contains(sourceLower) {
            return .keepOriginal
        }

        // For audio/video media types, prefer subtitles if configured.
        if isAudioVisual(mediaType), let subtitleLang = profile.subtitleLanguage {
            return .subtitle(language: subtitleLang)
        }

        // Auto-translate if enabled.
        if profile.autoTranslate {
            return .translate(to: profile.preferredLanguage)
        }

        // Not accepted, no auto-translate — ask user.
        return .askUser
    }

    private static func isAudioVisual(_ mediaType: MediaType) -> Bool {
        switch mediaType {
        case .anime, .series, .movie, .youtube, .podcast:
            return true
        case .manga, .manhwa, .bd, .article, .book:
            return false
        }
    }
}

/// Result of language resolution for a piece of content.
public enum TranslationDecision: Sendable, Equatable {
    /// No translation needed — source language is accepted.
    case keepOriginal
    /// Translate to target language.
    case translate(to: String)
    /// Show subtitles in target language (for video/audio).
    case subtitle(language: String)
    /// User must decide (no profile or no auto-translate for this media type).
    case askUser
}
