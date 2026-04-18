import Testing
@testable import AIKit

@Suite("MediaLanguageResolver")
struct MediaLanguageResolverTests {

    // French user who reads manga in FR, articles in EN, watches anime in JP with FR subs.
    private var preferences: MediaLanguagePreferences {
        MediaLanguagePreferences(
            profiles: [
                MediaProfile(
                    mediaType: .manga,
                    sourceBehavior: .autoTranslate,
                    preferredLanguage: "fr",
                    acceptedSourceLanguages: ["fr"],
                    autoTranslate: true
                ),
                MediaProfile(
                    mediaType: .article,
                    sourceBehavior: .keepOriginal,
                    preferredLanguage: "en",
                    acceptedSourceLanguages: ["en", "fr"],
                    autoTranslate: true
                ),
                MediaProfile(
                    mediaType: .anime,
                    sourceBehavior: .keepOriginal,
                    preferredLanguage: "fr",
                    acceptedSourceLanguages: ["fr"],
                    subtitleLanguage: "fr",
                    autoTranslate: false
                ),
            ],
            defaultSourceBehavior: .keepOriginal
        )
    }

    @Test("Manga in FR source — keepOriginal")
    func mangaFrench() {
        let decision = MediaLanguageResolver.resolve(
            mediaType: .manga, sourceLanguage: "fr", preferences: preferences
        )
        #expect(decision == .keepOriginal)
    }

    @Test("Manga in JP source — translate to FR")
    func mangaJapanese() {
        let decision = MediaLanguageResolver.resolve(
            mediaType: .manga, sourceLanguage: "jp", preferences: preferences
        )
        #expect(decision == .translate(to: "fr"))
    }

    @Test("Article in EN — keepOriginal (EN is accepted)")
    func articleEnglish() {
        let decision = MediaLanguageResolver.resolve(
            mediaType: .article, sourceLanguage: "en", preferences: preferences
        )
        #expect(decision == .keepOriginal)
    }

    @Test("Article in JP — translate to EN (autoTranslate enabled)")
    func articleJapaneseAutoTranslate() {
        let decision = MediaLanguageResolver.resolve(
            mediaType: .article, sourceLanguage: "jp", preferences: preferences
        )
        #expect(decision == .translate(to: "en"))
    }

    @Test("Article in JP — askUser when autoTranslate is off")
    func articleJapaneseNoAutoTranslate() {
        var prefs = preferences
        prefs.profiles = prefs.profiles.map { profile in
            guard profile.mediaType == .article else { return profile }
            var p = profile
            p.autoTranslate = false
            return p
        }

        let decision = MediaLanguageResolver.resolve(
            mediaType: .article, sourceLanguage: "jp", preferences: prefs
        )
        #expect(decision == .askUser)
    }

    @Test("Anime in JP — subtitle in FR")
    func animeJapanese() {
        let decision = MediaLanguageResolver.resolve(
            mediaType: .anime, sourceLanguage: "jp", preferences: preferences
        )
        #expect(decision == .subtitle(language: "fr"))
    }

    @Test("Unknown media type (no profile) — uses default keepOriginal")
    func unknownMediaTypeKeepOriginal() {
        let decision = MediaLanguageResolver.resolve(
            mediaType: .podcast, sourceLanguage: "de", preferences: preferences
        )
        #expect(decision == .keepOriginal)
    }

    @Test("No profile with autoTranslate default — askUser")
    func noProfileAutoTranslateDefault() {
        let prefs = MediaLanguagePreferences(
            profiles: [],
            defaultSourceBehavior: .autoTranslate
        )
        let decision = MediaLanguageResolver.resolve(
            mediaType: .book, sourceLanguage: "de", preferences: prefs
        )
        #expect(decision == .askUser)
    }

    @Test("Case insensitive language matching")
    func caseInsensitive() {
        let decision = MediaLanguageResolver.resolve(
            mediaType: .manga, sourceLanguage: "FR", preferences: preferences
        )
        #expect(decision == .keepOriginal)
    }
}
