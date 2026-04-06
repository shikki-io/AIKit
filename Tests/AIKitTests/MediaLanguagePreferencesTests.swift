import Testing
@testable import AIKit

@Suite("MediaLanguagePreferences")
struct MediaLanguagePreferencesTests {

    @Test("Default preferences are VO-first (keepOriginal)")
    func defaultIsKeepOriginal() {
        let prefs = MediaLanguagePreferences()
        #expect(prefs.defaultSourceBehavior == .keepOriginal)
        #expect(prefs.profiles.isEmpty)
    }

    @Test("Manga profile auto-translates to FR")
    func mangaAutoTranslate() {
        let prefs = MediaLanguagePreferences(profiles: [
            MediaProfile(
                mediaType: .manga,
                sourceBehavior: .autoTranslate,
                preferredLanguage: "fr",
                acceptedSourceLanguages: ["fr"],
                autoTranslate: true,
                qualityThreshold: 0.8
            ),
        ])

        let manga = prefs.profile(for: .manga)
        #expect(manga != nil)
        #expect(manga?.sourceBehavior == .autoTranslate)
        #expect(manga?.preferredLanguage == "fr")
        #expect(manga?.autoTranslate == true)
        #expect(manga?.qualityThreshold == 0.8)
    }

    @Test("Article profile keeps original")
    func articleKeepsOriginal() {
        let prefs = MediaLanguagePreferences(profiles: [
            MediaProfile(
                mediaType: .article,
                sourceBehavior: .keepOriginal,
                preferredLanguage: "en",
                acceptedSourceLanguages: ["fr", "en"],
                autoTranslate: false
            ),
        ])

        let article = prefs.profile(for: .article)
        #expect(article != nil)
        #expect(article?.sourceBehavior == .keepOriginal)
        #expect(article?.autoTranslate == false)
    }

    @Test("Anime profile: VO + subtitle")
    func animeVoWithSubtitle() {
        let prefs = MediaLanguagePreferences(profiles: [
            MediaProfile(
                mediaType: .anime,
                sourceBehavior: .keepOriginal,
                preferredLanguage: "ja",
                acceptedSourceLanguages: ["ja", "fr", "en"],
                subtitleLanguage: "fr"
            ),
        ])

        let anime = prefs.profile(for: .anime)
        #expect(anime != nil)
        #expect(anime?.sourceBehavior == .keepOriginal)
        #expect(anime?.preferredLanguage == "ja")
        #expect(anime?.subtitleLanguage == "fr")
    }

    @Test("Profile lookup returns nil for missing media type")
    func missingProfile() {
        let prefs = MediaLanguagePreferences(profiles: [
            MediaProfile(
                mediaType: .manga,
                sourceBehavior: .autoTranslate,
                preferredLanguage: "fr",
                acceptedSourceLanguages: ["fr"]
            ),
        ])

        #expect(prefs.profile(for: .podcast) == nil)
    }

    @Test("All MediaType cases exist")
    func allMediaTypes() {
        let allTypes = MediaType.allCases
        #expect(allTypes.count == 10)
        #expect(allTypes.contains(.manga))
        #expect(allTypes.contains(.manhwa))
        #expect(allTypes.contains(.bd))
        #expect(allTypes.contains(.article))
        #expect(allTypes.contains(.book))
        #expect(allTypes.contains(.anime))
        #expect(allTypes.contains(.series))
        #expect(allTypes.contains(.movie))
        #expect(allTypes.contains(.youtube))
        #expect(allTypes.contains(.podcast))
    }
}
