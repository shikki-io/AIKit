import Foundation
import TOMLKit

// MARK: - AIProviderSettings

/// The provider list as the operator wrote it: a `[[providers]]` array, in
/// file order.
///
/// BR-AIL-20. This is the SSoT for which providers exist, what they speak,
/// where they are and in what order they are tried — none of which is a code
/// branch any more. A new `openai-chat` server joins the fleet by adding a row
/// (BR-AIL-11); nothing is recompiled, and no environment variable is read
/// (the three that used to configure providers —`LMSTUDIO_URL`,
/// `LMSTUDIO_MODEL`, `MLX_LM_BASE_URL` — are gone from Sources).
///
/// ```toml
/// # providers.toml — tried in this order
/// [[providers]]
/// id      = "lmstudio"
/// family  = "openai-chat"        # or openai-compatible, which still decodes
/// address = "http://127.0.0.1:1234"
/// model   = "devstral-small-2507"
///
/// [[providers]]
/// id          = "claude-api"
/// family      = "anthropic-messages"
/// address     = "https://api.anthropic.com"
/// model       = "<model-id>"
/// api_key_ref = "broker://anthropic/api-key"   # a REFERENCE, never the key
/// ```
///
/// **AIKit does not decide where this file lives.** The host passes the
/// document's `URL` (shikki: `ShikkiResource.AIRoutingFile`), because a path
/// is the host's knowledge and a library that guessed one would be guessing
/// about somebody else's disk.
public struct AIProviderSettings: Sendable, Equatable, Codable {

    // MARK: - Decoder context

    /// Key under which the loader puts the settings file's name on the
    /// decoder, so a row that fails three levels down still names the file the
    /// operator has to edit.
    public static let fileNameKey = CodingUserInfoKey(rawValue: "AIKit.settingsFileName")!

    // MARK: - Properties

    /// Every configured provider, IN FILE ORDER.
    ///
    /// The order is the fallback order (BR-AIL-11): local first is the
    /// operator's default file, not a rule in code. Nothing here re-sorts it.
    public let providers: [AIProviderEntry]

    // MARK: - Init

    public init(providers: [AIProviderEntry]) {
        self.providers = providers
    }

    // MARK: - Loading

    /// Reads a settings document from disk.
    ///
    /// - Parameter url: the document the HOST chose.
    /// - Throws: ``AIKitError/settings(file:entry:reason:)`` naming the file,
    ///   and the row when one row is at fault. The whole file is refused on
    ///   the first bad entry (BR-AIL-18): a partially loaded provider list is
    ///   a configuration nobody wrote, and a silently dropped row is the
    ///   failure this spec exists to remove.
    public static func load(from url: URL) throws(AIKitError) -> AIProviderSettings {
        let text: String
        do {
            text = try String(contentsOf: url, encoding: .utf8)
        } catch {
            throw .settings(file: url.path, entry: nil, reason: "it could not be read: \(error.localizedDescription)")
        }
        return try parse(toml: text, file: url.path)
    }

    /// Reads a settings document from TOML text.
    ///
    /// - Parameters:
    ///   - toml: the document.
    ///   - file: the path to NAME in an error. Passed rather than derived so a
    ///     caller that already has the text (a test, a host that reads the
    ///     file itself) still produces an error an operator can act on.
    /// - Throws: as ``load(from:)``.
    public static func parse(toml: String, file: String) throws(AIKitError) -> AIProviderSettings {
        var decoder = TOMLDecoder()
        decoder.userInfo[fileNameKey] = file
        do {
            let settings = try decoder.decode(AIProviderSettings.self, from: toml)
            try settings.refuseDuplicateIDs(file: file)
            return settings
        } catch let error as AIKitError {
            // Already names the file and the row: an entry's own refusal
            // (a bad address, an unknown family) travels out whole.
            throw error
        } catch let error as TOMLParseError {
            throw .settings(file: file, entry: nil, reason: "it is not valid TOML: \(error)")
        } catch {
            throw .settings(file: file, entry: nil, reason: "\(error)")
        }
    }

    /// Refuses a document with two rows under one id.
    ///
    /// Two rows with one id is not a provider list: the second silently wins
    /// every lookup keyed by id, so the first is configured and unreachable.
    private func refuseDuplicateIDs(file: String) throws(AIKitError) {
        var seen: Set<AIProviderEntry.EntryID> = []
        for entry in providers {
            guard seen.insert(entry.id).inserted else {
                throw .settings(
                    file: file,
                    entry: entry.id,
                    reason: "two [[providers]] rows carry this id; an id names one provider"
                )
            }
        }
    }

    // MARK: - Reading

    /// The entry with this id, or nil.
    public func entry(id: AIProviderEntry.EntryID) -> AIProviderEntry? {
        providers.first { $0.id == id }
    }

    /// Every entry of one family, in file order.
    public func entries(family: AIProviderFamily) -> [AIProviderEntry] {
        providers.filter { $0.family == family }
    }
}
