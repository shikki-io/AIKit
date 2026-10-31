import AIKit
import Foundation
import MLXLMCommon
import Tokenizers

// MARK: - Downloader

/// `MLXLMCommon.Downloader` backed by AIKit's one model root and its checksums.
///
/// W13 of spec ai-layer-aikit-core-2026-10-31. mlx-swift-lm 3.x made
/// downloading a protocol with no shipped conformance, and that is the opening
/// the two measured problems needed:
///
/// - **one root.** 2.x reached HuggingFace through `HubApi`, which keeps its own
///   cache under `~/.cache/huggingface/hub`, while this fleet's other
///   downloader wrote `~/Library/Caches/models`. Measured on 2026-10-31:
///   `mlx-community/Mistral-7B-Instruct-v0.3-4bit`'s `model.safetensors` sits
///   in BOTH, 4077480215 bytes each, inodes 939711379 and 939711382, both
///   hashing to `d7182d88…503d`. 4 GB spent twice for one model. Everything
///   here writes under `ModelStore.root` and nowhere else.
/// - **a verified download.** `HubApi` compared a local hash to the expected
///   one in offline mode only, so a normal download was accepted on its HTTP
///   status. Every file that lands here is hashed against the digest the hub
///   published, and a mismatch is deleted rather than loaded.
///
/// Reuse comes first and costs nothing: a file already present under a
/// configured extra root — `~/.lmstudio/models` on a Mac — is verified and
/// **hard-linked**, so a model the operator already downloaded through LM
/// Studio is not downloaded again and takes no second space.
///
/// The two sides that touch the network are injected closures rather than new
/// protocols: `AIKit` already owns a `HuggingFaceClient` and a
/// `ModelDownloader`, and the defaults are wired to them. A test passes its own
/// and reaches no server (BR-AIL-12).
public struct MLXHubDownloader: Downloader {

    /// Everything the hub says a repository contains, with each file's
    /// published checksum.
    public typealias FileIndex =
        @Sendable (_ repository: String, _ revision: String) async throws -> [ModelFile]

    /// Move one file's bytes to `destination`, reporting `(written, total)`.
    public typealias FileTransfer =
        @Sendable (
            _ repository: String,
            _ revision: String,
            _ path: String,
            _ destination: URL,
            _ onBytes: @Sendable @escaping (Int64, Int64) -> Void
        ) async throws -> Void

    /// Told `(file path, bytes hashed, bytes total)` while a file is verified.
    ///
    /// Separate from the download's own progress on purpose: hashing 4 GB is
    /// seconds of work that moves no bytes over the network, and a caller with
    /// only a download bar to show would render it as a freeze.
    public typealias HashProgress = @Sendable (String, Int64, Int64) -> Void

    private let store: ModelStore
    private let index: FileIndex
    private let transfer: FileTransfer
    private let onHashProgress: HashProgress?

    /// - Parameters:
    ///   - store: the one root to write under and the roots to reuse from.
    ///   - index: how to list a repository's files. Defaults to
    ///     `HuggingFaceClient.files(repoId:revision:)`.
    ///   - transfer: how to fetch one file. Defaults to `ModelDownloader`.
    ///   - onHashProgress: told how far a verification has read.
    public init(
        store: ModelStore = ModelStore(),
        index: FileIndex? = nil,
        transfer: FileTransfer? = nil,
        onHashProgress: HashProgress? = nil
    ) {
        self.store = store
        self.index =
            index ?? { repository, revision in
                try await HuggingFaceClient().files(repoId: repository, revision: revision)
            }
        self.transfer =
            transfer ?? { repository, revision, path, destination, onBytes in
                let url = HuggingFaceClient()
                    .downloadURL(repoId: repository, filename: path, revision: revision)
                try await ModelDownloader().download(from: url, to: destination) { progress in
                    onBytes(progress.bytesDownloaded, progress.totalBytes ?? 0)
                }
            }
        self.onHashProgress = onHashProgress
    }

    // MARK: - Downloader

    /// The local directory holding every file of `id` that `patterns` selects.
    ///
    /// Per file, in order: already held and verified → nothing happens; found
    /// under an extra root → hard-linked after its checksum is checked;
    /// otherwise downloaded and then checked. A file the hub published no
    /// checksum for is refused — see `ModelStoreError.unverifiableFile`.
    ///
    /// - Throws: `ModelStoreError.digestMismatch` naming the file and both
    ///   digests, after deleting the bytes that failed; `.unverifiableFile`;
    ///   or whatever the index or transfer threw.
    public func download(
        id: String,
        revision: String?,
        matching patterns: [String],
        useLatest: Bool,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL {
        let revision = revision ?? "main"
        let directory = store.directory(forRepository: id)

        let files = try await selectedFiles(
            id: id,
            revision: revision,
            patterns: patterns,
            useLatest: useLatest
        )

        // Already complete: return without moving a byte. This is the second
        // load of the same model, and it is the point of having one root.
        if !useLatest, store.holdsVerified(files, ofRepository: id) {
            report(progressHandler, completed: files.totalBytes, total: files.totalBytes)
            return directory
        }

        let held = store.heldPaths(among: files, ofRepository: id)
        let pending = files.filter { !held.contains($0.path) }
        // The unit is BYTES, not a count of files: nine small JSON files and
        // one 4 GB weights file is not "90% done" after the JSON.
        let totalBytes = pending.totalBytes
        var movedBytes: Int64 = 0
        report(progressHandler, completed: 0, total: totalBytes)

        for file in pending {
            // The path came from a remote listing, so where it resolves to is
            // checked rather than trusted.
            guard let destination = store.destination(for: file, ofRepository: id) else {
                throw ModelStoreError.pathEscapesStore(file.path)
            }

            if let existing = store.locate(file, ofRepository: id, onProgress: hashReporter(file)) {
                // Verified somewhere else on this machine. A link, so the
                // bytes are not spent twice.
                try store.adopt(existing, as: destination)
            } else {
                // `movedBytes` is bound to a `let` for this file's transfer:
                // the callback is `@Sendable` and runs on the transfer's own
                // task, so it cannot read the mutating accumulator. Bytes
                // already moved are a constant for the duration of one file.
                let alreadyMoved = movedBytes
                try await transfer(id, revision, file.path, destination) { written, _ in
                    report(
                        progressHandler,
                        completed: alreadyMoved + written,
                        total: totalBytes
                    )
                }
                try store.acceptDownloaded(file, at: destination, onProgress: hashReporter(file))
            }

            movedBytes += file.sizeBytes
            report(progressHandler, completed: movedBytes, total: totalBytes)
        }

        try store.writeReceipt(files, forRepository: id)
        return directory
    }

    // MARK: - Private

    /// The repository's files, filtered by `patterns`.
    ///
    /// When the hub cannot be reached and this store holds a receipt covering
    /// every file it recorded, that receipt answers instead: a model already on
    /// disk must still load on a machine with no network. With no receipt there
    /// is nothing to fall back to and the index error is the answer.
    private func selectedFiles(
        id: String,
        revision: String,
        patterns: [String],
        useLatest: Bool
    ) async throws -> [ModelFile] {
        do {
            return try await index(id, revision).filter { Self.matches($0.path, patterns) }
        } catch {
            let held = store.receipt(forRepository: id).heldFiles
            guard !useLatest, !held.isEmpty else { throw error }
            let selected = held.filter { Self.matches($0.path, patterns) }
            guard store.holdsVerified(selected, ofRepository: id) else { throw error }
            return selected
        }
    }

    /// Whether `path` is selected by any of `patterns`.
    ///
    /// `fnmatch` with no flags, so `*` crosses `/` — `*.json` is meant to
    /// select `subdir/config.json` too. An empty pattern list selects
    /// everything, matching what "no filter" means to every caller.
    static func matches(_ path: String, _ patterns: [String]) -> Bool {
        guard !patterns.isEmpty else { return true }
        return patterns.contains { fnmatch($0, path, 0) == 0 }
    }

    private func hashReporter(_ file: ModelFile) -> (@Sendable (Int64, Int64) -> Void)? {
        guard let onHashProgress else { return nil }
        let path = file.path
        return { hashed, total in onHashProgress(path, hashed, total) }
    }

    /// Hand the caller a fresh `Progress` carrying bytes.
    ///
    /// A new object per report rather than one mutated in place: `Progress` is
    /// a reference type, and a caller that keeps the one it was handed would
    /// otherwise see it change under them from another task.
    private func report(
        _ handler: @Sendable (Progress) -> Void,
        completed: Int64,
        total: Int64
    ) {
        let progress = Progress(totalUnitCount: max(total, 0))
        progress.completedUnitCount = min(completed, max(total, 0))
        handler(progress)
    }
}

extension [ModelFile] {
    /// Bytes every file in this list weighs.
    var totalBytes: Int64 {
        reduce(0) { $0 + $1.sizeBytes }
    }
}

extension ModelStore.Receipt {
    /// The files this receipt vouches for, as `ModelFile` values.
    ///
    /// A receipt records SHA-256 for an LFS file and a git blob SHA-1 for the
    /// rest, and does not record which. The digest length tells them apart —
    /// 64 hex characters is SHA-256, 40 is SHA-1 — which is enough to rebuild
    /// the list for an offline load.
    var heldFiles: [ModelFile] {
        verified.compactMap { path, hex in
            guard let size = sizes[path] else { return nil }
            let digest: ModelFileDigest? =
                switch hex.count {
                case 64: .sha256(hex)
                case 40: .gitBlobSHA1(hex)
                default: nil
                }
            guard let digest else { return nil }
            return ModelFile(path: path, sizeBytes: size, digest: digest)
        }
        .sorted { $0.path < $1.path }
    }
}

// MARK: - TokenizerLoader

/// `MLXLMCommon.TokenizerLoader` over swift-transformers' `AutoTokenizer`.
///
/// mlx-swift-lm 3.x turned `Tokenizer` and `TokenizerLoader` into protocols and
/// dropped its HuggingFace dependency, so a consumer brings its own. Upstream
/// offers the `MLXHuggingFace` macros for this, and they are not taken: that
/// target pulls `MLXFoundationModels` and `MLXGuidedGeneration`, and with them
/// a vendored xgrammar C++ library and a swift-syntax macro plugin — a large
/// build for a bridge that is this long written out. Keeping `AIKitMLX` light
/// is BR-AIL-14's whole point.
///
/// The bridge is the same one upstream's `#adaptHuggingFaceTokenizer` expands
/// to, written by hand.
public struct TransformersTokenizerLoader: TokenizerLoader {

    public init() {}

    /// Load the tokenizer from a model directory.
    ///
    /// `AutoTokenizer.from(modelFolder:)` reads `tokenizer_config.json` and
    /// `tokenizer.json` from `directory` and reaches no network. Those files
    /// are in the snapshot `MLXHubDownloader` just verified.
    public func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        let upstream = try await Tokenizers.AutoTokenizer.from(modelFolder: directory)
        return TransformersTokenizerBridge(upstream)
    }
}

/// `MLXLMCommon.Tokenizer` over `Tokenizers.Tokenizer`.
///
/// The two protocols agree on everything but one name: swift-transformers
/// spells it `decode(tokens:)`, and 3.x renamed its own to `decode(tokenIds:)`
/// to match the Python `transformers` library.
struct TransformersTokenizerBridge: MLXLMCommon.Tokenizer {

    private let upstream: any Tokenizers.Tokenizer

    init(_ upstream: any Tokenizers.Tokenizer) {
        self.upstream = upstream
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        upstream.encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        upstream.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens)
    }

    func convertTokenToId(_ token: String) -> Int? {
        upstream.convertTokenToId(token)
    }

    func convertIdToToken(_ id: Int) -> String? {
        upstream.convertIdToToken(id)
    }

    var bosToken: String? { upstream.bosToken }
    var eosToken: String? { upstream.eosToken }
    var unknownToken: String? { upstream.unknownToken }

    /// Render the chat template.
    ///
    /// A model with no chat template is reported as MLX's own
    /// `TokenizerError.missingChatTemplate`, not swift-transformers': the
    /// caller is MLX's generation path and catches MLX's error. Letting the
    /// upstream one through would surface as an unrecognised failure.
    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        do {
            return try upstream.applyChatTemplate(
                messages: messages,
                tools: tools,
                additionalContext: additionalContext
            )
        } catch Tokenizers.TokenizerError.missingChatTemplate {
            throw MLXLMCommon.TokenizerError.missingChatTemplate
        }
    }
}
