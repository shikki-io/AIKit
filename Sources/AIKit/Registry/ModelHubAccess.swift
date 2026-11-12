import Foundation

// MARK: - The hub, reached through door 3

/// Listing and fetching a model repository's published files.
///
/// W27 of spec ai-layer-aikit-core-2026-10-31, BR-AIL-21 door 3: *"The Hugging
/// Face endpoint stays internal behind the catalog."* `HuggingFaceClient`,
/// `HuggingFaceEndPoint` and `ModelDownloader` are internal, and before this
/// wave `AIKitMLX` reached all three by name to build its own default file
/// index and transfer. These two methods are that default, moved behind the
/// door: a consumer asks the store for a repository's files and for one file's
/// bytes, and never learns which registry answered.
///
/// On ``ModelStore`` rather than in a new type, because the store is already
/// door 3's noun for *do we have this model and where does it go* — a new
/// `ModelHub` would be a second noun for the same job.
extension ModelStore {

    /// Every file the hub publishes for a repository revision, with its size
    /// and the digest the hub published for it.
    ///
    /// A file with no published digest comes back with `digest == nil`, and the
    /// caller refuses it (``ModelStoreError/unverifiableFile(_:)``): an
    /// unverifiable file is not a verified one.
    public static func publishedFiles(
        repository: String,
        revision: String = "main"
    ) async throws -> [ModelFile] {
        try await HuggingFaceClient().files(repoId: repository, revision: revision)
    }

    /// Fetches one published file to a destination this store resolved,
    /// reporting `(bytesWritten, totalBytes)`.
    ///
    /// `totalBytes` is `0` while the server has not said how long the body is;
    /// a caller showing a bar reads it as unknown rather than as zero work.
    /// The bytes are NOT verified here — ``digest(of:at:onProgress:)`` and
    /// ``isVerified(_:at:)`` are the verification half, and the caller runs
    /// them after this returns.
    public static func fetchPublishedFile(
        repository: String,
        revision: String = "main",
        path: String,
        to destination: ModelFileDestination,
        onBytes: @Sendable @escaping (Int64, Int64) -> Void
    ) async throws {
        let url = try HuggingFaceClient()
            .downloadURL(repoId: repository, filename: path, revision: revision)
        try await ModelDownloader().download(from: url, to: destination) { progress in
            onBytes(progress.bytesDownloaded, progress.totalBytes ?? 0)
        }
    }
}
