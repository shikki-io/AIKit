import CoreKit
import Foundation

#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

/// Progress information for an ongoing download.
struct DownloadProgress: Sendable {
    let bytesDownloaded: Int64
    let totalBytes: Int64?

    init(bytesDownloaded: Int64, totalBytes: Int64?) {
        self.bytesDownloaded = bytesDownloaded
        self.totalBytes = totalBytes
    }

    /// Fraction complete (0.0 to 1.0), or 0 if total is unknown.
    var fraction: Double {
        guard let total = totalBytes, total > 0 else { return 0.0 }
        return Double(bytesDownloaded) / Double(total)
    }

    /// Human-readable size string, e.g. "4.23 GB".
    var formattedSize: String {
        Self.formatBytes(bytesDownloaded)
    }

    static func formatBytes(_ bytes: Int64) -> String {
        let units = ["B", "KB", "MB", "GB", "TB"]
        var size = Double(bytes)
        var unitIndex = 0
        while size >= 1024 && unitIndex < units.count - 1 {
            size /= 1024
            unitIndex += 1
        }
        if unitIndex == 0 {
            return "\(bytes) B"
        }
        return String(format: "%.2f %@", size, units[unitIndex])
    }
}

// MARK: - Resume

/// Where a partial transfer's resume token is kept: beside the file it belongs
/// to.
///
/// W33 of spec ai-layer-aikit-core-2026-10-31, BR-AIL-23: *"Interrupted
/// downloads resume from where they stopped."* `URLSession` hands back an
/// opaque `Data` token when a download task fails part-way, and before this
/// wave AIKit threw it away — so a 4 GB model interrupted at 3.9 GB started
/// again at zero, which on a metered path is the same bill twice.
///
/// Beside the file rather than in one index, because that is what makes the
/// token impossible to orphan: the store's own layout already places the file,
/// deleting the model's directory takes its tokens with it, and there is no
/// second place to keep consistent. The name is the destination's own with one
/// extension appended — no separator is spelled here, and the location is
/// derived from a ``ModelFileDestination`` the path layer resolved.
struct DownloadResumeStore: Sendable {

    /// The extension appended to a destination's name.
    ///
    /// Prefixed, so a token cannot be mistaken for the model file it belongs to
    /// by a human reading the directory or by a pattern selecting weights.
    static let tokenExtension = "aikit-resume"

    private var fileManager: FileManager { .default }

    /// Where the token for `destination` lives.
    func location(for destination: URL) -> URL {
        destination.appendingPathExtension(Self.tokenExtension)
    }

    /// The stored token, or `nil` when there is none to resume from.
    ///
    /// An unreadable token is `nil` rather than an error: it is an
    /// optimisation, and the honest fall-back for one that cannot be read is
    /// to fetch the file again.
    func token(for destination: URL) -> Data? {
        let url = location(for: destination)
        guard let data = try? Data(contentsOf: url), !data.isEmpty else { return nil }
        return data
    }

    /// Keep `token` so the next attempt can continue.
    func save(_ token: Data, for destination: URL) throws {
        let url = location(for: destination)
        try fileManager.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try token.write(to: url, options: .atomic)
    }

    /// Forget any token for `destination`.
    func clear(for destination: URL) {
        try? fileManager.removeItem(at: location(for: destination))
    }
}

// MARK: - Downloader

/// Async model downloader with progress reporting and resume.
///
/// W23 of spec ai-layer-aikit-core-2026-10-31 narrowed what it will write to.
/// The destination is a ``ModelFileDestination``, which only ``ModelStore``
/// mints and only by resolving a ``VirtualPathOwning`` declaration through the
/// injected CoreKit path layer. A bare `URL` is no longer accepted, so the one
/// type that moves bytes onto the disk cannot be handed a location somebody
/// composed by hand — and the containment of that location is settled where it
/// is RESOLVED (a `PathSegment` cannot hold `..` or a separator) rather than
/// re-checked, or forgotten, here.
///
/// It still names no directory of its own: it creates the parent of the
/// location it was given and nothing else. Resolution creates nothing, the
/// writer creates what it needs (CoreKit BR-RP-01).
///
/// W33 added resume (BR-AIL-23). The decision ABOVE the transfer — read a
/// stored token, pass it to the transfer, store a fresh one when the transfer
/// fails carrying one, start over once when a stored token is rejected, forget
/// it on success — is this type's own logic, and ``Transport`` is the seam that
/// makes it provable without a server (BR-AIL-12).
struct ModelDownloader: Sendable {

    /// How bytes actually move.
    ///
    /// Injected so the resume decision above it can be exercised against a
    /// transfer that fails on purpose. The default is `URLSession`, so a host
    /// that configures nothing runs the real one.
    ///
    /// `resumeToken` is the token to continue from, or `nil` to fetch the whole
    /// file. A transport that cannot resume ignores it and fetches the file —
    /// the result is correct either way, only slower.
    typealias Transport = @Sendable (
        _ url: URL,
        _ destination: URL,
        _ resumeToken: Data?,
        _ onProgress: @Sendable @escaping (DownloadProgress) -> Void
    ) async throws -> Void

    private let transport: Transport
    private let resume: DownloadResumeStore

    /// - Parameters:
    ///   - transport: how bytes move. Defaults to `URLSession`.
    ///   - resume: where resume tokens are kept. Defaults to beside the file.
    init(transport: Transport? = nil, resume: DownloadResumeStore = DownloadResumeStore()) {
        self.transport = transport ?? { url, destination, token, onProgress in
            try await Self.urlSessionTransfer(
                from: url, to: destination, resumeToken: token, onProgress: onProgress)
        }
        self.resume = resume
    }

    /// Download a file from URL to the resolved location, with progress.
    ///
    /// Resumes from a stored token when there is one. A stored token is a guess
    /// about a partial file on a server that may have moved on, so a failure
    /// WITH a token is retried once from the start: a token that has gone stale
    /// must not wedge a model permanently. A failure without one is the news,
    /// and any fresh token it carries is kept for next time.
    func download(
        from url: URL,
        to destination: ModelFileDestination,
        onProgress: @Sendable @escaping (DownloadProgress) -> Void
    ) async throws {
        let target = destination.url
        // The writer creates the directory it writes into.
        let parentDir = target.deletingLastPathComponent()
        if !FileManager.default.fileExists(atPath: parentDir.path) {
            try FileManager.default.createDirectory(
                at: parentDir,
                withIntermediateDirectories: true
            )
        }

        guard let token = resume.token(for: target) else {
            try await attempt(url, target, nil, onProgress)
            resume.clear(for: target)
            return
        }

        do {
            try await attempt(url, target, token, onProgress)
        } catch {
            // A cancelled task is not a stale token, and retrying it would
            // ignore the cancellation.
            try Task.checkCancellation()
            resume.clear(for: target)
            try await attempt(url, target, nil, onProgress)
        }
        resume.clear(for: target)
    }

    // MARK: - Private

    /// One transfer, keeping whatever resume token its failure carried.
    private func attempt(
        _ url: URL,
        _ target: URL,
        _ token: Data?,
        _ onProgress: @Sendable @escaping (DownloadProgress) -> Void
    ) async throws {
        do {
            try await transport(url, target, token, onProgress)
        } catch {
            if let fresh = Self.resumeToken(from: error) {
                // Best effort: failing to KEEP a token costs the next attempt
                // its head start and must not replace the transfer's own error,
                // which is the one the caller has to act on.
                try? resume.save(fresh, for: target)
            }
            throw error
        }
    }

    /// The resume token a failed download carries, if any.
    ///
    /// `URLSession` puts it on the error's `userInfo` under
    /// `NSURLSessionDownloadTaskResumeData`. Read from the `NSError` bridge
    /// rather than by matching on `URLError`, because the same key is how both
    /// a `URLError` and a plain `NSError` from the same failure carry it.
    static func resumeToken(from error: any Error) -> Data? {
        (error as NSError).userInfo[NSURLSessionDownloadTaskResumeData] as? Data
    }

    /// The production transfer: one `URLSession` download, resumed when a token
    /// was given.
    private static func urlSessionTransfer(
        from url: URL,
        to destination: URL,
        resumeToken: Data?,
        onProgress: @Sendable @escaping (DownloadProgress) -> Void
    ) async throws {
        let delegate = DownloadDelegate(onProgress: onProgress)
        let session = URLSession(configuration: .default, delegate: delegate, delegateQueue: nil)

        defer { session.invalidateAndCancel() }

        let tempURL: URL
        let response: URLResponse
        if let resumeToken {
            (tempURL, response) = try await session.download(resumeFrom: resumeToken)
        } else {
            (tempURL, response) = try await session.download(from: url)
        }

        // 206 Partial Content is what a resumed transfer answers, and it is
        // inside this range: a resumed download must not read as a failure.
        guard let httpResponse = response as? HTTPURLResponse,
            200..<400 ~= httpResponse.statusCode
        else {
            let statusCode = (response as? HTTPURLResponse)?.statusCode ?? -1
            throw AIKitError.downloadFailed("HTTP \(statusCode)")
        }

        // Move temp file to destination
        if FileManager.default.fileExists(atPath: destination.path) {
            try FileManager.default.removeItem(at: destination)
        }
        try FileManager.default.moveItem(at: tempURL, to: destination)
    }
}

// MARK: - URLSession Delegate

private final class DownloadDelegate: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    private let onProgress: @Sendable (DownloadProgress) -> Void

    init(onProgress: @Sendable @escaping (DownloadProgress) -> Void) {
        self.onProgress = onProgress
        super.init()
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {
        // Handled by the caller via the async download(from:) return value.
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        let total: Int64? = totalBytesExpectedToWrite > 0 ? totalBytesExpectedToWrite : nil
        let progress = DownloadProgress(bytesDownloaded: totalBytesWritten, totalBytes: total)
        onProgress(progress)
    }
}
