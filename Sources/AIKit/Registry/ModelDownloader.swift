import Foundation

/// Progress information for an ongoing download.
public struct DownloadProgress: Sendable {
    public let bytesDownloaded: Int64
    public let totalBytes: Int64?

    public init(bytesDownloaded: Int64, totalBytes: Int64?) {
        self.bytesDownloaded = bytesDownloaded
        self.totalBytes = totalBytes
    }

    /// Fraction complete (0.0 to 1.0), or 0 if total is unknown.
    public var fraction: Double {
        guard let total = totalBytes, total > 0 else { return 0.0 }
        return Double(bytesDownloaded) / Double(total)
    }

    /// Human-readable size string, e.g. "4.23 GB".
    public var formattedSize: String {
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

/// Async model downloader with progress reporting.
public struct ModelDownloader: Sendable {

    public init() {}

    /// Download a file from URL to local path with progress.
    public func download(
        from url: URL,
        to destination: URL,
        onProgress: @Sendable @escaping (DownloadProgress) -> Void
    ) async throws {
        // Ensure parent directory exists
        let parentDir = destination.deletingLastPathComponent()
        if !FileManager.default.fileExists(atPath: parentDir.path) {
            try FileManager.default.createDirectory(
                at: parentDir,
                withIntermediateDirectories: true
            )
        }

        let delegate = DownloadDelegate(onProgress: onProgress)
        let session = URLSession(configuration: .default, delegate: delegate, delegateQueue: nil)

        defer { session.invalidateAndCancel() }

        let (tempURL, response) = try await session.download(from: url)

        guard let httpResponse = response as? HTTPURLResponse,
              200..<400 ~= httpResponse.statusCode else {
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
