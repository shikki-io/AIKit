// kagami-scope: exempt — AIKit (moved from Irori) declares its scope in .kagami/scopes.override.toml; file touched only for resolver-exempt marker (no behavior change).
// MARK: - ModelWeightLoader
// Lazy download + resume + SHA-256 verification for V-JEPA weights.
//
// Spec §5 Phase 1 deliverable 4 + BR-01, BR-07, BR-08, BR-09.
//
//   • Weights live at `~/.shikki/models/<modelId>/<fileName>` by default
//     (configurable via `cacheRoot` for tests).
//   • If the file exists and its SHA-256 matches, return the path — no I/O.
//   • If a `.part` file exists, resume via HTTP Range: bytes=<offset>-.
//   • Stream the response to disk in chunks to keep memory bounded.
//   • Verify SHA-256 on completion. Mismatch → delete + throw.
//
// The HTTP fetcher is injected so tests can stub the network without
// touching the real HuggingFace endpoint.

import CryptoKit
import Foundation

// MARK: - Byte stream

/// Minimal async byte stream abstraction — lets tests feed chunks without
/// going through URLSession.
public struct AsyncByteStream: Sendable {
    public typealias Continuation = AsyncStream<Data>.Continuation
    public let stream: AsyncStream<Data>

    public init(builder: @Sendable (Continuation) -> Void) {
        var cont: Continuation!
        self.stream = AsyncStream { c in cont = c }
        builder(cont)
    }
}

// MARK: - HTTP fetcher protocol

public protocol ModelWeightHTTPFetcher: Sendable {
    /// Fetch `url`. If `rangeStart` is set, request Range: bytes=<rangeStart>-.
    func fetch(url: URL, rangeStart: Int64?) async throws -> AsyncByteStream
}

// MARK: - Production URLSession fetcher

public struct URLSessionWeightFetcher: ModelWeightHTTPFetcher {
    public let session: URLSession

    public init(session: URLSession = .shared) {
        self.session = session
    }

    public func fetch(url: URL, rangeStart: Int64?) async throws -> AsyncByteStream {
        var request = URLRequest(url: url)
        if let start = rangeStart {
            request.setValue("bytes=\(start)-", forHTTPHeaderField: "Range")
        }
        let (bytes, _) = try await session.bytes(for: request)
        return AsyncByteStream { continuation in
            Task {
                var buffer = Data()
                buffer.reserveCapacity(64 * 1024)
                do {
                    for try await byte in bytes {
                        buffer.append(byte)
                        if buffer.count >= 64 * 1024 {
                            continuation.yield(buffer)
                            buffer.removeAll(keepingCapacity: true)
                        }
                    }
                    if !buffer.isEmpty {
                        continuation.yield(buffer)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish()
                }
            }
        }
    }
}

// MARK: - Loader

public struct ModelWeightLoader: Sendable {

    // MARK: Descriptor

    public struct ModelDescriptor: Sendable, Equatable {
        public let modelId: String
        public let fileName: String
        public let remoteURL: URL
        public let expectedSHA256: String
        public let expectedBytes: Int64

        public init(
            modelId: String,
            fileName: String,
            remoteURL: URL,
            expectedSHA256: String,
            expectedBytes: Int64
        ) {
            self.modelId = modelId
            self.fileName = fileName
            self.remoteURL = remoteURL
            self.expectedSHA256 = expectedSHA256
            self.expectedBytes = expectedBytes
        }
    }

    public enum Error: Swift.Error, Equatable, Sendable {
        case checksumMismatch(expected: String, actual: String)
        case downloadShort(expected: Int64, actual: Int64)
        case cacheIOFailed(String)
    }

    // MARK: Stored

    public let cacheRoot: URL
    public let httpFetcher: ModelWeightHTTPFetcher

    // MARK: Init

    /// Default cacheRoot: `~/.shikki/models/` (BR-07).
    public init(
        cacheRoot: URL? = nil,
        httpFetcher: ModelWeightHTTPFetcher = URLSessionWeightFetcher()
    ) {
        if let cacheRoot {
            self.cacheRoot = cacheRoot
        } else {
            let shikkiRoot = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".shikki")
            self.cacheRoot = shikkiRoot.appendingPathComponent("models")
        }
        self.httpFetcher = httpFetcher
    }

    // MARK: Resolve

    /// Return a local path to the weights, downloading if necessary.
    @discardableResult
    public func resolve(_ descriptor: ModelDescriptor) async throws -> URL {
        let modelDir = cacheRoot.appendingPathComponent(descriptor.modelId)
        let finalPath = modelDir.appendingPathComponent(descriptor.fileName)
        let partPath = modelDir.appendingPathComponent(descriptor.fileName + ".part")

        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)

        // Cache hit fast path.
        if FileManager.default.fileExists(atPath: finalPath.path) {
            let data = try Data(contentsOf: finalPath)
            let actual = Self.sha256Hex(of: data)
            if actual == descriptor.expectedSHA256 {
                return finalPath
            }
            // Corrupt cache — drop it and re-download.
            try? FileManager.default.removeItem(at: finalPath)
        }

        // Resume if `.part` exists.
        var rangeStart: Int64?
        if FileManager.default.fileExists(atPath: partPath.path) {
            let attrs = try FileManager.default.attributesOfItem(atPath: partPath.path)
            let existingSize = (attrs[.size] as? NSNumber)?.int64Value ?? 0
            if existingSize > 0 && existingSize < descriptor.expectedBytes {
                rangeStart = existingSize
            } else {
                try? FileManager.default.removeItem(at: partPath)
            }
        }

        let stream = try await httpFetcher.fetch(url: descriptor.remoteURL, rangeStart: rangeStart)

        // Append streamed bytes to the .part file.
        if !FileManager.default.fileExists(atPath: partPath.path) {
            FileManager.default.createFile(atPath: partPath.path, contents: nil)
        }
        guard let handle = try? FileHandle(forWritingTo: partPath) else {
            throw Error.cacheIOFailed(partPath.path)
        }
        defer { try? handle.close() }

        if let start = rangeStart {
            try handle.seek(toOffset: UInt64(start))
        } else {
            try handle.truncate(atOffset: 0)
        }

        for await chunk in stream.stream {
            try handle.write(contentsOf: chunk)
        }
        try handle.close()

        // Verify on-disk SHA-256.
        let finalData = try Data(contentsOf: partPath)
        let actualSHA = Self.sha256Hex(of: finalData)
        if actualSHA != descriptor.expectedSHA256 {
            try? FileManager.default.removeItem(at: partPath)
            throw Error.checksumMismatch(expected: descriptor.expectedSHA256, actual: actualSHA)
        }

        // Promote .part → final.
        try FileManager.default.moveItem(at: partPath, to: finalPath)
        return finalPath
    }

    // MARK: SHA helper

    public static func sha256Hex(of data: Data) -> String {
        let digest = SHA256.hash(data: data)
        return digest.map { String(format: "%02x", $0) }.joined()
    }
}
