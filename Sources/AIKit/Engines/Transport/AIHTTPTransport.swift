import Foundation
import NetKit

#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

// MARK: - AIHTTPReply

/// One complete HTTP answer: the status, and the bytes, both kept.
public struct AIHTTPReply: Sendable, Equatable {
    public let status: Int
    public let body: Data

    public init(status: Int, body: Data) {
        self.status = status
        self.body = body
    }

    /// The body as text, for an error that has to carry the server's own words.
    public var text: String {
        String(data: body, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
    }
}

// MARK: - AIHTTPStream

/// What opening a streaming request produced, decided by the status BEFORE any
/// body byte is handed to a caller (BR-AIL-19).
public enum AIHTTPStream: Sendable {
    /// 2xx: the line stream is live.
    case open(status: Int, lines: AsyncThrowingStream<String, any Error>)
    /// Non-2xx: nothing was streamed, and this is what the server said
    /// instead, read up to the caller's byte cap.
    case rejected(status: Int, body: String)
}

// MARK: - AIHTTPTransportError

/// What the transport itself can fail at, as opposed to what a server says.
public enum AIHTTPTransportError: Error, Sendable, Equatable {
    /// One line passed the byte cap with no newline in sight. The provider
    /// re-throws this as ``AIKitError/streamLineTooLong(cap:target:)``, which
    /// can name where the request went.
    case lineExceededCap(Int)
    /// The answer was not an HTTP answer at all, so it has no status.
    case notAnHTTPResponse
}

// MARK: - AIHTTPTransport

/// How AIKit sends an HTTP request and reads what comes back.
///
/// It refines `NetKit.NetworkProtocol` rather than replacing it, so NetKit
/// stays the one place a `URLRequest` is built from an `EndPoint`: URL,
/// headers and body construction are inherited, and a conformer adds only the
/// sending.
///
/// WHY SENDING IS NOT NetKit's `sendRequest`. BR-AIL-19 requires a non-2xx
/// error to carry the server's own text, and requires `stream` to know the
/// status before the first byte of body. NetKit 3.0.0's
/// `sendRequest(endpoint:)` does neither: it throws
/// `NetworkError.unexpectedStatusCode(code, headers:)` and DISCARDS the body,
/// and it has no byte-stream API at all (measured at NetKit 3.0.0 —
/// `NetworkProtocol` declares `createRequest`, `sendRequest` and its callback
/// form, nothing else). Giving AIKit its own two-method seam is what lets a
/// 429 keep its Retry-After text and a 401 fail a stream instead of finishing
/// it empty. Adding a byte stream to NetKit is a change to a shared package
/// and would go through its own reviewed PR.
public protocol AIHTTPTransport: NetworkProtocol {

    /// Sends a request and returns the status with the whole body.
    ///
    /// Does NOT treat a non-2xx as an error: the caller decides, because only
    /// the caller knows the target to name in the error.
    func send(_ request: URLRequest) async throws -> AIHTTPReply

    /// Opens a streaming request, reading the status before any body byte.
    ///
    /// - Parameter lineByteCap: Bytes one line may reach before
    ///   ``AIHTTPTransportError/lineExceededCap(_:)``. Also the ceiling on how
    ///   much of a rejected reply's body is read.
    func open(_ request: URLRequest, lineByteCap: Int) async throws -> AIHTTPStream
}

// MARK: - URLSessionHTTPTransport

/// The production transport: `URLSession`, with the entry's timeout on every
/// request it builds.
///
/// There is no `URLSessionDataDelegate` and no subclass here. The reader this
/// replaces was a delegate holding a growing `Data` buffer that it re-decoded
/// from the start on every chunk, and that could not report a status because a
/// delegate is handed bytes after the fact.
public struct URLSessionHTTPTransport: AIHTTPTransport {

    /// Seconds a single request may take.
    public let timeout: TimeInterval
    /// The `URLSession` requests go out on. Its own `timeoutIntervalForRequest`
    /// is left alone; the per-request value below is what applies.
    private let session: URLSession
    /// NetKit's own request builder, reached through a type that does not
    /// override it, so URL, header and body construction stay in one place.
    private let builder = NetworkService()

    public init(timeout: TimeInterval = AIProviderEntry.defaultRequestTimeout, session: URLSession = .shared) {
        self.timeout = timeout
        self.session = session
    }

    /// `throws(NetworkError)` since NetKit 3.0.0: the inherited builder reports
    /// an endpoint that forms no URL, and a body JSON cannot represent, instead
    /// of trapping on the first and dropping the second. This override only
    /// adds the timeout, so it forwards that typed error untouched rather than
    /// deciding anything about it.
    public func createRequest(endPoint: EndPoint) throws(NetworkError) -> URLRequest {
        var request = try builder.createRequest(endPoint: endPoint)
        request.timeoutInterval = timeout
        return request
    }

    public func send(_ request: URLRequest) async throws -> AIHTTPReply {
        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw AIHTTPTransportError.notAnHTTPResponse
        }
        return AIHTTPReply(status: http.statusCode, body: data)
    }

    public func open(_ request: URLRequest, lineByteCap: Int) async throws -> AIHTTPStream {
        // `bytes(for:)` resolves as soon as the response HEAD has arrived, so
        // the status below is read before any body byte is consumed. That
        // ordering is the whole point of BR-AIL-19 on the streaming path.
        let (bytes, response) = try await session.bytes(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw AIHTTPTransportError.notAnHTTPResponse
        }

        guard 200..<300 ~= http.statusCode else {
            // Read what the server said INSTEAD of a stream, bounded by the
            // same cap, then stop. The connection is dropped by letting the
            // sequence go out of scope.
            var collected = Data()
            for try await byte in bytes {
                collected.append(byte)
                if collected.count >= lineByteCap { break }
            }
            let text = String(data: collected, encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            return .rejected(status: http.statusCode, body: text)
        }

        return .open(status: http.statusCode, lines: Self.lines(of: bytes, cap: lineByteCap))
    }

    /// Splits a byte sequence into newline-terminated lines, refusing a line
    /// that passes `cap` without one.
    ///
    /// The cap is what makes the buffer bounded: a server that answers 200 and
    /// then sends a megabyte of JSON on one line is stopped at the cap instead
    /// of being accumulated forever and re-parsed per chunk.
    private static func lines(
        of bytes: URLSession.AsyncBytes,
        cap: Int
    ) -> AsyncThrowingStream<String, any Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                var line = Data()
                do {
                    for try await byte in bytes {
                        if byte == UInt8(ascii: "\n") {
                            continuation.yield(Self.text(of: line))
                            line.removeAll(keepingCapacity: true)
                            continue
                        }
                        line.append(byte)
                        guard line.count <= cap else {
                            throw AIHTTPTransportError.lineExceededCap(cap)
                        }
                    }
                    // A final line with no trailing newline is still a line.
                    if !line.isEmpty {
                        continuation.yield(Self.text(of: line))
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }

    /// One line's text. A line that is not UTF-8 is handed over as empty rather
    /// than throwing: SSE keep-alives and comment lines are not payload, and a
    /// codec reading an empty line returns no chunk.
    private static func text(of line: Data) -> String {
        var bytes = line
        // Tolerate CRLF, which some proxies introduce.
        if bytes.last == UInt8(ascii: "\r") { bytes.removeLast() }
        return String(data: bytes, encoding: .utf8) ?? ""
    }
}

// MARK: - TimeoutNetworkService

/// A `NetworkProtocol` that puts a caller's timeout on the request it builds.
///
/// Kept as its own type because it is public API outside this package: a caller
/// that only needs NetKit's decode-the-2xx behaviour with a longer deadline
/// uses this, and does not need the status-and-stream seam above.
/// `createRequest(endPoint:)` is a `NetworkProtocol` requirement and not only
/// an extension default, so the override here is the one the inherited
/// `sendRequest` calls.
public struct TimeoutNetworkService: NetworkProtocol {
    /// Seconds a single request may take.
    public let timeoutInterval: TimeInterval
    private let base = NetworkService()

    public init(timeoutInterval: TimeInterval) {
        self.timeoutInterval = timeoutInterval
    }

    public func createRequest(endPoint: EndPoint) throws(NetworkError) -> URLRequest {
        var request = try base.createRequest(endPoint: endPoint)
        request.timeoutInterval = timeoutInterval
        return request
    }
}
