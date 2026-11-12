import Foundation
import NetKit

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

// MARK: - AIHTTPReply

/// One complete HTTP answer: the status, and the bytes, both kept.
struct AIHTTPReply: Sendable, Equatable {
    let status: Int
    let body: Data

    init(status: Int, body: Data) {
        self.status = status
        self.body = body
    }

    /// The body as text, for an error that has to carry the server's own words.
    var text: String {
        String(data: body, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
    }
}

// MARK: - AIHTTPStream

/// What opening a streaming request produced, decided by the status BEFORE any
/// body byte is handed to a caller (BR-AIL-19).
enum AIHTTPStream: Sendable {
    /// 2xx: the line stream is live.
    case open(status: Int, lines: AsyncThrowingStream<String, any Error>)
    /// Non-2xx: nothing was streamed, and this is what the server said
    /// instead, read up to the caller's byte cap.
    case rejected(status: Int, body: String)
}

// MARK: - AIHTTPTransportError

/// What the transport itself can fail at, as opposed to what a server says.
enum AIHTTPTransportError: Error, Sendable, Equatable {
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
protocol AIHTTPTransport: NetworkProtocol {

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
struct URLSessionHTTPTransport: AIHTTPTransport {

    /// Seconds a single request may take.
    let timeout: TimeInterval
    /// The `URLSession` requests go out on. Its own `timeoutIntervalForRequest`
    /// is left alone; the per-request value below is what applies.
    private let session: URLSession
    /// NetKit's own request builder, reached through a type that does not
    /// override it, so URL, header and body construction stay in one place.
    private let builder = NetworkService()

    init(timeout: TimeInterval = AIProviderEntry.defaultRequestTimeout, session: URLSession = .shared) {
        self.timeout = timeout
        self.session = session
    }

    /// `throws(NetworkError)` since NetKit 3.0.0: the inherited builder reports
    /// an endpoint that forms no URL, and a body JSON cannot represent, instead
    /// of trapping on the first and dropping the second. This override only
    /// adds the timeout, so it forwards that typed error untouched rather than
    /// deciding anything about it.
    func createRequest(endPoint: EndPoint) throws(NetworkError) -> URLRequest {
        var request = try builder.createRequest(endPoint: endPoint)
        request.timeoutInterval = timeout
        return request
    }

    func send(_ request: URLRequest) async throws -> AIHTTPReply {
        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw AIHTTPTransportError.notAnHTTPResponse
        }
        return AIHTTPReply(status: http.statusCode, body: data)
    }

    func open(_ request: URLRequest, lineByteCap: Int) async throws -> AIHTTPStream {
        #if canImport(FoundationNetworking)
            // swift-corelibs-foundation has no `URLSession.bytes(for:)` and no
            // `URLSession.AsyncBytes` (measured on swift:6.3-jammy, W30:
            // `value of type 'URLSession' has no member 'bytes'`). The delegate
            // below is the same contract built from the pieces that platform
            // does have, and it resolves on the response HEAD exactly as
            // `bytes(for:)` does.
            let (status, bytes) = try await Self.openByDelegate(request, timeout: timeout)
            return try await Self.reading(status: status, bytes: bytes, cap: lineByteCap)
        #else
            // `bytes(for:)` resolves as soon as the response HEAD has arrived,
            // so the status below is read before any body byte is consumed.
            // That ordering is the whole point of BR-AIL-19 on the streaming
            // path.
            let (bytes, response) = try await session.bytes(for: request)
            guard let http = response as? HTTPURLResponse else {
                throw AIHTTPTransportError.notAnHTTPResponse
            }
            return try await Self.reading(status: http.statusCode, bytes: bytes, cap: lineByteCap)
        #endif
    }

    /// Turns a status and a byte sequence into the stream a caller sees.
    ///
    /// Generic over the sequence so the two platform paths above share every
    /// decision that follows the HEAD: what counts as open, how much of a
    /// rejected body is read, and where the line splitting happens. A platform
    /// difference that reached past this point would be a behaviour
    /// difference, and this is the seam that stops it.
    private static func reading<Bytes: AsyncSequence & Sendable>(
        status: Int,
        bytes: Bytes,
        cap: Int
    ) async throws -> AIHTTPStream where Bytes.Element == UInt8 {
        guard 200..<300 ~= status else {
            // Read what the server said INSTEAD of a stream, bounded by the
            // same cap, then stop. The connection is dropped by letting the
            // sequence go out of scope.
            var collected = Data()
            for try await byte in bytes {
                collected.append(byte)
                if collected.count >= cap { break }
            }
            let text =
                String(data: collected, encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            return .rejected(status: status, body: text)
        }

        return .open(status: status, lines: Self.lines(of: bytes, cap: cap))
    }

    /// Splits a byte sequence into newline-terminated lines, refusing a line
    /// that passes `cap` without one.
    ///
    /// The cap is what makes the buffer bounded: a server that answers 200 and
    /// then sends a megabyte of JSON on one line is stopped at the cap instead
    /// of being accumulated forever and re-parsed per chunk.
    private static func lines<Bytes: AsyncSequence & Sendable>(
        of bytes: Bytes,
        cap: Int
    ) -> AsyncThrowingStream<String, any Error> where Bytes.Element == UInt8 {
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

// MARK: - The corelibs byte stream

#if canImport(FoundationNetworking)

    extension URLSessionHTTPTransport {

        /// Sends `request` and resolves when the response HEAD arrives, with
        /// the status and a live byte stream of the body (W30).
        ///
        /// This is `URLSession.bytes(for:)` rebuilt from what
        /// swift-corelibs-foundation has. Two properties matter and both are
        /// kept:
        ///
        ///  * **The status arrives before any body byte.** The continuation
        ///    below is resumed from `didReceive response:`, which corelibs
        ///    calls before the first `didReceive data:` — so BR-AIL-19 holds
        ///    here exactly as it does on Apple, and a 401 fails a stream
        ///    instead of finishing it empty.
        ///  * **The body really streams.** `session.data(for:)` would have
        ///    compiled on Linux with no `#if` at all, and would have buffered
        ///    the whole answer before returning: a chat stream would arrive in
        ///    one lump at the end, looking like a slow success. That is the
        ///    silent degradation this delegate exists to avoid.
        ///
        /// The injected `session` is NOT used on this path: a delegate can
        /// only be given to a session at construction, so this makes its own
        /// and invalidates it when the task completes or the stream is
        /// dropped. Nothing in the suite depends on the injected session for
        /// streaming — tests reach a mock transport, never URLSession
        /// (BR-AIL-12).
        fileprivate static func openByDelegate(
            _ request: URLRequest,
            timeout: TimeInterval
        ) async throws -> (status: Int, bytes: AsyncThrowingStream<UInt8, any Error>) {
            let (bytes, continuation) = AsyncThrowingStream<UInt8, any Error>.makeStream()
            let delegate = ByteStreamDelegate(continuation: continuation)

            let configuration = URLSessionConfiguration.default
            configuration.timeoutIntervalForRequest = timeout
            let session = URLSession(configuration: configuration, delegate: delegate, delegateQueue: nil)
            let task = session.dataTask(with: request)

            // A caller that stops reading — a `break`, a cancelled Task, the
            // bounded read of a rejected body above — must not leave the
            // connection open or the session retained.
            continuation.onTermination = { @Sendable _ in
                task.cancel()
                session.invalidateAndCancel()
            }

            let status: Int = try await withCheckedThrowingContinuation { head in
                delegate.awaitHead(head)
                task.resume()
            }
            return (status, bytes)
        }
    }

    /// Feeds a byte stream from `URLSessionDataDelegate` callbacks, and
    /// resolves the status once.
    ///
    /// `@unchecked Sendable` with an explicit lock, in the shape
    /// `ModelDownloader`'s delegate already uses in this package: the
    /// callbacks arrive on a delegate queue this type does not own, so its two
    /// pieces of state are guarded rather than isolated.
    private final class ByteStreamDelegate: NSObject, URLSessionDataDelegate, @unchecked Sendable {

        private let lock = NSLock()
        private let continuation: AsyncThrowingStream<UInt8, any Error>.Continuation

        /// The waiting caller, before the HEAD arrives.
        private var head: CheckedContinuation<Int, any Error>?
        /// The HEAD's verdict, if it arrived before anyone waited on it.
        private var settled: Result<Int, any Error>?

        init(continuation: AsyncThrowingStream<UInt8, any Error>.Continuation) {
            self.continuation = continuation
            super.init()
        }

        /// Takes the caller's continuation, or resumes it at once if the HEAD
        /// already landed. The task is resumed after this returns, so the
        /// second case should not happen — it is handled anyway, because a
        /// `CheckedContinuation` that is never resumed hangs forever and a
        /// delegate queue is not ours to reason about.
        func awaitHead(_ continuation: CheckedContinuation<Int, any Error>) {
            lock.lock()
            if let settled {
                lock.unlock()
                continuation.resume(with: settled)
                return
            }
            head = continuation
            lock.unlock()
        }

        /// Resumes the waiting caller exactly once; later calls do nothing.
        private func settle(_ result: Result<Int, any Error>) {
            lock.lock()
            guard settled == nil else {
                lock.unlock()
                return
            }
            settled = result
            let waiting = head
            head = nil
            lock.unlock()
            waiting?.resume(with: result)
        }

        func urlSession(
            _ session: URLSession,
            dataTask: URLSessionDataTask,
            didReceive response: URLResponse,
            completionHandler: @escaping (URLSession.ResponseDisposition) -> Void
        ) {
            guard let http = response as? HTTPURLResponse else {
                settle(.failure(AIHTTPTransportError.notAnHTTPResponse))
                completionHandler(.cancel)
                return
            }
            settle(.success(http.statusCode))
            completionHandler(.allow)
        }

        func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
            for byte in data {
                continuation.yield(byte)
            }
        }

        func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: (any Error)?) {
            if let error {
                // A connection refused or a timeout BEFORE any HEAD: the
                // caller is still waiting on the status, and must get the
                // transport error rather than hang.
                settle(.failure(error))
                continuation.finish(throwing: error)
            } else {
                settle(.failure(AIHTTPTransportError.notAnHTTPResponse))
                continuation.finish()
            }
            session.finishTasksAndInvalidate()
        }
    }

#endif

// MARK: - TimeoutNetworkService

/// A `NetworkProtocol` that puts a caller's timeout on the request it builds.
///
/// Kept as its own type because it is public API outside this package: a caller
/// that only needs NetKit's decode-the-2xx behaviour with a longer deadline
/// uses this, and does not need the status-and-stream seam above.
/// `createRequest(endPoint:)` is a `NetworkProtocol` requirement and not only
/// an extension default, so the override here is the one the inherited
/// `sendRequest` calls.
struct TimeoutNetworkService: NetworkProtocol {
    /// Seconds a single request may take.
    let timeoutInterval: TimeInterval
    private let base = NetworkService()

    init(timeoutInterval: TimeInterval) {
        self.timeoutInterval = timeoutInterval
    }

    func createRequest(endPoint: EndPoint) throws(NetworkError) -> URLRequest {
        var request = try base.createRequest(endPoint: endPoint)
        request.timeoutInterval = timeoutInterval
        return request
    }
}
