import Foundation
import NetKit

@testable import AIKit

/// A stub ``AIHTTPTransport``: answers with what a test configured, and records
/// every request it was asked to send.
///
/// No test reaches a real provider (BR-AIL-12). Everything above the transport
/// — endpoint, codec, DTO, status handling — is the real code path a dispatch
/// takes.
///
/// The counter matters as much as the answer: a codec that REFUSES a request
/// must send nothing, and `sentRequests.isEmpty` is how that is asserted
/// (T-AIL-40). `createRequest` deliberately does NOT count as a send — a
/// refusal happens before the request is even built, and a test that wants to
/// inspect the bytes of a built request should still see zero sends.
final class MockHTTPTransport: AIHTTPTransport, @unchecked Sendable {

    // MARK: - Configuration

    /// Status `send` and `open` report. Default 200.
    var status = 200
    /// Body `send` answers with.
    var body = Data()
    /// Error `send` and `open` throw instead of answering.
    var error: (any Error)?
    /// Lines `open` yields, in order, when the status is 2xx.
    var lines: [String] = []
    /// Error the LINE STREAM finishes with, after yielding `lines`.
    var lineStreamError: (any Error)?
    /// Seconds the transport puts on a request it builds.
    var timeout: TimeInterval = AIProviderEntry.defaultRequestTimeout

    // MARK: - Recording

    /// Every request actually SENT, in order. Not every request built.
    private(set) var sentRequests: [URLRequest] = []
    /// The byte cap `open` was called with, if it was.
    private(set) var observedLineByteCap: Int?

    /// The first request sent, for a test that only sends one.
    var sentRequest: URLRequest? { sentRequests.first }

    /// The JSON body of the first request sent, as a dictionary.
    func sentBodyObject() throws -> [String: Any]? {
        guard let data = sentRequests.first?.httpBody else { return nil }
        return try JSONSerialization.jsonObject(with: data) as? [String: Any]
    }

    // MARK: - Convenience

    /// Answers `send` with this JSON text and a 200.
    func answer(json: String, status: Int = 200) {
        self.status = status
        self.body = Data(json.utf8)
    }

    // MARK: - NetworkProtocol

    func createRequest(endPoint: EndPoint) throws(NetworkError) -> URLRequest {
        var request = try NetworkService().createRequest(endPoint: endPoint)
        request.timeoutInterval = timeout
        return request
    }

    /// Answers from the configured state, and records the request.
    ///
    /// Overriding this is NOT optional. `NetworkProtocol`'s default
    /// implementation sends on `URLSession.shared`, so a stub that inherited it
    /// would reach a real server — which is how a first draft of this wave's
    /// `OpenAICompatibleEngine` tests ended up talking to the operator's live
    /// LM Studio and to api.openai.com (BR-AIL-12).
    func sendRequest<T: Decodable & Sendable>(endpoint: EndPoint) async throws -> T {
        let reply = try await send(try createRequest(endPoint: endpoint))
        guard 200..<300 ~= reply.status else {
            throw NetworkError.unexpectedStatusCode(reply.status)
        }
        do {
            return try JSONDecoder().decode(T.self, from: reply.body)
        } catch let error as DecodingError {
            throw NetworkError.jsonParsingFailed(error)
        }
    }

    func sendRequest<T: Decodable>(
        endpoint: EndPoint,
        resultHandler: @escaping @Sendable (Result<T, NetworkError>) -> Void
    ) {
        resultHandler(.failure(.requestFailed(description: "MockHTTPTransport has no callback path")))
    }

    // MARK: - AIHTTPTransport

    func send(_ request: URLRequest) async throws -> AIHTTPReply {
        sentRequests.append(request)
        if let error { throw error }
        return AIHTTPReply(status: status, body: body)
    }

    func open(_ request: URLRequest, lineByteCap: Int) async throws -> AIHTTPStream {
        sentRequests.append(request)
        observedLineByteCap = lineByteCap
        if let error { throw error }

        guard 200..<300 ~= status else {
            // Mirrors the production transport: a non-2xx reply is read, up to
            // the cap, and nothing is streamed.
            let text =
                String(data: body.prefix(lineByteCap), encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            return .rejected(status: status, body: text)
        }

        let lines = self.lines
        let streamError = self.lineStreamError
        let cap = lineByteCap
        return .open(
            status: status,
            lines: AsyncThrowingStream { continuation in
                for line in lines {
                    // The production reader refuses a line that passes the cap
                    // before a newline; a stub that yielded one anyway would
                    // let a test pass against behaviour the real transport
                    // does not have.
                    guard line.utf8.count <= cap else {
                        continuation.finish(throwing: AIHTTPTransportError.lineExceededCap(cap))
                        return
                    }
                    continuation.yield(line)
                }
                continuation.finish(throwing: streamError)
            }
        )
    }
}
