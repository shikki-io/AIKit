import Foundation
import Logging

// MARK: - LMStudioProvider

/// AI provider connecting to LM Studio's OpenAI-compatible API.
/// Enables local model fallback when cloud providers hit rate limits.
///
/// LM Studio exposes POST /v1/chat/completions at http://127.0.0.1:1234 by default.
/// Supports any model loaded in LM Studio — model name is configurable.
public struct LMStudioProvider: AgentProviding, Sendable {
    /// Base URL for the LM Studio server (no trailing slash).
    public let baseURL: String
    /// Model identifier as loaded in LM Studio.
    public let model: String
    /// Request timeout in seconds for the URLSession.
    public let requestTimeout: TimeInterval
    /// Logger for diagnostics.
    private let logger: Logger

    // MARK: - Errors

    public enum LMStudioError: Error, Equatable, Sendable {
        case connectionRefused(String)
        case rateLimited
        case invalidResponse(String)
        case httpError(Int, String)
        case emptyContent
    }

    // MARK: - Response Models

    struct ChatCompletionResponse: Decodable {
        struct Choice: Decodable {
            struct Message: Decodable {
                let role: String
                let content: String
            }
            let message: Message
        }
        let choices: [Choice]
    }

    struct ErrorResponse: Decodable {
        struct ErrorDetail: Decodable {
            let message: String
        }
        let error: ErrorDetail
    }

    // MARK: - Init

    /// Create a new LMStudioProvider.
    /// - Parameters:
    ///   - baseURL: LM Studio server URL. Defaults to env `LMSTUDIO_URL` or `http://127.0.0.1:1234`.
    ///   - model: Model name. Defaults to env `LMSTUDIO_MODEL` or `local-model`.
    ///   - requestTimeout: Timeout for individual requests in seconds. Defaults to 300.
    public init(
        baseURL: String? = nil,
        model: String? = nil,
        requestTimeout: TimeInterval = 300,
        logger: Logger = Logger(label: "shikki.lmstudio-provider")
    ) {
        self.baseURL = baseURL
            ?? ProcessInfo.processInfo.environment["LMSTUDIO_URL"]
            ?? "http://127.0.0.1:1234"
        self.model = model
            ?? ProcessInfo.processInfo.environment["LMSTUDIO_MODEL"]
            ?? "local-model"
        self.requestTimeout = requestTimeout
        self.logger = logger
    }

    // MARK: - AgentProviding

    public func run(prompt: String, timeout: TimeInterval) async throws -> String {
        let url = URL(string: "\(baseURL)/v1/chat/completions")!

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = min(timeout, requestTimeout)

        let body = buildRequestBody(prompt: prompt)
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        logger.info("LM Studio request", metadata: [
            "model": "\(model)",
            "url": "\(baseURL)",
        ])

        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await URLSession.shared.data(for: request)
        } catch let urlError as URLError {
            switch urlError.code {
            case .cannotConnectToHost, .networkConnectionLost, .timedOut, .cannotFindHost:
                throw LMStudioError.connectionRefused(
                    "Cannot connect to LM Studio at \(baseURL): \(urlError.localizedDescription)"
                )
            default:
                throw LMStudioError.connectionRefused(
                    "Network error: \(urlError.localizedDescription)"
                )
            }
        }

        guard let httpResponse = response as? HTTPURLResponse else {
            throw LMStudioError.invalidResponse("Non-HTTP response received")
        }

        switch httpResponse.statusCode {
        case 200:
            return try parseResponse(data)
        case 429:
            throw LMStudioError.rateLimited
        default:
            let errorMessage = parseErrorMessage(data) ?? "HTTP \(httpResponse.statusCode)"
            throw LMStudioError.httpError(httpResponse.statusCode, errorMessage)
        }
    }

    // MARK: - Request Building

    func buildRequestBody(prompt: String) -> [String: Any] {
        [
            "model": model,
            "messages": [
                ["role": "user", "content": prompt]
            ],
            "temperature": 0.7,
            "max_tokens": -1,
            "stream": false,
        ]
    }

    // MARK: - Response Parsing

    func parseResponse(_ data: Data) throws -> String {
        let decoded: ChatCompletionResponse
        do {
            decoded = try JSONDecoder().decode(ChatCompletionResponse.self, from: data)
        } catch {
            throw LMStudioError.invalidResponse("Failed to decode response: \(error)")
        }

        guard let firstChoice = decoded.choices.first else {
            throw LMStudioError.emptyContent
        }

        let content = firstChoice.message.content.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !content.isEmpty else {
            throw LMStudioError.emptyContent
        }

        return content
    }

    // MARK: - Error Parsing

    private func parseErrorMessage(_ data: Data) -> String? {
        let decoded = try? JSONDecoder().decode(ErrorResponse.self, from: data)
        return decoded?.error.message
    }
}
