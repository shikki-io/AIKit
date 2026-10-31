import Foundation
import NetKit

/// EndPoint for OpenAI-compatible /v1/chat/completions.
struct OpenAIChatCompletionEndPoint: EndPoint {
    let host: String
    let port: Int?
    let scheme: String
    let apiKey: String?
    let requestBody: ChatCompletionRequest

    var apiPath: String { "" }
    var apiFilePath: String { "" }
    var path: String { "/v1/chat/completions" }
    var method: RequestMethod { .POST }

    var header: [String: String]? {
        var headers = ["Content-Type": "application/json"]
        if let apiKey, !apiKey.isEmpty {
            headers["Authorization"] = "Bearer \(apiKey)"
        }
        return headers
    }

    var queryParams: [String: Any]? { nil }

    var body: [String: Any]? {
        guard let data = try? JSONEncoder().encode(requestBody),
              let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return nil }
        return dict
    }
}

// MARK: - Reply

/// One reply from `/v1/chat/completions`, in either shape the surface answers with.
///
/// A completion is the normal shape. The other one is a 200 carrying an error
/// body, which an OpenAI-compatible server returns for a request it accepted
/// and then could not serve — a model name it has not loaded, a `/v1` left in
/// the server root. NetKit decodes a 2xx body straight into the type the caller
/// asks for, so asking for `ChatCompletionResponseDTO` alone turns that answer
/// into a `DecodingError` and loses the server's own words. BR-AIL-10 requires
/// those words to reach the caller, so the reply type admits both shapes and
/// `OpenAIProvider` decides which one is a failure.
enum ChatCompletionReply: Decodable, Sendable {
    /// The server answered with choices.
    case completion(ChatCompletionResponseDTO)
    /// The server answered 200 and named what went wrong.
    case serverError(message: String)

    /// An error body in either form seen in the wild: `{"error": {"message": "…"}}`
    /// and the bare `{"error": "…"}`.
    private struct ErrorBody: Decodable {
        let message: String

        init(from decoder: Decoder) throws {
            if let single = try? decoder.singleValueContainer(),
               let text = try? single.decode(String.self) {
                message = text
                return
            }
            let container = try decoder.container(keyedBy: ChatCompletionReplyKey.self)
            message = try container.decode(String.self, forKey: .message)
        }
    }

    init(from decoder: Decoder) throws {
        if let container = try? decoder.container(keyedBy: ChatCompletionReplyKey.self),
           let body = try? container.decode(ErrorBody.self, forKey: .error) {
            self = .serverError(message: body.message)
            return
        }
        self = .completion(try ChatCompletionResponseDTO(from: decoder))
    }
}

/// The two keys the reply envelope reads: the error object on the body, and the
/// text inside it.
private enum ChatCompletionReplyKey: String, CodingKey {
    case error
    case message
}
