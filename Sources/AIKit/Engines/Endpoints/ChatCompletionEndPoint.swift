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
