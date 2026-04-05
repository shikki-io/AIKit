import Foundation
import NetKit

/// EndPoint for OpenAI-compatible /v1/models.
struct OpenAIModelsListEndPoint: EndPoint {
    let host: String
    let port: Int?
    let scheme: String
    let apiKey: String?

    var apiPath: String { "" }
    var apiFilePath: String { "" }
    var path: String { "/v1/models" }
    var method: RequestMethod { .GET }

    var header: [String: String]? {
        var headers = ["Content-Type": "application/json"]
        if let apiKey, !apiKey.isEmpty {
            headers["Authorization"] = "Bearer \(apiKey)"
        }
        return headers
    }

    var queryParams: [String: Any]? { nil }
    var body: [String: Any]? { nil }
}
