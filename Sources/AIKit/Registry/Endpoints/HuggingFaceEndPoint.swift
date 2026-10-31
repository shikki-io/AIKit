import Foundation
import NetKit

/// EndPoint definitions for the HuggingFace Hub API.
enum HuggingFaceEndPoint: EndPoint {
    case searchModels(query: String, filter: HFSearchFilter?)
    case getModel(id: String)
    /// The repository's file tree, which is where the published checksums are.
    ///
    /// `/api/models/<id>/tree/<revision>` returns one entry per file carrying
    /// `size`, the git blob `oid`, and — for an LFS-tracked file — an `lfs`
    /// object whose own `oid` is the file's SHA-256. `ModelStore` verifies
    /// against those two (W13 of spec ai-layer-aikit-core-2026-10-31).
    case listTree(id: String, revision: String)

    var host: String { "huggingface.co" }
    var scheme: String { "https" }
    var port: Int? { nil }
    var apiPath: String { "/api" }
    var apiFilePath: String { "" }
    var method: RequestMethod { .GET }
    var header: [String: String]? { nil }
    var body: [String: Any]? { nil }

    var path: String {
        switch self {
        case .searchModels:
            "/models"
        case .getModel(let id):
            "/models/\(id)"
        case .listTree(let id, let revision):
            "/models/\(id)/tree/\(revision)"
        }
    }

    var queryParams: [String: Any]? {
        switch self {
        case .searchModels(let query, let filter):
            var params: [String: Any] = ["search": query]
            if let tags = filter?.tags {
                for tag in tags {
                    // HuggingFace uses repeated filter params; for simplicity use comma-separated
                    params["filter"] = tag
                }
            }
            if let pipelineTag = filter?.pipelineTag {
                params["pipeline_tag"] = pipelineTag
            }
            if let sort = filter?.sort {
                params["sort"] = sort
            }
            if let direction = filter?.direction {
                params["direction"] = direction
            }
            if let limit = filter?.limit {
                params["limit"] = String(limit)
            }
            return params
        case .getModel, .listTree:
            return nil
        }
    }
}
