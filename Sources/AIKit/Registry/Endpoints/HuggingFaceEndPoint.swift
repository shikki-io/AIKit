import Foundation
import NetKit

/// The HuggingFace Hub API: one case per route AIKit calls.
///
/// BR-AIL-16, answering AIKit#18 r4500469694 ("Why this is not part of the
/// hugginface endpoint?" — about `HuggingFaceClient.downloadURL`, which built
/// `https://huggingface.co/\(repoId)/resolve/\(revision)/\(filename)` by
/// interpolation and force-unwrapped the result).
///
/// Route table, from <https://huggingface.co/docs/hub/api>:
///
/// | Route                                   | Method | Host path   |
/// |-----------------------------------------|--------|-------------|
/// | `/api/models`                           | GET    | under `/api` |
/// | `/api/models/{id}`                      | GET    | under `/api` |
/// | `/api/models/{id}/tree/{revision}`      | GET    | under `/api` |
/// | `/{id}/resolve/{revision}/{filename}`   | GET    | at the root |
///
/// The last one is why `apiPath` is per-case rather than a constant `"/api"`:
/// the file-resolve route is served from the host root, and a shared `/api`
/// prefix would have made it `/api/{id}/resolve/…`.
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
    /// `GET /{id}/resolve/{revision}/{filename}` — the bytes of one file.
    ///
    /// Served from the host ROOT, not from `/api`. `ModelDownloader` reaches
    /// this through ``HuggingFaceClient/downloadURL(repoId:filename:revision:)``.
    case resolveFile(id: String, revision: String, filename: String)

    var host: String { "huggingface.co" }
    var scheme: String { "https" }
    var port: Int? { nil }

    var apiPath: String {
        switch self {
        case .searchModels, .getModel, .listTree: "/api"
        // The file-resolve route is not an API route: it is the repository
        // itself, at the host root.
        case .resolveFile: ""
        }
    }

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
        case .resolveFile(let id, let revision, let filename):
            "/\(id)/resolve/\(revision)/\(filename)"
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
        case .getModel, .listTree, .resolveFile:
            return nil
        }
    }
}
