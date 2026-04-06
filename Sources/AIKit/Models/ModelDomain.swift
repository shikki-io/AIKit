/// The primary domain/task a model is designed for.
public enum ModelDomain: String, Sendable, Codable, CaseIterable {
    case llm
    case embedding
    case voice
    case vision
    case inpainting
    case video
}
