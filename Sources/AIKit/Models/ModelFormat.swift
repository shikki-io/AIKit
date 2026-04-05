/// On-disk format of an AI model.
public enum ModelFormat: String, Sendable, Codable, CaseIterable {
    /// GGML/GGUF quantized format (llama.cpp).
    case gguf
    /// Apple MLX format.
    case mlx
    /// Apple Core ML compiled model.
    case coreml
    /// HuggingFace safetensors format.
    case safetensors
    /// Remote API — no local file.
    case api
}
