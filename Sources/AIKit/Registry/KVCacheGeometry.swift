import Foundation

/// The shape of a transformer's key/value cache, read from the model's own
/// `config.json`, so the memory a context costs is computed from the model and
/// never hard-coded per model name.
///
/// bytes per token = 2 (K and V) x layers x kvHeads x headDim x element bytes.
/// Llama-3.1-8B: 2 x 32 x 8 x 128 x 2 = 128 KiB per token; Llama-3.2-3B:
/// 2 x 28 x 8 x 128 x 2 = 112 KiB.
package struct KVCacheGeometry: Sendable, Equatable {
    package let layers: Int
    package let kvHeads: Int
    package let headDim: Int

    package init(layers: Int, kvHeads: Int, headDim: Int) {
        self.layers = layers
        self.kvHeads = kvHeads
        self.headDim = headDim
    }

    /// Bytes of K and V one token occupies. The cache is fp16 unless the
    /// runtime quantises it, so 2 bytes per element is the safe figure.
    package func bytesPerToken(elementBytes: Int = 2) -> Int {
        2 * layers * kvHeads * headDim * elementBytes
    }

    /// Read the geometry out of a `config.json`. A multimodal config nests the
    /// language model under `text_config`. Nil when a field is missing: an
    /// unknown geometry is "unmeasured", never "free".
    package init?(configJSON: Data) {
        guard let root = try? JSONSerialization.jsonObject(with: configJSON) as? [String: Any] else {
            return nil
        }
        let config = (root["text_config"] as? [String: Any]) ?? root
        guard let layers = config["num_hidden_layers"] as? Int else { return nil }
        let heads = config["num_attention_heads"] as? Int
        guard let kvHeads = (config["num_key_value_heads"] as? Int) ?? heads else { return nil }
        let headDim: Int
        if let declared = config["head_dim"] as? Int {
            headDim = declared
        } else if let hidden = config["hidden_size"] as? Int, let heads, heads > 0 {
            headDim = hidden / heads
        } else {
            return nil
        }
        guard layers > 0, kvHeads > 0, headDim > 0 else { return nil }
        self.init(layers: layers, kvHeads: kvHeads, headDim: headDim)
    }
}
