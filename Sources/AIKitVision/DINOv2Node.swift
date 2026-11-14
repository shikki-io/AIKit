// MARK: - DINOv2Node
// Spec: features/shikki-vjepa-mlx-node.md §5 Phase 4 + §14.3 + BR-06 + BR-16.
//
// Companion to VJEPANode: handles still images (no temporal dimension).
// Wraps a pluggable `DINOv2Backend` (real CoreML once the conversion
// pipeline lands, `MockDINOv2Backend` in the meantime) and exposes the
// same heartbeat contract as VJEPANode so arena registration is uniform.
//
// Stack per §14.3:
//   Model    : DINOv2 ViT-S/14, 21M params, Apache-2.0
//   Runtime  : CoreML (.mlpackage), ANE-first, GPU fallback
//   Input    : CGImage → 224×224 RGB → ImageNet normalize
//   Output   : [1, 384] fp16 embedding
//   Cache    : ~/.shikki/models/dinov2-vits14/
//
// The backend abstraction also serves the World Model Phase 1 spec, which
// already expects a `DINOv2Backend` + `MockDINOv2Backend` (see
// features/shikki-world-model-hybrid.md §12 item 3).

#if canImport(CoreGraphics)
import CoreGraphics
import Foundation

// MARK: - Embedding value type

/// DINOv2 output: `[1, D]` flat embedding. `D` is fixed at 384 for ViT-S/14.
public struct DINOv2Embedding: Sendable, Equatable {
    public let featureDim: Int
    /// Row-major float32 embedding. For ViT-S/14 this is `[384]`.
    public let data: [Float]

    public init(featureDim: Int, data: [Float]) {
        self.featureDim = featureDim
        self.data = data
    }
}

// MARK: - Errors

public enum DINOv2Error: Error, Sendable, Equatable {
    case preprocessingFailed
    case backendUnavailable(String)
    case invalidInput(String)
}

// MARK: - Backend protocol

/// Pluggable DINOv2 image-embedding backend.
///
/// Allows swapping between:
///   - `MockDINOv2Backend` (deterministic, hash-based) for CI + World Model Phase 1
///   - `CoreMLDINOv2Backend` (real ANE inference, once .mlpackage is shipped)
///   - any experimental backend (MLX port, CPU reference, etc.)
///
/// Implementations must be deterministic for identical normalized pixel inputs.
public protocol DINOv2Backend: Sendable {
    /// Forward pass over a normalized 224×224 RGB pixel tensor (`[1, 3, 224, 224]`,
    /// row-major in CHW order, already ImageNet-normalized).
    func forward(normalizedPixels: [Float]) throws -> [Float]

    /// Declared output feature dimension. ViT-S/14 = 384.
    var featureDim: Int { get }

    /// Declared input spatial size. ViT-S/14 = 224.
    var inputSize: Int { get }
}

// MARK: - Mock backend (deterministic, CoreML-free)

/// Deterministic in-process backend.
///
/// Hashes the normalized pixel buffer with FNV-1a (64-bit), then derives each
/// of the 384 output features from a mixed seed so the embedding is:
///   - identical for identical inputs (bit-exact),
///   - distinct for distinct inputs (even single-bit pixel changes cascade),
///   - bounded in roughly `[-1, 1]` (looks like a unit-norm ViT output).
///
/// This is NOT a real DINOv2 forward pass. It is sufficient for:
///   - pipeline / shape / determinism tests,
///   - cosine similarity over identical vs different screenshots (visual QC),
///   - World Model Phase 1's MaterialClassifier smoke tests.
///
/// Swap for `CoreMLDINOv2Backend` once the conversion pipeline ships.
public struct MockDINOv2Backend: DINOv2Backend {

    public let featureDim: Int
    public let inputSize: Int

    public init(featureDim: Int = 384, inputSize: Int = 224) {
        self.featureDim = featureDim
        self.inputSize = inputSize
    }

    public func forward(normalizedPixels: [Float]) throws -> [Float] {
        var h = Self.fnv1aOffset
        for value in normalizedPixels {
            // Quantize to int bits so tiny float noise cannot make two
            // visually-identical embeddings differ, but distinct real
            // pixel differences still change the hash.
            let bits = value.bitPattern
            h = (h ^ UInt64(bits)) &* Self.fnv1aPrime
        }
        var output = [Float](repeating: 0, count: featureDim)
        for i in 0..<featureDim {
            let mixed = h &+ UInt64(i) &* 0x9E37_79B9_7F4A_7C15
            // Map 64-bit state → [-1, 1] uniformly via top 24 bits.
            let top = Float((mixed >> 40) & 0x00FF_FFFF) / Float(0x00FF_FFFF)
            output[i] = top * 2.0 - 1.0
        }
        // L2-normalize so cosine similarity is well-defined.
        var sq: Float = 0
        for v in output { sq += v * v }
        let norm = sqrtf(max(sq, 1e-12))
        for i in 0..<output.count { output[i] /= norm }
        return output
    }

    // FNV-1a 64-bit constants.
    private static let fnv1aOffset: UInt64 = 0xCBF2_9CE4_8422_2325
    private static let fnv1aPrime: UInt64 = 0x0000_0100_0000_01B3
}

// MARK: - Node metadata (parallel to VJEPANodeMetadata)

/// Model metadata advertised in every DINOv2Node heartbeat (spec BR-16).
public struct DINOv2NodeMetadata: Codable, Sendable, Equatable {
    public let modelName: String
    public let modelVersion: String
    public let inputFrames: Int      // always 1 for single-image models
    public let embeddingDim: Int
    public let latencyP50Ms: Int
    public let memoryPeakMB: Int

    public init(
        modelName: String,
        modelVersion: String,
        inputFrames: Int,
        embeddingDim: Int,
        latencyP50Ms: Int,
        memoryPeakMB: Int
    ) {
        self.modelName = modelName
        self.modelVersion = modelVersion
        self.inputFrames = inputFrames
        self.embeddingDim = embeddingDim
        self.latencyP50Ms = latencyP50Ms
        self.memoryPeakMB = memoryPeakMB
    }
}

// MARK: - Heartbeat payload

public struct DINOv2ProviderHeartbeat: Codable, Sendable, Equatable {
    public let nodeId: String
    public let hostname: String
    public let nodeKind: String
    public let sentAt: Date
    public let metadata: DINOv2NodeMetadata

    public init(
        nodeId: String,
        hostname: String,
        nodeKind: String,
        sentAt: Date,
        metadata: DINOv2NodeMetadata
    ) {
        self.nodeId = nodeId
        self.hostname = hostname
        self.nodeKind = nodeKind
        self.sentAt = sentAt
        self.metadata = metadata
    }
}

// MARK: - Node

/// DINOv2 provider node. Wraps a `DINOv2Backend`, preprocesses CGImages to
/// the ImageNet-normalized tensor DINOv2 expects, tracks latency, and
/// publishes heartbeats over the same NATS subject as VJEPANode.
public actor DINOv2Node {

    // MARK: Identity

    nonisolated public let nodeId: String
    nonisolated public let hostname: String
    nonisolated public let modelVersion: String
    nonisolated public let backend: DINOv2Backend

    // MARK: Latency tracking

    private var latenciesMs: [Int] = []
    private let latencyCapacity: Int = 64

    // MARK: Constants

    public static let nodeKind: String = "provider"
    public static let heartbeatSubject: String = "shikki.arena.nodes.heartbeat"

    // MARK: Init

    public init(
        nodeId: String,
        hostname: String,
        backend: DINOv2Backend,
        modelVersion: String
    ) {
        self.nodeId = nodeId
        self.hostname = hostname
        self.backend = backend
        self.modelVersion = modelVersion
    }

    nonisolated public var name: String { "dinov2-vits14" }

    // MARK: Embedding API

    /// Embed a `CGImage` to a `[1, 384]` DINOv2 embedding.
    ///
    /// Pipeline:
    ///   1. resize to `backend.inputSize` × `backend.inputSize` RGBA
    ///   2. convert to CHW float32 and ImageNet-normalize
    ///   3. call `backend.forward(normalizedPixels:)`
    nonisolated public func embed(image: CGImage) throws -> DINOv2Embedding {
        let start = Date()
        let pixels = try Self.preprocess(image: image, size: backend.inputSize)
        let flat = try backend.forward(normalizedPixels: pixels)
        guard flat.count == backend.featureDim else {
            throw DINOv2Error.backendUnavailable(
                "backend returned \(flat.count) floats, expected \(backend.featureDim)"
            )
        }
        let ms = Int(Date().timeIntervalSince(start) * 1000)
        Task { await self.recordLatency(ms) }
        return DINOv2Embedding(featureDim: backend.featureDim, data: flat)
    }

    // MARK: Heartbeat

    public func makeHeartbeat(now: Date = Date()) -> DINOv2ProviderHeartbeat {
        DINOv2ProviderHeartbeat(
            nodeId: nodeId,
            hostname: hostname,
            nodeKind: Self.nodeKind,
            sentAt: now,
            metadata: makeMetadata()
        )
    }

    public func publishHeartbeat(via publisher: VisionNATSPublisher, now: Date = Date()) async throws {
        let hb = makeHeartbeat(now: now)
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(hb)
        try await publisher.publish(data, subject: Self.heartbeatSubject)
    }

    public func makeMetadata() -> DINOv2NodeMetadata {
        DINOv2NodeMetadata(
            modelName: name,
            modelVersion: modelVersion,
            inputFrames: 1,
            embeddingDim: backend.featureDim,
            latencyP50Ms: computeLatencyP50(),
            memoryPeakMB: VJEPANode.residentMemoryMB()
        )
    }

    // MARK: Internals

    private func recordLatency(_ ms: Int) {
        latenciesMs.append(ms)
        if latenciesMs.count > latencyCapacity {
            latenciesMs.removeFirst(latenciesMs.count - latencyCapacity)
        }
    }

    private func computeLatencyP50() -> Int {
        guard !latenciesMs.isEmpty else { return 0 }
        let sorted = latenciesMs.sorted()
        return sorted[sorted.count / 2]
    }

    // MARK: - Preprocessing (CGImage → normalized [1, 3, H, W] float tensor)

    static func preprocess(image: CGImage, size: Int) throws -> [Float] {
        let bytesPerRow = size * 4
        var buffer = [UInt8](repeating: 0, count: size * bytesPerRow)
        let space = CGColorSpaceCreateDeviceRGB()
        let info: UInt32 = CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
        try buffer.withUnsafeMutableBytes { raw in
            guard let base = raw.baseAddress, let ctx = CGContext(
                data: base,
                width: size,
                height: size,
                bitsPerComponent: 8,
                bytesPerRow: bytesPerRow,
                space: space,
                bitmapInfo: info
            ) else {
                throw DINOv2Error.preprocessingFailed
            }
            ctx.interpolationQuality = .high
            let rect = CGRect(x: 0, y: 0, width: size, height: size)
            ctx.draw(image, in: rect)
        }

        // ImageNet mean/std (same as VideoFramePreprocessor).
        let meanR: Float = 0.485, meanG: Float = 0.456, meanB: Float = 0.406
        let stdR: Float = 0.229, stdG: Float = 0.224, stdB: Float = 0.225

        var data = [Float](repeating: 0, count: 3 * size * size)
        for y in 0..<size {
            for x in 0..<size {
                let pixel = (y * size + x) * 4
                let r = Float(buffer[pixel + 0]) / 255.0
                let g = Float(buffer[pixel + 1]) / 255.0
                let b = Float(buffer[pixel + 2]) / 255.0
                let rIdx = (0 * size + y) * size + x
                let gIdx = (1 * size + y) * size + x
                let bIdx = (2 * size + y) * size + x
                data[rIdx] = (r - meanR) / stdR
                data[gIdx] = (g - meanG) / stdG
                data[bIdx] = (b - meanB) / stdB
            }
        }
        return data
    }
}

#endif
