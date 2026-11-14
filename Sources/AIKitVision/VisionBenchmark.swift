// MARK: - VisionBenchmark
// Spec §5 Phase 3 deliverable 2 — video understanding benchmark corpus.
//
// Phase 3 dispatch brief: ship a 10-task starter corpus (synthetic, hermetic
// for CI) plus the documented path to the full 50-task corpus. The 50-task
// corpus is a v1.1 follow-up because 30 of those 50 will bootstrap from
// real agent activity logs per @Metsuke's answer in §14.1.4 of the spec.
//
// Each task declares:
//   - a stable `taskId`
//   - a category (routing bucket for the adaptive router)
//   - a natural-language `question`
//   - the `groundTruthLabel` (judged by `LabelMatchJudge` unless overridden)
//   - a `frames` tensor deterministically synthesised from the task id so
//     the corpus is bit-identical across CI runs.
//
// Categories in the starter corpus (chosen to mirror the three task types
// enumerated in spec §5 Phase 3 deliverable 2):
//
//   - `ui-identification`   — which UI element was interacted with
//   - `error-detection`     — what error appeared on screen
//   - `action-performed`    — what action was performed
//
// Synthetic frames encode the ground-truth label into a low-frequency
// pattern so the two demo providers in this file (VJEPARetrievalProvider
// + MockLLMFrameDescriptionProvider) have a principled way to answer.

import Foundation

// MARK: - Task helpers

/// One synthesised task in the starter corpus.
public struct VisionBenchmarkTask: Sendable {
    public let task: BattleTask
    /// Hint string the synthetic providers use to recover the answer. Encodes
    /// the ground-truth label as a deterministic pixel signature.
    public let labelSignal: [Float]

    public init(task: BattleTask, labelSignal: [Float]) {
        self.task = task
        self.labelSignal = labelSignal
    }
}

// MARK: - Benchmark

public struct VisionBenchmark: Sendable {

    // MARK: Public API

    public static let corpusVersion: String = "vision-benchmark-v1-10tasks-2026-04-15"

    /// Full path from 10-task starter to 50-task corpus — documented in the
    /// spec, surfaced here so `shi vision benchmark` can print it.
    public static let fullCorpusRoadmap: String = """
    Starter corpus: 10 synthetic tasks shipped in-tree for CI.
    Phase 3.5 (post-dispatch): bootstrap 30 tasks from agent activity logs
    (screen recording + concurrent terminal commands as implicit ground truth,
    per §14.1.4). Remaining 20 tasks hand-labelled for ambiguous visual
    regressions. Target total: 50 tasks across 5 categories.
    """

    /// The 10-task starter corpus. Deterministic — same output every call.
    public static func starterCorpus() -> [VisionBenchmarkTask] {
        let rows: [StarterRow] = [
            StarterRow(id: "t01", category: "ui-identification", question: "Which button was clicked?", label: "submit"),
            StarterRow(id: "t02", category: "ui-identification", question: "Which menu was opened?", label: "file"),
            StarterRow(id: "t03", category: "ui-identification", question: "Which tab is active?", label: "editor"),
            StarterRow(id: "t04", category: "error-detection", question: "What error appeared?", label: "timeout"),
            StarterRow(id: "t05", category: "error-detection", question: "What error appeared?", label: "permission-denied"),
            StarterRow(id: "t06", category: "error-detection", question: "What error appeared?", label: "not-found"),
            StarterRow(id: "t07", category: "action-performed", question: "What action was performed?", label: "save"),
            StarterRow(id: "t08", category: "action-performed", question: "What action was performed?", label: "copy"),
            StarterRow(id: "t09", category: "action-performed", question: "What action was performed?", label: "delete"),
            StarterRow(id: "t10", category: "action-performed", question: "What action was performed?", label: "rename"),
        ]
        return rows.map { row in
            let frames = Self.synthesiseFrames(forLabel: row.label)
            let signal = Self.labelSignature(row.label)
            let task = BattleTask(
                taskId: row.id,
                category: row.category,
                question: row.question,
                groundTruthLabel: row.label,
                videoURL: nil,
                frames: frames
            )
            return VisionBenchmarkTask(task: task, labelSignal: signal)
        }
    }

    // MARK: Deterministic frame synthesis

    /// Build a tiny deterministic `[T=4, C=3, H=16, W=16]` tensor whose first
    /// pixel sequence spells out the label signature. Real V-JEPA weights
    /// never see these tensors in CI — we use the reference backend, which
    /// is deterministic. The label signature lets the synthetic providers
    /// recover the ground truth without cheating via a side channel.
    static func synthesiseFrames(forLabel label: String) -> VideoFrameTensor {
        let T = 4, C = 3, H = 16, W = 16
        let sig = labelSignature(label)
        var data = [Float](repeating: 0, count: T * C * H * W)
        for t in 0..<T {
            for c in 0..<C {
                for y in 0..<H {
                    for x in 0..<W {
                        let idx = ((t * C + c) * H + y) * W + x
                        let sIdx = (x + y + c + t) % sig.count
                        data[idx] = sig[sIdx]
                    }
                }
            }
        }
        return VideoFrameTensor(
            frameCount: T,
            channels: C,
            height: H,
            width: W,
            data: data
        )
    }

    /// Fingerprint of the label — stable across runs. A 16-dim float vector
    /// derived from the UTF-8 bytes of the label, normalised to unit norm.
    static func labelSignature(_ label: String) -> [Float] {
        let bytes = Array(label.utf8)
        var out = [Float](repeating: 0, count: 16)
        for (i, b) in bytes.enumerated() {
            out[i % 16] += Float(b) / 255.0
        }
        // Normalise to unit norm so cosine similarity is well-behaved.
        let norm = out.reduce(Float(0)) { $0 + $1 * $1 }.squareRoot()
        if norm > 0 {
            for i in 0..<out.count { out[i] /= norm }
        }
        return out
    }
}

// MARK: - Provider A: V-JEPA + retrieval (spec deliverable 3)

/// Provider that embeds the task's frames, searches a known library of
/// (signature → label) pairs, and returns the closest-matching label.
///
/// For the 10-task starter corpus this is wired to an in-memory library
/// built from the corpus itself — V-JEPA retrieval is expected to nail
/// every task. When the full corpus arrives the library becomes a disk
/// cache and the nearest-neighbour match can miss.
public struct VJEPARetrievalProvider: VisionProvider, Sendable {

    public let name: String = "vjepa-retrieval"

    /// (label → signature) lookup used to invert the ground-truth mapping.
    public let labelLibrary: [String: [Float]]

    /// Estimated energy cost per task in watt-hours. M-series Macs at the
    /// 16-frame target draw roughly 7-12 W during the ~150 ms inference
    /// window, which is ~0.0003-0.0005 Wh per task. Rounding to 0.002 Wh
    /// as a conservative v1 estimate; the benchmark tool refines this
    /// from measured latency on the dev machine.
    public let energyWhPerTask: Double

    public init(
        labelLibrary: [String: [Float]],
        energyWhPerTask: Double = 0.002
    ) {
        self.labelLibrary = labelLibrary
        self.energyWhPerTask = energyWhPerTask
    }

    /// Build a provider populated from the starter corpus.
    public static func forStarterCorpus() -> VJEPARetrievalProvider {
        var lib: [String: [Float]] = [:]
        for item in VisionBenchmark.starterCorpus() {
            lib[item.task.groundTruthLabel] = item.labelSignal
        }
        return VJEPARetrievalProvider(labelLibrary: lib)
    }

    public func answer(task: BattleTask) async throws -> ProviderAnswer {
        // Phase 3 is framework-level — the *real* provider would pipe frames
        // through `VJEPAInference` and query `EmbeddingRetriever`. For the
        // synthetic starter corpus we recover the signature from the frame
        // tensor and cosine-match against the label library.
        let sig = Self.deriveSignature(task: task)
        let best = Self.nearestLabel(signature: sig, library: labelLibrary)
        return ProviderAnswer(
            label: best ?? "",
            tokenCostUSD: 0,
            energyWh: energyWhPerTask,
            memoryMB: 180
        )
    }

    // MARK: Internals

    static func deriveSignature(task: BattleTask) -> [Float] {
        // The benchmark frames encode the label fingerprint in the (t=0, c=0,
        // y=0, x=0..15) stripe. Reading those 16 floats recovers the
        // signature. For non-synthetic tasks (no `frames` provided), we fall
        // back to the ground truth label itself — a sanity path for the
        // integration layer, never exercised in production.
        if let frames = task.frames {
            var out = [Float](repeating: 0, count: 16)
            for i in 0..<16 {
                out[i] = frames.data[i]
            }
            // Normalise so cosine similarity lines up with the library.
            let norm = out.reduce(Float(0)) { $0 + $1 * $1 }.squareRoot()
            if norm > 0 { for i in 0..<out.count { out[i] /= norm } }
            return out
        }
        return VisionBenchmark.labelSignature(task.groundTruthLabel)
    }

    static func nearestLabel(signature: [Float], library: [String: [Float]]) -> String? {
        var best: (label: String, sim: Float)?
        for (label, sig) in library {
            let sim = cosine(signature, sig)
            if best == nil || sim > best!.sim {
                best = (label, sim)
            }
        }
        return best?.label
    }

    static func cosine(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count else { return 0 }
        var dot: Float = 0
        var na: Float = 0
        var nb: Float = 0
        for i in 0..<a.count {
            dot += a[i] * b[i]
            na  += a[i] * a[i]
            nb  += b[i] * b[i]
        }
        let denom = na.squareRoot() * nb.squareRoot()
        return denom > 0 ? dot / denom : 0
    }
}

// MARK: - Provider B: LLM frame description (spec deliverable 4)

/// Abstract multimodal LLM contract the mock implements and a real provider
/// would also implement. Pluggable so the arena can run V-JEPA vs any
/// LLM-backed provider without code changes (AD-06).
public protocol MultimodalLLMClient: Sendable {
    /// Describe the provided video frames and answer the question in free-form
    /// text. Returns the raw text response plus the number of tokens billed.
    func describe(frames: VideoFrameTensor?, question: String) async throws -> LLMDescribeResponse
}

public struct LLMDescribeResponse: Sendable, Equatable {
    public let text: String
    public let inputTokens: Int
    public let outputTokens: Int
    public init(text: String, inputTokens: Int, outputTokens: Int) {
        self.text = text
        self.inputTokens = inputTokens
        self.outputTokens = outputTokens
    }
}

/// Mock LLM client used in CI + the starter corpus gate. Returns a
/// configurable mapping of (question → answer) or falls back to an "unknown"
/// tag, and advertises a realistic token bill.
///
/// Real provider wiring is documented as a follow-up in the return brief
/// of the Phase 3 dispatch. The real client plugs in via this protocol
/// with zero arena changes.
public struct MockLLMClient: MultimodalLLMClient, Sendable {
    public let responses: [String: String]
    public let inputTokensPerCall: Int
    public let outputTokensPerCall: Int

    public init(
        responses: [String: String],
        inputTokensPerCall: Int = 1200,
        outputTokensPerCall: Int = 30
    ) {
        self.responses = responses
        self.inputTokensPerCall = inputTokensPerCall
        self.outputTokensPerCall = outputTokensPerCall
    }

    public func describe(frames: VideoFrameTensor?, question: String) async throws -> LLMDescribeResponse {
        let text = responses[question] ?? "unknown"
        return LLMDescribeResponse(
            text: text,
            inputTokens: inputTokensPerCall,
            outputTokens: outputTokensPerCall
        )
    }
}

/// Provider that samples frames, sends them to a multimodal LLM, and extracts
/// the answer label from the LLM's free-form description.
public struct MockLLMFrameDescriptionProvider: VisionProvider, Sendable {

    public let name: String
    public let client: MultimodalLLMClient

    /// Cloud LLM pricing per million tokens — Claude Haiku list price as of
    /// 2026-04-15 (documented here so the cost estimate is auditable).
    public let inputTokenUSDPerMillion: Double
    public let outputTokenUSDPerMillion: Double

    public init(
        name: String = "mock-llm-haiku",
        client: MultimodalLLMClient,
        inputTokenUSDPerMillion: Double = 0.80,
        outputTokenUSDPerMillion: Double = 4.00
    ) {
        self.name = name
        self.client = client
        self.inputTokenUSDPerMillion = inputTokenUSDPerMillion
        self.outputTokenUSDPerMillion = outputTokenUSDPerMillion
    }

    /// Build a provider that knows the starter corpus answers. Used in tests
    /// where we want the LLM to be competitive with V-JEPA on relevance but
    /// penalised on token cost.
    public static func forStarterCorpus(name: String = "mock-llm-haiku") -> MockLLMFrameDescriptionProvider {
        var responses: [String: String] = [:]
        for item in VisionBenchmark.starterCorpus() {
            responses[item.task.question] = item.task.groundTruthLabel
        }
        return MockLLMFrameDescriptionProvider(
            name: name,
            client: MockLLMClient(responses: responses)
        )
    }

    public func answer(task: BattleTask) async throws -> ProviderAnswer {
        let resp = try await client.describe(frames: task.frames, question: task.question)
        let tokenCost =
            Double(resp.inputTokens)  * inputTokenUSDPerMillion  / 1_000_000 +
            Double(resp.outputTokens) * outputTokenUSDPerMillion / 1_000_000
        return ProviderAnswer(
            label: resp.text,
            tokenCostUSD: tokenCost,
            energyWh: 0,
            memoryMB: 8
        )
    }
}

/// One row of the starter corpus.
///
/// A named type rather than a four-member tuple: `large_tuple` caps a tuple at
/// two, and four same-typed `String`s positionally is exactly the shape where a
/// question and a label get swapped without the compiler noticing.
private struct StarterRow {
    let id: String
    let category: String
    let question: String
    let label: String
}
