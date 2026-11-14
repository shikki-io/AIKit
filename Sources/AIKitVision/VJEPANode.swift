// MARK: - VJEPANode
// Spec: features/shikki-vjepa-mlx-node.md §5 Phase 2 + BR-06 + BR-16.
//
// Wraps `VJEPAInference` as a first-class Shikki provider node:
//   - exposes video/frames → embedding over a stable interface,
//   - publishes heartbeat on `shikki.arena.nodes.heartbeat` as `NodeKind.provider`,
//   - reports model metadata (BR-16): modelName, modelVersion, inputFrames,
//     embeddingDim, latencyP50Ms, memoryPeakMB.
//
// Node-level NATS publishing is delegated to a minimal `VisionNATSPublisher`
// protocol so ShikkiKit can stay independent of ShikkiQuality and still be
// tested with a recording double. The heartbeat subject is stable and matches
// existing arena node convention (`shikki.arena.nodes.heartbeat`).

import Foundation

// MARK: - NATS publisher protocol (vision-local)

/// Minimal NATS publisher used by vision nodes to emit heartbeats.
///
/// Kept local to the vision module so `ShikkiKit` does not depend on the
/// quality-server NATS stack. Real builds can wire the same nats.swift
/// client behind this interface via a thin adapter.
public protocol VisionNATSPublisher: Sendable {
    func publish(_ data: Data, subject: String) async throws
    var isConnected: Bool { get async }
}

// MARK: - BR-16 metadata

/// Model metadata advertised in every VJEPANode heartbeat (spec BR-16).
public struct VJEPANodeMetadata: Codable, Sendable, Equatable {
    public let modelName: String
    public let modelVersion: String
    public let inputFrames: Int
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

// MARK: - Provider heartbeat payload

/// Heartbeat payload published by a VJEPANode.
///
/// This is a vision-provider-specific heartbeat — carries the BR-16
/// model metadata alongside the standard node identity. Shape is flat
/// JSON for easy consumption by `shi nodes ls` and the arena registry.
public struct VJEPAProviderHeartbeat: Codable, Sendable, Equatable {
    public let nodeId: String
    public let hostname: String
    public let nodeKind: String
    public let sentAt: Date
    public let metadata: VJEPANodeMetadata

    public init(
        nodeId: String,
        hostname: String,
        nodeKind: String,
        sentAt: Date,
        metadata: VJEPANodeMetadata
    ) {
        self.nodeId = nodeId
        self.hostname = hostname
        self.nodeKind = nodeKind
        self.sentAt = sentAt
        self.metadata = metadata
    }
}

// MARK: - Node

/// V-JEPA provider node. Wraps `VJEPAInference`, tracks latency, and publishes
/// periodic heartbeats so agents can discover it on the NATS mesh (BR-06).
///
/// Thread-safe: inference is pure and the node keeps a small rolling latency
/// buffer behind an actor-local lock. `publishHeartbeat` is a single async
/// round-trip — the caller picks the cadence (typically 30s, driven from the
/// orchestrator heartbeat loop).
public actor VJEPANode {

    // MARK: Identity

    nonisolated public let nodeId: String
    nonisolated public let hostname: String
    nonisolated public let modelVersion: String

    // MARK: Wrapped inference

    nonisolated public let inference: VJEPAInference

    // MARK: Latency tracking

    /// Ring buffer of recent per-call latencies in milliseconds. Used to
    /// compute P50 for heartbeat metadata (BR-16). Capped at 64 entries.
    private var latenciesMs: [Int] = []
    private let latencyCapacity: Int = 64

    /// Peak resident memory observed during the node's lifetime, in MB.
    private var memoryPeakMB: Int = 0

    // MARK: Constants

    public static let nodeKind: String = "provider"
    public static let heartbeatSubject: String = "shikki.arena.nodes.heartbeat"

    // MARK: Init

    public init(
        nodeId: String,
        hostname: String,
        inference: VJEPAInference,
        modelVersion: String
    ) {
        self.nodeId = nodeId
        self.hostname = hostname
        self.inference = inference
        self.modelVersion = modelVersion
    }

    // MARK: Public name — AgentProvider-style identity

    nonisolated public var name: String {
        "vjepa-vitb16"
    }

    // MARK: Embedding API

    /// Embed already-preprocessed frames. Synchronous because the backend is
    /// pure CPU/GPU compute. Latency is recorded for heartbeat P50.
    nonisolated public func embed(frames: VideoFrameTensor) throws -> VJEPAEmbedding {
        let start = Date()
        let embedding = try inference.embed(frames: frames)
        let ms = Int(Date().timeIntervalSince(start) * 1000)
        Task { await self.recordLatency(ms) }
        return embedding
    }

    /// Embed a video file. Preprocesses then runs the encoder.
    nonisolated public func embed(videoAt url: URL) async throws -> VJEPAEmbedding {
        let start = Date()
        let embedding = try await inference.embed(videoAt: url)
        let ms = Int(Date().timeIntervalSince(start) * 1000)
        await recordLatency(ms)
        return embedding
    }

    // MARK: Heartbeat

    /// Build the current heartbeat payload reflecting the node's live state.
    public func makeHeartbeat(now: Date = Date()) -> VJEPAProviderHeartbeat {
        VJEPAProviderHeartbeat(
            nodeId: nodeId,
            hostname: hostname,
            nodeKind: Self.nodeKind,
            sentAt: now,
            metadata: makeMetadata()
        )
    }

    /// Serialize + publish the heartbeat to the given NATS publisher.
    public func publishHeartbeat(via publisher: VisionNATSPublisher, now: Date = Date()) async throws {
        let hb = makeHeartbeat(now: now)
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(hb)
        try await publisher.publish(data, subject: Self.heartbeatSubject)
    }

    // MARK: Metadata

    public func makeMetadata() -> VJEPANodeMetadata {
        VJEPANodeMetadata(
            modelName: name,
            modelVersion: modelVersion,
            inputFrames: inference.preprocessor.targetFrameCount,
            embeddingDim: inference.encoder.config.hiddenDim,
            latencyP50Ms: computeLatencyP50(),
            memoryPeakMB: currentMemoryPeakMB()
        )
    }

    // MARK: Internals

    private func recordLatency(_ ms: Int) {
        latenciesMs.append(ms)
        if latenciesMs.count > latencyCapacity {
            latenciesMs.removeFirst(latenciesMs.count - latencyCapacity)
        }
        let mem = Self.residentMemoryMB()
        if mem > memoryPeakMB { memoryPeakMB = mem }
    }

    private func computeLatencyP50() -> Int {
        guard !latenciesMs.isEmpty else { return 0 }
        let sorted = latenciesMs.sorted()
        return sorted[sorted.count / 2]
    }

    private func currentMemoryPeakMB() -> Int {
        max(memoryPeakMB, Self.residentMemoryMB())
    }

    /// Resident memory snapshot in MB. Best-effort — returns 0 if the syscall
    /// is not available in the current sandbox.
    static func residentMemoryMB() -> Int {
        #if canImport(Darwin)
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size)
        let kerr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        guard kerr == KERN_SUCCESS else { return 0 }
        return Int(info.resident_size / (1024 * 1024))
        #else
        return 0
        #endif
    }
}

// MARK: - Darwin import (for resident memory)

#if canImport(Darwin)
import Darwin.Mach
#endif
