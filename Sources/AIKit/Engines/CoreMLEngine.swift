#if canImport(CoreML)
    import AVFoundation
    import CoreML
    import Foundation

    /// Runtime engine for CoreML compiled models (.mlmodel/.mlpackage).
    /// Best for vision tasks on Neural Engine (ANE): OCR, layout detection, inpainting.
    ///
    /// Since W6 of spec `ai-layer-aikit-core-2026-10-31` the engine also serves
    /// voice: a descriptor in ``ModelDomain/voice``, or one advertising
    /// `.voiceToText`, loads as a transcriber instead of as a generic model.
    /// That is the iOS half of S3.5 — Core ML runs on the Neural Engine on both
    /// platforms, so an iPhone needs no CLI to transcribe.
    public final class CoreMLEngine: RuntimeEngine, @unchecked Sendable {
        public let id = "coreml"
        public let displayName = "Apple CoreML"
        public let supportedFormats: [ModelFormat] = [.coreml]

        /// Directory containing .mlmodelc files.
        let modelsDirectory: URL

        /// Currently loaded model identifiers.
        private var _loadedModels: [ModelIdentifier] = []

        public init(modelsDirectory: URL? = nil) {
            if let dir = modelsDirectory {
                self.modelsDirectory = dir
            } else {
                #if os(macOS)
                    let home = FileManager.default.homeDirectoryForCurrentUser
                    self.modelsDirectory = home.appendingPathComponent(".aikit/coreml-models", isDirectory: true)
                #else
                    let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
                    self.modelsDirectory = docs.appendingPathComponent("aikit-coreml-models")
                #endif
            }
        }

        public var isAvailable: Bool {
            // CoreML is available on all Apple platforms.
            true
        }

        // MARK: - RuntimeEngine

        public func loadModel(_ descriptor: ModelDescriptor) async throws -> any AIProvider {
            let model = try await compiledModel(for: descriptor)

            let provider: any AIProvider
            if Self.isVoice(descriptor) {
                // Which engine serves a task is config at the point of use,
                // never a code branch (BR-AIL-08): the descriptor the caller
                // passed is what decides, not a name match inside AIKit.
                provider = CoreMLWhisperProvider(
                    id: "\(id)/\(descriptor.id.modelId)",
                    displayName: "\(displayName) — \(descriptor.name)",
                    capabilities: descriptor.capabilities.union(.voiceToText),
                    modelName: descriptor.id.modelId,
                    model: model
                )
            } else {
                provider = CoreMLProvider(
                    id: "\(id)/\(descriptor.id.modelId)",
                    displayName: "\(displayName) — \(descriptor.name)",
                    capabilities: descriptor.capabilities,
                    model: model
                )
            }

            if !_loadedModels.contains(descriptor.id) {
                _loadedModels.append(descriptor.id)
            }

            return provider
        }

        public func unloadModel(_ id: ModelIdentifier) async throws {
            _loadedModels.removeAll { $0 == id }
        }

        public func loadedModels() -> [ModelIdentifier] {
            _loadedModels
        }

        // MARK: - Loading

        /// Whether a descriptor asks for transcription.
        ///
        /// Either signal is enough: a catalog that fills `domain` and one that
        /// fills `capabilities` both reach the voice path, so a transcriber is
        /// never loaded as a generic model because one field was left blank.
        static func isVoice(_ descriptor: ModelDescriptor) -> Bool {
            descriptor.domain == .voice || descriptor.capabilities.contains(.voiceToText)
        }

        /// Resolves the descriptor to a compiled model on disk and loads it.
        private func compiledModel(for descriptor: ModelDescriptor) async throws -> MLModel {
            let modelURL: URL
            if let localPath = descriptor.localPath {
                modelURL = localPath
            } else {
                modelURL = modelsDirectory.appendingPathComponent("\(descriptor.id.modelId).mlmodelc")
            }

            guard FileManager.default.fileExists(atPath: modelURL.path) else {
                throw AIKitError.modelNotFound(descriptor.id)
            }

            let compiledURL: URL
            if modelURL.pathExtension == "mlmodelc" {
                compiledURL = modelURL
            } else {
                compiledURL = try await MLModel.compileModel(at: modelURL)
            }

            let config = MLModelConfiguration()
            config.computeUnits = .all  // Use ANE + GPU + CPU as needed.
            return try MLModel(contentsOf: compiledURL, configuration: config)
        }

        // MARK: - Voice weights, from config

        /// Core ML Whisper packages in descending preference.
        ///
        /// `small` leads because the Flsh voice spec
        /// (`features/flsh-whisper-e4b-voice-stack.md`, BR-01) chose
        /// Whisper-small (~460 MB, ~4.4% WER) as the on-device default, with
        /// large reserved as the Mac upgrade. A caller on a machine with the
        /// headroom passes its own list to ``resolveVoiceWeights(preferring:)``
        /// rather than AIKit guessing from the host — the point of BR-AIL-08.
        public static let voiceWeightsPreference = [
            "whisper-small.mlmodelc",
            "whisper-small.mlpackage",
            "whisper-base.mlmodelc",
            "whisper-base.mlpackage",
        ]

        /// The most preferred voice weights present in ``modelsDirectory``, or
        /// `nil` when none are installed.
        ///
        /// This is what makes the model reachable from config: a caller asks
        /// what is on disk and passes the answer in a `ModelDescriptor`,
        /// instead of a path being decided inside the engine.
        public func resolveVoiceWeights(
            preferring preference: [String] = CoreMLEngine.voiceWeightsPreference
        ) -> URL? {
            for name in preference {
                let candidate = modelsDirectory.appendingPathComponent(name)
                if FileManager.default.fileExists(atPath: candidate.path) { return candidate }
            }
            return nil
        }
    }

    // MARK: - CoreMLProvider

    /// AIProvider that wraps a loaded CoreML model for inference.
    struct CoreMLProvider: AIProvider, @unchecked Sendable {
        let id: String
        let displayName: String
        let capabilities: AICapabilities

        private let model: MLModel

        init(id: String, displayName: String, capabilities: AICapabilities, model: MLModel) {
            self.id = id
            self.displayName = displayName
            self.capabilities = capabilities
            self.model = model
        }

        var status: AIProviderStatus {
            get async { .ready }
        }

        func complete(request: AIRequest) async throws -> AIResponse {
            // S3.2, second half. A vision or OCR model handed a `.wav` would
            // otherwise read `messages`, ignore the audio entirely and answer
            // about nothing — a wrong answer wearing a right answer's clothes.
            try request.rejectUnservableAudio(by: capabilities)

            let startTime = ContinuousClock.now

            // CoreML models have model-specific I/O. For now, pass the user message
            // as a generic text input and return the model output as text.
            // Real implementations will specialize per model type (vision, OCR, etc.).
            guard let prompt = request.messages.first(where: { $0.role == .user })?.content else {
                throw AIKitError.requestFailed("No user message found for CoreML inference")
            }

            // Generic feature provider with a "text" input.
            let input = try MLDictionaryFeatureProvider(dictionary: ["text": prompt as NSString])
            let output = try await model.prediction(from: input)

            // Extract first output feature as string.
            var resultText = ""
            for name in output.featureNames {
                if let value = output.featureValue(for: name)?.stringValue {
                    resultText = value
                    break
                }
            }

            let elapsed = startTime.duration(to: .now)
            let latencyMs = Int(
                elapsed.components.seconds * 1000 + elapsed.components.attoseconds / 1_000_000_000_000_000)

            return AIResponse(
                content: resultText,
                model: id,
                tokensUsed: TokenUsage(prompt: 0, completion: 0),
                latencyMs: latencyMs,
                providerID: id
            )
        }
    }

    // MARK: - CoreMLWhisperProvider

    /// Transcribes the audio an `AIRequest` carries through a Core ML model.
    ///
    /// The iOS half of S3.5. It answers through the same `AIProvider` surface
    /// as every other provider — a transcription is an `AIRequest` with an
    /// audio input, not a separate protocol (BR-AIL-01).
    ///
    /// ## What this provider fills, and what it refuses
    ///
    /// Core ML models declare their own I/O, and a Whisper conversion is a
    /// *pipeline*: a mel-spectrogram stage takes a window of raw 16 kHz mono
    /// samples, and the encoder stage after it takes that stage's mel output.
    /// This provider fills from an audio file exactly what an audio file can
    /// honestly fill:
    ///
    /// - a **string** input receives the audio's path, which is what an
    ///   end-to-end packaged converter asks for;
    /// - a **one-dimensional** multi-array receives the decoded samples,
    ///   padded or truncated to the declared window;
    /// - anything else — a mel-shaped array, an image — is **refused by name**
    ///   with ``AIKitError/requestFailed(_:)``.
    ///
    /// That last clause is the important one. Chaining a multi-stage Whisper
    /// pipeline is not part of W6, so rather than feed samples into a
    /// mel-shaped tensor and return whatever comes out, the provider says which
    /// feature it could not fill and what that feature wanted. An empty
    /// transcript is likewise an error, never a successful empty answer: the
    /// stub this provider replaces returned `""` forever and nothing noticed.
    struct CoreMLWhisperProvider: AIProvider, @unchecked Sendable {
        let id: String
        let displayName: String
        let capabilities: AICapabilities

        /// Sample rate every Whisper variant is trained on.
        static let sampleRate: Double = 16_000

        private let modelName: String
        private let model: MLModel

        init(
            id: String,
            displayName: String,
            capabilities: AICapabilities,
            modelName: String,
            model: MLModel
        ) {
            self.id = id
            self.displayName = displayName
            self.capabilities = capabilities
            self.modelName = modelName
            self.model = model
        }

        var status: AIProviderStatus {
            get async { .ready }
        }

        func complete(request: AIRequest) async throws -> AIResponse {
            let audio = try request.audioInput(servedBy: capabilities)
            let startTime = ContinuousClock.now

            let url = try Self.fileURL(for: audio)
            let input = try Self.featureProvider(
                for: model.modelDescription.inputDescriptionsByName,
                audioURL: url
            )
            let output = try await model.prediction(from: input)

            guard let transcript = Self.transcript(in: output) else {
                throw AIKitError.requestFailed(
                    "\(modelName) produced no transcript for \(url.lastPathComponent). "
                        + "Outputs: [\(output.featureNames.sorted().joined(separator: ", "))]"
                )
            }

            let elapsed = startTime.duration(to: .now)
            let latencyMs = Int(
                elapsed.components.seconds * 1000
                    + elapsed.components.attoseconds / 1_000_000_000_000_000)

            return AIResponse(
                content: transcript,
                model: modelName,
                tokensUsed: TokenUsage(prompt: 0, completion: 0),
                latencyMs: latencyMs,
                providerID: id
            )
        }

        // MARK: - Resolving the audio

        /// The audio input as a readable local file.
        ///
        /// Data carried in the request is written to a temporary file because
        /// `AVAudioFile` reads paths, not bytes.
        static func fileURL(for audio: AIInput) throws -> URL {
            switch audio.source {
            case .url(let url):
                guard url.isFileURL else {
                    throw AIKitError.requestFailed(
                        "\(url.absoluteString) is not a file URL; fetch it before transcribing"
                    )
                }
                guard FileManager.default.fileExists(atPath: url.path) else {
                    throw AIKitError.requestFailed("No audio file at \(url.path)")
                }
                return url

            case .data(let data):
                guard !data.isEmpty else {
                    throw AIKitError.requestFailed("The request carries 0 bytes of audio")
                }
                let url = URL(fileURLWithPath: NSTemporaryDirectory())
                    .appendingPathComponent("aikit-coreml-voice-\(UUID().uuidString).wav")
                do {
                    try data.write(to: url)
                } catch {
                    throw AIKitError.requestFailed(
                        "Could not stage the request's audio: \(error.localizedDescription)"
                    )
                }
                return url
            }
        }

        /// Decodes an audio file to mono `Float` samples at 16 kHz.
        ///
        /// Whisper is trained at one rate on one channel, so anything else has
        /// to be converted rather than reinterpreted.
        static func samples(at url: URL) throws -> [Float] {
            let buffer = try decode(url)

            // Already 16 kHz mono float: no conversion to do.
            if buffer.format.sampleRate == sampleRate,
                buffer.format.channelCount == 1,
                buffer.format.commonFormat == .pcmFormatFloat32,
                let channel = buffer.floatChannelData
            {
                return Array(UnsafeBufferPointer(start: channel[0], count: Int(buffer.frameLength)))
            }

            return try resample(buffer, named: url.lastPathComponent)
        }

        /// Reads the whole file into a buffer in its own format.
        private static func decode(_ url: URL) throws -> AVAudioPCMBuffer {
            let file: AVAudioFile
            do {
                file = try AVAudioFile(forReading: url)
            } catch {
                throw AIKitError.requestFailed(
                    "Could not read \(url.lastPathComponent) as audio: \(error.localizedDescription)"
                )
            }

            let frameCount = AVAudioFrameCount(file.length)
            guard frameCount > 0 else {
                throw AIKitError.requestFailed("\(url.lastPathComponent) carries no audio frames")
            }
            guard let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: frameCount)
            else {
                throw AIKitError.requestFailed("Could not allocate a read buffer for \(url.lastPathComponent)")
            }
            do {
                try file.read(into: buffer)
            } catch {
                throw AIKitError.requestFailed(
                    "Could not decode \(url.lastPathComponent): \(error.localizedDescription)"
                )
            }
            return buffer
        }

        /// Converts a buffer of any rate or channel count to 16 kHz mono.
        private static func resample(_ buffer: AVAudioPCMBuffer, named name: String) throws -> [Float] {
            guard
                let target = AVAudioFormat(
                    commonFormat: .pcmFormatFloat32,
                    sampleRate: sampleRate,
                    channels: 1,
                    interleaved: false
                )
            else {
                throw AIKitError.requestFailed("Could not build the 16 kHz mono format Whisper needs")
            }

            let source = buffer.format
            guard let converter = AVAudioConverter(from: source, to: target) else {
                throw AIKitError.requestFailed(
                    "No conversion from \(Int(source.sampleRate)) Hz / "
                        + "\(source.channelCount)ch to 16 kHz mono"
                )
            }

            let ratio = sampleRate / source.sampleRate
            let capacity = AVAudioFrameCount((Double(buffer.frameLength) * ratio).rounded(.up)) + 1
            guard let converted = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: capacity) else {
                throw AIKitError.requestFailed("Could not allocate a 16 kHz buffer for \(name)")
            }

            var supplied = false
            var conversionError: NSError?
            converter.convert(to: converted, error: &conversionError) { _, status in
                if supplied {
                    status.pointee = .noDataNow
                    return nil
                }
                supplied = true
                status.pointee = .haveData
                return buffer
            }
            if let conversionError {
                throw AIKitError.requestFailed(
                    "Could not resample \(name): \(conversionError.localizedDescription)"
                )
            }
            guard let channel = converted.floatChannelData else {
                throw AIKitError.requestFailed("Resampling \(name) produced no samples")
            }
            return Array(UnsafeBufferPointer(start: channel[0], count: Int(converted.frameLength)))
        }

        // MARK: - Filling the model's inputs

        /// Builds the feature provider for whatever the loaded package declares.
        ///
        /// - Throws: ``AIKitError/requestFailed(_:)`` naming the feature and
        ///   the shape it wanted, when an audio file cannot honestly fill it.
        static func featureProvider(
            for inputs: [String: MLFeatureDescription],
            audioURL: URL
        ) throws -> MLDictionaryFeatureProvider {
            guard !inputs.isEmpty else {
                throw AIKitError.requestFailed("The loaded Core ML model declares no inputs")
            }

            var features: [String: Any] = [:]
            var decoded: [Float]?

            for (name, description) in inputs {
                switch description.type {
                case .string:
                    features[name] = audioURL.path as NSString

                case .multiArray:
                    guard let constraint = description.multiArrayConstraint else {
                        throw Self.cannotFill(name, description)
                    }
                    guard let window = Self.sampleWindow(in: constraint.shape) else {
                        throw Self.cannotFill(name, description)
                    }
                    // Decoded at most once, however many windows the package
                    // declares.
                    let audio = try decoded ?? samples(at: audioURL)
                    decoded = audio
                    features[name] = try Self.multiArray(
                        from: audio,
                        window: window,
                        shape: constraint.shape,
                        dataType: constraint.dataType,
                        feature: name
                    )

                default:
                    throw Self.cannotFill(name, description)
                }
            }

            return try MLDictionaryFeatureProvider(dictionary: features)
        }

        /// The refusal for a feature an audio file cannot fill.
        ///
        /// Names the feature and what it wanted, so the operator sees which
        /// stage of a Whisper pipeline is missing rather than a bad transcript.
        private static func cannotFill(_ name: String, _ description: MLFeatureDescription) -> AIKitError {
            let wanted: String
            if let shape = description.multiArrayConstraint?.shape {
                wanted = "a \(shape.map(\.stringValue).joined(separator: "×")) multi-array"
            } else {
                wanted = "type \(description.type.rawValue)"
            }
            return AIKitError.requestFailed(
                "Core ML input '\(name)' wants \(wanted), which an audio file cannot fill. "
                    + "A mel-spectrogram stage has to run first, and chaining a multi-stage "
                    + "Whisper pipeline is outside this provider."
            )
        }

        /// The sample count a shape asks for, or `nil` when the shape is not a
        /// plain window of samples.
        ///
        /// Leading unit dimensions are a batch axis and are ignored, so both
        /// `[480000]` and `[1, 480000]` are a 30-second window. Anything with
        /// two real axes is a spectrogram, not samples.
        static func sampleWindow(in shape: [NSNumber]) -> Int? {
            let squeezed = Array(shape.map(\.intValue).drop(while: { $0 == 1 }))
            guard squeezed.count == 1, let window = squeezed.first, window > 1 else { return nil }
            return window
        }

        /// Fits the decoded samples to the declared window: zero-padded when
        /// the audio is shorter, truncated when it is longer, which is how a
        /// fixed Whisper window is fed.
        static func multiArray(
            from samples: [Float],
            window: Int,
            shape: [NSNumber],
            dataType: MLMultiArrayDataType,
            feature: String
        ) throws -> MLMultiArray {
            switch dataType {
            case .float32, .float16, .double:
                break
            default:
                throw AIKitError.requestFailed(
                    "Core ML input '\(feature)' wants data type \(dataType.rawValue), "
                        + "which is not a float format audio samples can fill"
                )
            }

            let array = try MLMultiArray(shape: shape, dataType: dataType)
            let count = min(window, samples.count)
            for index in 0..<count {
                array[index] = NSNumber(value: samples[index])
            }
            // Pad the tail explicitly. `MLMultiArray` happens to hand back
            // zeroed storage, but that is not promised anywhere, and silence
            // that is only silence by luck would be a hard bug to see.
            if count < window {
                let silence = NSNumber(value: Float.zero)
                for index in count..<window { array[index] = silence }
            }
            return array
        }

        // MARK: - Reading the result

        /// The first non-empty string output, or `nil` when the model said
        /// nothing — which the caller turns into a typed error rather than an
        /// empty transcript.
        static func transcript(in output: MLFeatureProvider) -> String? {
            for name in output.featureNames.sorted() {
                guard let value = output.featureValue(for: name), value.type == .string else { continue }
                let text = value.stringValue.trimmingCharacters(in: .whitespacesAndNewlines)
                if !text.isEmpty { return text }
            }
            return nil
        }
    }

#else

    import Foundation

    /// Linux stub — CoreML is not available outside Apple platforms.
    public final class CoreMLEngine: RuntimeEngine, @unchecked Sendable {
        public let id = "coreml"
        public let displayName = "Apple CoreML"
        public let supportedFormats: [ModelFormat] = [.coreml]

        public init(modelsDirectory: URL? = nil) {}

        public var isAvailable: Bool { false }

        public func loadModel(_ descriptor: ModelDescriptor) async throws -> any AIProvider {
            throw AIKitError.engineUnavailable("CoreML is only available on Apple platforms")
        }

        public func unloadModel(_ id: ModelIdentifier) async throws {
            throw AIKitError.engineUnavailable("CoreML is only available on Apple platforms")
        }

        public func loadedModels() -> [ModelIdentifier] { [] }
    }

#endif
