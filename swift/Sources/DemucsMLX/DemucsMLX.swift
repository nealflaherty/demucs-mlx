// DemucsMLX.swift — Public API for HTDemucs music source separation
//
// This API is designed to feel familiar to users of the Python demucs library.
//
// Basic usage:
//   let model = try Demucs.getModel("htdemucs")
//   let sources = try Demucs.applyModel(model, mix: audio)
//
// Or using the convenience class:
//   let separator = Separator()
//   try separator.loadModel(from: modelURL)
//   let result = try separator.separate(file: audioURL)

import MLX
import Foundation

// MARK: - Version

/// Library version
public let version = "1.0.0"

/// Default source names for HTDemucs
public let defaultSources = ["drums", "bass", "other", "vocals"]

/// Default model name
public let defaultModel = "htdemucs"

// MARK: - Model Protocol

/// Protocol for Demucs-family models
public protocol DemucsModel {
    /// Source names the model separates into
    var sources: [String] { get }
    /// Expected sample rate
    var sampleRate: Int { get }
    /// Number of audio channels (typically 2 for stereo)
    var audioChannels: Int { get }
    /// Segment length in seconds for chunked processing
    var segment: Float { get }
    /// Run the model forward pass
    mutating func forward(_ mix: MLXArray) -> MLXArray
}

// MARK: - HTDemucs Wrapper

/// HTDemucs model wrapper conforming to DemucsModel protocol
public final class HTDemucs: DemucsModel {
    var model: HTDemucsModel
    
    public var sources: [String] { model.sources }
    public var sampleRate: Int { model.samplerate }
    public var audioChannels: Int { model.audioChannels }
    public var segment: Float { model.segment }
    
    init(model: HTDemucsModel) {
        self.model = model
    }
    
    public func forward(_ mix: MLXArray) -> MLXArray {
        return model.forward(mix)
    }
}

// MARK: - Demucs Namespace

/// Main entry point for the DemucsMLX library.
/// Provides model loading and the core `applyModel` function.
public enum Demucs {
    
    // MARK: - Model Loading
    
    /// Load a pretrained model by name.
    ///
    /// Currently only "htdemucs" is supported. The model weights must be
    /// available at the default path or specified via `repo`.
    ///
    /// - Parameters:
    ///   - name: Model name (default: "htdemucs")
    ///   - repo: Optional path to model repository
    /// - Returns: A DemucsModel instance
    /// - Throws: `DemucsError.modelNotFound` if model cannot be loaded
    public static func getModel(
        _ name: String = defaultModel,
        repo: URL? = nil
    ) throws -> any DemucsModel {
        guard name == "htdemucs" else {
            throw DemucsError.modelNotFound("Only 'htdemucs' model is currently supported")
        }
        
        // Look for model weights
        let modelPath: String
        if let repo = repo {
            modelPath = repo.appendingPathComponent("htdemucs.safetensors").path
        } else {
            // Default locations to search
            let searchPaths = [
                "models/htdemucs.safetensors",
                "htdemucs.safetensors",
                FileManager.default.homeDirectoryForCurrentUser
                    .appendingPathComponent(".cache/demucs/htdemucs.safetensors").path
            ]
            guard let found = searchPaths.first(where: { FileManager.default.fileExists(atPath: $0) }) else {
                throw DemucsError.modelNotFound(
                    "Model weights not found. Searched: \(searchPaths.joined(separator: ", "))")
            }
            modelPath = found
        }
        
        return try loadModel(from: modelPath)
    }
    
    /// Load model weights from a SafeTensors file.
    ///
    /// - Parameter path: Path to the .safetensors file
    /// - Returns: A DemucsModel instance
    /// - Throws: `DemucsError.weightLoadFailed` if loading fails
    public static func loadModel(from path: String) throws -> any DemucsModel {
        guard FileManager.default.fileExists(atPath: path) else {
            throw DemucsError.fileNotFound(path)
        }
        
        var htModel = HTDemucsModel(
            sources: defaultSources,
            audioChannels: 2, channels: 48, channelsTime: -1,
            growth: 2.0, nfft: 4096, cac: true, depth: 4,
            rewrite: true, freqEmb: 0.2, embScale: 10.0,
            embSmooth: true, kernelSize: 8, timeStride: 2,
            stride: 4, context: 1, contextEnc: 0,
            normStarts: 4, normGroups: 4, dconvMode: 3,
            dconvComp: 8, dconvInit: 1e-3, bottomChannels: 512,
            tLayers: 5, tHeads: 8, tHiddenScale: 4.0,
            tNormIn: true, tNormOut: true, tCrossFirst: false,
            tLayerScale: true, tGelu: true,
            samplerate: 44100, segment: 7.8
        )
        
        guard WeightLoader.loadHTDemucs(&htModel, from: path) else {
            throw DemucsError.weightLoadFailed(path)
        }
        
        return HTDemucs(model: htModel)
    }
    
    /// Load model weights from a URL.
    public static func loadModel(from url: URL) throws -> any DemucsModel {
        return try loadModel(from: url.path)
    }

    // MARK: - Apply Model
    
    /// Apply a model to a given mixture, returning separated sources.
    ///
    /// This is the core separation function, matching the Python
    /// `demucs.apply.apply_model()` interface.
    ///
    /// - Parameters:
    ///   - model: A DemucsModel instance
    ///   - mix: Audio tensor of shape (batch, channels, length)
    ///   - shifts: Number of random shifts for equivariant stabilization.
    ///     Increases separation time but improves quality. 10 was used in
    ///     the original paper. Default is 1.
    ///   - split: If true, split audio into overlapping chunks. Reduces
    ///     memory usage for long tracks.
    ///   - overlap: Overlap ratio between chunks (0.0–1.0)
    ///   - transitionPower: Power for overlap-add weighting. Higher values
    ///     give sharper transitions between chunks.
    ///   - progress: Optional callback for progress updates
    ///   - segment: Override model segment length in seconds. nil uses
    ///     the model default.
    /// - Returns: MLXArray of shape (batch, sources, channels, length)
    public static func applyModel(
        _ model: any DemucsModel,
        mix: MLXArray,
        shifts: Int = 1,
        split: Bool = true,
        overlap: Float = 0.25,
        transitionPower: Float = 1.0,
        progress: ProgressCallback? = nil,
        segment: Float? = nil
    ) -> MLXArray {
        // We need the HTDemucs wrapper to access the internal model
        guard let htModel = (model as? HTDemucs) else {
            fatalError("Only HTDemucs models are currently supported")
        }
        
        let seg = segment ?? -1
        
        return _applyModelInternal(
            &htModel.model, mix: mix,
            shifts: shifts, split: split,
            overlap: overlap, transitionPower: transitionPower,
            segment: seg, progress: progress
        )
    }
}

// MARK: - Separator (Convenience)

/// High-level convenience class for audio source separation.
///
/// Wraps model loading and the normalize → separate → denormalize pipeline
/// into a simple interface.
///
/// ```swift
/// let separator = Separator()
/// try separator.loadModel(from: modelURL)
/// let result = try separator.separate(file: audioURL)
/// for stem in result.stems {
///     AudioIO.save(stem.url, audio: stem.audio)
/// }
/// ```
public class Separator {
    private var model: (any DemucsModel)?
    
    /// Source names the loaded model separates into.
    public var sources: [String] { model?.sources ?? defaultSources }
    
    /// Sample rate the model expects.
    public var sampleRate: Int { model?.sampleRate ?? 44100 }
    
    /// Whether a model has been loaded.
    public var isLoaded: Bool { model != nil }
    
    public init() {}
    
    /// Load model weights from a SafeTensors file URL.
    public func loadModel(from url: URL) throws {
        self.model = try Demucs.loadModel(from: url)
    }
    
    /// Load model weights from a SafeTensors file path.
    public func loadModel(from path: String) throws {
        self.model = try Demucs.loadModel(from: path)
    }
    
    /// Load a pretrained model by name.
    public func loadModel(name: String = defaultModel, repo: URL? = nil) throws {
        self.model = try Demucs.getModel(name, repo: repo)
    }
    
    /// Separate an audio file into stems.
    ///
    /// Handles the full pipeline: load audio → normalize → apply model →
    /// denormalize → extract stems → prevent clipping.
    ///
    /// - Parameters:
    ///   - url: Path to the audio file
    ///   - options: Separation options (shifts, overlap, etc.)
    ///   - progress: Optional callback for progress updates
    /// - Returns: A `SeparationResult` containing the separated stems
    public func separate(
        file url: URL,
        options: SeparationOptions = .default,
        progress: ProgressCallback? = nil
    ) throws -> SeparationResult {
        guard let audio = AudioIO.load(url) else {
            throw DemucsError.audioLoadFailed(url.path)
        }
        return try separate(audio: audio, options: options, progress: progress)
    }
    
    /// Separate raw audio into stems.
    ///
    /// - Parameters:
    ///   - audio: MLXArray of shape (channels, samples) at 44100 Hz
    ///   - options: Separation options
    ///   - progress: Optional callback for progress updates
    /// - Returns: A `SeparationResult` containing the separated stems
    public func separate(
        audio: MLXArray,
        options: SeparationOptions = .default,
        progress: ProgressCallback? = nil
    ) throws -> SeparationResult {
        guard let model = self.model else {
            throw DemucsError.modelNotLoaded
        }
        
        progress?(0.0, "Normalizing audio...")
        
        // Normalize — matches Python: ref = wav.mean(0); wav = (wav - ref.mean()) / ref.std()
        let ref = audio.mean(axis: 0)
        let refMean = ref.mean()
        let refStd = MLX.sqrt(ref.variance(ddof: 1))
        eval(refMean); eval(refStd)
        
        let audioNorm = (audio - refMean) / (refStd + MLXArray(Float(1e-8)))
        eval(audioNorm)
        
        let mix = expandedDimensions(audioNorm, axis: 0)
        
        progress?(0.05, "Separating stems...")
        
        // Run model
        var result = Demucs.applyModel(
            model, mix: mix,
            shifts: options.shifts,
            split: options.split,
            overlap: options.overlap,
            transitionPower: options.transitionPower,
            progress: { pct, msg in
                let mapped = 0.05 + pct * 0.85
                progress?(mapped, msg)
            },
            segment: options.segment > 0 ? options.segment : nil
        )
        eval(result)
        
        progress?(0.90, "Denormalizing...")
        
        // Denormalize
        result = result * (refStd + MLXArray(Float(1e-8))) + refMean
        eval(result)
        
        progress?(0.95, "Extracting stems...")
        
        // Extract stems with clip prevention
        var stems = [StemAudio]()
        for (idx, name) in model.sources.enumerated() {
            let source = sliceAxis(result, axis: 1, start: idx, end: idx + 1)
                .squeezed(axes: [0, 1])
            let clipped = preventClip(source, mode: options.clipMode)
            eval(clipped)
            stems.append(StemAudio(name: name, audio: clipped))
        }
        
        progress?(1.0, "Complete")
        
        return SeparationResult(stems: stems, sourceNames: model.sources)
    }
}

// MARK: - Types

/// A single separated audio stem.
public struct StemAudio {
    /// Stem name (e.g. "vocals", "drums", "bass", "other").
    public let name: String
    /// Audio data as MLXArray of shape (channels, samples).
    public let audio: MLXArray
    
    public init(name: String, audio: MLXArray) {
        self.name = name
        self.audio = audio
    }
}

/// Result of a separation operation.
public struct SeparationResult {
    /// The separated stems in source order.
    public let stems: [StemAudio]
    /// Source names in model order.
    public let sourceNames: [String]
    
    /// Get a stem by name. Returns nil if not found.
    public func stem(named name: String) -> StemAudio? {
        stems.first { $0.name == name }
    }
}

/// Options for the separation process.
public struct SeparationOptions {
    /// Number of random shifts for averaging (0 = no shifts).
    public var shifts: Int
    /// Whether to split audio into overlapping chunks.
    public var split: Bool
    /// Overlap ratio between chunks (0.0–1.0).
    public var overlap: Float
    /// Transition power for overlap-add weighting.
    public var transitionPower: Float
    /// Segment length in seconds (-1 = model default).
    public var segment: Float
    /// Clip prevention mode: "rescale", "clamp", or "none".
    public var clipMode: String
    
    public init(
        shifts: Int = 1,
        split: Bool = true,
        overlap: Float = 0.25,
        transitionPower: Float = 1.0,
        segment: Float = -1,
        clipMode: String = "rescale"
    ) {
        self.shifts = shifts
        self.split = split
        self.overlap = overlap
        self.transitionPower = transitionPower
        self.segment = segment
        self.clipMode = clipMode
    }
    
    /// Sensible defaults.
    public static let `default` = SeparationOptions()
}

/// Progress callback: (fraction 0..1, human-readable message).
public typealias ProgressCallback = (Float, String) -> Void

// MARK: - Errors

public enum DemucsError: LocalizedError {
    case fileNotFound(String)
    case weightLoadFailed(String)
    case audioLoadFailed(String)
    case modelNotLoaded
    case modelNotFound(String)
    
    public var errorDescription: String? {
        switch self {
        case .fileNotFound(let path):
            return "File not found: \(path)"
        case .weightLoadFailed(let path):
            return "Failed to load weights from: \(path)"
        case .audioLoadFailed(let path):
            return "Failed to load audio from: \(path)"
        case .modelNotLoaded:
            return "Model weights not loaded. Call loadModel() first."
        case .modelNotFound(let msg):
            return msg
        }
    }
}

// MARK: - Audio Utilities

/// Public audio utility functions matching the Python demucs.audio module.
extension AudioIO {
    
    /// Save audio with automatic clip prevention.
    ///
    /// Matches the Python `demucs.audio.save_audio()` function.
    ///
    /// - Parameters:
    ///   - url: Output file URL
    ///   - audio: Audio tensor of shape (channels, samples)
    ///   - sampleRate: Output sample rate
    ///   - clip: Clip prevention mode: "rescale", "clamp", or "none"
    ///   - bitsPerSample: Bits per sample for WAV output (16 or 24)
    ///   - asFloat: Save as 32-bit float WAV
    ///   - bitrate: Bitrate for compressed formats (kbps)
    ///   - codec: Codec for compressed formats ("aac", "flac", "alac")
    @discardableResult
    public static func saveAudio(
        _ url: URL,
        audio: MLXArray,
        sampleRate: Int = 44100,
        clip: String = "rescale",
        bitsPerSample: Int = 16,
        asFloat: Bool = false,
        bitrate: Int = 128,
        codec: String = ""
    ) -> Bool {
        let clipped = preventClip(audio, mode: clip)
        eval(clipped)
        return save(url.path, audio: clipped, sampleRate: sampleRate,
                    bitsPerSample: bitsPerSample, asFloat: asFloat,
                    bitrate: bitrate, codec: codec)
    }
}

// MARK: - Internal Bridge

/// Internal bridge to the applyModel implementation
fileprivate func _applyModelInternal(
    _ model: inout HTDemucsModel,
    mix: MLXArray,
    shifts: Int,
    split: Bool,
    overlap: Float,
    transitionPower: Float,
    segment: Float,
    progress: ProgressCallback?
) -> MLXArray {
    return applyModel(
        &model, mix: mix,
        shifts: shifts, split: split,
        overlap: overlap, transitionPower: transitionPower,
        segment: segment, progress: progress
    )
}
