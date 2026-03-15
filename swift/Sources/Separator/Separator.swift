// main.swift - CLI for separating audio tracks using HTDemucs
// 1:1 port of cpp/demucs/separate.cpp

import ArgumentParser
import DemucsMLX
import Foundation
import MLX

@main
struct Separator: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "demucs-separate",
        abstract: "Separate audio into stems using HTDemucs"
    )

    @Argument(help: "Audio files to separate")
    var tracks: [String]

    @Option(name: [.short, .customLong("out")],
            help: "Output directory")
    var outDir: String = "separated/htdemucs"

    @Option(help: "Model weights path (.safetensors)")
    var model: String = "models/htdemucs.safetensors"

    @Option(help: "Output filename pattern ({track}, {trackext}, {stem}, {ext})")
    var filename: String = "{track}/{stem}.{ext}"

    @Option(help: "Number of random shifts for averaging")
    var shifts: Int = 1

    @Option(help: "Overlap between chunks (0.0-1.0)")
    var overlap: Float = 0.25

    @Flag(name: .customLong("no-split"),
          help: "Don't split audio into chunks")
    var noSplit: Bool = false

    @Option(help: "Segment length in seconds (-1 = model default)")
    var segment: Float = -1.0

    @Option(name: .customLong("clip-mode"),
            help: "Clipping prevention: rescale, clamp, none")
    var clipMode: String = "rescale"

    @Option(name: .customLong("two-stems"),
            help: "Only separate into STEM and no_STEM")
    var twoStems: String?

    @Option(name: .customLong("other-method"),
            help: "How to compute complement: add, minus, none")
    var otherMethod: String = "add"

    // Output format flags
    @Flag(help: "Output as WAV (default)")
    var wav: Bool = false

    @Flag(help: "Output as AAC in M4A container")
    var m4a: Bool = false

    @Flag(help: "Output as FLAC lossless")
    var flac: Bool = false

    @Flag(help: "Output as ALAC lossless in M4A")
    var alac: Bool = false

    // WAV options
    @Flag(help: "Save WAV as 24-bit integer PCM")
    var int24: Bool = false

    @Flag(help: "Save WAV as 32-bit float PCM")
    var float32: Bool = false

    // AAC options
    @Option(help: "AAC bitrate in kbps")
    var bitrate: Int = 256

    // MARK: - Run

    func run() throws {
        let sources = ["drums", "bass", "other", "vocals"]

        // Validate two-stems
        if let stem = twoStems, !sources.contains(stem) {
            print("Error: stem \"\(stem)\" not in model. Must be one of: \(sources.joined(separator: ", "))")
            throw ExitCode.failure
        }

        // Determine output format
        let format: String
        let codec: String
        if m4a { format = "m4a"; codec = "aac" }
        else if flac { format = "flac"; codec = "flac" }
        else if alac { format = "m4a"; codec = "alac" }
        else { format = "wav"; codec = "" }

        let bitsPerSample = int24 ? 24 : 16

        // Create model
        var htModel = HTDemucsModel(
            sources: sources, audioChannels: 2, channels: 48,
            channelsTime: -1, growth: 2.0, nfft: 4096, cac: true,
            depth: 4, rewrite: true, freqEmb: 0.2, embScale: 10.0,
            embSmooth: true, kernelSize: 8, timeStride: 2, stride: 4,
            context: 1, contextEnc: 0, normStarts: 4, normGroups: 4,
            dconvMode: 3, dconvComp: 8, dconvInit: 1e-3,
            bottomChannels: 512, tLayers: 5, tHeads: 8,
            tHiddenScale: 4.0, tNormIn: true, tNormOut: true,
            tCrossFirst: false, tLayerScale: true, tGelu: true,
            samplerate: 44100, segment: 7.8
        )

        // Load weights
        guard FileManager.default.fileExists(atPath: model) else {
            print("Error: model file not found: \(model)")
            print("\nDownload the model first or specify a path with --model <path>")
            throw ExitCode.failure
        }

        print("Loading model weights from \(model)...")
        guard WeightLoader.loadHTDemucs(&htModel, from: model) else {
            print("Failed to load weights!")
            throw ExitCode.failure
        }
        print("Model loaded.")

        // Create output directory
        try FileManager.default.createDirectory(atPath: outDir,
            withIntermediateDirectories: true)
        print("Separated tracks will be stored in \(outDir)")

        let fileExt = (codec == "alac") ? "m4a" : format

        for trackPath in tracks {
            guard FileManager.default.fileExists(atPath: trackPath) else {
                print("File \(trackPath) does not exist.")
                continue
            }

            print("Separating track \(trackPath)")

            guard let audio = AudioIO.load(trackPath, targetSampleRate: 44100) else {
                print("Failed to load audio: \(trackPath)")
                continue
            }

            // Normalize: ref = wav.mean(0); wav = (wav - ref.mean()) / (ref.std() + 1e-8)
            let ref = audio.mean(axis: 0)
            let refMean = ref.mean()
            let refStd = MLX.sqrt(ref.variance(ddof: 1))
            eval(refMean); eval(refStd)

            var audioNorm = (audio - refMean) / (refStd + MLXArray(Float(1e-8)))
            eval(audioNorm)

            let mix = expandedDimensions(audioNorm, axis: 0)

            print("Running separation...")
            var result = applyModel(&htModel, mix: mix, shifts: shifts,
                                    split: !noSplit, overlap: overlap,
                                    transitionPower: 1.0, segment: segment)
            eval(result)

            // Denormalize
            result = result * (refStd + MLXArray(Float(1e-8))) + refMean
            eval(result)

            // Also denormalize original for minus method
            audioNorm = audioNorm * (refStd + MLXArray(Float(1e-8))) + refMean

            // Parse track name
            let trackURL = URL(fileURLWithPath: trackPath)
            let trackName = trackURL.deletingPathExtension().lastPathComponent
            let trackExtStr = String(trackURL.pathExtension)

            let outLength = result.shape[result.ndim - 1]

            if let stem = twoStems {
                // Two-stems mode
                let stemIdx = sources.firstIndex(of: stem)!
                let selected = sliceAxis(result, axis: 1, start: stemIdx, end: stemIdx + 1)
                    .squeezed(axes: [0, 1])

                let stemPath = outDir + "/" + formatFilename(filename,
                    track: trackName, trackext: trackExtStr, stem: stem, ext: fileExt)
                createParentDirs(stemPath)
                saveStem(selected, path: stemPath, bitsPerSample: bitsPerSample,
                         asFloat: float32, bitrate: bitrate, codec: codec)

                if otherMethod == "minus" {
                    let complement = audioNorm - selected
                    let compPath = outDir + "/" + formatFilename(filename,
                        track: trackName, trackext: trackExtStr, stem: "no_\(stem)", ext: fileExt)
                    createParentDirs(compPath)
                    saveStem(complement, path: compPath, bitsPerSample: bitsPerSample,
                             asFloat: float32, bitrate: bitrate, codec: codec)
                } else if otherMethod == "add" {
                    var complement = MLXArray.zeros([2, outLength])
                    for s in 0..<sources.count where s != stemIdx {
                        let other = sliceAxis(result, axis: 1, start: s, end: s + 1)
                            .squeezed(axes: [0, 1])
                        complement = complement + other
                    }
                    eval(complement)
                    let compPath = outDir + "/" + formatFilename(filename,
                        track: trackName, trackext: trackExtStr, stem: "no_\(stem)", ext: fileExt)
                    createParentDirs(compPath)
                    saveStem(complement, path: compPath, bitsPerSample: bitsPerSample,
                             asFloat: float32, bitrate: bitrate, codec: codec)
                }
            } else {
                // Save all stems
                for (s, name) in sources.enumerated() {
                    let source = sliceAxis(result, axis: 1, start: s, end: s + 1)
                        .squeezed(axes: [0, 1])
                    
                    let stemPath = outDir + "/" + formatFilename(filename,
                        track: trackName, trackext: trackExtStr, stem: name, ext: fileExt)
                    createParentDirs(stemPath)
                    saveStem(source, path: stemPath, bitsPerSample: bitsPerSample,
                             asFloat: float32, bitrate: bitrate, codec: codec)
                }
            }

            print("Saved stems to \(outDir)/\(trackName)/")
        }
    }

    // MARK: - Helpers

    private func formatFilename(_ pattern: String, track: String,
                                trackext: String, stem: String, ext: String) -> String {
        pattern
            .replacingOccurrences(of: "{track}", with: track)
            .replacingOccurrences(of: "{trackext}", with: trackext)
            .replacingOccurrences(of: "{stem}", with: stem)
            .replacingOccurrences(of: "{ext}", with: ext)
    }

    private func createParentDirs(_ path: String) {
        let url = URL(fileURLWithPath: path).deletingLastPathComponent()
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    }

    private func saveStem(_ source: MLXArray, path: String,
                          bitsPerSample: Int, asFloat: Bool,
                          bitrate: Int, codec: String) {
        let clipped = preventClip(source, mode: clipMode)
        eval(clipped)
        _ = AudioIO.save(path, audio: clipped, sampleRate: 44100,
                         bitsPerSample: bitsPerSample, asFloat: asFloat,
                         bitrate: bitrate, codec: codec)
    }
}
