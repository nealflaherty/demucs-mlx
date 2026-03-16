// main.swift - CLI for separating audio tracks using HTDemucs
// 1:1 port of cpp/demucs/separate.cpp

import ArgumentParser
import DemucsMLX
import Foundation
import MLX

@main
struct SeparatorCLI: ParsableCommand {
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
        // Validate two-stems
        if let stem = twoStems, !defaultSources.contains(stem) {
            print("Error: stem \"\(stem)\" not in model. Must be one of: \(defaultSources.joined(separator: ", "))")
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
        let fileExt = (codec == "alac") ? "m4a" : format

        // Create separator and load model
        let separator = Separator()

        guard FileManager.default.fileExists(atPath: model) else {
            print("Error: model file not found: \(model)")
            print("\nDownload the model first or specify a path with --model <path>")
            throw ExitCode.failure
        }

        print("Loading model weights from \(model)...")
        try separator.loadModel(from: model)
        print("Model loaded.")

        // Create output directory
        try FileManager.default.createDirectory(atPath: outDir,
            withIntermediateDirectories: true)
        print("Separated tracks will be stored in \(outDir)")

        let options = SeparationOptions(
            shifts: shifts,
            split: !noSplit,
            overlap: overlap,
            transitionPower: 1.0,
            segment: segment,
            clipMode: clipMode
        )

        for trackPath in tracks {
            guard FileManager.default.fileExists(atPath: trackPath) else {
                print("File \(trackPath) does not exist.")
                continue
            }

            print("Separating track \(trackPath)")

            let result = try separator.separate(
                file: URL(fileURLWithPath: trackPath),
                options: options
            )

            // Parse track name
            let trackURL = URL(fileURLWithPath: trackPath)
            let trackName = trackURL.deletingPathExtension().lastPathComponent
            let trackExtStr = String(trackURL.pathExtension)

            if let stem = twoStems {
                // Two-stems mode
                guard let selected = result.stem(named: stem) else {
                    print("Error: stem \(stem) not found in result")
                    continue
                }

                let stemPath = outDir + "/" + formatFilename(filename,
                    track: trackName, trackext: trackExtStr, stem: stem, ext: fileExt)
                createParentDirs(stemPath)
                _ = AudioIO.save(stemPath, audio: selected.audio,
                                 sampleRate: 44100, bitsPerSample: bitsPerSample,
                                 asFloat: float32, bitrate: bitrate, codec: codec)

                if otherMethod != "none" {
                    // Compute complement by summing all other stems
                    var complement = MLXArray.zeros([2, selected.audio.shape[1]])
                    if otherMethod == "minus" {
                        // Load original audio and subtract
                        if let audio = AudioIO.load(trackPath, targetSampleRate: 44100) {
                            complement = audio - selected.audio
                        }
                    } else {
                        // "add" — sum all non-selected stems
                        for s in result.stems where s.name != stem {
                            complement = complement + s.audio
                        }
                    }
                    eval(complement)
                    let compPath = outDir + "/" + formatFilename(filename,
                        track: trackName, trackext: trackExtStr, stem: "no_\(stem)", ext: fileExt)
                    createParentDirs(compPath)
                    _ = AudioIO.save(compPath, audio: complement,
                                     sampleRate: 44100, bitsPerSample: bitsPerSample,
                                     asFloat: float32, bitrate: bitrate, codec: codec)
                }
            } else {
                // Save all stems
                for stem in result.stems {
                    let stemPath = outDir + "/" + formatFilename(filename,
                        track: trackName, trackext: trackExtStr, stem: stem.name, ext: fileExt)
                    createParentDirs(stemPath)
                    _ = AudioIO.save(stemPath, audio: stem.audio,
                                     sampleRate: 44100, bitsPerSample: bitsPerSample,
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
}
