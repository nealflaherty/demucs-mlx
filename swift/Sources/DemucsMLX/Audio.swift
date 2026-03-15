// Audio.swift - Audio file I/O using AVFoundation
// 1:1 port of cpp/demucs/audio.cpp

import MLX
import Foundation
import AVFoundation

public enum AudioIO {

    /// Load an audio file, convert to stereo float32 at the target sample rate.
    /// Returns MLXArray of shape (2, samples) or nil on error.
    public static func load(_ path: String, targetSampleRate: Int = 44100) -> MLXArray? {
        let url = URL(fileURLWithPath: path)
        guard FileManager.default.fileExists(atPath: path) else {
            print("Audio: file not found: \(path)")
            return nil
        }

        // Open with ExtAudioFile for format conversion
        var audioFile: ExtAudioFileRef?
        var status = ExtAudioFileOpenURL(url as CFURL, &audioFile)
        guard status == noErr, let file = audioFile else {
            print("Audio: failed to open \(path) (error \(status))")
            return nil
        }
        defer { ExtAudioFileDispose(file) }

        // Get source format
        var fileFormat = AudioStreamBasicDescription()
        var propSize = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
        status = ExtAudioFileGetProperty(file,
            kExtAudioFileProperty_FileDataFormat, &propSize, &fileFormat)
        guard status == noErr else {
            print("Audio: failed to get file format (error \(status))")
            return nil
        }

        let srcChannels = Int(fileFormat.mChannelsPerFrame)
        let srcSampleRate = fileFormat.mSampleRate

        // Set client format: float32, stereo, target sample rate
        var clientFormat = AudioStreamBasicDescription()
        clientFormat.mSampleRate = Float64(targetSampleRate)
        clientFormat.mFormatID = kAudioFormatLinearPCM
        clientFormat.mFormatFlags = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked
        clientFormat.mChannelsPerFrame = 2
        clientFormat.mBitsPerChannel = 32
        clientFormat.mBytesPerFrame = 2 * UInt32(MemoryLayout<Float>.size)
        clientFormat.mFramesPerPacket = 1
        clientFormat.mBytesPerPacket = clientFormat.mBytesPerFrame

        status = ExtAudioFileSetProperty(file,
            kExtAudioFileProperty_ClientDataFormat,
            UInt32(MemoryLayout<AudioStreamBasicDescription>.size), &clientFormat)
        guard status == noErr else {
            print("Audio: failed to set client format (error \(status))")
            return nil
        }

        // Get total frame count
        var fileLengthFrames: Int64 = 0
        propSize = UInt32(MemoryLayout<Int64>.size)
        status = ExtAudioFileGetProperty(file,
            kExtAudioFileProperty_FileLengthFrames, &propSize, &fileLengthFrames)
        guard status == noErr else {
            print("Audio: failed to get file length (error \(status))")
            return nil
        }

        let estimatedFrames = Int(ceil(Double(fileLengthFrames) *
            Double(targetSampleRate) / srcSampleRate))

        // Read all audio in chunks
        let chunkFrames: UInt32 = 8192
        var allSamples = [Float]()
        allSamples.reserveCapacity(estimatedFrames * 2)
        var buffer = [Float](repeating: 0, count: Int(chunkFrames) * 2)

        while true {
            var framesToRead = chunkFrames
            let readOk = buffer.withUnsafeMutableBytes { rawBuf -> Bool in
                var bufList = AudioBufferList(
                    mNumberBuffers: 1,
                    mBuffers: AudioBuffer(
                        mNumberChannels: 2,
                        mDataByteSize: chunkFrames * 2 * UInt32(MemoryLayout<Float>.size),
                        mData: rawBuf.baseAddress
                    )
                )
                let st = ExtAudioFileRead(file, &framesToRead, &bufList)
                return st == noErr
            }
            guard readOk else {
                print("Audio: read error")
                return nil
            }
            if framesToRead == 0 { break }
            allSamples.append(contentsOf: buffer[0..<Int(framesToRead) * 2])
        }

        let totalFrames = allSamples.count / 2
        guard totalFrames > 0 else {
            print("Audio: no audio data in \(path)")
            return nil
        }

        print("Loaded audio: \(path)")
        print("  Sample rate: \(targetSampleRate) Hz (source: \(srcSampleRate) Hz)")
        print("  Channels: 2 (source: \(srcChannels))")
        print("  Duration: \(Float(totalFrames) / Float(targetSampleRate)) seconds")

        // Convert interleaved (L R L R ...) to planar (LL...RR...)
        var planar = [Float](repeating: 0, count: totalFrames * 2)
        for i in 0..<totalFrames {
            planar[i] = allSamples[2 * i]
            planar[totalFrames + i] = allSamples[2 * i + 1]
        }

        return MLXArray(planar, [2, totalFrames])
    }

    // MARK: - Save

    /// Save audio to file. Supports .wav, .m4a (AAC), .flac, .alac.
    /// audio shape: (2, samples) or (1, 2, samples).
    public static func save(_ path: String, audio: MLXArray,
                            sampleRate: Int = 44100, bitsPerSample: Int = 16,
                            asFloat: Bool = false, bitrate: Int = 128,
                            codec: String = "") -> Bool {
        var audio2d = audio
        if audio.ndim == 3 {
            audio2d = audio.squeezed(axis: 0)
        }
        guard audio2d.ndim == 2 else {
            print("Audio: expected 2D (channels, samples), got \(audio2d.ndim)D")
            return false
        }

        let ext = (path as NSString).pathExtension.lowercased()

        if ext == "wav" {
            return saveWav(path, audio: audio2d, sampleRate: sampleRate,
                          bitsPerSample: bitsPerSample, asFloat: asFloat)
        }

        if ext == "m4a" || ext == "flac" {
            return saveCompressed(path, audio: audio2d, sampleRate: sampleRate,
                                  bitsPerSample: bitsPerSample, bitrate: bitrate,
                                  codec: codec.isEmpty ? (ext == "flac" ? "flac" : "aac") : codec)
        }

        print("Audio: unsupported format .\(ext)")
        return false
    }

    // MARK: - WAV save

    private static func saveWav(_ path: String, audio: MLXArray,
                                sampleRate: Int, bitsPerSample: Int,
                                asFloat: Bool) -> Bool {
        let numChannels = audio.shape[0]
        let samplesPerChannel = audio.shape[1]
        eval(audio)

        let flatData = audio.reshaped(-1).asArray(Float.self)

        // Convert planar to interleaved int16
        var interleaved: [Int16]
        interleaved = [Int16](repeating: 0, count: numChannels * samplesPerChannel)
        for i in 0..<samplesPerChannel {
            for ch in 0..<numChannels {
                let sample = flatData[ch * samplesPerChannel + i]
                let clamped = max(Float(-1.0), min(Float(1.0), sample))
                interleaved[i * numChannels + ch] = Int16(clamped * 32767.0)
            }
        }

        // Build WAV file manually for reliability
        let dataSize = interleaved.count * MemoryLayout<Int16>.size
        let fileSize = 36 + dataSize

        var wav = Data()
        wav.append(contentsOf: "RIFF".utf8)
        wav.append(withUnsafeBytes(of: UInt32(fileSize).littleEndian) { Data($0) })
        wav.append(contentsOf: "WAVE".utf8)

        // fmt chunk
        wav.append(contentsOf: "fmt ".utf8)
        wav.append(withUnsafeBytes(of: UInt32(16).littleEndian) { Data($0) })
        wav.append(withUnsafeBytes(of: UInt16(1).littleEndian) { Data($0) })  // PCM
        wav.append(withUnsafeBytes(of: UInt16(numChannels).littleEndian) { Data($0) })
        wav.append(withUnsafeBytes(of: UInt32(sampleRate).littleEndian) { Data($0) })
        let byteRate = UInt32(sampleRate * numChannels * 2)
        wav.append(withUnsafeBytes(of: byteRate.littleEndian) { Data($0) })
        let blockAlign = UInt16(numChannels * 2)
        wav.append(withUnsafeBytes(of: blockAlign.littleEndian) { Data($0) })
        wav.append(withUnsafeBytes(of: UInt16(16).littleEndian) { Data($0) })  // bits per sample

        // data chunk
        wav.append(contentsOf: "data".utf8)
        wav.append(withUnsafeBytes(of: UInt32(dataSize).littleEndian) { Data($0) })
        interleaved.withUnsafeBytes { wav.append(contentsOf: $0) }

        let url = URL(fileURLWithPath: path)
        do {
            try wav.write(to: url)
            print("Saved audio: \(path)")
            return true
        } catch {
            print("Audio: failed to write \(path): \(error)")
            return false
        }
    }

    // MARK: - Compressed save (AAC, FLAC, ALAC) via ExtAudioFile

    private static func saveCompressed(_ path: String, audio: MLXArray,
                                       sampleRate: Int, bitsPerSample: Int,
                                       bitrate: Int, codec: String) -> Bool {
        let numChannels = audio.shape[0]
        let samplesPerChannel = audio.shape[1]
        eval(audio)

        let flatData = audio.reshaped(-1).asArray(Float.self)

        // Convert planar to interleaved
        var interleaved = [Float](repeating: 0, count: numChannels * samplesPerChannel)
        for i in 0..<samplesPerChannel {
            for ch in 0..<numChannels {
                interleaved[i * numChannels + ch] = flatData[ch * samplesPerChannel + i]
            }
        }

        // Output format
        var outputFormat = AudioStreamBasicDescription()
        outputFormat.mSampleRate = Float64(sampleRate)
        outputFormat.mChannelsPerFrame = UInt32(numChannels)

        var fileType: AudioFileTypeID
        switch codec {
        case "aac":
            fileType = kAudioFileM4AType
            outputFormat.mFormatID = kAudioFormatMPEG4AAC
        case "flac":
            fileType = kAudioFileFLACType
            outputFormat.mFormatID = kAudioFormatFLAC
            outputFormat.mBitsPerChannel = UInt32(bitsPerSample <= 16 ? 16 : 24)
        case "alac":
            fileType = kAudioFileM4AType
            outputFormat.mFormatID = kAudioFormatAppleLossless
        default:
            print("Audio: unsupported codec \(codec)")
            return false
        }

        let url = URL(fileURLWithPath: path)
        var outFile: ExtAudioFileRef?
        var status = ExtAudioFileCreateWithURL(url as CFURL, fileType, &outputFormat,
                                                nil, AudioFileFlags.eraseFile.rawValue, &outFile)
        guard status == noErr, let file = outFile else {
            print("Audio: failed to create \(path) (error \(status))")
            return false
        }
        defer { ExtAudioFileDispose(file) }

        // Set AAC bitrate
        if outputFormat.mFormatID == kAudioFormatMPEG4AAC {
            var converter: AudioConverterRef?
            var converterSize = UInt32(MemoryLayout<AudioConverterRef>.size)
            status = ExtAudioFileGetProperty(file,
                kExtAudioFileProperty_AudioConverter, &converterSize, &converter)
            if status == noErr, let conv = converter {
                var aacBitrate = UInt32(bitrate * 1000)
                AudioConverterSetProperty(conv, kAudioConverterEncodeBitRate,
                    UInt32(MemoryLayout<UInt32>.size), &aacBitrate)
            }
        }

        // Client format: float32 interleaved
        var clientFormat = AudioStreamBasicDescription()
        clientFormat.mSampleRate = Float64(sampleRate)
        clientFormat.mFormatID = kAudioFormatLinearPCM
        clientFormat.mFormatFlags = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked
        clientFormat.mChannelsPerFrame = UInt32(numChannels)
        clientFormat.mBitsPerChannel = 32
        clientFormat.mBytesPerFrame = UInt32(numChannels) * UInt32(MemoryLayout<Float>.size)
        clientFormat.mFramesPerPacket = 1
        clientFormat.mBytesPerPacket = clientFormat.mBytesPerFrame

        status = ExtAudioFileSetProperty(file,
            kExtAudioFileProperty_ClientDataFormat,
            UInt32(MemoryLayout<AudioStreamBasicDescription>.size), &clientFormat)
        guard status == noErr else {
            print("Audio: failed to set client format (error \(status))")
            return false
        }

        // Write in chunks
        let chunkFrames: UInt32 = 8192
        var framesRemaining = UInt32(samplesPerChannel)
        var offset: UInt32 = 0

        while framesRemaining > 0 {
            let framesToWrite = min(chunkFrames, framesRemaining)
            let byteOffset = Int(offset) * numChannels
            let writeOk = interleaved.withUnsafeMutableBytes { rawBuf -> Bool in
                let ptr = rawBuf.baseAddress! + byteOffset * MemoryLayout<Float>.size
                var bufList = AudioBufferList(
                    mNumberBuffers: 1,
                    mBuffers: AudioBuffer(
                        mNumberChannels: UInt32(numChannels),
                        mDataByteSize: framesToWrite * UInt32(numChannels) * UInt32(MemoryLayout<Float>.size),
                        mData: ptr
                    )
                )
                return ExtAudioFileWrite(file, framesToWrite, &bufList) == noErr
            }
            guard writeOk else {
                print("Audio: write error")
                return false
            }
            offset += framesToWrite
            framesRemaining -= framesToWrite
        }

        print("Saved audio: \(path)")
        return true
    }
}
