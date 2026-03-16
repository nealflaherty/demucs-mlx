// Apply.swift - Chunked inference engine with overlap-add
// 1:1 port of cpp/demucs/apply.cpp

import MLX
import Foundation

// MARK: - TensorChunk

/// A view into a tensor along the last dimension.
struct TensorChunk {
    let tensor: MLXArray
    let offset: Int
    let length: Int

    init(_ tensor: MLXArray, offset: Int = 0, length: Int = -1) {
        self.tensor = tensor
        self.offset = offset
        let totalLength = tensor.shape[tensor.ndim - 1]
        if length < 0 {
            self.length = totalLength - offset
        } else {
            self.length = min(totalLength - offset, length)
        }
    }

    /// Pad/extract a region of target_length centered on this chunk.
    func padded(_ targetLength: Int) -> MLXArray {
        let delta = targetLength - length
        let totalLength = tensor.shape[tensor.ndim - 1]

        let start = offset - delta / 2
        let end = start + targetLength
        let correctStart = max(0, start)
        let correctEnd = min(totalLength, end)
        let padLeft = correctStart - start
        let padRight = end - correctEnd

        var out = sliceAxis(tensor, axis: tensor.ndim - 1,
                            start: correctStart, end: correctEnd)

        if padLeft > 0 || padRight > 0 {
            var padWidths = [(Int, Int)](repeating: (0, 0), count: tensor.ndim)
            padWidths[tensor.ndim - 1] = (padLeft, padRight)
            out = MLX.padded(out, widths: padWidths.map { .init($0) })
        }
        return out
    }
}

// MARK: - Utilities

/// Trim the last dimension to `reference` length, centered.
func centerTrim(_ tensor: MLXArray, _ reference: Int) -> MLXArray {
    let delta = tensor.shape[tensor.ndim - 1] - reference
    if delta <= 0 { return tensor }
    let trimLeft = delta / 2
    let trimRight = delta - trimLeft
    return sliceAxis(tensor, axis: tensor.ndim - 1,
                     start: trimLeft,
                     end: tensor.shape[tensor.ndim - 1] - trimRight)
}

/// Prevent clipping in output audio.
func preventClip(_ wav: MLXArray, mode: String = "rescale") -> MLXArray {
    switch mode {
    case "none":
        return wav
    case "rescale":
        let peak = MLX.abs(wav).max()
        eval(peak)
        let peakVal = peak.item(Float.self)
        let denom = max(1.01 * peakVal, 1.0)
        return wav / MLXArray(denom)
    case "clamp":
        return clip(wav, min: MLXArray(Float(-0.99)), max: MLXArray(Float(0.99)))
    default:
        print("Apply: invalid clip mode '\(mode)', using rescale")
        return preventClip(wav, mode: "rescale")
    }
}

// MARK: - Apply model

/// Run the model on a chunk, with optional shifts and splitting.
func applyModelChunk(
    _ model: inout HTDemucsModel,
    chunk: TensorChunk,
    shifts: Int = 0,
    split: Bool = true,
    overlap: Float = 0.25,
    transitionPower: Float = 1.0,
    segment: Float = -1,
    progress: ((Float, String) -> Void)? = nil
) -> MLXArray {
    let length = chunk.length
    let samplerate = 44100

    // Random shift averaging
    if shifts > 0 {
        let maxShift = Int(0.5 * Float(samplerate))
        let paddedMix = chunk.padded(length + 2 * maxShift)

        var rng = RandomNumberGenerator_LCG(seed: 42)
        var out = MLXArray.zeros([1, 4, 2, length])

        for _ in 0..<shifts {
            let offset = Int.random(in: 0...maxShift, using: &rng)
            let shifted = TensorChunk(paddedMix, offset: offset,
                                      length: length + maxShift - offset)
            var res = applyModelChunk(&model, chunk: shifted, shifts: 0,
                                      split: split, overlap: overlap,
                                      transitionPower: transitionPower,
                                      segment: segment, progress: progress)
            let trimStart = maxShift - offset
            res = sliceAxis(res, axis: -1, start: trimStart, end: trimStart + length)
            out = out + res
            eval(out)
        }
        return out / MLXArray(Float(shifts))
    }

    // Overlap-add chunked inference
    if split {
        let seg = segment < 0 ? Float(7.8) : segment
        let segmentLength = Int(Float(samplerate) * seg)
        let stride = Int((1.0 - overlap) * Float(segmentLength))

        var out = MLXArray.zeros([1, 4, 2, length])

        // Triangle weight
        let half = segmentLength / 2
        var weightData = [Float](repeating: 0, count: segmentLength)
        for i in 0..<half { weightData[i] = Float(i + 1) }
        for i in half..<segmentLength { weightData[i] = Float(segmentLength - i) }
        let maxW = weightData.max() ?? 1.0
        for i in 0..<segmentLength {
            weightData[i] = pow(weightData[i] / maxW, transitionPower)
        }

        var sumWeightData = [Float](repeating: 0, count: length)
        let mixTensor = chunk.padded(length)

        var offset = 0
        while offset < length {
            let subChunk = TensorChunk(mixTensor, offset: offset, length: segmentLength)

            print("  Processing chunk at \(offset)/\(length) (\(100 * offset / length)%)")
            let pct = Float(offset) / Float(length)
            progress?(pct, "Processing chunk \(offset)/\(length) (\(Int(pct * 100))%)")

            let chunkOut = applyModelChunk(&model, chunk: subChunk, shifts: 0,
                                           split: false, overlap: overlap,
                                           transitionPower: transitionPower,
                                           segment: segment, progress: progress)
            eval(chunkOut)

            let chunkLength = chunkOut.shape[chunkOut.ndim - 1]
            let chunkWeight = Array(weightData[0..<chunkLength])
            let weightArr = MLXArray(chunkWeight, [chunkLength]).reshaped(1, 1, 1, chunkLength)

            let weighted = weightArr * chunkOut
            eval(weighted)

            let padLeft = offset
            let padRight = max(0, length - offset - chunkLength)
            var pw = [(Int, Int)](repeating: (0, 0), count: 4)
            pw[3] = (padLeft, padRight)
            let paddedWeighted = MLX.padded(weighted, widths: pw.map { .init($0) })
            out = out + paddedWeighted
            eval(out)

            for i in 0..<chunkLength where (offset + i) < length {
                sumWeightData[offset + i] += chunkWeight[i]
            }
            offset += stride
        }

        let sumWeight = MLXArray(sumWeightData, [length]).reshaped(1, 1, 1, length)
        out = out / sumWeight
        eval(out)
        return out
    }

    // Direct model call (no split, no shifts)
    let seg = segment < 0 ? Float(7.8) : segment
    let validLength = Int(seg * Float(samplerate))
    let paddedMix = chunk.padded(validLength)
    let out = model.forward(paddedMix)
    eval(out)
    return centerTrim(out, length)
}

/// Top-level entry point.
func applyModel(
    _ model: inout HTDemucsModel,
    mix: MLXArray,
    shifts: Int = 0,
    split: Bool = true,
    overlap: Float = 0.25,
    transitionPower: Float = 1.0,
    segment: Float = -1,
    progress: ((Float, String) -> Void)? = nil
) -> MLXArray {
    let chunk = TensorChunk(mix)
    return applyModelChunk(&model, chunk: chunk, shifts: shifts,
                           split: split, overlap: overlap,
                           transitionPower: transitionPower,
                           segment: segment, progress: progress)
}

// MARK: - Simple LCG RNG for reproducibility

struct RandomNumberGenerator_LCG: RandomNumberGenerator {
    var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return state
    }
}
