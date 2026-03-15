// Spec.swift — STFT and inverse STFT
// 1:1 port of cpp/demucs/spec.cpp
//
// Uses Accelerate vDSP for FFT, matching the C++ implementation exactly.

import MLX
import Accelerate
import Foundation

// MARK: - Window normalization helper

/// Compute sum of squared windows at each output position for overlap-add normalization.
private func computeWindowNormalization(
    _ window: [Float], nFFT: Int, hopLength: Int, numFrames: Int
) -> [Float] {
    let outputLength = (numFrames - 1) * hopLength + nFFT
    var norm = [Float](repeating: 0.0, count: outputLength)
    for frameIdx in 0..<numFrames {
        let startPos = frameIdx * hopLength
        for i in 0..<nFFT where (startPos + i) < outputLength {
            norm[startPos + i] += window[i] * window[i]
        }
    }
    return norm
}

/// Create periodic Hann window of length N: `0.5 * (1 - cos(2π * i / N))`
private func periodicHannWindow(_ n: Int) -> [Float] {
    let floatN = Float(n)
    return (0..<n).map { i in
        0.5 * (1.0 - cos(2.0 * .pi * Float(i) / floatN))
    }
}

// MARK: - spectro (STFT)

/// Short-Time Fourier Transform matching the C++ `spectro()`.
public func spectro(_ x: MLXArray, nFFT: Int, hopLength: Int = -1, pad: Int = 0) -> MLXArray {
    let hopLen = hopLength < 0 ? nFFT / 4 : hopLength
    let shape = x.shape
    let length = shape.last!

    var batchSize = 1
    for i in 0..<(shape.count - 1) { batchSize *= shape[i] }

    let x2d = x.reshaped(batchSize, length)
    eval(x2d)
    let xData = x2d.asArray(Float.self)

    // Reflect padding of nFFT/2 on each side
    let padAmount = nFFT / 2
    let paddedLength = length + 2 * padAmount
    var paddedData = [Float](repeating: 0.0, count: batchSize * paddedLength)

    for b in 0..<batchSize {
        let srcOffset = b * length
        let dstOffset = b * paddedLength

        // Copy original signal to middle
        for i in 0..<length {
            paddedData[dstOffset + padAmount + i] = xData[srcOffset + i]
        }
        // Left reflect: x[pad], x[pad-1], ..., x[1]
        for i in 0..<padAmount {
            paddedData[dstOffset + i] = xData[srcOffset + padAmount - i]
        }
        // Right reflect: x[-2], x[-3], ..., x[-(pad+1)]
        for i in 0..<padAmount {
            paddedData[dstOffset + padAmount + length + i] = xData[srcOffset + length - 2 - i]
        }
    }

    let numFrames = 1 + (paddedLength - nFFT) / hopLen
    let freqBins = nFFT / 2 + 1
    let window = periodicHannWindow(nFFT)

    // Setup FFT
    let log2n = vDSP_Length(log2(Double(nFFT)))
    guard let fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(FFT_RADIX2)) else {
        fatalError("Failed to create FFT setup")
    }
    defer { vDSP_destroy_fftsetup(fftSetup) }

    var outputReal = [Float](repeating: 0.0, count: batchSize * freqBins * numFrames)
    var outputImag = [Float](repeating: 0.0, count: batchSize * freqBins * numFrames)

    var frameBuffer = [Float](repeating: 0.0, count: nFFT)
    var realPart = [Float](repeating: 0.0, count: nFFT / 2 + 1)
    var imagPart = [Float](repeating: 0.0, count: nFFT / 2 + 1)
    let normalization: Float = 1.0 / sqrt(Float(nFFT))

    for b in 0..<batchSize {
        let signalOffset = b * paddedLength

        for frameIdx in 0..<numFrames {
            let startPos = frameIdx * hopLen

            // Extract and window the frame
            for i in 0..<nFFT {
                frameBuffer[i] = paddedData[signalOffset + startPos + i] * window[i]
            }

            // Perform FFT using proper unsafe pointer scoping
            realPart.withUnsafeMutableBufferPointer { rBuf in
                imagPart.withUnsafeMutableBufferPointer { iBuf in
                    // Convert to split complex format
                    var splitComplex = DSPSplitComplex(realp: rBuf.baseAddress!, imagp: iBuf.baseAddress!)
                    frameBuffer.withUnsafeBufferPointer { fBuf in
                        fBuf.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: nFFT / 2) { ptr in
                            vDSP_ctoz(ptr, 2, &splitComplex, 1, vDSP_Length(nFFT / 2))
                        }
                    }

                    // Perform FFT
                    vDSP_fft_zrip(fftSetup, &splitComplex, 1, log2n, FFTDirection(FFT_FORWARD))

                    // Scale by 0.5 (vDSP convention)
                    var scale: Float = 0.5
                    vDSP_vsmul(rBuf.baseAddress!, 1, &scale, rBuf.baseAddress!, 1, vDSP_Length(nFFT / 2))
                    vDSP_vsmul(iBuf.baseAddress!, 1, &scale, iBuf.baseAddress!, 1, vDSP_Length(nFFT / 2))
                }
            }

            // Extract results with normalization
            for f in 0..<freqBins {
                let outIdx = b * freqBins * numFrames + f * numFrames + frameIdx
                if f == 0 {
                    outputReal[outIdx] = realPart[0] * normalization
                    outputImag[outIdx] = 0.0
                } else if f == freqBins - 1 {
                    outputReal[outIdx] = imagPart[0] * normalization
                    outputImag[outIdx] = 0.0
                } else {
                    outputReal[outIdx] = realPart[f] * normalization
                    outputImag[outIdx] = imagPart[f] * normalization
                }
            }
        }
    }

    // Create MLX arrays and combine into complex
    let realArray = MLXArray(outputReal, [batchSize, freqBins, numFrames])
    let imagArray = MLXArray(outputImag, [batchSize, freqBins, numFrames])

    var result = realArray + MLXArray(real: 0, imaginary: 1) * imagArray

    // Reshape back to original batch dimensions + (freqBins, numFrames)
    var outputShape = Array(shape.dropLast())
    outputShape.append(freqBins)
    outputShape.append(numFrames)
    result = result.reshaped(outputShape)
    eval(result)

    return result
}


// MARK: - ispectro (inverse STFT)

/// Inverse Short-Time Fourier Transform matching the C++ `ispectro()`.
public func ispectro(_ z: MLXArray, hopLength: Int = -1, length: Int = 0, pad: Int = 0) -> MLXArray {
    let shape = z.shape
    let freqs = shape[shape.count - 2]
    let frames = shape[shape.count - 1]
    let nFFT = 2 * freqs - 2
    let hopLen = hopLength < 0 ? nFFT / 4 : hopLength

    var batchSize = 1
    for i in 0..<(shape.count - 2) { batchSize *= shape[i] }

    // Reshape and extract real/imag
    let z3d = z.reshaped(batchSize, freqs, frames)
    let zReal = z3d.realPart()
    let zImag = z3d.imaginaryPart()
    eval(zReal)
    eval(zImag)

    let realData = zReal.asArray(Float.self)
    let imagData = zImag.asArray(Float.self)

    let window = periodicHannWindow(nFFT)
    let windowNorm = computeWindowNormalization(window, nFFT: nFFT, hopLength: hopLen, numFrames: frames)

    // Setup FFT
    let log2n = vDSP_Length(log2(Double(nFFT)))
    guard let fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(FFT_RADIX2)) else {
        fatalError("Failed to create FFT setup")
    }
    defer { vDSP_destroy_fftsetup(fftSetup) }

    let outputLength = (frames - 1) * hopLen + nFFT
    var outputData = [Float](repeating: 0.0, count: batchSize * outputLength)

    var realPart = [Float](repeating: 0.0, count: nFFT / 2 + 1)
    var imagPart = [Float](repeating: 0.0, count: nFFT / 2 + 1)
    var frameBuffer = [Float](repeating: 0.0, count: nFFT)
    let normalization: Float = sqrt(Float(nFFT))

    for b in 0..<batchSize {
        let outOffset = b * outputLength

        for frameIdx in 0..<frames {
            // Prepare split complex input and undo normalization
            for f in 0..<freqs {
                let inIdx = b * freqs * frames + f * frames + frameIdx
                if f == 0 {
                    realPart[0] = realData[inIdx] * normalization
                } else if f == freqs - 1 {
                    imagPart[0] = realData[inIdx] * normalization
                } else {
                    realPart[f] = realData[inIdx] * normalization
                    imagPart[f] = imagData[inIdx] * normalization
                }
            }

            // Perform inverse FFT and convert from split complex
            realPart.withUnsafeMutableBufferPointer { rBuf in
                imagPart.withUnsafeMutableBufferPointer { iBuf in
                    var splitComplex = DSPSplitComplex(realp: rBuf.baseAddress!, imagp: iBuf.baseAddress!)
                    vDSP_fft_zrip(fftSetup, &splitComplex, 1, log2n, FFTDirection(FFT_INVERSE))

                    // Scale by 0.5 (vDSP convention)
                    var scale: Float = 0.5
                    vDSP_vsmul(rBuf.baseAddress!, 1, &scale, rBuf.baseAddress!, 1, vDSP_Length(nFFT / 2))
                    vDSP_vsmul(iBuf.baseAddress!, 1, &scale, iBuf.baseAddress!, 1, vDSP_Length(nFFT / 2))

                    // Convert from split complex
                    frameBuffer.withUnsafeMutableBufferPointer { fBuf in
                        fBuf.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: nFFT / 2) { ptr in
                            vDSP_ztoc(&splitComplex, 1, ptr, 2, vDSP_Length(nFFT / 2))
                        }
                    }
                }
            }

            // Overlap-add with window and normalization
            let startPos = frameIdx * hopLen
            for i in 0..<nFFT {
                let pos = startPos + i
                if pos < outputLength {
                    let normFactor = windowNorm[pos] + 1e-8
                    outputData[outOffset + pos] += frameBuffer[i] * window[i] / Float(nFFT / 2) / normFactor
                }
            }
        }
    }

    // Remove center padding and trim to requested length
    let padAmount = nFFT / 2
    let finalLength = length > 0 ? length : (outputLength - 2 * padAmount)

    var trimmedData = [Float](repeating: 0.0, count: batchSize * finalLength)
    for b in 0..<batchSize {
        let srcStart = b * outputLength + padAmount
        let dstStart = b * finalLength
        for i in 0..<finalLength {
            trimmedData[dstStart + i] = outputData[srcStart + i]
        }
    }

    var result = MLXArray(trimmedData, [batchSize, finalLength])

    // Reshape back to original dimensions (minus the last 2, plus finalLength)
    var outputShape = Array(shape.dropLast(2))
    outputShape.append(finalLength)
    result = result.reshaped(outputShape)
    eval(result)

    return result
}
