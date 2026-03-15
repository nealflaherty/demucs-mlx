// HTDemucs.swift - Hybrid Transformer Demucs model
// 1:1 port of cpp/demucs/htdemucs.cpp

import MLX
import Foundation

/// HTDemucs - Hybrid Transformer Demucs model.
/// Extends HDemucs with cross-transformer layers between frequency and time branches.
public struct HTDemucsModel {

    // Model parameters
    public let sources: [String]
    public let audioChannels: Int
    public let channels: Int
    public let depth: Int
    public let nfft: Int
    public let hopLength: Int
    public let cac: Bool
    public let stride_: Int
    public let kernelSize: Int
    public let samplerate: Int
    public let segment: Float
    public let freqEmbScale: Float
    public let bottomChannels: Int

    // Layers
    public var encoder: [HEncLayer]
    public var decoder: [HDecLayer]
    public var tencoder: [HEncLayer]
    public var tdecoder: [HDecLayer]
    public var freqEmb: ScaledEmbedding?
    public var crossTransformer: CrossTransformerEncoder?

    // Channel up/downsampler weights
    public var channelUpsamplerWeight: MLXArray
    public var channelUpsamplerBias: MLXArray
    public var channelUpsamplerTWeight: MLXArray
    public var channelUpsamplerTBias: MLXArray
    public var channelDownsamplerWeight: MLXArray
    public var channelDownsamplerBias: MLXArray
    public var channelDownsamplerTWeight: MLXArray
    public var channelDownsamplerTBias: MLXArray

    public init(
        sources: [String] = ["drums", "bass", "other", "vocals"],
        audioChannels: Int = 2, channels: Int = 48, channelsTime: Int = -1,
        growth: Float = 2.0, nfft: Int = 4096, cac: Bool = true,
        depth: Int = 6, rewrite: Bool = true, freqEmb: Float = 0.2,
        embScale: Float = 10.0, embSmooth: Bool = true,
        kernelSize: Int = 8, timeStride: Int = 2, stride: Int = 4,
        context: Int = 1, contextEnc: Int = 0,
        normStarts: Int = 4, normGroups: Int = 4,
        dconvMode: Int = 3, dconvComp: Int = 8, dconvInit: Float = 1e-3,
        bottomChannels: Int = 0,
        tLayers: Int = 5, tHeads: Int = 8, tHiddenScale: Float = 4.0,
        tNormIn: Bool = true, tNormOut: Bool = true,
        tCrossFirst: Bool = false, tLayerScale: Bool = true, tGelu: Bool = true,
        samplerate: Int = 44100, segment: Float = 40.0
    ) {
        self.sources = sources
        self.audioChannels = audioChannels
        self.channels = channels
        self.depth = depth
        self.nfft = nfft
        self.hopLength = nfft / 4
        self.cac = cac
        self.stride_ = stride
        self.kernelSize = kernelSize
        self.samplerate = samplerate
        self.segment = segment
        self.freqEmbScale = freqEmb
        self.bottomChannels = bottomChannels

        // Initialize channel samplers to placeholder
        channelUpsamplerWeight = MLXArray.zeros([1])
        channelUpsamplerBias = MLXArray.zeros([1])
        channelUpsamplerTWeight = MLXArray.zeros([1])
        channelUpsamplerTBias = MLXArray.zeros([1])
        channelDownsamplerWeight = MLXArray.zeros([1])
        channelDownsamplerBias = MLXArray.zeros([1])
        channelDownsamplerTWeight = MLXArray.zeros([1])
        channelDownsamplerTBias = MLXArray.zeros([1])

        // Build encoder/decoder layers
        var chin = audioChannels
        var chinZ = cac ? chin * 2 : chin
        var chout = channelsTime > 0 ? channelsTime : channels
        var choutZ = channels
        var freqs = nfft / 2

        var encoderArr = [HEncLayer]()
        var decoderArr = [HDecLayer]()
        var tencoderArr = [HEncLayer]()
        var tdecoderArr = [HDecLayer]()
        var freqEmbOpt: ScaledEmbedding? = nil

        for index in 0..<depth {
            let norm = index >= normStarts
            let freq = freqs > 1
            var stri = stride
            var ker = kernelSize

            if !freq {
                ker = timeStride * 2
                stri = timeStride
            }

            var pad = true
            var lastFreq = false
            if freq && freqs <= kernelSize {
                ker = freqs
                pad = false
                lastFreq = true
            }

            // Frequency encoder
            encoderArr.append(HEncLayer(
                chin: chinZ, chout: choutZ, kernelSize: ker, stride: stri,
                normGroups: normGroups, empty: false, freq: freq,
                dconvEnabled: (dconvMode & 1) != 0, norm: norm, context: contextEnc,
                pad: pad, rewrite: rewrite,
                dconvComp: Float(dconvComp), dconvInit: dconvInit))

            // Time encoder
            if freq {
                tencoderArr.append(HEncLayer(
                    chin: chin, chout: chout, kernelSize: kernelSize, stride: stride,
                    normGroups: normGroups, empty: lastFreq, freq: false,
                    dconvEnabled: (dconvMode & 1) != 0, norm: norm, context: contextEnc,
                    pad: true, rewrite: rewrite,
                    dconvComp: Float(dconvComp), dconvInit: dconvInit))
            }

            // Update chin for decoder (first layer outputs S*audioChannels)
            if index == 0 {
                chin = audioChannels * sources.count
                chinZ = cac ? chin * 2 : chin
            }

            // Frequency decoder (prepend)
            decoderArr.insert(HDecLayer(
                chin: choutZ, chout: chinZ, last: index == 0, kernelSize: ker,
                stride: stri, normGroups: normGroups, empty: false, freq: freq,
                dconvEnabled: (dconvMode & 2) != 0, norm: norm, context: context,
                pad: pad, contextFreq: true, rewrite: rewrite,
                dconvComp: Float(dconvComp), dconvInit: dconvInit), at: 0)

            // Time decoder (prepend)
            if freq {
                tdecoderArr.insert(HDecLayer(
                    chin: chout, chout: chin, last: index == 0, kernelSize: kernelSize,
                    stride: stride, normGroups: normGroups, empty: lastFreq, freq: false,
                    dconvEnabled: (dconvMode & 2) != 0, norm: norm, context: context,
                    pad: true, contextFreq: true, rewrite: rewrite,
                    dconvComp: Float(dconvComp), dconvInit: dconvInit), at: 0)
            }

            // Update for next iteration
            chin = chout
            chinZ = choutZ
            chout = Int(growth * Float(chout))
            choutZ = Int(growth * Float(choutZ))

            if freq {
                if freqs <= kernelSize { freqs = 1 } else { freqs /= stride }
            }

            // Frequency embedding after first layer
            if index == 0 && freqEmb > 0 {
                freqEmbOpt = ScaledEmbedding(
                    numEmbeddings: freqs, embeddingDim: chinZ,
                    scale: embScale, smooth: embSmooth)
            }
        }

        self.encoder = encoderArr
        self.decoder = decoderArr
        self.tencoder = tencoderArr
        self.tdecoder = tdecoderArr
        self.freqEmb = freqEmbOpt

        // Create transformer
        if tLayers > 0 {
            var transformerChannels = channels
            for _ in 0..<(depth - 1) {
                transformerChannels = Int(growth * Float(transformerChannels))
            }
            if bottomChannels > 0 {
                channelUpsamplerWeight = MLXArray.zeros([bottomChannels, transformerChannels, 1])
                channelUpsamplerBias = MLXArray.zeros([bottomChannels])
                channelUpsamplerTWeight = MLXArray.zeros([bottomChannels, transformerChannels, 1])
                channelUpsamplerTBias = MLXArray.zeros([bottomChannels])
                channelDownsamplerWeight = MLXArray.zeros([transformerChannels, bottomChannels, 1])
                channelDownsamplerBias = MLXArray.zeros([transformerChannels])
                channelDownsamplerTWeight = MLXArray.zeros([transformerChannels, bottomChannels, 1])
                channelDownsamplerTBias = MLXArray.zeros([transformerChannels])
                transformerChannels = bottomChannels
            }
            crossTransformer = CrossTransformerEncoder(
                dim: transformerChannels, hiddenScale: tHiddenScale,
                numHeads: tHeads, numLayers: tLayers,
                crossFirst: tCrossFirst, normIn: tNormIn,
                normFirst: true, normOut: tNormOut,
                layerScale: tLayerScale, useGelu: tGelu)
        }
    }

    // MARK: - STFT helpers

    public func spec(_ x: MLXArray) -> MLXArray {
        let lastDim = x.shape[x.ndim - 1]
        let le = (lastDim + hopLength - 1) / hopLength
        let padL = hopLength / 2 * 3
        let padR = padL + le * hopLength - x.shape[x.ndim - 1]
        let xPadded = pad1d(x, paddingLeft: padL, paddingRight: padR, mode: "reflect")
        var z = spectro(xPadded, nFFT: nfft, hopLength: hopLength)
        // Remove last freq bin
        z = sliceAxis(z, axis: -2, start: 0, end: z.shape[z.ndim - 2] - 1)
        // Remove padding frames
        z = sliceAxis(z, axis: -1, start: 2, end: 2 + le)
        return z
    }

    public func ispec(_ z: MLXArray, length: Int) -> MLXArray {
        // Add back last freq bin (zeros)
        let freqShape = z.shape
        var padWidths = [(Int, Int)](repeating: (0, 0), count: freqShape.count)
        padWidths[freqShape.count - 2] = (0, 1)
        var zPadded = MLX.padded(z, widths: padWidths.map { .init($0) })
        // Add time padding
        var timePad = [(Int, Int)](repeating: (0, 0), count: zPadded.ndim)
        timePad[zPadded.ndim - 1] = (2, 2)
        zPadded = MLX.padded(zPadded, widths: timePad.map { .init($0) })

        let padL = hopLength / 2 * 3
        let le = hopLength * ((length + hopLength - 1) / hopLength) + 2 * padL
        var x = ispectro(zPadded, hopLength: hopLength, length: le)
        x = sliceAxis(x, axis: -1, start: padL, end: padL + length)
        return x
    }

    public func magnitude(_ z: MLXArray) -> MLXArray {
        if cac {
            let realPart = z.realPart()
            let imagPart = z.imaginaryPart()
            let B = z.shape[0], C = z.shape[1], Fr = z.shape[2], T = z.shape[3]
            let realExp = expandedDimensions(realPart, axis: 2)
            let imagExp = expandedDimensions(imagPart, axis: 2)
            let stacked = concatenated([realExp, imagExp], axis: 2)
            return stacked.reshaped(B, C * 2, Fr, T)
        } else {
            return MLX.abs(z)
        }
    }

    public func mask(_ z: MLXArray, _ m: MLXArray) -> MLXArray {
        if cac {
            let B = m.shape[0], S = m.shape[1], Fr = m.shape[3], T = m.shape[4]
            let C = m.shape[2] / 2
            let mReshaped = m.reshaped(B, S, C, 2, Fr, T)
            let realPart = sliceAxis(mReshaped, axis: 3, start: 0, end: 1)
                .squeezed(axis: 3)
            let imagPart = sliceAxis(mReshaped, axis: 3, start: 1, end: 2)
                .squeezed(axis: 3)
            return realPart + MLXArray(real: Float(0), imaginary: Float(1)) * imagPart
        } else {
            let zExp = expandedDimensions(z, axis: 1)
            let zAbs = MLX.abs(zExp) + MLXArray(Float(1e-8))
            return zExp * (m / zAbs)
        }
    }

    // MARK: - Forward pass

    public mutating func forward(_ mix: MLXArray) -> MLXArray {
        var length = mix.shape[mix.ndim - 1]
        var lengthPrePad = -1
        let trainingLength = Int(segment * Float(samplerate))
        var paddedMix = mix

        if mix.shape[mix.ndim - 1] < trainingLength {
            lengthPrePad = mix.shape[mix.ndim - 1]
            let padAmount = trainingLength - lengthPrePad
            var pw = [(Int, Int)](repeating: (0, 0), count: mix.ndim)
            pw[mix.ndim - 1] = (0, padAmount)
            paddedMix = MLX.padded(mix, widths: pw.map { .init($0) })
            eval(paddedMix)
            length = trainingLength
        }

        // Step 1: STFT
        let z = spec(paddedMix)
        eval(z)
        
        let mag = magnitude(z)
        eval(mag)
        
        var x = mag

        // Step 2: Normalize freq branch (ddof=1 to match PyTorch std)
        let mean = x.mean(axes: [1, 2, 3], keepDims: true)
        let variance = x.variance(axes: [1, 2, 3], keepDims: true, ddof: 1)
        let std = MLX.sqrt(variance)
        x = (x - mean) / (std + MLXArray(Float(1e-5)))
        eval(x)

        // Step 3: Normalize time branch (ddof=1 to match PyTorch std)
        var xt = paddedMix
        let meant = xt.mean(axes: [1, 2], keepDims: true)
        let vart = xt.variance(axes: [1, 2], keepDims: true, ddof: 1)
        let stdt = MLX.sqrt(vart)
        xt = (xt - meant) / (stdt + MLXArray(Float(1e-5)))
        eval(xt)

        // Step 4: Encoding loop
        var saved = [MLXArray]()
        var savedT = [MLXArray]()
        var lengths = [Int]()
        var lengthsT = [Int]()

        for idx in 0..<encoder.count {
            lengths.append(x.shape[x.ndim - 1])
            var inject: MLXArray? = nil

            if idx < tencoder.count {
                lengthsT.append(xt.shape[xt.ndim - 1])
                xt = tencoder[idx].forward(xt)
                eval(xt)
                if !tencoder[idx].empty {
                    savedT.append(xt)
                } else {
                    inject = xt
                }
            }

            x = encoder[idx].forward(x, inject: inject)
            eval(x)

            if idx == 0, let fe = freqEmb {
                let freqSize = x.shape[x.ndim - 2]
                let frs = MLXArray(Array(0..<freqSize))
                var emb = fe.forward(frs)
                emb = emb.transposed(1, 0).reshaped(1, emb.shape[1], emb.shape[0], 1)
                x = x + freqEmbScale * emb
                eval(x)
            }
            saved.append(x)
        }

        // Step 5: Cross-transformer
        if crossTransformer != nil {
            if bottomChannels > 0 {
                let B = x.shape[0], C = x.shape[1]
                let F = x.shape[2], TF = x.shape[3]
                x = x.reshaped(B, C, F * TF)
                x = conv1d(x, weight: channelUpsamplerWeight, bias: channelUpsamplerBias)
                let cNew = x.shape[1]
                x = x.reshaped(B, cNew, F, TF)
                xt = conv1d(xt, weight: channelUpsamplerTWeight, bias: channelUpsamplerTBias)
                eval(x); eval(xt)
            }

            let (xf, xtf) = crossTransformer!.forward(x, xt)
            x = xf; xt = xtf
            eval(x); eval(xt)

            if bottomChannels > 0 {
                let B = x.shape[0], C = x.shape[1]
                let F = x.shape[2], TF = x.shape[3]
                x = x.reshaped(B, C, F * TF)
                x = conv1d(x, weight: channelDownsamplerWeight, bias: channelDownsamplerBias)
                let cNew = x.shape[1]
                x = x.reshaped(B, cNew, F, TF)
                xt = conv1d(xt, weight: channelDownsamplerTWeight, bias: channelDownsamplerTBias)
                eval(x); eval(xt)
            }
        }

        // Step 6: Decoding loop
        let offset = depth - tdecoder.count
        for idx in 0..<decoder.count {
            let skip = saved.removeLast()
            let len = lengths.removeLast()
            let (xOut, pre) = decoder[idx].forward(x, skip: skip, length: len)
            x = xOut
            eval(x)

            if idx >= offset {
                let tdecIdx = idx - offset
                if tdecIdx < tdecoder.count {
                    let lengthT = lengthsT.removeLast()
                    if tdecoder[tdecIdx].empty {
                        let preSqueezed = pre.squeezed(axis: 2)
                        let (xtOut, _) = tdecoder[tdecIdx].forward(
                            preSqueezed, skip: MLXArray.zeros(like: preSqueezed), length: lengthT)
                        xt = xtOut
                    } else {
                        let skipT = savedT.removeLast()
                        let (xtOut, _) = tdecoder[tdecIdx].forward(xt, skip: skipT, length: lengthT)
                        xt = xtOut
                    }
                    eval(xt)
                }
            }
        }

        // Step 7: Reshape to (B, S, C, Fr, T)
        let B = x.shape[0]
        let S = sources.count
        let C = x.shape[1] / S
        let Fr = x.shape[2]
        let T = x.shape[3]
        x = x.reshaped(B, S, C, Fr, T)

        // Step 8: Denormalize freq branch
        x = x * expandedDimensions(std, axis: 1) + expandedDimensions(mean, axis: 1)
        eval(x)

        // Step 9: Mask and iSTFT
        let zout = mask(z, x)
        eval(zout)
        x = ispec(zout, length: length)
        eval(x)

        // Step 10: Add time branch
        xt = xt.reshaped(B, S, -1, length)
        xt = xt * expandedDimensions(stdt, axis: 1) + expandedDimensions(meant, axis: 1)
        x = xt + x
        eval(x)

        if lengthPrePad >= 0 {
            x = sliceAxis(x, axis: -1, start: 0, end: lengthPrePad)
        }

        return x
    }
}
