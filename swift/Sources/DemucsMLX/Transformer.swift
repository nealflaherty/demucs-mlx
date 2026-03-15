// Transformer.swift — Positional embeddings, self-attention, cross-attention
// 1:1 port of cpp/demucs/transformer.cpp

import MLX
import Foundation

// MARK: - Sinusoidal Embeddings

/// Create 1D sinusoidal positional embedding of shape (length, 1, dim).
public func createSinEmbedding(length: Int, dim: Int, shift: Int = 0,
                                maxPeriod: Float = 10000.0) -> MLXArray {
    let pos = MLXArray(Array(Swift.stride(from: Float(shift), to: Float(shift + length), by: 1)))
        .reshaped(length, 1, 1)

    let halfDim = dim / 2
    let adim = MLXArray(Array(Swift.stride(from: Float(0), to: Float(halfDim), by: 1)))
        .reshaped(1, 1, halfDim)

    let exponent = adim / Float(halfDim - 1)
    let divisor = pow(MLXArray(maxPeriod), exponent)
    let phase = pos / divisor

    return concatenated([cos(phase), sin(phase)], axis: -1)
}

/// Create 2D sinusoidal positional embedding of shape (1, dModel, height, width).
public func create2DSinEmbedding(dModel: Int, height: Int, width: Int,
                                  maxPeriod: Float = 10000.0) -> MLXArray {
    precondition(dModel % 4 == 0)

    let dModelHalf = dModel / 2
    let arangeVals = MLXArray(Array(Swift.stride(from: Float(0), to: Float(dModelHalf), by: 2)))
    let divTerm = exp(arangeVals * (-log(maxPeriod) / Float(dModelHalf)))
    let divTermRow = divTerm.reshaped(1, -1)

    let posW = MLXArray(Array(Swift.stride(from: Float(0), to: Float(width), by: 1)))
        .reshaped(width, 1)
    let posH = MLXArray(Array(Swift.stride(from: Float(0), to: Float(height), by: 1)))
        .reshaped(height, 1)

    // Width embeddings: (numFreqs, 1, width) -> broadcast to (numFreqs, height, width)
    let sinW = sin(matmul(posW, divTermRow)).transposed(1, 0)
        .reshaped(-1, 1, width)
    let sinWB = broadcast(sinW, to: [sinW.shape[0], height, width])
    let cosW = cos(matmul(posW, divTermRow)).transposed(1, 0)
        .reshaped(-1, 1, width)
    let cosWB = broadcast(cosW, to: [cosW.shape[0], height, width])

    // Height embeddings: (numFreqs, height, 1) -> broadcast
    let sinH = sin(matmul(posH, divTermRow)).transposed(1, 0)
        .reshaped(-1, height, 1)
    let sinHB = broadcast(sinH, to: [sinH.shape[0], height, width])
    let cosH = cos(matmul(posH, divTermRow)).transposed(1, 0)
        .reshaped(-1, height, 1)
    let cosHB = broadcast(cosH, to: [cosH.shape[0], height, width])

    // Interleave sin/cos for width and height
    let numFreqs = sinWB.shape[0]
    var parts = [MLXArray]()
    for i in 0..<numFreqs {
        parts.append(sliceAxis(sinWB, axis: 0, start: i, end: i + 1))
        parts.append(sliceAxis(cosWB, axis: 0, start: i, end: i + 1))
    }
    for i in 0..<numFreqs {
        parts.append(sliceAxis(sinHB, axis: 0, start: i, end: i + 1))
        parts.append(sliceAxis(cosHB, axis: 0, start: i, end: i + 1))
    }

    let pe = concatenated(parts, axis: 0)
    return expandedDimensions(pe, axis: 0)
}

// MARK: - MyGroupNorm

/// GroupNorm that expects (B, T, C) format.
public struct MyGroupNorm {
    public var weight: MLXArray
    public var bias: MLXArray
    let numGroups: Int
    let eps: Float
    let layerNormMode: Bool

    public init(numGroups: Int, numChannels: Int, eps: Float = 1e-5,
                layerNormMode: Bool = false) {
        self.numGroups = numGroups
        self.eps = eps
        self.layerNormMode = layerNormMode
        self.weight = MLXArray.ones([numChannels])
        self.bias = MLXArray.zeros([numChannels])
    }

    public func forward(_ x: MLXArray) -> MLXArray {
        if layerNormMode {
            let mean = x.mean(axes: [-1], keepDims: true)
            let variance = x.variance(axes: [-1], keepDims: true)
            let normalized = (x - mean) / sqrt(variance + MLXArray(eps))
            return normalized * weight + bias
        }
        let xt = x.transposed(0, 2, 1)
        let normalized = groupNorm(xt, weight: weight, bias: bias, numGroups: numGroups, eps: eps)
        return normalized.transposed(0, 2, 1)
    }
}

// MARK: - MyTransformerEncoderLayer

public struct MyTransformerEncoderLayer {
    public var selfAttnInProjWeight: MLXArray
    public var selfAttnInProjBias: MLXArray
    public var selfAttnOutProjWeight: MLXArray
    public var selfAttnOutProjBias: MLXArray
    public var linear1Weight: MLXArray
    public var linear1Bias: MLXArray
    public var linear2Weight: MLXArray
    public var linear2Bias: MLXArray
    public var norm1: MyGroupNorm
    public var norm2: MyGroupNorm
    public var normOut: MyGroupNorm?
    public var gamma1: LayerScale?
    public var gamma2: LayerScale?

    let dModel: Int
    let nHead: Int
    let useGelu: Bool
    let normFirst: Bool

    public init(dModel: Int, nHead: Int, dimFeedforward: Int = 2048,
                useGelu: Bool = true, groupNormGroups: Int = 0,
                normFirst: Bool = false, normOut: Bool = false,
                layerScale: Bool = false, initValues: Float = 1e-4) {
        self.dModel = dModel
        self.nHead = nHead
        self.useGelu = useGelu
        self.normFirst = normFirst

        self.selfAttnInProjWeight = MLXArray.zeros([3 * dModel, dModel])
        self.selfAttnInProjBias = MLXArray.zeros([3 * dModel])
        self.selfAttnOutProjWeight = MLXArray.zeros([dModel, dModel])
        self.selfAttnOutProjBias = MLXArray.zeros([dModel])
        self.linear1Weight = MLXArray.zeros([dimFeedforward, dModel])
        self.linear1Bias = MLXArray.zeros([dimFeedforward])
        self.linear2Weight = MLXArray.zeros([dModel, dimFeedforward])
        self.linear2Bias = MLXArray.zeros([dModel])

        if groupNormGroups > 0 {
            self.norm1 = MyGroupNorm(numGroups: groupNormGroups, numChannels: dModel)
            self.norm2 = MyGroupNorm(numGroups: groupNormGroups, numChannels: dModel)
        } else {
            self.norm1 = MyGroupNorm(numGroups: 1, numChannels: dModel, layerNormMode: true)
            self.norm2 = MyGroupNorm(numGroups: 1, numChannels: dModel, layerNormMode: true)
        }

        if normFirst && normOut {
            self.normOut = MyGroupNorm(numGroups: 1, numChannels: dModel)
        }
        if layerScale {
            self.gamma1 = LayerScale(channels: dModel, init: initValues, channelLast: true)
            self.gamma2 = LayerScale(channels: dModel, init: initValues, channelLast: true)
        }
    }

    public func forward(_ src: MLXArray) -> MLXArray {
        var x = src
        if normFirst {
            var saOut = saBlock(norm1.forward(x))
            if let g1 = gamma1 { saOut = g1.forward(saOut) }
            x = x + saOut
            var ffOut = ffBlock(norm2.forward(x))
            if let g2 = gamma2 { ffOut = g2.forward(ffOut) }
            x = x + ffOut
            if let no = normOut { x = no.forward(x) }
        } else {
            var saOut = saBlock(x)
            if let g1 = gamma1 { saOut = g1.forward(saOut) }
            x = norm1.forward(x + saOut)
            var ffOut = ffBlock(x)
            if let g2 = gamma2 { ffOut = g2.forward(ffOut) }
            x = norm2.forward(x + ffOut)
        }
        return x
    }

    private func saBlock(_ x: MLXArray) -> MLXArray {
        let B = x.shape[0], T = x.shape[1]
        let headDim = dModel / nHead

        let qkv = addMM(selfAttnInProjBias, x, selfAttnInProjWeight.transposed(1, 0))
        var q = sliceAxis(qkv, axis: -1, start: 0, end: dModel)
        var k = sliceAxis(qkv, axis: -1, start: dModel, end: 2 * dModel)
        var v = sliceAxis(qkv, axis: -1, start: 2 * dModel, end: 3 * dModel)

        q = q.reshaped(B, T, nHead, headDim).transposed(0, 2, 1, 3)
        k = k.reshaped(B, T, nHead, headDim).transposed(0, 2, 1, 3)
        v = v.reshaped(B, T, nHead, headDim).transposed(0, 2, 1, 3)

        var scores = matmul(q, k.transposed(0, 1, 3, 2))
        scores = scores / MLXArray(Foundation.sqrt(Float(headDim)))
        let attnWeights = softmax(scores, axis: -1)
        var attnOutput = matmul(attnWeights, v)
        attnOutput = attnOutput.transposed(0, 2, 1, 3).reshaped(B, T, dModel)

        return addMM(selfAttnOutProjBias, attnOutput, selfAttnOutProjWeight.transposed(1, 0))
    }

    private func ffBlock(_ x: MLXArray) -> MLXArray {
        var out = addMM(linear1Bias, x, linear1Weight.transposed(1, 0))
        out = useGelu ? gelu(out) : maximum(out, MLXArray(Float(0)))
        return addMM(linear2Bias, out, linear2Weight.transposed(1, 0))
    }
}

// MARK: - CrossTransformerEncoderLayer

public struct CrossTransformerEncoderLayer {
    public var crossAttnQProjWeight: MLXArray
    public var crossAttnQProjBias: MLXArray
    public var crossAttnKProjWeight: MLXArray
    public var crossAttnKProjBias: MLXArray
    public var crossAttnVProjWeight: MLXArray
    public var crossAttnVProjBias: MLXArray
    public var crossAttnOutProjWeight: MLXArray
    public var crossAttnOutProjBias: MLXArray
    public var linear1Weight: MLXArray
    public var linear1Bias: MLXArray
    public var linear2Weight: MLXArray
    public var linear2Bias: MLXArray
    public var norm1: MyGroupNorm
    public var norm2: MyGroupNorm
    public var norm3: MyGroupNorm
    public var normOut: MyGroupNorm?
    public var gamma1: LayerScale?
    public var gamma2: LayerScale?

    let dModel: Int
    let nHead: Int
    let useGelu: Bool
    let normFirst: Bool

    public init(dModel: Int, nHead: Int, dimFeedforward: Int = 2048,
                useGelu: Bool = true, groupNormGroups: Int = 0,
                normFirst: Bool = false, normOut: Bool = false,
                layerScale: Bool = false, initValues: Float = 1e-4) {
        self.dModel = dModel
        self.nHead = nHead
        self.useGelu = useGelu
        self.normFirst = normFirst

        self.crossAttnQProjWeight = MLXArray.zeros([dModel, dModel])
        self.crossAttnQProjBias = MLXArray.zeros([dModel])
        self.crossAttnKProjWeight = MLXArray.zeros([dModel, dModel])
        self.crossAttnKProjBias = MLXArray.zeros([dModel])
        self.crossAttnVProjWeight = MLXArray.zeros([dModel, dModel])
        self.crossAttnVProjBias = MLXArray.zeros([dModel])
        self.crossAttnOutProjWeight = MLXArray.zeros([dModel, dModel])
        self.crossAttnOutProjBias = MLXArray.zeros([dModel])
        self.linear1Weight = MLXArray.zeros([dimFeedforward, dModel])
        self.linear1Bias = MLXArray.zeros([dimFeedforward])
        self.linear2Weight = MLXArray.zeros([dModel, dimFeedforward])
        self.linear2Bias = MLXArray.zeros([dModel])

        if groupNormGroups > 0 {
            self.norm1 = MyGroupNorm(numGroups: groupNormGroups, numChannels: dModel)
            self.norm2 = MyGroupNorm(numGroups: groupNormGroups, numChannels: dModel)
            self.norm3 = MyGroupNorm(numGroups: groupNormGroups, numChannels: dModel)
        } else {
            self.norm1 = MyGroupNorm(numGroups: 1, numChannels: dModel, layerNormMode: true)
            self.norm2 = MyGroupNorm(numGroups: 1, numChannels: dModel, layerNormMode: true)
            self.norm3 = MyGroupNorm(numGroups: 1, numChannels: dModel, layerNormMode: true)
        }

        if normFirst && normOut {
            self.normOut = MyGroupNorm(numGroups: 1, numChannels: dModel)
        }
        if layerScale {
            self.gamma1 = LayerScale(channels: dModel, init: initValues, channelLast: true)
            self.gamma2 = LayerScale(channels: dModel, init: initValues, channelLast: true)
        }
    }

    public func forward(_ q: MLXArray, _ k: MLXArray) -> MLXArray {
        var x = q
        if normFirst {
            var caOut = caBlock(norm1.forward(q), norm2.forward(k))
            if let g1 = gamma1 { caOut = g1.forward(caOut) }
            x = q + caOut
            var ffOut = ffBlock(norm3.forward(x))
            if let g2 = gamma2 { ffOut = g2.forward(ffOut) }
            x = x + ffOut
            if let no = normOut { x = no.forward(x) }
        } else {
            var caOut = caBlock(q, k)
            if let g1 = gamma1 { caOut = g1.forward(caOut) }
            x = norm1.forward(q + caOut)
            var ffOut = ffBlock(x)
            if let g2 = gamma2 { ffOut = g2.forward(ffOut) }
            x = norm2.forward(x + ffOut)
        }
        return x
    }

    private func caBlock(_ q: MLXArray, _ k: MLXArray) -> MLXArray {
        let Bq = q.shape[0], Tq = q.shape[1]
        let Bk = k.shape[0], Tk = k.shape[1]
        let headDim = dModel / nHead

        var query = addMM(crossAttnQProjBias, q, crossAttnQProjWeight.transposed(1, 0))
        var key = addMM(crossAttnKProjBias, k, crossAttnKProjWeight.transposed(1, 0))
        var value = addMM(crossAttnVProjBias, k, crossAttnVProjWeight.transposed(1, 0))

        query = query.reshaped(Bq, Tq, nHead, headDim).transposed(0, 2, 1, 3)
        key = key.reshaped(Bk, Tk, nHead, headDim).transposed(0, 2, 1, 3)
        value = value.reshaped(Bk, Tk, nHead, headDim).transposed(0, 2, 1, 3)

        var scores = matmul(query, key.transposed(0, 1, 3, 2))
        scores = scores / MLXArray(Foundation.sqrt(Float(headDim)))
        let attnWeights = softmax(scores, axis: -1)
        var attnOutput = matmul(attnWeights, value)
        attnOutput = attnOutput.transposed(0, 2, 1, 3).reshaped(Bq, Tq, dModel)

        return addMM(crossAttnOutProjBias, attnOutput, crossAttnOutProjWeight.transposed(1, 0))
    }

    private func ffBlock(_ x: MLXArray) -> MLXArray {
        var out = addMM(linear1Bias, x, linear1Weight.transposed(1, 0))
        out = useGelu ? gelu(out) : maximum(out, MLXArray(Float(0)))
        return addMM(linear2Bias, out, linear2Weight.transposed(1, 0))
    }
}

// MARK: - CrossTransformerEncoder

public struct CrossTransformerEncoder {
    public var classicLayers: [MyTransformerEncoderLayer]
    public var classicLayersT: [MyTransformerEncoderLayer]
    public var crossLayers: [CrossTransformerEncoderLayer]
    public var crossLayersT: [CrossTransformerEncoderLayer]
    public var normIn: MyGroupNorm?
    public var normInT: MyGroupNorm?

    let numLayers: Int
    let classicParity: Int
    let maxPeriod: Float
    let weightPosEmbed: Float

    public init(dim: Int, hiddenScale: Float = 4.0, numHeads: Int = 8,
                numLayers: Int = 6, crossFirst: Bool = false,
                normIn: Bool = true, groupNorm: Int = 0,
                normFirst: Bool = false, normOut: Bool = false,
                maxPeriod: Float = 10000.0, layerScale: Bool = false,
                useGelu: Bool = true, weightPosEmbed: Float = 1.0) {
        self.numLayers = numLayers
        self.classicParity = crossFirst ? 1 : 0
        self.maxPeriod = maxPeriod
        self.weightPosEmbed = weightPosEmbed

        let hiddenDim = Int(Float(dim) * hiddenScale)

        if normIn {
            self.normIn = MyGroupNorm(numGroups: 1, numChannels: dim, layerNormMode: true)
            self.normInT = MyGroupNorm(numGroups: 1, numChannels: dim, layerNormMode: true)
        }

        var classic = [MyTransformerEncoderLayer]()
        var classicT = [MyTransformerEncoderLayer]()
        var cross = [CrossTransformerEncoderLayer]()
        var crossT = [CrossTransformerEncoderLayer]()

        for idx in 0..<numLayers {
            if idx % 2 == (crossFirst ? 1 : 0) {
                classic.append(MyTransformerEncoderLayer(
                    dModel: dim, nHead: numHeads, dimFeedforward: hiddenDim,
                    useGelu: useGelu, groupNormGroups: groupNorm,
                    normFirst: normFirst, normOut: normOut, layerScale: layerScale))
                classicT.append(MyTransformerEncoderLayer(
                    dModel: dim, nHead: numHeads, dimFeedforward: hiddenDim,
                    useGelu: useGelu, groupNormGroups: groupNorm,
                    normFirst: normFirst, normOut: normOut, layerScale: layerScale))
            } else {
                cross.append(CrossTransformerEncoderLayer(
                    dModel: dim, nHead: numHeads, dimFeedforward: hiddenDim,
                    useGelu: useGelu, groupNormGroups: groupNorm,
                    normFirst: normFirst, normOut: normOut, layerScale: layerScale))
                crossT.append(CrossTransformerEncoderLayer(
                    dModel: dim, nHead: numHeads, dimFeedforward: hiddenDim,
                    useGelu: useGelu, groupNormGroups: groupNorm,
                    normFirst: normFirst, normOut: normOut, layerScale: layerScale))
            }
        }
        self.classicLayers = classic
        self.classicLayersT = classicT
        self.crossLayers = cross
        self.crossLayersT = crossT
    }

    public func forward(_ x: MLXArray, _ xt: MLXArray) -> (MLXArray, MLXArray) {
        let B = x.shape[0], C = x.shape[1], Fr = x.shape[2], T1 = x.shape[3]

        // 2D positional embedding for freq branch
        let posEmb2D = create2DSinEmbedding(dModel: C, height: Fr, width: T1,
                                             maxPeriod: maxPeriod)
        // (1, C, Fr, T1) -> (1, C, T1, Fr) -> (1, C, T1*Fr) -> (1, T1*Fr, C)
        let posEmb2DFlat = posEmb2D.transposed(0, 1, 3, 2)
            .reshaped(1, C, T1 * Fr).transposed(0, 2, 1)

        // (B, C, Fr, T1) -> (B, C, T1, Fr) -> (B, C, T1*Fr) -> (B, T1*Fr, C)
        var xFlat = x.transposed(0, 1, 3, 2).reshaped(B, C, T1 * Fr).transposed(0, 2, 1)

        if let ni = normIn { xFlat = ni.forward(xFlat) }
        xFlat = xFlat + weightPosEmbed * posEmb2DFlat

        // Time branch
        let T2 = xt.shape[2]
        var xtFlat = xt.transposed(0, 2, 1)

        let posEmb = createSinEmbedding(length: T2, dim: C, maxPeriod: maxPeriod)
            .transposed(1, 0, 2)  // (1, T2, C)

        if let niT = normInT { xtFlat = niT.forward(xtFlat) }
        xtFlat = xtFlat + weightPosEmbed * posEmb

        // Apply alternating layers
        var classicIdx = 0, crossIdx = 0
        for idx in 0..<numLayers {
            if idx % 2 == classicParity {
                xFlat = classicLayers[classicIdx].forward(xFlat)
                xtFlat = classicLayersT[classicIdx].forward(xtFlat)
                classicIdx += 1
            } else {
                let oldX = xFlat
                xFlat = crossLayers[crossIdx].forward(xFlat, xtFlat)
                xtFlat = crossLayersT[crossIdx].forward(xtFlat, oldX)
                crossIdx += 1
            }
        }

        // (B, T1*Fr, C) -> (B, C, T1*Fr) -> (B, C, T1, Fr) -> (B, C, Fr, T1)
        let xOut = xFlat.transposed(0, 2, 1).reshaped(B, C, T1, Fr).transposed(0, 1, 3, 2)
        let xtOut = xtFlat.transposed(0, 2, 1)

        return (xOut, xtOut)
    }
}
