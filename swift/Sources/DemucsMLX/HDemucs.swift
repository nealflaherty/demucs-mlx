// HDemucs.swift - pad1d, ScaledEmbedding, HEncLayer, HDecLayer
// 1:1 port of cpp/demucs/hdemucs.cpp (encoder/decoder layers)

import MLX

// MARK: - Reflect padding helper

/// Manual reflect padding for 1D since MLX doesn't support reflect mode.
private func reflectPad1D(_ x: MLXArray, padLeft: Int, padRight: Int) -> MLXArray {
    let shape = x.shape
    let length = shape.last!

    if padLeft == 0 && padRight == 0 { return x }

    var parts = [MLXArray]()

    // Left reflect: [x[padLeft], x[padLeft-1], ..., x[1]]
    if padLeft > 0 {
        let sliceLen = min(padLeft, length - 1)
        var leftParts = [MLXArray]()
        for i in Swift.stride(from: sliceLen, through: 1, by: -1) {
            leftParts.append(sliceAxis(x, axis: -1, start: i, end: i + 1))
        }
        if !leftParts.isEmpty {
            parts.append(concatenated(leftParts, axis: -1))
        }
    }

    parts.append(x)

    // Right reflect: [x[-2], x[-3], ..., x[-(padRight+1)]]
    if padRight > 0 {
        let sliceLen = min(padRight, length - 1)
        var rightParts = [MLXArray]()
        for i in Swift.stride(from: length - 2, through: length - 1 - sliceLen, by: -1) {
            rightParts.append(sliceAxis(x, axis: -1, start: i, end: i + 1))
        }
        if !rightParts.isEmpty {
            parts.append(concatenated(rightParts, axis: -1))
        }
    }

    return concatenated(parts, axis: -1)
}

// MARK: - pad1d

/// Pad a 1D tensor with reflection or constant padding.
/// Matches pad1d() from hdemucs.py / hdemucs.cpp.
func pad1d(_ x: MLXArray, paddingLeft: Int, paddingRight: Int,
                   mode: String = "constant", value: Float = 0.0) -> MLXArray {
    let shape = x.shape
    let length = shape.last!
    var xPadded = x
    var pLeft = paddingLeft
    var pRight = paddingRight

    if mode == "reflect" {
        let maxPad = max(pLeft, pRight)
        if length <= maxPad {
            let extraPad = maxPad - length + 1
            let extraPadRight = min(pRight, extraPad)
            let extraPadLeft = extraPad - extraPadRight

            var padWidths = [(Int, Int)](repeating: (0, 0), count: shape.count)
            padWidths[shape.count - 1] = (extraPadLeft, extraPadRight)
            xPadded = MLX.padded(xPadded, widths: padWidths.map { .init($0) },
                                  value: MLXArray(value))
            pLeft -= extraPadLeft
            pRight -= extraPadRight
        }

        if pLeft > 0 || pRight > 0 {
            xPadded = reflectPad1D(xPadded, padLeft: pLeft, padRight: pRight)
        }
    } else {
        // constant padding
        var padWidths = [(Int, Int)](repeating: (0, 0), count: shape.count)
        padWidths[shape.count - 1] = (pLeft, pRight)
        xPadded = MLX.padded(x, widths: padWidths.map { .init($0) },
                              value: MLXArray(value))
    }

    return xPadded
}


// MARK: - ScaledEmbedding

/// Scaled embedding layer matching C++ ScaledEmbedding.
struct ScaledEmbedding {
    var embeddingWeight: MLXArray
    let scale: Float

    init(numEmbeddings: Int, embeddingDim: Int,
                scale: Float = 10.0, smooth: Bool = false) {
        var w = MLXRandom.normal([numEmbeddings, embeddingDim])
        if smooth {
            w = MLX.cumsum(w, axis: 0)
            let n = MLXArray(Array(Swift.stride(from: Float(1), through: Float(numEmbeddings), by: 1)))
            let sqrtN = MLX.sqrt(n).reshaped(numEmbeddings, 1)
            w = w / sqrtN
        }
        self.embeddingWeight = w / MLXArray(scale)
        self.scale = scale
    }

    func forward(_ x: MLXArray) -> MLXArray {
        let emb = take(embeddingWeight, x, axis: 0)
        return emb * MLXArray(scale)
    }

    func weight() -> MLXArray {
        return embeddingWeight * MLXArray(scale)
    }
}


// MARK: - HEncLayer

/// Encoder layer matching C++ HEncLayer.
struct HEncLayer {
    var convWeight: MLXArray
    var convBias: MLXArray
    var norm1Weight: MLXArray
    var norm1Bias: MLXArray
    var rewriteWeight: MLXArray
    var rewriteBias: MLXArray
    var norm2Weight: MLXArray
    var norm2Bias: MLXArray
    var dconv: DConvBlock?

    let freq: Bool
    let kernelSize: Int
    let stride_: Int
    let empty: Bool
    let norm: Bool
    let pad: Int
    let context: Int
    let rewrite: Bool
    let normGroups: Int

    init(chin: Int, chout: Int, kernelSize: Int = 8, stride: Int = 4,
                normGroups: Int = 1, empty: Bool = false, freq: Bool = false,
                dconvEnabled: Bool = true, norm: Bool = true, context: Int = 0,
                pad: Bool = true, rewrite: Bool = true,
                dconvComp: Float = 8.0, dconvInit: Float = 1e-3) {
        self.freq = freq
        self.kernelSize = kernelSize
        self.stride_ = stride
        self.empty = empty
        self.norm = norm
        self.pad = pad ? kernelSize / 4 : 0
        self.context = context
        self.rewrite = rewrite
        self.normGroups = normGroups

        if freq {
            self.convWeight = MLXArray.zeros([chout, chin, kernelSize, 1])
        } else {
            self.convWeight = MLXArray.zeros([chout, chin, kernelSize])
        }
        self.convBias = MLXArray.zeros([chout])
        self.norm1Weight = MLXArray.zeros([chout])
        self.norm1Bias = MLXArray.zeros([chout])

        if freq {
            self.rewriteWeight = MLXArray.zeros([2 * chout, chout,
                                                  1 + 2 * context, 1 + 2 * context])
        } else {
            self.rewriteWeight = MLXArray.zeros([2 * chout, chout, 1 + 2 * context])
        }
        self.rewriteBias = MLXArray.zeros([2 * chout])
        self.norm2Weight = MLXArray.zeros([2 * chout])
        self.norm2Bias = MLXArray.zeros([2 * chout])

        if dconvEnabled && !empty {
            self.dconv = DConvBlock(channels: chout, compress: dconvComp, depth: 2,
                                    initScale: dconvInit, norm: norm, useGelu: true,
                                    kernel: 3, dilate: true)
        }
    }

    func forward(_ x: MLXArray, inject: MLXArray? = nil) -> MLXArray {
        var y = x

        // Flatten freq dims for time branch
        if !freq && y.ndim == 4 {
            let B = y.shape[0], C = y.shape[1], Fr = y.shape[2], T = y.shape[3]
            y = y.reshaped(B, C * Fr, T)
        }

        // Pad to stride alignment
        if !freq {
            let le = y.shape[y.ndim - 1]
            if le % stride_ != 0 {
                let padAmount = stride_ - (le % stride_)
                y = pad1d(y, paddingLeft: 0, paddingRight: padAmount, mode: "constant")
            }
        }

        // Conv
        if freq {
            y = conv2d(y, weight: convWeight, bias: convBias,
                       stride: (stride_, 1), padding: (self.pad, 0))
        } else {
            y = conv1d(y, weight: convWeight, bias: convBias,
                       stride: stride_, padding: self.pad)
        }

        if empty { return y }

        // Inject
        if var inj = inject {
            if inj.ndim == 3 && y.ndim == 4 {
                inj = expandedDimensions(inj, axis: 2)
            }
            y = y + inj
        }

        // Norm + GELU
        if norm {
            y = groupNorm(y, weight: norm1Weight, bias: norm1Bias, numGroups: normGroups)
        }
        y = gelu(y)

        // DConv
        if let dc = dconv {
            if freq {
                let B = y.shape[0], C = y.shape[1], Fr = y.shape[2], T = y.shape[3]
                y = y.transposed(0, 2, 1, 3).reshaped(B * Fr, C, T)
                y = dc.forward(y)
                y = y.reshaped(B, Fr, C, T).transposed(0, 2, 1, 3)
            } else {
                y = dc.forward(y)
            }
        }

        // Rewrite + GLU
        var z = y
        if rewrite {
            if freq {
                z = conv2d(y, weight: rewriteWeight, bias: rewriteBias,
                           stride: (1, 1), padding: (context, context))
            } else {
                z = conv1d(y, weight: rewriteWeight, bias: rewriteBias,
                           stride: 1, padding: context)
            }
            if norm {
                z = groupNorm(z, weight: norm2Weight, bias: norm2Bias, numGroups: normGroups)
            }
            z = glu(z, axis: 1)
        }

        return z
    }
}


// MARK: - HDecLayer

/// Decoder layer matching C++ HDecLayer.
struct HDecLayer {
    var convTrWeight: MLXArray
    var convTrBias: MLXArray
    var norm2Weight: MLXArray
    var norm2Bias: MLXArray
    var rewriteWeight: MLXArray
    var rewriteBias: MLXArray
    var norm1Weight: MLXArray
    var norm1Bias: MLXArray
    var dconv: DConvBlock?

    let last: Bool
    let freq: Bool
    let chin: Int
    let empty: Bool
    let stride_: Int
    let kernelSize: Int
    let norm: Bool
    let contextFreq: Bool
    let context: Int
    let rewrite: Bool
    let normGroups: Int
    let pad: Int

    init(chin: Int, chout: Int, last: Bool = false, kernelSize: Int = 8,
                stride: Int = 4, normGroups: Int = 1, empty: Bool = false,
                freq: Bool = false, dconvEnabled: Bool = true, norm: Bool = true,
                context: Int = 1, pad: Bool = true, contextFreq: Bool = true,
                rewrite: Bool = true, dconvComp: Float = 8.0, dconvInit: Float = 1e-3) {
        self.last = last
        self.freq = freq
        self.chin = chin
        self.empty = empty
        self.stride_ = stride
        self.kernelSize = kernelSize
        self.norm = norm
        self.contextFreq = contextFreq
        self.context = context
        self.rewrite = rewrite
        self.normGroups = normGroups
        self.pad = pad ? kernelSize / 4 : 0

        if freq {
            self.convTrWeight = MLXArray.zeros([chin, chout, kernelSize, 1])
        } else {
            self.convTrWeight = MLXArray.zeros([chin, chout, kernelSize])
        }
        self.convTrBias = MLXArray.zeros([chout])
        self.norm2Weight = MLXArray.zeros([chout])
        self.norm2Bias = MLXArray.zeros([chout])

        // Rewrite weight shape depends on freq and contextFreq
        if freq {
            if contextFreq {
                self.rewriteWeight = MLXArray.zeros([2 * chin, chin,
                                                      1 + 2 * context, 1 + 2 * context])
            } else {
                self.rewriteWeight = MLXArray.zeros([2 * chin, chin, 1, 1 + 2 * context])
            }
        } else {
            self.rewriteWeight = MLXArray.zeros([2 * chin, chin, 1 + 2 * context])
        }
        self.rewriteBias = MLXArray.zeros([2 * chin])
        self.norm1Weight = MLXArray.zeros([2 * chin])
        self.norm1Bias = MLXArray.zeros([2 * chin])

        if dconvEnabled && !empty {
            self.dconv = DConvBlock(channels: chin, compress: dconvComp, depth: 2,
                                    initScale: dconvInit, norm: norm, useGelu: true,
                                    kernel: 3, dilate: true)
        }
    }

    /// Returns (output, pre-conv-transpose tensor for skip connections).
    func forward(_ x: MLXArray, skip: MLXArray, length: Int) -> (MLXArray, MLXArray) {
        var y = x

        // Reshape for freq branch
        if freq && y.ndim == 3 {
            let B = y.shape[0], T = y.shape[2]
            y = y.reshaped(B, chin, -1, T)
        }

        if !empty {
            let xWithSkip = y + skip

            // Rewrite + GLU
            if rewrite {
                var rewriteOut: MLXArray
                if freq {
                    if contextFreq {
                        rewriteOut = conv2d(xWithSkip, weight: rewriteWeight, bias: rewriteBias,
                                            stride: (1, 1), padding: (context, context))
                    } else {
                        rewriteOut = conv2d(xWithSkip, weight: rewriteWeight, bias: rewriteBias,
                                            stride: (1, 1), padding: (0, context))
                    }
                } else {
                    rewriteOut = conv1d(xWithSkip, weight: rewriteWeight, bias: rewriteBias,
                                        stride: 1, padding: context)
                }
                if norm {
                    rewriteOut = groupNorm(rewriteOut, weight: norm1Weight, bias: norm1Bias,
                                           numGroups: normGroups)
                }
                y = glu(rewriteOut, axis: 1)
            } else {
                y = xWithSkip
            }

            // DConv
            if let dc = dconv {
                if freq {
                    let B = y.shape[0], C = y.shape[1], Fr = y.shape[2], T = y.shape[3]
                    y = y.transposed(0, 2, 1, 3).reshaped(B * Fr, C, T)
                    y = dc.forward(y)
                    y = y.reshaped(B, Fr, C, T).transposed(0, 2, 1, 3)
                } else {
                    y = dc.forward(y)
                }
            }
        }

        // Transposed conv
        var z: MLXArray
        if freq {
            z = convTranspose2d(y, weight: convTrWeight, bias: convTrBias,
                                stride: (stride_, 1))
        } else {
            z = convTranspose1d(y, weight: convTrWeight, bias: convTrBias,
                                stride: stride_)
        }

        if norm {
            z = groupNorm(z, weight: norm2Weight, bias: norm2Bias, numGroups: normGroups)
        }

        // Trim
        if freq {
            if pad > 0 {
                z = sliceAxis(z, axis: -2, start: pad, end: z.shape[z.ndim - 2] - pad)
            }
        } else {
            z = sliceAxis(z, axis: -1, start: pad, end: pad + length)
        }

        if !last {
            z = gelu(z)
        }

        return (z, y)
    }
}
