// DConv.swift — Dilated convolution residual block
// 1:1 port of cpp/demucs/demucs.cpp
//
// Each DConv layer: dilated conv1d → GroupNorm → GELU → 1×1 conv1d → GroupNorm → GLU → LayerScale → residual add

import MLX

// MARK: - LayerScale

/// Rescales residual outputs close to 0 initially.
/// Matches LayerScale from transformer.py / transformer.cpp.
public struct LayerScale {
    public var scale: MLXArray

    let channelLast: Bool

    public init(channels: Int, init: Float = 1e-4, channelLast: Bool = true) {
        self.scale = MLXArray.full([channels], values: MLXArray(`init`))
        self.channelLast = channelLast
    }

    public func forward(_ x: MLXArray) -> MLXArray {
        if channelLast {
            return scale * x
        } else {
            let scaleReshaped = scale.reshaped(-1, 1)
            return scaleReshaped * x
        }
    }
}

// MARK: - DConv Layer

/// A single layer in the DConv residual branch.
public struct DConvLayer {
    // Conv1d: channels → hidden (with dilation)
    public var conv1Weight: MLXArray
    public var conv1Bias: MLXArray
    public var norm1Weight: MLXArray
    public var norm1Bias: MLXArray

    // Conv1d: hidden → 2*channels (1x1)
    public var conv2Weight: MLXArray
    public var conv2Bias: MLXArray
    public var norm2Weight: MLXArray
    public var norm2Bias: MLXArray

    public var layerScale: LayerScale

    let dilation: Int
    let padding: Int

    public init(channels: Int, hidden: Int, kernel: Int, dilation: Int,
                initScale: Float, norm: Bool) {
        self.conv1Weight = MLXArray.zeros([hidden, channels, kernel])
        self.conv1Bias = MLXArray.zeros([hidden])
        self.norm1Weight = MLXArray.zeros([hidden])
        self.norm1Bias = MLXArray.zeros([hidden])
        self.conv2Weight = MLXArray.zeros([2 * channels, hidden, 1])
        self.conv2Bias = MLXArray.zeros([2 * channels])
        self.norm2Weight = MLXArray.zeros([2 * channels])
        self.norm2Bias = MLXArray.zeros([2 * channels])
        self.layerScale = LayerScale(channels: channels, init: initScale, channelLast: false)
        self.dilation = dilation
        self.padding = dilation * (kernel / 2)
    }

    public func forward(_ x: MLXArray, useGelu: Bool) -> MLXArray {
        // Conv1d with dilation
        var y = conv1d(x, weight: conv1Weight, bias: conv1Bias,
                       stride: 1, padding: padding, dilation: dilation)

        // GroupNorm (1 group = LayerNorm-like)
        y = groupNorm(y, weight: norm1Weight, bias: norm1Bias, numGroups: 1)

        // Activation
        if useGelu {
            y = gelu(y)
        } else {
            y = MLX.maximum(y, MLXArray(Float(0)))
        }

        // 1x1 conv
        y = conv1d(y, weight: conv2Weight, bias: conv2Bias)

        // GroupNorm
        y = groupNorm(y, weight: norm2Weight, bias: norm2Bias, numGroups: 1)

        // GLU
        y = glu(y, axis: 1)

        // LayerScale
        y = layerScale.forward(y)

        return y
    }
}

// MARK: - DConv Block

/// Dilated convolution residual block.
/// Matches DConv class from demucs.py / demucs.cpp.
public struct DConvBlock {
    public var layers: [DConvLayer]
    let useGelu: Bool

    public init(channels: Int, compress: Float = 4.0, depth: Int = 2,
                initScale: Float = 1e-4, norm: Bool = true, useGelu: Bool = true,
                kernel: Int = 3, dilate: Bool = true) {
        precondition(kernel % 2 == 1, "DConv kernel size must be odd")

        let useDilation = depth > 0
        let actualDepth = abs(depth)
        let hidden = Int(Float(channels) / compress)

        var layers = [DConvLayer]()
        for d in 0..<actualDepth {
            let dilation = useDilation ? (1 << d) : 1
            layers.append(DConvLayer(channels: channels, hidden: hidden, kernel: kernel,
                                     dilation: dilation, initScale: initScale, norm: norm))
        }
        self.layers = layers
        self.useGelu = useGelu
    }

    public func forward(_ x: MLXArray) -> MLXArray {
        var y = x
        for layer in layers {
            let residual = layer.forward(y, useGelu: useGelu)
            y = y + residual
        }
        return y
    }
}
