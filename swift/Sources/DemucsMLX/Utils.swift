// Utils.swift — Conv wrappers, GroupNorm, GELU, GLU
// 1:1 port of cpp/demucs/utils.cpp
//
// All conv wrappers handle PyTorch↔MLX layout transposition:
//   PyTorch: channels-first (N, C, L) / (N, C, H, W)
//   MLX:     channels-last  (N, L, C) / (N, H, W, C)

import MLX

// MARK: - Slice helper

/// Slice an array along a specific axis from `start` to `end`.
/// This replaces `x[startVec, endVec]` style indexing which MLX Swift doesn't support.
public func sliceAxis(_ x: MLXArray, axis: Int, start: Int, end: Int) -> MLXArray {
    let ax = axis < 0 ? x.ndim + axis : axis
    var indices = [MLXArrayIndex]()
    for i in 0..<x.ndim {
        if i == ax {
            indices.append(start..<end)
        } else {
            indices.append(0...)
        }
    }
    return x[indices]
}

// MARK: - GELU activation (tanh approximation)

/// GELU activation using the tanh approximation matching PyTorch.
/// `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`
public func gelu(_ x: MLXArray) -> MLXArray {
    let sqrt2OverPi: Float = 0.7978845608
    let coeff: Float = 0.044715

    let xCubed = x * x * x
    let inner = x + MLXArray(coeff) * xCubed
    let tanhArg = MLXArray(sqrt2OverPi) * inner
    let tanhOut = MLX.tanh(tanhArg)
    let onePlusTanh = MLXArray(Float(1.0)) + tanhOut
    let halfX = MLXArray(Float(0.5)) * x

    return halfX * onePlusTanh
}

// MARK: - GLU (Gated Linear Unit)

/// Splits input in half along `axis` and returns `a * sigmoid(b)`.
public func glu(_ x: MLXArray, axis: Int = 1) -> MLXArray {
    let splits = MLX.split(x, parts: 2, axis: axis)
    let a = splits[0]
    let b = splits[1]
    return a * MLX.sigmoid(b)
}

// MARK: - Conv1d wrapper

/// Conv1d with PyTorch format conversion.
/// Input: (N, C, L), Weight: (O, I, K) → MLX layout → result back to (N, C, L).
public func conv1d(
    _ input: MLXArray, weight: MLXArray, bias: MLXArray? = nil,
    stride: Int = 1, padding: Int = 0, dilation: Int = 1, groups: Int = 1
) -> MLXArray {
    // Transpose input from (N, C, L) to (N, L, C)
    let inputT = input.transposed(0, 2, 1)
    // Transpose weight from (O, I, K) to (O, K, I)
    let weightT = weight.transposed(0, 2, 1)

    var result = MLX.conv1d(inputT, weightT, stride: stride, padding: padding,
                            dilation: dilation, groups: groups)

    if let bias = bias {
        let biasReshaped = bias.reshaped(1, 1, -1)
        result = result + biasReshaped
    }

    // Transpose back from (N, L, C) to (N, C, L)
    return result.transposed(0, 2, 1)
}


// MARK: - ConvTranspose1d wrapper

/// ConvTranspose1d with PyTorch format conversion.
/// Input: (N, C, L), Weight: (C_in, C_out, K) → MLX layout → result back to (N, C, L).
public func convTranspose1d(
    _ input: MLXArray, weight: MLXArray, bias: MLXArray? = nil,
    stride: Int = 1, padding: Int = 0, outputPadding: Int = 0,
    groups: Int = 1, dilation: Int = 1
) -> MLXArray {
    // Transpose input from (N, C, L) to (N, L, C)
    let inputT = input.transposed(0, 2, 1)
    // Transpose weight from (C_in, C_out, K) to (C_out, K, C_in)
    let weightT = weight.transposed(1, 2, 0)

    var result = MLX.convTransposed1d(inputT, weightT, stride: stride, padding: padding,
                                      dilation: dilation, outputPadding: outputPadding,
                                      groups: groups)

    if let bias = bias {
        let biasReshaped = bias.reshaped(1, 1, -1)
        result = result + biasReshaped
    }

    // Transpose back from (N, L, C) to (N, C, L)
    return result.transposed(0, 2, 1)
}

// MARK: - Conv2d wrapper

/// Conv2d with PyTorch format conversion.
/// Input: (N, C, H, W), Weight: (O, I, kH, kW) → MLX layout → result back to (N, C, H, W).
public func conv2d(
    _ input: MLXArray, weight: MLXArray, bias: MLXArray? = nil,
    stride: (Int, Int) = (1, 1), padding: (Int, Int) = (0, 0),
    dilation: (Int, Int) = (1, 1), groups: Int = 1
) -> MLXArray {
    // Transpose input from (N, C, H, W) to (N, H, W, C)
    let inputT = input.transposed(0, 2, 3, 1)
    // Transpose weight from (O, I, kH, kW) to (O, kH, kW, I)
    let weightT = weight.transposed(0, 2, 3, 1)

    var result = MLX.conv2d(inputT, weightT,
                            stride: [stride.0, stride.1],
                            padding: [padding.0, padding.1],
                            dilation: [dilation.0, dilation.1],
                            groups: groups)

    if let bias = bias {
        let biasReshaped = bias.reshaped(1, 1, 1, -1)
        result = result + biasReshaped
    }

    // Transpose back from (N, H, W, C) to (N, C, H, W)
    return result.transposed(0, 3, 1, 2)
}

// MARK: - ConvTranspose2d wrapper

/// ConvTranspose2d with PyTorch format conversion.
/// Input: (N, C, H, W), Weight: (C_in, C_out, kH, kW) → MLX layout → result back to (N, C, H, W).
public func convTranspose2d(
    _ input: MLXArray, weight: MLXArray, bias: MLXArray? = nil,
    stride: (Int, Int) = (1, 1), padding: (Int, Int) = (0, 0),
    outputPadding: (Int, Int) = (0, 0), groups: Int = 1,
    dilation: (Int, Int) = (1, 1)
) -> MLXArray {
    // Transpose input from (N, C, H, W) to (N, H, W, C)
    let inputT = input.transposed(0, 2, 3, 1)
    // Transpose weight from (C_in, C_out, kH, kW) to (C_out, kH, kW, C_in)
    let weightT = weight.transposed(1, 2, 3, 0)

    var result = MLX.convTransposed2d(inputT, weightT,
                                      stride: [stride.0, stride.1],
                                      padding: [padding.0, padding.1],
                                      dilation: [dilation.0, dilation.1],
                                      outputPadding: [outputPadding.0, outputPadding.1],
                                      groups: groups)

    if let bias = bias {
        let biasReshaped = bias.reshaped(1, 1, 1, -1)
        result = result + biasReshaped
    }

    // Transpose back from (N, H, W, C) to (N, C, H, W)
    return result.transposed(0, 3, 1, 2)
}

// MARK: - GroupNorm

/// GroupNorm matching PyTorch's nn.GroupNorm.
/// Input shape: (N, C, ...) where C is divisible by numGroups.
public func groupNorm(
    _ x: MLXArray, weight: MLXArray, bias: MLXArray,
    numGroups: Int, eps: Float = 1e-5
) -> MLXArray {
    let shape = x.shape
    let N = shape[0]
    let C = shape[1]

    // If no weights, return identity
    guard weight.size > 0, bias.size > 0 else { return x }

    let channelsPerGroup = C / numGroups

    // Reshape to (N, numGroups, channelsPerGroup, ...)
    var groupShape: [Int] = [N, numGroups, channelsPerGroup]
    for i in 2..<shape.count {
        groupShape.append(shape[i])
    }
    let xGrouped = x.reshaped(groupShape)

    // Reduce axes: everything from axis 2 onward
    let reduceAxes = Array(2..<groupShape.count)

    let mean = xGrouped.mean(axes: reduceAxes, keepDims: true)
    let variance = xGrouped.variance(axes: reduceAxes, keepDims: true)

    // Normalize
    var normalized = (xGrouped - mean) / MLX.sqrt(variance + MLXArray(eps))

    // Reshape back to original shape
    normalized = normalized.reshaped(shape)

    // Apply affine transform: weight and bias shaped as (1, C, 1, 1, ...)
    var broadcastShape: [Int] = [1, C]
    for _ in 2..<shape.count {
        broadcastShape.append(1)
    }
    let weightReshaped = weight.reshaped(broadcastShape)
    let biasReshaped = bias.reshaped(broadcastShape)

    return normalized * weightReshaped + biasReshaped
}
