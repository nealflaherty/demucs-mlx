#include "utils.hpp"
#include <cmath>

namespace demucs {
namespace utils {

namespace mx = mlx::core;

// GELU activation
mx::array gelu(const mx::array& x) {
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    
    auto x_cubed = x * x * x;
    auto inner = x + mx::array(coeff) * x_cubed;
    auto tanh_arg = mx::array(sqrt_2_over_pi) * inner;
    auto tanh_out = mx::tanh(tanh_arg);
    auto one_plus_tanh = mx::array(1.0f) + tanh_out;
    auto half_x = mx::array(0.5f) * x;
    
    return half_x * one_plus_tanh;
}

// GLU (Gated Linear Unit): splits input in half along axis and applies a * sigmoid(b)
mx::array glu(const mx::array& x, int axis) {
    auto splits = mx::split(x, 2, axis);
    auto a = splits[0];
    auto b = splits[1];
    return a * mx::sigmoid(b);
}

// Conv1d with PyTorch format conversion
mx::array conv1d(const mx::array& input, const mx::array& weight, const mx::array& bias,
                 int stride, int padding, int dilation, int groups) {
    // PyTorch format: input (N, C, L), weight (O, I, K)
    // MLX format: input (N, L, C), weight (O, K, I)
    
    // Transpose input from (N, C, L) to (N, L, C)
    auto input_t = mx::transpose(input, {0, 2, 1});
    
    // Transpose weight from (O, I, K) to (O, K, I)
    auto weight_t = mx::transpose(weight, {0, 2, 1});
    
    auto result = mx::conv1d(input_t, weight_t, stride, padding, dilation, groups);
    
    if (bias.size() > 0) {
        // Add bias: reshape to (1, 1, out_channels) for broadcasting
        auto bias_reshaped = mx::reshape(bias, {1, 1, -1});
        result = result + bias_reshaped;
    }
    
    // Transpose result back from (N, L, C) to (N, C, L)
    result = mx::transpose(result, {0, 2, 1});
    
    return result;
}

// ConvTranspose1d with PyTorch format conversion
mx::array conv_transpose1d(const mx::array& input, const mx::array& weight, const mx::array& bias,
                           int stride, int padding, int output_padding, int groups, int dilation) {
    // PyTorch format: input (N, C, L), weight (C_in, C_out, K)
    // MLX format: input (N, L, C), weight (C_out, K, C_in)
    
    // Transpose input from (N, C, L) to (N, L, C)
    auto input_t = mx::transpose(input, {0, 2, 1});
    
    // Transpose weight from (C_in, C_out, K) to (C_out, K, C_in)
    auto weight_t = mx::transpose(weight, {1, 2, 0});
    
    // MLX conv_transpose1d signature: (input, weight, stride, padding, dilation, output_padding, groups)
    // Our wrapper signature:          (input, weight, bias, stride, padding, output_padding, groups, dilation)
    // Note the different parameter order!
    auto result = mx::conv_transpose1d(input_t, weight_t, stride, padding, dilation, output_padding, groups);
    
    if (bias.size() > 0) {
        // Add bias: reshape to (1, 1, out_channels) for broadcasting
        auto bias_reshaped = mx::reshape(bias, {1, 1, -1});
        result = result + bias_reshaped;
    }
    
    // Transpose result back from (N, L, C) to (N, C, L)
    result = mx::transpose(result, {0, 2, 1});
    
    return result;
}

// GroupNorm (1 group = LayerNorm-like)
mx::array group_norm(const mx::array& x, const mx::array& weight, const mx::array& bias,
                     int num_groups, float eps) {
    // Input shape: (N, C, L) for 1D or (N, C, H, W) for 2D
    auto shape = x.shape();
    int N = shape[0];
    int C = shape[1];
    
    // If no weights, return identity
    if (weight.size() == 0 || bias.size() == 0) {
        return x;
    }
    
    // Reshape to (N, num_groups, C/num_groups, ...)
    int channels_per_group = C / num_groups;
    std::vector<int> group_shape_vec = {N, num_groups, channels_per_group};
    for (size_t i = 2; i < shape.size(); ++i) {
        group_shape_vec.push_back(shape[i]);
    }
    
    // Convert std::vector to Shape using iterator constructor
    auto x_grouped = mx::reshape(x, mx::Shape(group_shape_vec.begin(), group_shape_vec.end()));
    
    // Compute mean and variance over (C/num_groups, ...) dimensions
    std::vector<int> reduce_axes;
    for (size_t i = 2; i < group_shape_vec.size(); ++i) {
        reduce_axes.push_back(i);
    }
    
    auto mean = mx::mean(x_grouped, reduce_axes, /* keepdims= */ true);
    auto var = mx::var(x_grouped, reduce_axes, /* keepdims= */ true);
    
    // Normalize
    auto normalized = (x_grouped - mean) / mx::sqrt(var + mx::array(eps));
    
    // Reshape back to original shape
    normalized = mx::reshape(normalized, shape);
    
    // Apply affine transform
    std::vector<int> broadcast_shape_vec = {1, C};
    for (size_t i = 2; i < shape.size(); ++i) {
        broadcast_shape_vec.push_back(1);
    }
    
    // Convert std::vector to Shape using iterator constructor
    auto weight_reshaped = mx::reshape(weight, mx::Shape(broadcast_shape_vec.begin(), broadcast_shape_vec.end()));
    auto bias_reshaped = mx::reshape(bias, mx::Shape(broadcast_shape_vec.begin(), broadcast_shape_vec.end()));
    
    return normalized * weight_reshaped + bias_reshaped;
}

// Conv2d with PyTorch format conversion
mx::array conv2d(const mx::array& input, const mx::array& weight, const mx::array& bias,
                 std::pair<int,int> stride, std::pair<int,int> padding,
                 std::pair<int,int> dilation, int groups) {
    // PyTorch format: input (N, C, H, W), weight (O, I, kH, kW)
    // MLX format: input (N, H, W, C), weight (O, kH, kW, I)
    
    // Transpose input from (N, C, H, W) to (N, H, W, C)
    auto input_t = mx::transpose(input, {0, 2, 3, 1});
    
    // Transpose weight from (O, I, kH, kW) to (O, kH, kW, I)
    auto weight_t = mx::transpose(weight, {0, 2, 3, 1});
    
    auto result = mx::conv2d(input_t, weight_t, stride, padding, dilation, groups);
    
    if (bias.size() > 0) {
        // Add bias: reshape to (1, 1, 1, out_channels) for broadcasting
        auto bias_reshaped = mx::reshape(bias, {1, 1, 1, -1});
        result = result + bias_reshaped;
    }
    
    // Transpose result back from (N, H, W, C) to (N, C, H, W)
    result = mx::transpose(result, {0, 3, 1, 2});
    
    return result;
}

// ConvTranspose2d with PyTorch format conversion
mx::array conv_transpose2d(const mx::array& input, const mx::array& weight, const mx::array& bias,
                           std::pair<int,int> stride, std::pair<int,int> padding,
                           std::pair<int,int> output_padding, int groups,
                           std::pair<int,int> dilation) {
    // PyTorch format: input (N, C, H, W), weight (C_in, C_out, kH, kW)
    // MLX format: input (N, H, W, C), weight (C_out, kH, kW, C_in)
    
    // Transpose input from (N, C, H, W) to (N, H, W, C)
    auto input_t = mx::transpose(input, {0, 2, 3, 1});
    
    // Transpose weight from (C_in, C_out, kH, kW) to (C_out, kH, kW, C_in)
    auto weight_t = mx::transpose(weight, {1, 2, 3, 0});
    
    auto result = mx::conv_transpose2d(input_t, weight_t, stride, padding, dilation, output_padding, groups);
    
    if (bias.size() > 0) {
        // Add bias: reshape to (1, 1, 1, out_channels) for broadcasting
        auto bias_reshaped = mx::reshape(bias, {1, 1, 1, -1});
        result = result + bias_reshaped;
    }
    
    // Transpose result back from (N, H, W, C) to (N, C, H, W)
    result = mx::transpose(result, {0, 3, 1, 2});
    
    return result;
}

} // namespace utils
} // namespace demucs