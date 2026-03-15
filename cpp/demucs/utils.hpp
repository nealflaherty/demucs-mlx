#pragma once

#include <mlx/mlx.h>
#include <utility>

namespace demucs {
namespace utils {

namespace mx = mlx::core;

// GELU activation
mx::array gelu(const mx::array& x);

// GLU (Gated Linear Unit): splits input in half along axis and applies a * sigmoid(b)
mx::array glu(const mx::array& x, int axis = 1);

// Conv1d with PyTorch format conversion
mx::array conv1d(const mx::array& input, const mx::array& weight, const mx::array& bias,
                 int stride = 1, int padding = 0, int dilation = 1, int groups = 1);

// ConvTranspose1d with PyTorch format conversion
mx::array conv_transpose1d(const mx::array& input, const mx::array& weight, const mx::array& bias,
                           int stride = 1, int padding = 0, int output_padding = 0, int groups = 1, int dilation = 1);

// GroupNorm (1 group = LayerNorm-like)
mx::array group_norm(const mx::array& x, const mx::array& weight, const mx::array& bias,
                     int num_groups, float eps = 1e-5f);

// Conv2d with PyTorch format conversion
// PyTorch: input (N, C, H, W), weight (O, I, kH, kW)
// For freq branch: kernel=[K,1], stride=[S,1], pad=[P,0]
mx::array conv2d(const mx::array& input, const mx::array& weight, const mx::array& bias,
                 std::pair<int,int> stride = {1,1}, std::pair<int,int> padding = {0,0},
                 std::pair<int,int> dilation = {1,1}, int groups = 1);

// ConvTranspose2d with PyTorch format conversion
mx::array conv_transpose2d(const mx::array& input, const mx::array& weight, const mx::array& bias,
                           std::pair<int,int> stride = {1,1}, std::pair<int,int> padding = {0,0},
                           std::pair<int,int> output_padding = {0,0}, int groups = 1,
                           std::pair<int,int> dilation = {1,1});

} // namespace utils
} // namespace demucs
