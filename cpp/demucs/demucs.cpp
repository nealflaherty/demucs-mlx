#include "demucs.hpp"
#include "utils.hpp"
#include <cmath>
#include <stdexcept>

namespace demucs {

namespace mx = mlx::core;

// ============================================================================
// DConv::Layer Implementation
// ============================================================================

DConv::Layer::Layer(int channels, int hidden, int kernel, int dilation, 
                    float init, bool norm)
    : conv1_weight(mx::zeros({hidden, channels, kernel})),
      conv1_bias(mx::zeros({hidden})),
      norm1_weight(mx::zeros({hidden})),
      norm1_bias(mx::zeros({hidden})),
      conv2_weight(mx::zeros({2 * channels, hidden, 1})),
      conv2_bias(mx::zeros({2 * channels})),
      norm2_weight(mx::zeros({2 * channels})),
      norm2_bias(mx::zeros({2 * channels})),
      layer_scale(channels, init, false),
      dilation(dilation),
      kernel_size(kernel),
      padding(dilation * (kernel / 2)) {
}

mx::array DConv::Layer::forward(const mx::array& x, bool use_gelu) const {
    // Python DConv layer structure:
    // Conv1d(channels, hidden, kernel, dilation=dilation, padding=padding)
    // GroupNorm(1, hidden)
    // GELU() or ReLU()
    // Conv1d(hidden, 2*channels, 1)
    // GroupNorm(1, 2*channels)
    // GLU(1)
    // LayerScale(channels, init)
    
    // First conv with dilation
    auto y = utils::conv1d(x, conv1_weight, conv1_bias, 1, padding, dilation);
    
    // GroupNorm (1 group = LayerNorm-like)
    if (norm1_weight.size() > 0) {
        y = utils::group_norm(y, norm1_weight, norm1_bias, 1);
    }
    
    // Activation
    if (use_gelu) {
        y = utils::gelu(y);
    } else {
        y = mx::maximum(y, mx::array(0.0f));  // ReLU
    }
    
    // Second conv (1x1)
    y = utils::conv1d(y, conv2_weight, conv2_bias, 1, 0, 1);
    
    // GroupNorm
    if (norm2_weight.size() > 0) {
        y = utils::group_norm(y, norm2_weight, norm2_bias, 1);
    }
    
    // GLU
    y = utils::glu(y, 1);
    
    // LayerScale
    y = layer_scale.forward(y);
    
    return y;
}

// ============================================================================
// DConv Implementation
// ============================================================================

DConv::DConv(int channels, float compress, int depth, float init,
             bool norm, bool attn, int heads, int ndecay,
             bool lstm, bool gelu, int kernel, bool dilate)
    : channels_(channels), depth_(abs(depth)), gelu_(gelu) {
    
    // Python: assert kernel % 2 == 1
    if (kernel % 2 != 1) {
        throw std::runtime_error("DConv kernel size must be odd");
    }
    
    // Python: dilate = depth > 0
    bool use_dilation = depth > 0;
    
    // Python: hidden = int(channels / compress)
    int hidden = static_cast<int>(channels / compress);
    
    // Python: for d in range(self.depth):
    for (int d = 0; d < depth_; ++d) {
        // Python: dilation = 2 ** d if dilate else 1
        int dilation = use_dilation ? (1 << d) : 1;  // 2^d
        
        // Create layer
        layers_.emplace_back(channels, hidden, kernel, dilation, init, norm);
        
        // TODO: Add LSTM and LocalState support when needed
        if (attn || lstm) {
            throw std::runtime_error("DConv: LSTM and LocalState not yet implemented");
        }
    }
}

mx::array DConv::forward(const mx::array& x) const {
    // Python: for layer in self.layers:
    //             x = x + layer(x)
    //         return x
    
    auto y = x;
    for (const auto& layer : layers_) {
        auto residual = layer.forward(y, gelu_);
        y = y + residual;
    }
    
    return y;
}

} // namespace demucs
