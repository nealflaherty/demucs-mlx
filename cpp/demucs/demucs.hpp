#pragma once

#include <mlx/mlx.h>
#include <vector>
#include <memory>
#include "transformer.hpp"  // For LayerScale

namespace demucs {

namespace mx = mlx::core;

/**
 * Dilated convolution residual branch
 * Matches: demucs/demucs.py class DConv
 * 
 * This alternates dilated convolutions, potentially with LSTMs and attention.
 * For now, we implement the basic version without LSTM/attention.
 */
class DConv {
public:
    /**
     * Args:
     *   channels: input/output channels for residual branch
     *   compress: amount of channel compression inside the branch
     *   depth: number of layers in the residual branch
     *   init: initial scale for LayerScale
     *   norm: use GroupNorm
     *   attn: use LocalAttention (not implemented yet)
     *   heads: number of heads for LocalAttention (not implemented yet)
     *   ndecay: number of decay controls (not implemented yet)
     *   lstm: use LSTM (not implemented yet)
     *   gelu: use GELU activation (vs ReLU)
     *   kernel: kernel size for dilated convolutions
     *   dilate: if true, use dilation increasing with depth
     */
    DConv(int channels, float compress = 4.0f, int depth = 2, float init = 1e-4f,
          bool norm = true, bool attn = false, int heads = 4, int ndecay = 4,
          bool lstm = false, bool gelu = true, int kernel = 3, bool dilate = true);
    
    mx::array forward(const mx::array& x) const;
    
    // Access to layers for weight loading
    struct Layer {
        // Conv1d: channels → hidden (with dilation)
        mx::array conv1_weight;  // Shape: (hidden, channels, kernel)
        mx::array conv1_bias;    // Shape: (hidden,)
        
        // GroupNorm after conv1 (1 group = LayerNorm-like)
        mx::array norm1_weight;  // Shape: (hidden,)
        mx::array norm1_bias;    // Shape: (hidden,)
        
        // Conv1d: hidden → 2*channels (1x1 conv)
        mx::array conv2_weight;  // Shape: (2*channels, hidden, 1)
        mx::array conv2_bias;    // Shape: (2*channels,)
        
        // GroupNorm after conv2
        mx::array norm2_weight;  // Shape: (2*channels,)
        mx::array norm2_bias;    // Shape: (2*channels,)
        
        // LayerScale
        LayerScale layer_scale;
        
        int dilation;
        int kernel_size;
        int padding;
        
        Layer(int channels, int hidden, int kernel, int dilation, float init, bool norm);
        
        mx::array forward(const mx::array& x, bool use_gelu) const;
    };
    
    std::vector<Layer>& layers() { return layers_; }
    const std::vector<Layer>& layers() const { return layers_; }
    
private:
    std::vector<Layer> layers_;
    int channels_;
    int depth_;
    bool gelu_;
};

} // namespace demucs
