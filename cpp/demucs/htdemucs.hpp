#pragma once

#include "hdemucs.hpp"
#include "transformer.hpp"
#include <mlx/mlx.h>
#include <string>
#include <vector>
#include <memory>

namespace demucs {

namespace mx = mlx::core;

/**
 * HTDemucs - Hybrid Transformer Demucs model
 * Extends HDemucs with cross-transformer layers between frequency and time branches
 * Matches HTDemucs class from htdemucs.py
 */
class HTDemucs {
public:
    HTDemucs(const std::vector<std::string>& sources,
             int audio_channels = 2,
             int channels = 48,
             int channels_time = -1,
             float growth = 2.0f,
             int nfft = 4096,
             bool cac = true,
             int depth = 6,
             bool rewrite = true,
             float freq_emb = 0.2f,
             float emb_scale = 10.0f,
             bool emb_smooth = true,
             int kernel_size = 8,
             int time_stride = 2,
             int stride = 4,
             int context = 1,
             int context_enc = 0,
             int norm_starts = 4,
             int norm_groups = 4,
             int dconv_mode = 3,
             int dconv_depth = 2,
             int dconv_comp = 8,
             int dconv_attn = 4,
             int dconv_lstm = 4,
             float dconv_init = 1e-3f,
             // Transformer parameters
             int bottom_channels = 0,
             int t_layers = 5,
             int t_heads = 8,
             float t_hidden_scale = 4.0f,
             float t_dropout = 0.0f,
             bool t_norm_in = true,
             bool t_norm_out = true,
             bool t_cross_first = false,
             bool t_layer_scale = true,
             bool t_gelu = true,
             // Model metadata
             int samplerate = 44100,
             float segment = 4.0f * 10.0f);
    
    // Forward pass: mix -> separated sources
    mx::array forward(const mx::array& mix);
    
    // Access to layers for weight loading
    std::vector<std::unique_ptr<HEncLayer>>& encoder() { return encoder_; }
    std::vector<std::unique_ptr<HDecLayer>>& decoder() { return decoder_; }
    std::vector<std::unique_ptr<HEncLayer>>& tencoder() { return tencoder_; }
    std::vector<std::unique_ptr<HDecLayer>>& tdecoder() { return tdecoder_; }
    std::unique_ptr<ScaledEmbedding>& freq_emb() { return freq_emb_; }
    float freq_emb_scale() const { return freq_emb_scale_; }
    
    // Transformer access
    std::unique_ptr<CrossTransformerEncoder>& crosstransformer() { return crosstransformer_; }
    
    // Channel up/downsampler access
    mx::array& channel_upsampler_weight() { return channel_upsampler_weight_; }
    mx::array& channel_upsampler_bias() { return channel_upsampler_bias_; }
    mx::array& channel_upsampler_t_weight() { return channel_upsampler_t_weight_; }
    mx::array& channel_upsampler_t_bias() { return channel_upsampler_t_bias_; }
    mx::array& channel_downsampler_weight() { return channel_downsampler_weight_; }
    mx::array& channel_downsampler_bias() { return channel_downsampler_bias_; }
    mx::array& channel_downsampler_t_weight() { return channel_downsampler_t_weight_; }
    mx::array& channel_downsampler_t_bias() { return channel_downsampler_t_bias_; }

    // STFT/iSTFT helpers (same as HDemucs) - public for testing
    mx::array _spec(const mx::array& x);
    mx::array _ispec(const mx::array& z, int length);
    mx::array _magnitude(const mx::array& z);
    mx::array _mask(const mx::array& z, const mx::array& m);

private:
    
    // Model parameters
    bool cac_;
    int audio_channels_;
    std::vector<std::string> sources_;
    int kernel_size_;
    int context_;
    int stride_;
    int depth_;
    int channels_;
    int samplerate_;
    float segment_;
    int nfft_;
    int hop_length_;
    float freq_emb_scale_;
    int bottom_channels_;
    
    // Layers (same as HDemucs)
    std::vector<std::unique_ptr<HEncLayer>> encoder_;
    std::vector<std::unique_ptr<HDecLayer>> decoder_;
    std::vector<std::unique_ptr<HEncLayer>> tencoder_;
    std::vector<std::unique_ptr<HDecLayer>> tdecoder_;
    std::unique_ptr<ScaledEmbedding> freq_emb_;
    
    // Transformer layers (new for HTDemucs)
    std::unique_ptr<CrossTransformerEncoder> crosstransformer_;
    
    // Channel up/downsampling (for transformer input/output)
    mx::array channel_upsampler_weight_;
    mx::array channel_upsampler_bias_;
    mx::array channel_upsampler_t_weight_;
    mx::array channel_upsampler_t_bias_;
    mx::array channel_downsampler_weight_;
    mx::array channel_downsampler_bias_;
    mx::array channel_downsampler_t_weight_;
    mx::array channel_downsampler_t_bias_;
};

} // namespace demucs
