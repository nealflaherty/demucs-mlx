#pragma once

#include <mlx/mlx.h>
#include <string>
#include <vector>
#include <memory>
#include "demucs.hpp"

namespace demucs {

namespace mx = mlx::core;

/**
 * Pad a 1D tensor with reflection or constant padding.
 * Matches pad1d() from hdemucs.py
 */
mx::array pad1d(
    const mx::array& x,
    int padding_left,
    int padding_right,
    const std::string& mode = "constant",
    float value = 0.0f
);

/**
 * ScaledEmbedding layer
 * Matches ScaledEmbedding class from hdemucs.py
 */
class ScaledEmbedding {
public:
    ScaledEmbedding(int num_embeddings, int embedding_dim, 
                    float scale = 10.0f, bool smooth = false);
    
    mx::array forward(const mx::array& x);
    mx::array weight() const;
    
    // Weight access for loading
    mx::array& embedding_weight() { return embedding_weight_; }
    const mx::array& embedding_weight() const { return embedding_weight_; }

private:
    mx::array embedding_weight_;
    float scale_;
};

/**
 * HEncLayer - Encoder layer
 * Matches HEncLayer class from hdemucs.py
 * 
 * This is used by both time and frequency branches.
 * For now, we implement the 1D (time) version. 2D (freq) version will be added later.
 */
class HEncLayer {
public:
    HEncLayer(int chin, int chout, int kernel_size = 8, int stride = 4,
              int norm_groups = 1, bool empty = false, bool freq = false,
              bool dconv = true, bool norm = true, int context = 0,
              bool pad = true, bool rewrite = true,
              float dconv_comp = 8.0f, float dconv_init = 1e-3f);
    
    mx::array forward(const mx::array& x, const mx::array* inject = nullptr);
    
    // Weight access for loading
    mx::array& conv_weight() { return conv_weight_; }
    mx::array& conv_bias() { return conv_bias_; }
    mx::array& norm1_weight() { return norm1_weight_; }
    mx::array& norm1_bias() { return norm1_bias_; }
    mx::array& rewrite_weight() { return rewrite_weight_; }
    mx::array& rewrite_bias() { return rewrite_bias_; }
    mx::array& norm2_weight() { return norm2_weight_; }
    mx::array& norm2_bias() { return norm2_bias_; }
    std::unique_ptr<DConv>& dconv() { return dconv_; }
    
    // Property access
    bool empty() const { return empty_; }

private:
    bool freq_;
    int kernel_size_;
    int stride_;
    bool empty_;
    bool norm_;
    int pad_;
    int context_;
    bool rewrite_;
    int norm_groups_;
    
    // Conv layer weights
    mx::array conv_weight_;  // Shape: (chout, chin, kernel_size) for 1D
    mx::array conv_bias_;    // Shape: (chout,)
    
    // Norm1 (GroupNorm after conv)
    mx::array norm1_weight_;  // Shape: (chout,)
    mx::array norm1_bias_;    // Shape: (chout,)
    
    // Rewrite layer (1x1 conv with context)
    mx::array rewrite_weight_;  // Shape: (2*chout, chout, 1+2*context)
    mx::array rewrite_bias_;    // Shape: (2*chout,)
    
    // Norm2 (GroupNorm after rewrite)
    mx::array norm2_weight_;  // Shape: (2*chout,)
    mx::array norm2_bias_;    // Shape: (2*chout,)
    
    // DConv residual branch
    std::unique_ptr<DConv> dconv_;
};

/**
 * HDecLayer - Decoder layer
 * Matches HDecLayer class from hdemucs.py
 */
class HDecLayer {
public:
    HDecLayer(int chin, int chout, bool last = false, int kernel_size = 8,
              int stride = 4, int norm_groups = 1, bool empty = false,
              bool freq = false, bool dconv = true, bool norm = true,
              int context = 1, bool pad = true, bool context_freq = true,
              bool rewrite = true,
              float dconv_comp = 8.0f, float dconv_init = 1e-3f);
    
    std::pair<mx::array, mx::array> forward(const mx::array& x, 
                                             const mx::array& skip,
                                             int length);
    
    // Weight access for loading
    mx::array& conv_tr_weight() { return conv_tr_weight_; }
    mx::array& conv_tr_bias() { return conv_tr_bias_; }
    mx::array& norm2_weight() { return norm2_weight_; }
    mx::array& norm2_bias() { return norm2_bias_; }
    mx::array& rewrite_weight() { return rewrite_weight_; }
    mx::array& rewrite_bias() { return rewrite_bias_; }
    mx::array& norm1_weight() { return norm1_weight_; }
    mx::array& norm1_bias() { return norm1_bias_; }
    std::unique_ptr<DConv>& dconv() { return dconv_; }
    
    // Property access
    bool empty() const { return empty_; }
    bool freq() const { return freq_; }
    bool last() const { return last_; }
    bool norm() const { return norm_; }
    bool context_freq() const { return context_freq_; }
    int context() const { return context_; }
    int stride() const { return stride_; }
    int pad() const { return pad_; }
    bool has_dconv() const { return dconv_ != nullptr; }
    DConv& dconv_ref() { return *dconv_; }

private:
    int pad_;
    bool last_;
    bool freq_;
    int chin_;
    bool empty_;
    int stride_;
    int kernel_size_;
    bool norm_;
    bool context_freq_;
    int context_;
    bool rewrite_;
    int norm_groups_;
    
    // ConvTranspose layer weights
    mx::array conv_tr_weight_;  // Shape: (chin, chout, kernel_size) for 1D
    mx::array conv_tr_bias_;    // Shape: (chout,)
    
    // Norm2 (GroupNorm after conv_tr)
    mx::array norm2_weight_;  // Shape: (chout,)
    mx::array norm2_bias_;    // Shape: (chout,)
    
    // Rewrite layer (1x1 conv with context)
    mx::array rewrite_weight_;  // Shape: (2*chin, chin, 1+2*context)
    mx::array rewrite_bias_;    // Shape: (2*chin,)
    
    // Norm1 (GroupNorm after rewrite)
    mx::array norm1_weight_;  // Shape: (2*chin,)
    mx::array norm1_bias_;    // Shape: (2*chin,)
    
    // DConv residual branch
    std::unique_ptr<DConv> dconv_;
};

/**
 * HDemucs - Full hybrid Demucs model
 * Matches HDemucs class from hdemucs.py
 * 
 * Spectrogram and hybrid Demucs model with parallel time and frequency branches.
 */
class HDemucs {
public:
    HDemucs(const std::vector<std::string>& sources,
            int audio_channels = 2,
            int channels = 48,
            int channels_time = -1,  // -1 means use channels
            float growth = 2.0f,
            int nfft = 4096,
            bool cac = true,
            int depth = 6,
            bool rewrite = true,
            bool hybrid = true,
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
            int dconv_mode = 1,
            int dconv_depth = 2,
            int dconv_comp = 4,
            int dconv_attn = 4,
            int dconv_lstm = 4,
            float dconv_init = 1e-4f,
            int samplerate = 44100,
            float segment = 4.0f * 10.0f);
    
    // Forward pass: mix -> separated sources
    mx::array forward(const mx::array& mix);
    
    // Access to encoder/decoder layers for weight loading
    std::vector<std::unique_ptr<HEncLayer>>& encoder() { return encoder_; }
    std::vector<std::unique_ptr<HDecLayer>>& decoder() { return decoder_; }
    std::vector<std::unique_ptr<HEncLayer>>& tencoder() { return tencoder_; }
    std::vector<std::unique_ptr<HDecLayer>>& tdecoder() { return tdecoder_; }
    std::unique_ptr<ScaledEmbedding>& freq_emb() { return freq_emb_; }

private:
    // STFT/iSTFT helpers
    mx::array _spec(const mx::array& x);
    mx::array _ispec(const mx::array& z, int length);
    mx::array _magnitude(const mx::array& z);
    mx::array _mask(const mx::array& z, const mx::array& m);
    
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
    bool hybrid_;
    float freq_emb_scale_;
    
    // Layers
    std::vector<std::unique_ptr<HEncLayer>> encoder_;
    std::vector<std::unique_ptr<HDecLayer>> decoder_;
    std::vector<std::unique_ptr<HEncLayer>> tencoder_;  // time branch encoder
    std::vector<std::unique_ptr<HDecLayer>> tdecoder_;  // time branch decoder
    std::unique_ptr<ScaledEmbedding> freq_emb_;
};

} // namespace demucs
