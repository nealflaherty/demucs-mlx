#include "htdemucs.hpp"
#include "spec.hpp"
#include "utils.hpp"
#include <stdexcept>
#include <cmath>
#include <iostream>

namespace demucs {

// ============================================================================
// HTDemucs - Full hybrid transformer model
// ============================================================================

HTDemucs::HTDemucs(
    const std::vector<std::string>& sources,
    int audio_channels,
    int channels,
    int channels_time,
    float growth,
    int nfft,
    bool cac,
    int depth,
    bool rewrite,
    float freq_emb,
    float emb_scale,
    bool emb_smooth,
    int kernel_size,
    int time_stride,
    int stride,
    int context,
    int context_enc,
    int norm_starts,
    int norm_groups,
    int dconv_mode,
    int dconv_depth,
    int dconv_comp,
    int dconv_attn,
    int dconv_lstm,
    float dconv_init,
    int bottom_channels,
    int t_layers,
    int t_heads,
    float t_hidden_scale,
    float t_dropout,
    bool t_norm_in,
    bool t_norm_out,
    bool t_cross_first,
    bool t_layer_scale,
    bool t_gelu,
    int samplerate,
    float segment)
    : cac_(cac),
      audio_channels_(audio_channels),
      sources_(sources),
      kernel_size_(kernel_size),
      context_(context),
      stride_(stride),
      depth_(depth),
      channels_(channels),
      samplerate_(samplerate),
      segment_(segment),
      nfft_(nfft),
      hop_length_(nfft / 4),
      freq_emb_scale_(freq_emb),
      bottom_channels_(bottom_channels),
      channel_upsampler_weight_(mx::zeros({1})),
      channel_upsampler_bias_(mx::zeros({1})),
      channel_upsampler_t_weight_(mx::zeros({1})),
      channel_upsampler_t_bias_(mx::zeros({1})),
      channel_downsampler_weight_(mx::zeros({1})),
      channel_downsampler_bias_(mx::zeros({1})),
      channel_downsampler_t_weight_(mx::zeros({1})),
      channel_downsampler_t_bias_(mx::zeros({1}))
{
    // Build encoder and decoder layers (same as HDemucs)
    int chin = audio_channels;
    int chin_z = cac_ ? chin * 2 : chin;
    int chout = (channels_time > 0) ? channels_time : channels;
    int chout_z = channels;
    int freqs = nfft / 2;
    
    for (int index = 0; index < depth; ++index) {
        bool lstm = index >= dconv_lstm;
        bool attn = index >= dconv_attn;
        bool norm = index >= norm_starts;
        bool freq = freqs > 1;
        int stri = stride;
        int ker = kernel_size;
        
        if (!freq) {
            ker = time_stride * 2;
            stri = time_stride;
        }
        
        bool pad = true;
        bool last_freq = false;
        if (freq && freqs <= kernel_size) {
            ker = freqs;
            pad = false;
            last_freq = true;
        }
        
        // Encoder layer (frequency branch)
        auto enc = std::make_unique<HEncLayer>(
            chin_z, chout_z, ker, stri, norm_groups,
            false, freq, dconv_mode & 1, norm, context_enc, pad, rewrite,
            static_cast<float>(dconv_comp), dconv_init
        );
        encoder_.push_back(std::move(enc));
        
        // Time branch encoder
        if (freq) {
            auto tenc = std::make_unique<HEncLayer>(
                chin, chout, kernel_size, stride, norm_groups,
                last_freq, false, dconv_mode & 1, norm, context_enc, true, rewrite,
                static_cast<float>(dconv_comp), dconv_init
            );
            tencoder_.push_back(std::move(tenc));
        }
        
        // Update channels for next layer
        if (index == 0) {
            chin = audio_channels * sources_.size();
            chin_z = cac_ ? chin * 2 : chin;
        }
        
        // Decoder layer (frequency branch)
        auto dec = std::make_unique<HDecLayer>(
            chout_z, chin_z, index == 0, ker, stri, norm_groups,
            false, freq, dconv_mode & 2, norm, context, pad, true, rewrite,
            static_cast<float>(dconv_comp), dconv_init
        );
        decoder_.insert(decoder_.begin(), std::move(dec));
        
        // Time branch decoder
        if (freq) {
            auto tdec = std::make_unique<HDecLayer>(
                chout, chin, index == 0, kernel_size, stride, norm_groups,
                last_freq, false, dconv_mode & 2, norm, context, true, true, rewrite,
                static_cast<float>(dconv_comp), dconv_init
            );
            tdecoder_.insert(tdecoder_.begin(), std::move(tdec));
        }
        
        // Update for next iteration
        chin = chout;
        chin_z = chout_z;
        chout = static_cast<int>(growth * chout);
        chout_z = static_cast<int>(growth * chout_z);
        
        if (freq) {
            if (freqs <= kernel_size) {
                freqs = 1;
            } else {
                freqs /= stride;
            }
        }
        
        // Add frequency embedding after first layer
        if (index == 0 && freq_emb > 0) {
            freq_emb_ = std::make_unique<ScaledEmbedding>(
                freqs, chin_z, emb_scale, emb_smooth
            );
        }
    }
    
    // Create transformer if requested
    if (t_layers > 0) {
        int transformer_channels = channels * static_cast<int>(std::pow(growth, depth - 1));
        
        // If bottom_channels specified, we'll need channel up/downsampling
        if (bottom_channels > 0) {
            // Initialize channel up/downsampler weights (will be loaded from file)
            channel_upsampler_weight_ = mx::zeros({bottom_channels, transformer_channels, 1});
            channel_upsampler_bias_ = mx::zeros({bottom_channels});
            channel_upsampler_t_weight_ = mx::zeros({bottom_channels, transformer_channels, 1});
            channel_upsampler_t_bias_ = mx::zeros({bottom_channels});
            
            channel_downsampler_weight_ = mx::zeros({transformer_channels, bottom_channels, 1});
            channel_downsampler_bias_ = mx::zeros({transformer_channels});
            channel_downsampler_t_weight_ = mx::zeros({transformer_channels, bottom_channels, 1});
            channel_downsampler_t_bias_ = mx::zeros({transformer_channels});
            
            transformer_channels = bottom_channels;
        }
        
        crosstransformer_ = std::make_unique<CrossTransformerEncoder>(
            transformer_channels,
            "sin",  // emb
            t_hidden_scale,
            t_heads,
            t_layers,
            t_cross_first,
            t_dropout,
            10000,  // max_positions
            t_norm_in,
            false,  // norm_in_group
            0,      // group_norm
            true,   // norm_first
            t_norm_out,
            10000.0f,  // max_period
            0.0f,   // weight_decay
            t_layer_scale,
            t_gelu,
            0,      // sin_random_shift
            1.0f    // weight_pos_embed
        );
    }
}

// STFT/iSTFT helpers (same as HDemucs)
mx::array HTDemucs::_spec(const mx::array& x) {
    // Matches HDemucs::_spec implementation
    mx::array x_padded = x;
    int le = 0;
    
    // Pad to align with hop_length stride
    le = static_cast<int>(std::ceil(static_cast<float>(x.shape(-1)) / hop_length_));
    int pad = hop_length_ / 2 * 3;
    int pad_right = pad + le * hop_length_ - x.shape(-1);
    x_padded = pad1d(x, pad, pad_right, "reflect");
    
    // Call spectro from spec.hpp
    auto z = spectro(x_padded, nfft_, hop_length_);
    
    // Remove last frequency bin: z[..., :-1, :]
    std::vector<int> start_vec(z.shape().size(), 0);
    std::vector<int> end_vec(z.shape().begin(), z.shape().end());
    end_vec[end_vec.size() - 2] = z.shape(-2) - 1;
    
    mx::Shape start_shape(start_vec.begin(), start_vec.end());
    mx::Shape end_shape(end_vec.begin(), end_vec.end());
    z = mx::slice(z, start_shape, end_shape);
    
    // Slice to remove padding: z[..., 2:2+le]
    std::vector<int> start_vec2(z.shape().size(), 0);
    std::vector<int> end_vec2(z.shape().begin(), z.shape().end());
    start_vec2.back() = 2;
    end_vec2.back() = 2 + le;
    
    mx::Shape start_shape2(start_vec2.begin(), start_vec2.end());
    mx::Shape end_shape2(end_vec2.begin(), end_vec2.end());
    z = mx::slice(z, start_shape2, end_shape2);
    
    return z;
}

mx::array HTDemucs::_ispec(const mx::array& z, int length) {
    // Add back the last frequency bin
    std::vector<std::pair<int, int>> pad_width(z.shape().size(), {0, 0});
    pad_width[pad_width.size() - 2].second = 1;
    auto z_padded = mx::pad(z, pad_width);
    
    // Add time padding: 2 on both sides
    std::vector<std::pair<int, int>> time_pad(z_padded.shape().size(), {0, 0});
    time_pad.back() = {2, 2};
    z_padded = mx::pad(z_padded, time_pad);
    
    int pad = hop_length_ / 2 * 3;
    int le = hop_length_ * static_cast<int>(std::ceil(static_cast<float>(length) / hop_length_)) + 2 * pad;
    
    // Call ispectro from spec.hpp
    auto x = ispectro(z_padded, hop_length_, le);
    
    // Remove padding: x[..., pad:pad+length]
    std::vector<int> start_vec(x.shape().size(), 0);
    std::vector<int> end_vec(x.shape().begin(), x.shape().end());
    start_vec.back() = pad;
    end_vec.back() = pad + length;
    
    mx::Shape start_shape(start_vec.begin(), start_vec.end());
    mx::Shape end_shape(end_vec.begin(), end_vec.end());
    x = mx::slice(x, start_shape, end_shape);
    
    return x;
}

mx::array HTDemucs::_magnitude(const mx::array& z) {
    if (cac_) {
        // Complex as channels: interleave real/imag per channel
        // Python: view_as_real(z) -> [B, C, Fr, T, 2]
        //         .permute(0, 1, 4, 2, 3) -> [B, C, 2, Fr, T]
        //         .reshape(B, C*2, Fr, T) -> [real_ch0, imag_ch0, real_ch1, imag_ch1, ...]
        auto real_part = mx::real(z);  // [B, C, Fr, T]
        auto imag_part = mx::imag(z);  // [B, C, Fr, T]
        int B = z.shape(0);
        int C = z.shape(1);
        int Fr = z.shape(2);
        int T = z.shape(3);
        // Stack real/imag along a new axis then reshape to interleave
        // [B, C, Fr, T] -> expand to [B, C, 1, Fr, T] -> concat -> [B, C, 2, Fr, T]
        auto real_exp = mx::expand_dims(real_part, 2);  // [B, C, 1, Fr, T]
        auto imag_exp = mx::expand_dims(imag_part, 2);  // [B, C, 1, Fr, T]
        auto stacked = mx::concatenate({real_exp, imag_exp}, 2);  // [B, C, 2, Fr, T]
        return mx::reshape(stacked, {B, C * 2, Fr, T});
    } else {
        return mx::abs(z);
    }
}

mx::array HTDemucs::_mask(const mx::array& z, const mx::array& m) {
    if (cac_) {
        // m shape: (B, S, C*2, Fr, T) where channels are interleaved [real_ch0, imag_ch0, ...]
        // Python: m.view(B, S, -1, 2, Fr, T).permute(0, 1, 2, 4, 5, 3)
        //         -> view_as_complex -> (B, S, C, Fr, T) complex
        int B = m.shape(0);
        int S = m.shape(1);
        int C = m.shape(2) / 2;
        int Fr = m.shape(3);
        int T = m.shape(4);
        
        // Reshape to [B, S, C, 2, Fr, T]
        auto m_reshaped = mx::reshape(m, {B, S, C, 2, Fr, T});
        
        // Extract real (index 0 along dim 3) and imag (index 1 along dim 3)
        auto real_part = mx::slice(m_reshaped, {0, 0, 0, 0, 0, 0}, {B, S, C, 1, Fr, T});
        auto imag_part = mx::slice(m_reshaped, {0, 0, 0, 1, 0, 0}, {B, S, C, 2, Fr, T});
        real_part = mx::squeeze(real_part, 3);  // [B, S, C, Fr, T]
        imag_part = mx::squeeze(imag_part, 3);  // [B, S, C, Fr, T]
        
        // Combine into complex
        auto result = real_part + mx::array(std::complex<float>(0, 1)) * imag_part;
        return result;
    } else {
        auto z_expanded = mx::expand_dims(z, 1);
        auto z_abs = mx::abs(z_expanded) + 1e-8f;
        return z_expanded * (m / z_abs);
    }
}

mx::array HTDemucs::forward(const mx::array& mix) {
    // Python: length = mix.shape[-1]
    int length = mix.shape(-1);
    int length_pre_pad = -1;
    
    int training_length = static_cast<int>(segment_ * samplerate_);
    auto padded_mix = mix;
    if (mix.shape(-1) < training_length) {
        length_pre_pad = mix.shape(-1);
        int pad_amount = training_length - length_pre_pad;
        padded_mix = mx::pad(mix, {{0, 0}, {0, 0}, {0, pad_amount}});
        mx::eval(padded_mix);
        length = training_length;
    }
    
    // Step 1: STFT and magnitude extraction
    auto z = _spec(padded_mix);
    mx::eval(z);
    
    auto mag = _magnitude(z);
    mx::eval(mag);
    
    auto x = mag;
    
    // Step 2: Normalize frequency branch
    auto mean = mx::mean(x, {1, 2, 3}, true);
    auto var = mx::var(x, {1, 2, 3}, true, /* ddof= */ 1);
    auto std = mx::sqrt(var);
    x = (x - mean) / (std + 1e-5f);
    mx::eval(x);
    
    // Step 3: Prepare and normalize time branch
    auto xt = padded_mix;
    auto meant = mx::mean(xt, {1, 2}, true);
    auto vart = mx::var(xt, {1, 2}, true, /* ddof= */ 1);
    auto stdt = mx::sqrt(vart);
    xt = (xt - meant) / (stdt + 1e-5f);
    mx::eval(xt);
    
    // Step 4: Dual-branch encoding loop
    std::vector<mx::array> saved;
    std::vector<mx::array> saved_t;
    std::vector<int> lengths;
    std::vector<int> lengths_t;
    
    for (size_t idx = 0; idx < encoder_.size(); ++idx) {
        lengths.push_back(x.shape(-1));
        
        mx::array* inject = nullptr;
        
        // Time branch encoding
        if (idx < tencoder_.size()) {
            lengths_t.push_back(xt.shape(-1));
            xt = tencoder_[idx]->forward(xt, nullptr);
            mx::eval(xt);
            
            if (!tencoder_[idx]->empty()) {
                saved_t.push_back(xt);
            } else {
                inject = &xt;
            }
        }
        
        // Frequency branch encoding
        x = encoder_[idx]->forward(x, inject);
        mx::eval(x);
        
        // Add frequency embedding after first layer
        if (idx == 0 && freq_emb_) {
            int freq_size = x.shape(-2);
            auto frs = mx::arange(freq_size);
            auto emb = freq_emb_->forward(frs);
            emb = mx::transpose(emb, {1, 0});
            emb = mx::reshape(emb, {1, emb.shape(0), emb.shape(1), 1});
            x = x + freq_emb_scale_ * emb;
            mx::eval(x);
        }
        
        saved.push_back(x);
    }
    
    // Step 5: Cross-transformer
    if (crosstransformer_) {
        // Apply channel upsampling if bottom_channels specified
        if (bottom_channels_ > 0) {
            // Frequency branch: (B, C, F, T) -> (B, C, F*T) for 1D conv
            auto x_shape = x.shape();
            int B = x_shape[0], C = x_shape[1], F = x_shape[2], T_frames = x_shape[3];
            x = mx::reshape(x, {B, C, F * T_frames});
            
            // Apply 1D conv for channel upsampling
            x = utils::conv1d(x, channel_upsampler_weight_, channel_upsampler_bias_, 1, 0, 1, 1);
            mx::eval(x);
            
            // Reshape back: (B, C_new, F*T) -> (B, C_new, F, T)
            auto x_upsampled_shape = x.shape();
            int C_new = x_upsampled_shape[1];
            x = mx::reshape(x, {B, C_new, F, T_frames});
            
            // Time branch: already in (B, C, T) format
            xt = utils::conv1d(xt, channel_upsampler_t_weight_, channel_upsampler_t_bias_, 1, 0, 1, 1);
            mx::eval(xt);
        }
        
        // Apply transformer
        auto [transformed_freq, transformed_time] = crosstransformer_->forward(x, xt);
        x = transformed_freq;
        xt = transformed_time;
        mx::eval(x);
        mx::eval(xt);
        
        // Apply channel downsampling if bottom_channels specified
        if (bottom_channels_ > 0) {
            // Frequency branch: (B, C, F, T) -> (B, C, F*T) for 1D conv
            auto x_shape = x.shape();
            int B = x_shape[0], C = x_shape[1], F = x_shape[2], T_frames = x_shape[3];
            x = mx::reshape(x, {B, C, F * T_frames});
            
            x = utils::conv1d(x, channel_downsampler_weight_, channel_downsampler_bias_, 1, 0, 1, 1);
            mx::eval(x);
            
            auto x_downsampled_shape = x.shape();
            int C_new = x_downsampled_shape[1];
            x = mx::reshape(x, {B, C_new, F, T_frames});
            
            // Time branch
            xt = utils::conv1d(xt, channel_downsampler_t_weight_, channel_downsampler_t_bias_, 1, 0, 1, 1);
            mx::eval(xt);
        }
    }
    
    // Step 6: Dual-branch decoding loop
    for (size_t idx = 0; idx < decoder_.size(); ++idx) {
        auto skip = saved.back();
        saved.pop_back();
        
        int len = lengths.back();
        lengths.pop_back();
        
        auto [x_out, pre] = decoder_[idx]->forward(x, skip, len);
        x = x_out;
        mx::eval(x);
        
        // Handle time branch decoder
        int offset = depth_ - tdecoder_.size();
        if (static_cast<int>(idx) >= offset) {
            int tdec_idx = idx - offset;
            if (tdec_idx < static_cast<int>(tdecoder_.size())) {
                int length_t = lengths_t.back();
                lengths_t.pop_back();
                
                if (tdecoder_[tdec_idx]->empty()) {
                    auto pre_squeezed = mx::squeeze(pre, 2);
                    auto [xt_out, _] = tdecoder_[tdec_idx]->forward(pre_squeezed, mx::zeros_like(pre_squeezed), length_t);
                    xt = xt_out;
                    mx::eval(xt);
                } else {
                    auto skip_t = saved_t.back();
                    saved_t.pop_back();
                    auto [xt_out, _] = tdecoder_[tdec_idx]->forward(xt, skip_t, length_t);
                    xt = xt_out;
                    mx::eval(xt);
                }
            }
        }
    }
    
    // Step 7: Reshape to (B, S, C, Fr, T) for sources
    int B = x.shape(0);
    int S = sources_.size();
    int C = x.shape(1) / S;
    int Fr = x.shape(2);
    int T = x.shape(3);
    
    x = mx::reshape(x, {B, S, C, Fr, T});
    mx::eval(x);
    
    // Step 8: Denormalize frequency branch
    x = x * mx::expand_dims(std, 1) + mx::expand_dims(mean, 1);
    mx::eval(x);
    
    // Step 9: Apply mask and iSTFT
    auto zout = _mask(z, x);
    mx::eval(zout);
    
    x = _ispec(zout, length);
    mx::eval(x);
    
    // Step 10: Add time branch
    xt = mx::reshape(xt, {B, S, -1, length});
    mx::eval(xt);
    
    xt = xt * mx::expand_dims(stdt, 1) + mx::expand_dims(meant, 1);
    mx::eval(xt);
    
    x = xt + x;
    mx::eval(x);
    
    if (length_pre_pad >= 0) {
        x = mx::slice(x, {0, 0, 0, 0}, {x.shape(0), x.shape(1), x.shape(2), length_pre_pad});
        mx::eval(x);
    }
    
    return x;
}

} // namespace demucs
