#include "hdemucs.hpp"
#include "spec.hpp"
#include "utils.hpp"
#include <stdexcept>
#include <cmath>

namespace demucs {

// ============================================================================
// pad1d - matches pad1d() from hdemucs.py
// ============================================================================

// Helper: Manual reflect padding since MLX doesn't support it
mx::array reflect_pad_1d(const mx::array& x, int pad_left, int pad_right) {
    auto shape = x.shape();
    int length = shape.back();
    
    if (pad_left == 0 && pad_right == 0) {
        return x;
    }
    
    // Build slices for reflection
    std::vector<mx::array> parts;
    
    // Left padding: reflect from [1:pad_left+1] and reverse it
    if (pad_left > 0) {
        int slice_len = std::min(pad_left, length - 1);
        
        // Manually reverse by extracting elements in reverse order
        std::vector<mx::array> left_parts;
        for (int i = slice_len; i >= 1; --i) {
            std::vector<int> start_vec(shape.size(), 0);
            std::vector<int> end_vec(shape.begin(), shape.end());
            start_vec.back() = i;
            end_vec.back() = i + 1;
            
            mx::Shape start_shape(start_vec.begin(), start_vec.end());
            mx::Shape end_shape(end_vec.begin(), end_vec.end());
            auto elem = mx::slice(x, start_shape, end_shape);
            left_parts.push_back(elem);
        }
        
        if (!left_parts.empty()) {
            auto left_pad = mx::concatenate(left_parts, -1);
            parts.push_back(left_pad);
        }
    }
    
    // Original signal
    parts.push_back(x);
    
    // Right padding: reflect from [length-pad_right-1:length-1] and reverse it
    if (pad_right > 0) {
        int slice_len = std::min(pad_right, length - 1);
        
        // Manually reverse by extracting elements in reverse order
        std::vector<mx::array> right_parts;
        for (int i = length - 2; i >= length - 1 - slice_len; --i) {
            std::vector<int> start_vec(shape.size(), 0);
            std::vector<int> end_vec(shape.begin(), shape.end());
            start_vec.back() = i;
            end_vec.back() = i + 1;
            
            mx::Shape start_shape(start_vec.begin(), start_vec.end());
            mx::Shape end_shape(end_vec.begin(), end_vec.end());
            auto elem = mx::slice(x, start_shape, end_shape);
            right_parts.push_back(elem);
        }
        
        if (!right_parts.empty()) {
            auto right_pad = mx::concatenate(right_parts, -1);
            parts.push_back(right_pad);
        }
    }
    
    // Concatenate along last axis
    return mx::concatenate(parts, -1);
}

mx::array pad1d(
    const mx::array& x,
    int padding_left,
    int padding_right,
    const std::string& mode,
    float value
) {
    // Get input shape
    auto shape = x.shape();
    int length = shape.back();
    
    mx::array x_padded = x;
    
    // Handle reflect mode with small input (matches Python logic exactly)
    if (mode == "reflect") {
        int max_pad = std::max(padding_left, padding_right);
        if (length <= max_pad) {
            // Need to add extra constant padding first
            int extra_pad = max_pad - length + 1;
            int extra_pad_right = std::min(padding_right, extra_pad);
            int extra_pad_left = extra_pad - extra_pad_right;
            
            // Apply constant padding first
            std::vector<std::pair<int, int>> pad_widths(shape.size(), {0, 0});
            pad_widths.back() = {extra_pad_left, extra_pad_right};
            x_padded = mx::pad(x_padded, pad_widths, mx::array(value));
            
            // Adjust padding amounts for reflection
            padding_left -= extra_pad_left;
            padding_right -= extra_pad_right;
        }
        
        // Apply reflection padding manually (MLX doesn't support reflect mode)
        if (padding_left > 0 || padding_right > 0) {
            x_padded = reflect_pad_1d(x_padded, padding_left, padding_right);
        }
    } else if (mode == "constant") {
        std::vector<std::pair<int, int>> pad_widths(shape.size(), {0, 0});
        pad_widths.back() = {padding_left, padding_right};
        x_padded = mx::pad(x, pad_widths, mx::array(value));
    } else {
        throw std::invalid_argument("Unsupported padding mode: " + mode);
    }
    
    return x_padded;
}

// ============================================================================
// ScaledEmbedding - matches ScaledEmbedding class from hdemucs.py
// ============================================================================

ScaledEmbedding::ScaledEmbedding(int num_embeddings, int embedding_dim,
                                 float scale, bool smooth)
    : scale_(scale), embedding_weight_(mx::random::normal({num_embeddings, embedding_dim})) {
    
    if (smooth) {
        // Cumulative sum along axis 0
        embedding_weight_ = mx::cumsum(embedding_weight_, 0);
        
        // Normalize by sqrt(n) where n is the position
        // Create [1, 2, 3, ..., num_embeddings]
        auto n = mx::arange(1, num_embeddings + 1, mx::float32);
        auto sqrt_n = mx::sqrt(n);
        // Reshape to (num_embeddings, 1) for broadcasting
        sqrt_n = mx::reshape(sqrt_n, {num_embeddings, 1});
        embedding_weight_ = embedding_weight_ / sqrt_n;
    }
    
    // Scale down initial weights
    embedding_weight_ = embedding_weight_ / scale;
}

mx::array ScaledEmbedding::forward(const mx::array& x) {
    // x: indices to look up, shape: (...)
    // embedding_weight_: (num_embeddings, embedding_dim)
    // return: (..., embedding_dim) * scale
    
    // Use take to index into embedding_weight_
    // take(array, indices, axis) - take elements along axis
    auto emb = mx::take(embedding_weight_, x, 0);
    
    // Scale by scale factor
    return emb * scale_;
}

mx::array ScaledEmbedding::weight() const {
    return embedding_weight_ * scale_;
}

// ============================================================================
// HEncLayer - matches HEncLayer class from hdemucs.py
// ============================================================================

HEncLayer::HEncLayer(int chin, int chout, int kernel_size, int stride,
                     int norm_groups, bool empty, bool freq, bool dconv_enabled,
                     bool norm, int context, bool pad, bool rewrite,
                     float dconv_comp, float dconv_init)
    : freq_(freq), kernel_size_(kernel_size), stride_(stride),
      empty_(empty), norm_(norm), context_(context), rewrite_(rewrite),
      norm_groups_(norm_groups),
      conv_weight_(freq ? mx::zeros({chout, chin, kernel_size, 1}) : mx::zeros({chout, chin, kernel_size})),
      conv_bias_(mx::zeros({chout})),
      norm1_weight_(mx::zeros({chout})),
      norm1_bias_(mx::zeros({chout})),
      rewrite_weight_(freq ? mx::zeros({2 * chout, chout, 1 + 2 * context, 1 + 2 * context}) : mx::zeros({2 * chout, chout, 1 + 2 * context})),
      rewrite_bias_(mx::zeros({2 * chout})),
      norm2_weight_(mx::zeros({2 * chout})),
      norm2_bias_(mx::zeros({2 * chout})) {
    
    pad_ = pad ? kernel_size / 4 : 0;
    
    // Create DConv if requested
    if (dconv_enabled && !empty) {
        dconv_ = std::make_unique<DConv>(chout, dconv_comp, 2, dconv_init, norm, false, 4, 4, false, true, 3, true);
    }
}

mx::array HEncLayer::forward(const mx::array& x, const mx::array* inject) {
    auto y = x;
    
    // Python: if not self.freq and x.dim() == 4:
    //             B, C, Fr, T = x.shape
    //             x = x.view(B, -1, T)
    if (!freq_ && y.ndim() == 4) {
        int B = y.shape(0);
        int C = y.shape(1);
        int Fr = y.shape(2);
        int T = y.shape(3);
        y = mx::reshape(y, {B, C * Fr, T});
    }
    
    // Python: if not self.freq:
    //             le = x.shape[-1]
    //             if not le % self.stride == 0:
    //                 x = F.pad(x, (0, self.stride - (le % self.stride)))
    if (!freq_) {
        int le = y.shape(-1);
        if (le % stride_ != 0) {
            int pad_amount = stride_ - (le % stride_);
            y = pad1d(y, 0, pad_amount, "constant", 0.0f);
        }
    }
    
    // Python: y = self.conv(x)
    if (freq_) {
        // Conv2d with kernel [K, 1], stride [S, 1], pad [P, 0]
        y = utils::conv2d(y, conv_weight_, conv_bias_, {stride_, 1}, {pad_, 0});
    } else {
        y = utils::conv1d(y, conv_weight_, conv_bias_, stride_, pad_, 1, 1);
    }
    
    // Python: if self.empty: return y
    if (empty_) {
        return y;
    }
    
    // Python: if inject is not None:
    //             assert inject.shape[-1] == y.shape[-1]
    //             if inject.dim() == 3 and y.dim() == 4:
    //                 inject = inject[:, :, None]
    //             y = y + inject
    if (inject != nullptr) {
        auto inj = *inject;
        if (inj.ndim() == 3 && y.ndim() == 4) {
            inj = mx::expand_dims(inj, 2);
        }
        y = y + inj;
    }
    
    // Python: y = F.gelu(self.norm1(y))
    if (norm_) {
        y = utils::group_norm(y, norm1_weight_, norm1_bias_, norm_groups_);
    }
    y = utils::gelu(y);
    
    // Python: if self.dconv:
    //             if self.freq:
    //                 B, C, Fr, T = y.shape
    //                 y = y.permute(0, 2, 1, 3).reshape(-1, C, T)
    //             y = self.dconv(y)
    //             if self.freq:
    //                 y = y.view(B, Fr, C, T).permute(0, 2, 1, 3)
    if (dconv_) {
        if (freq_) {
            int B = y.shape(0);
            int C = y.shape(1);
            int Fr = y.shape(2);
            int T = y.shape(3);
            // permute (B, C, Fr, T) -> (B, Fr, C, T) then reshape to (B*Fr, C, T)
            y = mx::transpose(y, {0, 2, 1, 3});
            y = mx::reshape(y, {B * Fr, C, T});
            y = dconv_->forward(y);
            // reshape back to (B, Fr, C, T) then permute to (B, C, Fr, T)
            y = mx::reshape(y, {B, Fr, C, T});
            y = mx::transpose(y, {0, 2, 1, 3});
        } else {
            y = dconv_->forward(y);
        }
        
    }
    
    // Python: if self.rewrite:
    //             z = self.norm2(self.rewrite(y))
    //             z = F.glu(z, dim=1)
    //         else:
    //             z = y
    mx::array z = y;
    if (rewrite_) {
        if (freq_) {
            // PyTorch Conv2d with scalar kernel_size=1+2*ctx means square kernel
            // so padding is (context, context) not (context, 0)
            z = utils::conv2d(y, rewrite_weight_, rewrite_bias_, {1, 1}, {context_, context_});
        } else {
            z = utils::conv1d(y, rewrite_weight_, rewrite_bias_, 1, context_, 1, 1);
        }
        if (norm_) {
            z = utils::group_norm(z, norm2_weight_, norm2_bias_, norm_groups_);
        }
        z = utils::glu(z, 1);
        
    }
    
    return z;
}

// ============================================================================
// HDecLayer - matches HDecLayer class from hdemucs.py
// ============================================================================

HDecLayer::HDecLayer(int chin, int chout, bool last, int kernel_size,
                     int stride, int norm_groups, bool empty, bool freq,
                     bool dconv_enabled, bool norm, int context, bool pad,
                     bool context_freq, bool rewrite,
                     float dconv_comp, float dconv_init)
    : last_(last), freq_(freq), chin_(chin), empty_(empty),
      stride_(stride), kernel_size_(kernel_size), norm_(norm),
      context_freq_(context_freq), context_(context), rewrite_(rewrite),
      norm_groups_(norm_groups),
      conv_tr_weight_(freq ? mx::zeros({chin, chout, kernel_size, 1}) : mx::zeros({chin, chout, kernel_size})),
      conv_tr_bias_(mx::zeros({chout})),
      norm2_weight_(mx::zeros({chout})),
      norm2_bias_(mx::zeros({chout})),
      rewrite_weight_(mx::zeros({1})),
      rewrite_bias_(mx::zeros({2 * chin})),
      norm1_weight_(mx::zeros({2 * chin})),
      norm1_bias_(mx::zeros({2 * chin})) {
    
    pad_ = pad ? kernel_size / 4 : 0;
    
    // Initialize rewrite weight with proper shape based on freq and context_freq
    if (freq) {
        if (context_freq) {
            // Conv2d: kernel [1+2*context, 1+2*context] (scalar kernel_size in PyTorch means square kernel)
            rewrite_weight_ = mx::zeros({2 * chin, chin, 1 + 2 * context, 1 + 2 * context});
        } else {
            // Conv2d: kernel [1, 1+2*context] (explicit list in PyTorch)
            rewrite_weight_ = mx::zeros({2 * chin, chin, 1, 1 + 2 * context});
        }
    } else {
        rewrite_weight_ = mx::zeros({2 * chin, chin, 1 + 2 * context});
    }
    
    // Create DConv if requested
    if (dconv_enabled && !empty) {
        dconv_ = std::make_unique<DConv>(chin, dconv_comp, 2, dconv_init, norm, false, 4, 4, false, true, 3, true);
    }
}

std::pair<mx::array, mx::array> HDecLayer::forward(const mx::array& x,
                                                     const mx::array& skip,
                                                     int length) {
    auto y = x;
    
    // Python: if self.freq and x.dim() == 3:
    //             B, C, T = x.shape
    //             x = x.view(B, self.chin, -1, T)
    if (freq_ && y.ndim() == 3) {
        int B = y.shape(0);
        int C = y.shape(1);
        int T = y.shape(2);
        y = mx::reshape(y, {B, chin_, -1, T});
    }
    
    if (!empty_) {
        // Python: x = x + skip
        auto x_with_skip = y + skip;
        
        // Python: if self.rewrite:
        //             y = F.glu(self.norm1(self.rewrite(x)), dim=1)
        //         else:
        //             y = x
        if (rewrite_) {
            mx::array rewrite_out = mx::zeros({1});
            if (freq_) {
                if (context_freq_) {
                    // Conv2d with scalar kernel_size=1+2*ctx means square kernel (1+2*ctx, 1+2*ctx)
                    // padding is (ctx, ctx)
                    rewrite_out = utils::conv2d(x_with_skip, rewrite_weight_, rewrite_bias_, {1, 1}, {context_, context_});
                } else {
                    // Conv2d with kernel [1, 1+2*ctx], pad [0, ctx]
                    rewrite_out = utils::conv2d(x_with_skip, rewrite_weight_, rewrite_bias_, {1, 1}, {0, context_});
                }
            } else {
                rewrite_out = utils::conv1d(x_with_skip, rewrite_weight_, rewrite_bias_, 1, context_, 1, 1);
            }
            
            if (norm_) {
                rewrite_out = utils::group_norm(rewrite_out, norm1_weight_, norm1_bias_, norm_groups_);
            }
            y = utils::glu(rewrite_out, 1);
        } else {
            y = x_with_skip;
        }
        
        // Python: if self.dconv:
        //             if self.freq:
        //                 B, C, Fr, T = y.shape
        //                 y = y.permute(0, 2, 1, 3).reshape(-1, C, T)
        //             y = self.dconv(y)
        //             if self.freq:
        //                 y = y.view(B, Fr, C, T).permute(0, 2, 1, 3)
        if (dconv_) {
            if (freq_) {
                int B = y.shape(0);
                int C = y.shape(1);
                int Fr = y.shape(2);
                int T = y.shape(3);
                y = mx::transpose(y, {0, 2, 1, 3});
                y = mx::reshape(y, {B * Fr, C, T});
                y = dconv_->forward(y);
                y = mx::reshape(y, {B, Fr, C, T});
                y = mx::transpose(y, {0, 2, 1, 3});
            } else {
                y = dconv_->forward(y);
            }
        }
    } else {
        // empty decoder: y = x, skip is None
    }
    
    // Python: z = self.norm2(self.conv_tr(y))
    mx::array z = mx::zeros({1});
    if (freq_) {
        z = utils::conv_transpose2d(y, conv_tr_weight_, conv_tr_bias_, {stride_, 1});
    } else {
        z = utils::conv_transpose1d(y, conv_tr_weight_, conv_tr_bias_, stride_, 0, 0, 1, 1);
    }
    
    if (norm_) {
        z = utils::group_norm(z, norm2_weight_, norm2_bias_, norm_groups_);
    }
    
    // Python: if self.freq:
    //             if self.pad:
    //                 z = z[..., self.pad:-self.pad, :]
    //         else:
    //             z = z[..., self.pad:self.pad + length]
    if (freq_) {
        if (pad_ > 0) {
            // Slice freq axis: z[..., pad:-pad, :]
            std::vector<int> start_vec(z.shape().size(), 0);
            std::vector<int> end_vec(z.shape().begin(), z.shape().end());
            start_vec[z.ndim() - 2] = pad_;
            end_vec[z.ndim() - 2] = z.shape(z.ndim() - 2) - pad_;
            
            mx::Shape start_shape(start_vec.begin(), start_vec.end());
            mx::Shape end_shape(end_vec.begin(), end_vec.end());
            z = mx::slice(z, start_shape, end_shape);
        }
    } else {
        // Slice time axis to match target length
        if (pad_ > 0 || z.shape(-1) != pad_ + length) {
            std::vector<int> start_vec(z.shape().size(), 0);
            std::vector<int> end_vec(z.shape().begin(), z.shape().end());
            start_vec.back() = pad_;
            end_vec.back() = pad_ + length;
            
            mx::Shape start_shape(start_vec.begin(), start_vec.end());
            mx::Shape end_shape(end_vec.begin(), end_vec.end());
            z = mx::slice(z, start_shape, end_shape);
        }
    }
    
    // Python: if not self.last:
    //             z = F.gelu(z)
    if (!last_) {
        z = utils::gelu(z);
    }
    
    return {z, y};
}

// ============================================================================
// HDemucs - Full hybrid model
// ============================================================================

HDemucs::HDemucs(
    const std::vector<std::string>& sources,
    int audio_channels,
    int channels,
    int channels_time,
    float growth,
    int nfft,
    bool cac,
    int depth,
    bool rewrite,
    bool hybrid,
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
      hybrid_(hybrid),
      freq_emb_scale_(freq_emb)
{
    // Build encoder and decoder layers
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
        
        // Time branch encoder (if hybrid and freq)
        if (hybrid && freq) {
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
        
        // Time branch decoder (if hybrid and freq)
        if (hybrid && freq) {
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
}

mx::array HDemucs::_spec(const mx::array& x) {
    // Python: z = spectro(x, nfft, hl)[..., :-1, :]
    // For hybrid models, we pad the input to align time and freq branches
    
    mx::array x_padded = x;
    int le = 0;
    
    if (hybrid_) {
        // Pad to align with hop_length stride
        le = static_cast<int>(std::ceil(static_cast<float>(x.shape(-1)) / hop_length_));
        int pad = hop_length_ / 2 * 3;
        int pad_right = pad + le * hop_length_ - x.shape(-1);
        x_padded = pad1d(x, pad, pad_right, "reflect");
    }
    
    // Call spectro from spec.hpp (implemented with vDSP)
    auto z = spectro(x_padded, nfft_, hop_length_);
    
    // Remove last frequency bin: z[..., :-1, :]
    std::vector<int> start_vec(z.shape().size(), 0);
    std::vector<int> end_vec(z.shape().begin(), z.shape().end());
    end_vec[end_vec.size() - 2] = z.shape(-2) - 1;  // -2 is freq axis
    
    mx::Shape start_shape(start_vec.begin(), start_vec.end());
    mx::Shape end_shape(end_vec.begin(), end_vec.end());
    z = mx::slice(z, start_shape, end_shape);
    
    if (hybrid_) {
        // Slice to remove padding: z[..., 2:2+le]
        std::vector<int> start_vec2(z.shape().size(), 0);
        std::vector<int> end_vec2(z.shape().begin(), z.shape().end());
        start_vec2.back() = 2;
        end_vec2.back() = 2 + le;
        
        mx::Shape start_shape2(start_vec2.begin(), start_vec2.end());
        mx::Shape end_shape2(end_vec2.begin(), end_vec2.end());
        z = mx::slice(z, start_shape2, end_shape2);
    }
    
    return z;
}

mx::array HDemucs::_ispec(const mx::array& z, int length) {
    // Add back the last frequency bin: pad with zeros
    std::vector<std::pair<int, int>> pad_width(z.shape().size(), {0, 0});
    pad_width[pad_width.size() - 2].second = 1;  // Pad freq axis by 1 on the right
    auto z_padded = mx::pad(z, pad_width);
    
    if (hybrid_) {
        // Add time padding: 2 on both sides
        std::vector<std::pair<int, int>> time_pad(z_padded.shape().size(), {0, 0});
        time_pad.back() = {2, 2};  // Pad time axis
        z_padded = mx::pad(z_padded, time_pad);
        
        int pad = hop_length_ / 2 * 3;
        int le = hop_length_ * static_cast<int>(std::ceil(static_cast<float>(length) / hop_length_));
        
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
    } else {
        return ispectro(z_padded, hop_length_, length);
    }
}

mx::array HDemucs::_magnitude(const mx::array& z) {
    if (cac_) {
        // Complex as channels: interleave real/imag per channel
        // Python: view_as_real(z) -> [B, C, Fr, T, 2]
        //         .permute(0, 1, 4, 2, 3) -> [B, C, 2, Fr, T]
        //         .reshape(B, C*2, Fr, T) -> [real_ch0, imag_ch0, real_ch1, imag_ch1, ...]
        auto real_part = mx::real(z);
        auto imag_part = mx::imag(z);
        int B = z.shape(0);
        int C = z.shape(1);
        int Fr = z.shape(2);
        int T = z.shape(3);
        auto real_exp = mx::expand_dims(real_part, 2);
        auto imag_exp = mx::expand_dims(imag_part, 2);
        auto stacked = mx::concatenate({real_exp, imag_exp}, 2);
        return mx::reshape(stacked, {B, C * 2, Fr, T});
    } else {
        return mx::abs(z);
    }
}

mx::array HDemucs::_mask(const mx::array& z, const mx::array& m) {
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
        // Simple masking: z * (m / (|z| + eps))
        auto z_expanded = mx::expand_dims(z, 1);  // (B, 1, C, Fr, T)
        auto z_abs = mx::abs(z_expanded) + 1e-8f;
        return z_expanded * (m / z_abs);
    }
}

mx::array HDemucs::forward(const mx::array& mix) {
    int length = mix.shape(-1);
    
    // Compute spectrogram
    auto z = _spec(mix);
    auto mag = _magnitude(z);
    auto x = mag;
    
    // Normalize frequency branch
    auto mean = mx::mean(x, {1, 2, 3}, true);
    auto var = mx::var(x, {1, 2, 3}, true, /* ddof= */ 1);
    auto std = mx::sqrt(var);
    x = (x - mean) / (std + 1e-5f);
    
    // Prepare time branch if hybrid
    mx::array xt = mix;  // Initialize with mix, will be overwritten if hybrid
    mx::array meant = mean;  // Initialize with dummy values
    mx::array stdt = std;
    if (hybrid_) {
        xt = mix;
        meant = mx::mean(xt, {1, 2}, true);
        auto vart = mx::var(xt, {1, 2}, true, /* ddof= */ 1);
        stdt = mx::sqrt(vart);
        xt = (xt - meant) / (stdt + 1e-5f);
    }
    
    // Encoder pass
    std::vector<mx::array> saved;
    std::vector<mx::array> saved_t;
    std::vector<int> lengths;
    std::vector<int> lengths_t;
    
    for (size_t idx = 0; idx < encoder_.size(); ++idx) {
        lengths.push_back(x.shape(-1));
        
        mx::array* inject = nullptr;
        if (hybrid_ && idx < tencoder_.size()) {
            lengths_t.push_back(xt.shape(-1));
            xt = tencoder_[idx]->forward(xt, nullptr);
            
            // If this is the merge point (empty encoder), use as injection
            // For now, simplified: no injection
        }
        
        x = encoder_[idx]->forward(x, inject);
        
        // Add frequency embedding after first layer
        if (idx == 0 && freq_emb_) {
            int freq_size = x.shape(-2);
            auto frs = mx::arange(freq_size);
            auto emb = freq_emb_->forward(frs);
            // emb is [Fr, C], transpose to [C, Fr] then reshape to [1, C, Fr, 1]
            emb = mx::transpose(emb, {1, 0});
            emb = mx::reshape(emb, {1, emb.shape(0), emb.shape(1), 1});
            x = x + freq_emb_scale_ * emb;
        }
        
        saved.push_back(x);
        if (hybrid_ && idx < tencoder_.size()) {
            saved_t.push_back(xt);
        }
    }
    
    // Initialize decoder input to zeros
    x = mx::zeros_like(x);
    if (hybrid_) {
        xt = mx::zeros_like(xt);
    }
    
    // Decoder pass
    for (size_t idx = 0; idx < decoder_.size(); ++idx) {
        auto skip = saved.back();
        saved.pop_back();
        
        int len = lengths.back();
        lengths.pop_back();
        
        auto [x_out, pre] = decoder_[idx]->forward(x, skip, len);
        x = x_out;
        
        // Handle time branch decoder
        if (hybrid_) {
            int offset = depth_ - tdecoder_.size();
            if (static_cast<int>(idx) >= offset) {
                int tdec_idx = idx - offset;
                if (tdec_idx < static_cast<int>(tdecoder_.size())) {
                    int length_t = lengths_t.back();
                    lengths_t.pop_back();
                    
                    // Simplified: no empty check for now
                    auto skip_t = saved_t.back();
                    saved_t.pop_back();
                    
                    auto [xt_out, _] = tdecoder_[tdec_idx]->forward(xt, skip_t, length_t);
                    xt = xt_out;
                }
            }
        }
    }
    
    // Reshape to (B, S, C, Fr, T) for sources
    int B = x.shape(0);
    int S = sources_.size();
    int C = x.shape(1) / S;
    int Fr = x.shape(2);
    int T = x.shape(3);
    
    x = mx::reshape(x, {B, S, C, Fr, T});
    
    // Denormalize
    x = x * mx::expand_dims(std, 1) + mx::expand_dims(mean, 1);
    
    // Apply mask and iSTFT
    auto zout = _mask(z, x);
    x = _ispec(zout, length);
    
    // Add time branch if hybrid
    if (hybrid_) {
        xt = mx::reshape(xt, {B, S, -1, length});
        xt = xt * mx::expand_dims(stdt, 1) + mx::expand_dims(meant, 1);
        x = x + xt;
    }
    
    return x;
}

} // namespace demucs
