#include "transformer.hpp"
#include "utils.hpp"
#include <cmath>
#include <stdexcept>

namespace demucs {

// ============================================================================
// Positional Embeddings
// ============================================================================

mx::array create_sin_embedding(
    int length,
    int dim,
    int shift,
    float max_period
) {
    // Python: pos = shift + torch.arange(length, device=device).view(-1, 1, 1)
    auto pos = mx::arange(shift, shift + length, mx::float32);
    pos = mx::reshape(pos, {length, 1, 1});
    
    // Python: half_dim = dim // 2
    int half_dim = dim / 2;
    
    // Python: adim = torch.arange(dim // 2, device=device).view(1, 1, -1)
    auto adim = mx::arange(0, half_dim, mx::float32);
    adim = mx::reshape(adim, {1, 1, half_dim});
    
    // Python: phase = pos / (max_period ** (adim / (half_dim - 1)))
    auto exponent = adim / static_cast<float>(half_dim - 1);
    auto divisor = mx::power(mx::array(max_period), exponent);
    auto phase = pos / divisor;
    
    // Python: return torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)
    auto cos_part = mx::cos(phase);
    auto sin_part = mx::sin(phase);
    
    return mx::concatenate({cos_part, sin_part}, -1);
}

mx::array create_2d_sin_embedding(
    int d_model,
    int height,
    int width,
    float max_period
) {
    // Python implementation creates separate embeddings for height and width
    // pe = torch.zeros(d_model, height, width)
    
    if (d_model % 4 != 0) {
        throw std::invalid_argument(
            "Cannot use sin/cos positional encoding with dimension not divisible by 4"
        );
    }
    
    // Each dimension uses half of d_model
    int d_model_half = d_model / 2;
    
    // Python: div_term = torch.exp(torch.arange(0.0, d_model, 2) * -(math.log(max_period) / d_model))
    auto arange_vals = mx::arange(0.0f, static_cast<float>(d_model_half), 2.0f, mx::float32);
    auto div_term = mx::exp(arange_vals * (-std::log(max_period) / d_model_half));
    
    // Python: pos_w = torch.arange(0.0, width).unsqueeze(1)
    auto pos_w = mx::arange(0.0f, static_cast<float>(width), mx::float32);
    pos_w = mx::reshape(pos_w, {width, 1});
    
    // Python: pos_h = torch.arange(0.0, height).unsqueeze(1)
    auto pos_h = mx::arange(0.0f, static_cast<float>(height), mx::float32);
    pos_h = mx::reshape(pos_h, {height, 1});
    
    // Create the positional encoding tensor
    auto pe = mx::zeros({d_model, height, width});
    
    // Python: pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    auto sin_w = mx::sin(mx::matmul(pos_w, mx::reshape(div_term, {1, -1})));  // (width, d_model_half/2)
    sin_w = mx::transpose(sin_w, {1, 0});  // (d_model_half/2, width)
    sin_w = mx::reshape(sin_w, {-1, 1, width});  // (d_model_half/2, 1, width)
    sin_w = mx::broadcast_to(sin_w, {sin_w.shape(0), height, width});
    
    // Python: pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    auto cos_w = mx::cos(mx::matmul(pos_w, mx::reshape(div_term, {1, -1})));
    cos_w = mx::transpose(cos_w, {1, 0});
    cos_w = mx::reshape(cos_w, {-1, 1, width});
    cos_w = mx::broadcast_to(cos_w, {cos_w.shape(0), height, width});
    
    // Python: pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    auto sin_h = mx::sin(mx::matmul(pos_h, mx::reshape(div_term, {1, -1})));  // (height, d_model_half/2)
    sin_h = mx::transpose(sin_h, {1, 0});  // (d_model_half/2, height)
    sin_h = mx::reshape(sin_h, {-1, height, 1});  // (d_model_half/2, height, 1)
    sin_h = mx::broadcast_to(sin_h, {sin_h.shape(0), height, width});
    
    // Python: pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    auto cos_h = mx::cos(mx::matmul(pos_h, mx::reshape(div_term, {1, -1})));
    cos_h = mx::transpose(cos_h, {1, 0});
    cos_h = mx::reshape(cos_h, {-1, height, 1});
    cos_h = mx::broadcast_to(cos_h, {cos_h.shape(0), height, width});
    
    // Interleave sin/cos for width and height dimensions
    std::vector<mx::array> pe_parts;
    int num_freqs = sin_w.shape(0);
    
    for (int i = 0; i < num_freqs; ++i) {
        pe_parts.push_back(mx::slice(sin_w, {i, 0, 0}, {i+1, height, width}));
        pe_parts.push_back(mx::slice(cos_w, {i, 0, 0}, {i+1, height, width}));
    }
    for (int i = 0; i < num_freqs; ++i) {
        pe_parts.push_back(mx::slice(sin_h, {i, 0, 0}, {i+1, height, width}));
        pe_parts.push_back(mx::slice(cos_h, {i, 0, 0}, {i+1, height, width}));
    }
    
    pe = mx::concatenate(pe_parts, 0);
    
    // Add batch dimension: (1, d_model, height, width)
    return mx::expand_dims(pe, 0);
}

// ============================================================================
// LayerScale
// ============================================================================

LayerScale::LayerScale(int channels, float init, bool channel_last)
    : channel_last_(channel_last),
      scale_(mx::full({channels}, init, mx::float32)) {
}

mx::array LayerScale::forward(const mx::array& x) const {
    if (channel_last_) {
        // (B, T, C) format: scale is (C,), broadcast automatically
        return scale_ * x;
    } else {
        // (B, C, T) format: need to reshape scale to (C, 1)
        auto scale_reshaped = mx::reshape(scale_, {-1, 1});
        return scale_reshaped * x;
    }
}

// ============================================================================
// MyGroupNorm
// ============================================================================

MyGroupNorm::MyGroupNorm(int num_groups, int num_channels, float eps, bool layer_norm_mode)
    : num_groups_(num_groups),
      num_channels_(num_channels),
      eps_(eps),
      layer_norm_mode_(layer_norm_mode),
      weight_(mx::ones({num_channels})),
      bias_(mx::zeros({num_channels})) {
}

mx::array MyGroupNorm::forward(const mx::array& x) {
    if (layer_norm_mode_) {
        // LayerNorm mode: normalize over last dimension (C) for each (B, T) position
        // x: (B, T, C)
        auto mean = mx::mean(x, -1, true);
        auto var = mx::var(x, -1, true);
        auto normalized = (x - mean) / mx::sqrt(var + eps_);
        return normalized * weight_ + bias_;
    }
    
    // GroupNorm mode: transpose to (B, C, T), apply GroupNorm, transpose back
    // Python MyGroupNorm: x.transpose(1, 2) -> GroupNorm -> transpose back
    auto x_transposed = mx::transpose(x, {0, 2, 1});  // (B, C, T)
    auto normalized = utils::group_norm(x_transposed, weight_, bias_, num_groups_, eps_);
    return mx::transpose(normalized, {0, 2, 1});
}

// ============================================================================
// MyTransformerEncoderLayer
// ============================================================================

MyTransformerEncoderLayer::MyTransformerEncoderLayer(
    int d_model,
    int nhead,
    int dim_feedforward,
    float dropout,
    bool gelu,
    int group_norm,
    bool norm_first,
    bool norm_out,
    float layer_norm_eps,
    bool layer_scale,
    float init_values,
    bool batch_first
) : d_model_(d_model),
    nhead_(nhead),
    dim_feedforward_(dim_feedforward),
    dropout_(dropout),
    gelu_(gelu),
    norm_first_(norm_first),
    batch_first_(batch_first),
    self_attn_in_proj_weight_(mx::zeros({3 * d_model, d_model})),
    self_attn_in_proj_bias_(mx::zeros({3 * d_model})),
    self_attn_out_proj_weight_(mx::zeros({d_model, d_model})),
    self_attn_out_proj_bias_(mx::zeros({d_model})),
    linear1_weight_(mx::zeros({dim_feedforward, d_model})),
    linear1_bias_(mx::zeros({dim_feedforward})),
    linear2_weight_(mx::zeros({d_model, dim_feedforward})),
    linear2_bias_(mx::zeros({d_model}))
{
    // Create normalization layers
    // Python: when group_norm > 0, uses MyGroupNorm(group_norm, d_model)
    //         when group_norm == 0, inherits nn.LayerNorm from nn.TransformerEncoderLayer
    if (group_norm > 0) {
        norm1_ = std::make_unique<MyGroupNorm>(group_norm, d_model, layer_norm_eps, false);
        norm2_ = std::make_unique<MyGroupNorm>(group_norm, d_model, layer_norm_eps, false);
    } else {
        norm1_ = std::make_unique<MyGroupNorm>(1, d_model, layer_norm_eps, true);  // LayerNorm mode
        norm2_ = std::make_unique<MyGroupNorm>(1, d_model, layer_norm_eps, true);  // LayerNorm mode
    }
    
    // Create norm_out if requested
    // Python: self.norm_out = MyGroupNorm(num_groups=int(norm_out), num_channels=d_model)
    // This is always MyGroupNorm, not LayerNorm
    if (norm_first && norm_out) {
        norm_out_ = std::make_unique<MyGroupNorm>(1, d_model, layer_norm_eps, false);
    }
    
    // Create layer scale
    if (layer_scale) {
        gamma_1_ = std::make_unique<LayerScale>(d_model, init_values, true);
        gamma_2_ = std::make_unique<LayerScale>(d_model, init_values, true);
    }
}

mx::array MyTransformerEncoderLayer::forward(const mx::array& src) {
    // Python: x = src (T, B, C) or (B, T, C) if batch_first
    auto x = src;
    
    if (norm_first_) {
        // Python: x = x + self.gamma_1(self._sa_block(self.norm1(x), src_mask, src_key_padding_mask))
        auto normed = norm1_->forward(x);
        auto sa_out = _sa_block(normed);
        if (gamma_1_) {
            sa_out = gamma_1_->forward(sa_out);
        }
        x = x + sa_out;
        
        // Python: x = x + self.gamma_2(self._ff_block(self.norm2(x)))
        normed = norm2_->forward(x);
        auto ff_out = _ff_block(normed);
        if (gamma_2_) {
            ff_out = gamma_2_->forward(ff_out);
        }
        x = x + ff_out;
        
        // Python: if self.norm_out: x = self.norm_out(x)
        if (norm_out_) {
            x = norm_out_->forward(x);
        }
    } else {
        // Python: x = self.norm1(x + self.gamma_1(self._sa_block(x, src_mask, src_key_padding_mask)))
        auto sa_out = _sa_block(x);
        if (gamma_1_) {
            sa_out = gamma_1_->forward(sa_out);
        }
        x = norm1_->forward(x + sa_out);
        
        // Python: x = self.norm2(x + self.gamma_2(self._ff_block(x)))
        auto ff_out = _ff_block(x);
        if (gamma_2_) {
            ff_out = gamma_2_->forward(ff_out);
        }
        x = norm2_->forward(x + ff_out);
    }
    
    return x;
}

mx::array MyTransformerEncoderLayer::_sa_block(const mx::array& x) {
    // Multi-head self-attention
    // x shape: (B, T, C) if batch_first
    
    int B = x.shape(0);
    int T = x.shape(1);
    int C = x.shape(2);
    
    // Project to Q, K, V using combined weight matrix
    // Python: qkv = F.linear(x, self.self_attn.in_proj_weight, self.self_attn.in_proj_bias)
    auto qkv = mx::addmm(self_attn_in_proj_bias_, x, mx::transpose(self_attn_in_proj_weight_, {1, 0}));
    
    // Split into Q, K, V: (B, T, 3*C) -> 3 x (B, T, C)
    auto q = mx::slice(qkv, {0, 0, 0}, {B, T, d_model_});
    auto k = mx::slice(qkv, {0, 0, d_model_}, {B, T, 2 * d_model_});
    auto v = mx::slice(qkv, {0, 0, 2 * d_model_}, {B, T, 3 * d_model_});
    
    // Reshape for multi-head attention: (B, T, C) -> (B, num_heads, T, head_dim)
    int head_dim = d_model_ / nhead_;
    q = mx::reshape(q, {B, T, nhead_, head_dim});
    q = mx::transpose(q, {0, 2, 1, 3});  // (B, num_heads, T, head_dim)
    
    k = mx::reshape(k, {B, T, nhead_, head_dim});
    k = mx::transpose(k, {0, 2, 1, 3});
    
    v = mx::reshape(v, {B, T, nhead_, head_dim});
    v = mx::transpose(v, {0, 2, 1, 3});
    
    // Scaled dot-product attention
    // scores = (Q @ K^T) / sqrt(head_dim)
    auto k_t = mx::transpose(k, {0, 1, 3, 2});  // (B, num_heads, head_dim, T)
    auto scores = mx::matmul(q, k_t);  // (B, num_heads, T, T)
    scores = scores / std::sqrt(static_cast<float>(head_dim));
    
    // Softmax over last dimension
    auto attn_weights = mx::softmax(scores, -1);
    
    // Apply dropout (simplified: skip during inference)
    // attn_weights = dropout(attn_weights)
    
    // Weighted sum of values
    auto attn_output = mx::matmul(attn_weights, v);  // (B, num_heads, T, head_dim)
    
    // Reshape back: (B, num_heads, T, head_dim) -> (B, T, C)
    attn_output = mx::transpose(attn_output, {0, 2, 1, 3});  // (B, T, num_heads, head_dim)
    attn_output = mx::reshape(attn_output, {B, T, d_model_});
    
    // Output projection
    auto output = mx::addmm(self_attn_out_proj_bias_, attn_output, 
                            mx::transpose(self_attn_out_proj_weight_, {1, 0}));
    
    // Dropout (simplified: skip)
    return output;
}

mx::array MyTransformerEncoderLayer::_ff_block(const mx::array& x) {
    // Feed-forward network
    // Python: x = self.linear2(self.dropout(self.activation(self.linear1(x))))
    
    // Linear 1
    auto out = mx::addmm(linear1_bias_, x, mx::transpose(linear1_weight_, {1, 0}));
    
    // Activation
    if (gelu_) {
        out = utils::gelu(out);
    } else {
        out = mx::maximum(out, mx::array(0.0f));  // ReLU
    }
    
    // Dropout (simplified: skip)
    
    // Linear 2
    out = mx::addmm(linear2_bias_, out, mx::transpose(linear2_weight_, {1, 0}));
    
    // Dropout (simplified: skip)
    
    return out;
}

// ============================================================================
// CrossTransformerEncoderLayer
// ============================================================================

CrossTransformerEncoderLayer::CrossTransformerEncoderLayer(
    int d_model,
    int nhead,
    int dim_feedforward,
    float dropout,
    bool gelu,
    float layer_norm_eps,
    bool layer_scale,
    float init_values,
    bool norm_first,
    int group_norm,
    bool norm_out,
    bool batch_first
) : d_model_(d_model),
    nhead_(nhead),
    dim_feedforward_(dim_feedforward),
    dropout_(dropout),
    gelu_(gelu),
    norm_first_(norm_first),
    batch_first_(batch_first),
    cross_attn_q_proj_weight_(mx::zeros({d_model, d_model})),
    cross_attn_q_proj_bias_(mx::zeros({d_model})),
    cross_attn_k_proj_weight_(mx::zeros({d_model, d_model})),
    cross_attn_k_proj_bias_(mx::zeros({d_model})),
    cross_attn_v_proj_weight_(mx::zeros({d_model, d_model})),
    cross_attn_v_proj_bias_(mx::zeros({d_model})),
    cross_attn_out_proj_weight_(mx::zeros({d_model, d_model})),
    cross_attn_out_proj_bias_(mx::zeros({d_model})),
    linear1_weight_(mx::zeros({dim_feedforward, d_model})),
    linear1_bias_(mx::zeros({dim_feedforward})),
    linear2_weight_(mx::zeros({d_model, dim_feedforward})),
    linear2_bias_(mx::zeros({d_model}))
{
    // Create normalization layers
    // Python: when group_norm > 0, uses MyGroupNorm(group_norm, d_model)
    //         when group_norm == 0, uses nn.LayerNorm(d_model)
    if (group_norm > 0) {
        norm1_ = std::make_unique<MyGroupNorm>(group_norm, d_model, layer_norm_eps, false);
        norm2_ = std::make_unique<MyGroupNorm>(group_norm, d_model, layer_norm_eps, false);
        norm3_ = std::make_unique<MyGroupNorm>(group_norm, d_model, layer_norm_eps, false);
    } else {
        norm1_ = std::make_unique<MyGroupNorm>(1, d_model, layer_norm_eps, true);  // LayerNorm mode
        norm2_ = std::make_unique<MyGroupNorm>(1, d_model, layer_norm_eps, true);  // LayerNorm mode
        norm3_ = std::make_unique<MyGroupNorm>(1, d_model, layer_norm_eps, true);  // LayerNorm mode
    }
    
    // Create norm_out if requested
    // Python: self.norm_out = MyGroupNorm(num_groups=int(norm_out), num_channels=d_model)
    // This is always MyGroupNorm, not LayerNorm
    if (norm_first && norm_out) {
        norm_out_ = std::make_unique<MyGroupNorm>(1, d_model, layer_norm_eps, false);
    }
    
    // Create layer scale
    if (layer_scale) {
        gamma_1_ = std::make_unique<LayerScale>(d_model, init_values, true);
        gamma_2_ = std::make_unique<LayerScale>(d_model, init_values, true);
    }
}

mx::array CrossTransformerEncoderLayer::forward(const mx::array& q, const mx::array& k) {
    // Python: q: (T, B, C) or (B, T, C) if batch_first
    //         k: (S, B, C) or (B, S, C) if batch_first
    auto x = q;
    
    if (norm_first_) {
        // Python: x = q + self.gamma_1(self._ca_block(self.norm1(q), self.norm2(k), mask))
        auto q_normed = norm1_->forward(q);
        auto k_normed = norm2_->forward(k);
        auto ca_out = _ca_block(q_normed, k_normed);
        if (gamma_1_) {
            ca_out = gamma_1_->forward(ca_out);
        }
        x = q + ca_out;
        
        // Python: x = x + self.gamma_2(self._ff_block(self.norm3(x)))
        auto normed = norm3_->forward(x);
        auto ff_out = _ff_block(normed);
        if (gamma_2_) {
            ff_out = gamma_2_->forward(ff_out);
        }
        x = x + ff_out;
        
        // Python: if self.norm_out: x = self.norm_out(x)
        if (norm_out_) {
            x = norm_out_->forward(x);
        }
    } else {
        // Python: x = self.norm1(q + self.gamma_1(self._ca_block(q, k, mask)))
        auto ca_out = _ca_block(q, k);
        if (gamma_1_) {
            ca_out = gamma_1_->forward(ca_out);
        }
        x = norm1_->forward(q + ca_out);
        
        // Python: x = self.norm2(x + self.gamma_2(self._ff_block(x)))
        auto ff_out = _ff_block(x);
        if (gamma_2_) {
            ff_out = gamma_2_->forward(ff_out);
        }
        x = norm2_->forward(x + ff_out);
    }
    
    return x;
}

mx::array CrossTransformerEncoderLayer::_ca_block(const mx::array& q, const mx::array& k) {
    // Multi-head cross-attention: q queries k
    // q shape: (B, T_q, C)
    // k shape: (B, T_k, C)
    
    int B_q = q.shape(0);
    int T_q = q.shape(1);
    int C = q.shape(2);
    
    int B_k = k.shape(0);
    int T_k = k.shape(1);
    
    // Project Q from query
    auto query = mx::addmm(cross_attn_q_proj_bias_, q, 
                           mx::transpose(cross_attn_q_proj_weight_, {1, 0}));
    
    // Project K and V from key
    auto key = mx::addmm(cross_attn_k_proj_bias_, k,
                         mx::transpose(cross_attn_k_proj_weight_, {1, 0}));
    auto value = mx::addmm(cross_attn_v_proj_bias_, k,
                           mx::transpose(cross_attn_v_proj_weight_, {1, 0}));
    
    // Reshape for multi-head attention
    int head_dim = d_model_ / nhead_;
    
    query = mx::reshape(query, {B_q, T_q, nhead_, head_dim});
    query = mx::transpose(query, {0, 2, 1, 3});  // (B, num_heads, T_q, head_dim)
    
    key = mx::reshape(key, {B_k, T_k, nhead_, head_dim});
    key = mx::transpose(key, {0, 2, 1, 3});  // (B, num_heads, T_k, head_dim)
    
    value = mx::reshape(value, {B_k, T_k, nhead_, head_dim});
    value = mx::transpose(value, {0, 2, 1, 3});  // (B, num_heads, T_k, head_dim)
    
    // Scaled dot-product attention
    auto key_t = mx::transpose(key, {0, 1, 3, 2});  // (B, num_heads, head_dim, T_k)
    auto scores = mx::matmul(query, key_t);  // (B, num_heads, T_q, T_k)
    scores = scores / std::sqrt(static_cast<float>(head_dim));
    
    // Softmax
    auto attn_weights = mx::softmax(scores, -1);
    
    // Dropout (simplified: skip)
    
    // Weighted sum
    auto attn_output = mx::matmul(attn_weights, value);  // (B, num_heads, T_q, head_dim)
    
    // Reshape back
    attn_output = mx::transpose(attn_output, {0, 2, 1, 3});  // (B, T_q, num_heads, head_dim)
    attn_output = mx::reshape(attn_output, {B_q, T_q, d_model_});
    
    // Output projection
    auto output = mx::addmm(cross_attn_out_proj_bias_, attn_output,
                            mx::transpose(cross_attn_out_proj_weight_, {1, 0}));
    
    // Dropout (simplified: skip)
    return output;
}

mx::array CrossTransformerEncoderLayer::_ff_block(const mx::array& x) {
    // Feed-forward network (same as MyTransformerEncoderLayer)
    
    // Linear 1
    auto out = mx::addmm(linear1_bias_, x, mx::transpose(linear1_weight_, {1, 0}));
    
    // Activation
    if (gelu_) {
        out = utils::gelu(out);
    } else {
        out = mx::maximum(out, mx::array(0.0f));  // ReLU
    }
    
    // Dropout (simplified: skip)
    
    // Linear 2
    out = mx::addmm(linear2_bias_, out, mx::transpose(linear2_weight_, {1, 0}));
    
    // Dropout (simplified: skip)
    
    return out;
}

// ============================================================================
// CrossTransformerEncoder
// ============================================================================

CrossTransformerEncoder::CrossTransformerEncoder(
    int dim,
    std::string emb,
    float hidden_scale,
    int num_heads,
    int num_layers,
    bool cross_first,
    float dropout,
    int max_positions,
    bool norm_in,
    bool norm_in_group,
    int group_norm,
    bool norm_first,
    bool norm_out,
    float max_period,
    float weight_decay,
    bool layer_scale,
    bool gelu,
    int sin_random_shift,
    float weight_pos_embed
) : num_layers_(num_layers),
    classic_parity_(cross_first ? 1 : 0),
    emb_(emb),
    max_period_(max_period),
    weight_pos_embed_(weight_pos_embed),
    sin_random_shift_(sin_random_shift)
{
    int hidden_dim = static_cast<int>(dim * hidden_scale);
    
    // Create norm_in layers
    // Python: norm_in uses nn.LayerNorm(dim), not MyGroupNorm
    if (norm_in) {
        norm_in_ = std::make_unique<MyGroupNorm>(1, dim, 1e-5f, true);  // LayerNorm mode
        norm_in_t_ = std::make_unique<MyGroupNorm>(1, dim, 1e-5f, true);  // LayerNorm mode
    } else if (norm_in_group) {
        norm_in_ = std::make_unique<MyGroupNorm>(norm_in_group, dim);
        norm_in_t_ = std::make_unique<MyGroupNorm>(norm_in_group, dim);
    }
    
    // Create transformer layers (alternating classic and cross)
    for (int idx = 0; idx < num_layers; ++idx) {
        if (idx % 2 == classic_parity_) {
            // Classic layer (self-attention)
            auto layer = std::make_unique<MyTransformerEncoderLayer>(
                dim, num_heads, hidden_dim, dropout, gelu,
                group_norm, norm_first, norm_out, 1e-5f, layer_scale, 1e-4f, true
            );
            classic_layers_.push_back(std::move(layer));
            
            auto layer_t = std::make_unique<MyTransformerEncoderLayer>(
                dim, num_heads, hidden_dim, dropout, gelu,
                group_norm, norm_first, norm_out, 1e-5f, layer_scale, 1e-4f, true
            );
            classic_layers_t_.push_back(std::move(layer_t));
        } else {
            // Cross layer (cross-attention)
            auto layer = std::make_unique<CrossTransformerEncoderLayer>(
                dim, num_heads, hidden_dim, dropout, gelu,
                1e-5f, layer_scale, 1e-4f, norm_first, group_norm, norm_out, true
            );
            cross_layers_.push_back(std::move(layer));
            
            auto layer_t = std::make_unique<CrossTransformerEncoderLayer>(
                dim, num_heads, hidden_dim, dropout, gelu,
                1e-5f, layer_scale, 1e-4f, norm_first, group_norm, norm_out, true
            );
            cross_layers_t_.push_back(std::move(layer_t));
        }
    }
}

std::pair<mx::array, mx::array> CrossTransformerEncoder::forward(
    const mx::array& x,
    const mx::array& xt
) {
    // Python: B, C, Fr, T1 = x.shape
    int B = x.shape(0);
    int C = x.shape(1);
    int Fr = x.shape(2);
    int T1 = x.shape(3);
    
    // Python: pos_emb_2d = create_2d_sin_embedding(C, Fr, T1, x.device, self.max_period)
    auto pos_emb_2d = create_2d_sin_embedding(C, Fr, T1, max_period_);  // (1, C, Fr, T1)
    
    // Python: pos_emb_2d = rearrange(pos_emb_2d, "b c fr t1 -> b (t1 fr) c")
    // Swap Fr and T1 first so T1 varies slowly, Fr varies fast when flattened
    pos_emb_2d = mx::transpose(pos_emb_2d, {0, 1, 3, 2});  // (1, C, T1, Fr)
    pos_emb_2d = mx::reshape(pos_emb_2d, {1, C, T1 * Fr});
    pos_emb_2d = mx::transpose(pos_emb_2d, {0, 2, 1});  // (1, T1*Fr, C)
    
    // Python: x = rearrange(x, "b c fr t1 -> b (t1 fr) c")
    auto x_reshaped = mx::transpose(x, {0, 1, 3, 2});  // (B, C, T1, Fr)
    x_reshaped = mx::reshape(x_reshaped, {B, C, T1 * Fr});
    x_reshaped = mx::transpose(x_reshaped, {0, 2, 1});  // (B, T1*Fr, C)
    
    // Python: x = self.norm_in(x)
    if (norm_in_) {
        x_reshaped = norm_in_->forward(x_reshaped);
    }
    
    // Python: x = x + self.weight_pos_embed * pos_emb_2d
    x_reshaped = x_reshaped + weight_pos_embed_ * pos_emb_2d;
    
    // Process time branch
    // Python: B, C, T2 = xt.shape
    int T2 = xt.shape(2);
    
    // Python: xt = rearrange(xt, "b c t2 -> b t2 c")
    auto xt_reshaped = mx::transpose(xt, {0, 2, 1});  // (B, T2, C)
    
    // Python: pos_emb = self._get_pos_embedding(T2, B, C, x.device)
    auto pos_emb = _get_pos_embedding(T2, B, C);  // (T2, B, C) or (B, T2, C)
    
    // Python: pos_emb = rearrange(pos_emb, "t2 b c -> b t2 c")
    // (already in batch_first format from our implementation)
    
    // Python: xt = self.norm_in_t(xt)
    if (norm_in_t_) {
        xt_reshaped = norm_in_t_->forward(xt_reshaped);
    }
    
    // Python: xt = xt + self.weight_pos_embed * pos_emb
    xt_reshaped = xt_reshaped + weight_pos_embed_ * pos_emb;
    
    // Apply transformer layers
    int classic_idx = 0;
    int cross_idx = 0;
    
    for (int idx = 0; idx < num_layers_; ++idx) {
        if (idx % 2 == classic_parity_) {
            // Classic layer (self-attention on each branch)
            x_reshaped = classic_layers_[classic_idx]->forward(x_reshaped);
            xt_reshaped = classic_layers_t_[classic_idx]->forward(xt_reshaped);
            classic_idx++;
        } else {
            // Cross layer (cross-attention between branches)
            auto old_x = x_reshaped;
            x_reshaped = cross_layers_[cross_idx]->forward(x_reshaped, xt_reshaped);
            xt_reshaped = cross_layers_t_[cross_idx]->forward(xt_reshaped, old_x);
            cross_idx++;
        }
    }
    
    // Reshape back to original format
    // Python: x = rearrange(x, "b (t1 fr) c -> b c fr t1", t1=T1)
    x_reshaped = mx::transpose(x_reshaped, {0, 2, 1});  // (B, C, T1*Fr)
    x_reshaped = mx::reshape(x_reshaped, {B, C, T1, Fr});
    x_reshaped = mx::transpose(x_reshaped, {0, 1, 3, 2});  // (B, C, Fr, T1)
    
    // Python: xt = rearrange(xt, "b t2 c -> b c t2")
    xt_reshaped = mx::transpose(xt_reshaped, {0, 2, 1});  // (B, C, T2)
    
    return {x_reshaped, xt_reshaped};
}

mx::array CrossTransformerEncoder::_get_pos_embedding(int T, int B, int C) {
    // Python: if self.emb == "sin":
    if (emb_ == "sin") {
        // Python: shift = random.randrange(self.sin_random_shift + 1)
        int shift = 0;  // Simplified: no random shift during inference
        
        // Python: pos_emb = create_sin_embedding(T, C, shift=shift, device=device, max_period=self.max_period)
        auto pos_emb = create_sin_embedding(T, C, shift, max_period_);  // (T, 1, C)
        
        // Convert to batch_first format: (T, 1, C) -> (1, T, C) -> broadcast to (B, T, C)
        pos_emb = mx::transpose(pos_emb, {1, 0, 2});  // (1, T, C)
        pos_emb = mx::broadcast_to(pos_emb, {B, T, C});
        
        return pos_emb;
    }
    // TODO: Add support for "cape" and "scaled" embeddings if needed
    
    throw std::runtime_error("Unsupported embedding type: " + emb_);
}

} // namespace demucs
