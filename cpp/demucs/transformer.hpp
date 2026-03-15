#pragma once

#include <mlx/mlx.h>
#include <string>
#include <vector>
#include <memory>

namespace demucs {

namespace mx = mlx::core;

/**
 * Create sinusoidal positional embeddings (1D)
 * Matches create_sin_embedding() from transformer.py
 */
mx::array create_sin_embedding(
    int length,
    int dim,
    int shift = 0,
    float max_period = 10000.0f
);

/**
 * Create 2D sinusoidal positional embeddings for frequency-time grid
 * Matches create_2d_sin_embedding() from transformer.py
 */
mx::array create_2d_sin_embedding(
    int d_model,
    int height,
    int width,
    float max_period = 10000.0f
);

/**
 * LayerScale - rescales residual outputs close to 0 initially
 * Matches LayerScale class from transformer.py
 */
class LayerScale {
public:
    LayerScale(int channels, float init = 1e-4f, bool channel_last = true);
    
    mx::array forward(const mx::array& x) const;
    
    // Weight access for loading
    mx::array& scale() { return scale_; }
    const mx::array& scale() const { return scale_; }

private:
    mx::array scale_;
    bool channel_last_;
};

/**
 * MyGroupNorm - GroupNorm that expects (B, T, C) format
 * Matches MyGroupNorm class from transformer.py
 */
class MyGroupNorm {
public:
    MyGroupNorm(int num_groups, int num_channels, float eps = 1e-5f, bool layer_norm_mode = false);
    
    mx::array forward(const mx::array& x);
    
    // Weight access for loading
    mx::array& weight() { return weight_; }
    mx::array& bias() { return bias_; }
    const mx::array& weight() const { return weight_; }
    const mx::array& bias() const { return bias_; }

private:
    int num_groups_;
    int num_channels_;
    float eps_;
    bool layer_norm_mode_;
    mx::array weight_;
    mx::array bias_;
};

/**
 * MyTransformerEncoderLayer - Self-attention transformer layer
 * Matches MyTransformerEncoderLayer class from transformer.py
 */
class MyTransformerEncoderLayer {
public:
    MyTransformerEncoderLayer(
        int d_model,
        int nhead,
        int dim_feedforward = 2048,
        float dropout = 0.1f,
        bool gelu = true,
        int group_norm = 0,
        bool norm_first = false,
        bool norm_out = false,
        float layer_norm_eps = 1e-5f,
        bool layer_scale = false,
        float init_values = 1e-4f,
        bool batch_first = true
    );
    
    mx::array forward(const mx::array& src);
    
    // Weight access for loading
    mx::array& self_attn_in_proj_weight() { return self_attn_in_proj_weight_; }
    mx::array& self_attn_in_proj_bias() { return self_attn_in_proj_bias_; }
    mx::array& self_attn_out_proj_weight() { return self_attn_out_proj_weight_; }
    mx::array& self_attn_out_proj_bias() { return self_attn_out_proj_bias_; }
    mx::array& linear1_weight() { return linear1_weight_; }
    mx::array& linear1_bias() { return linear1_bias_; }
    mx::array& linear2_weight() { return linear2_weight_; }
    mx::array& linear2_bias() { return linear2_bias_; }
    std::unique_ptr<MyGroupNorm>& norm1() { return norm1_; }
    std::unique_ptr<MyGroupNorm>& norm2() { return norm2_; }
    std::unique_ptr<MyGroupNorm>& norm_out() { return norm_out_; }
    std::unique_ptr<LayerScale>& gamma_1() { return gamma_1_; }
    std::unique_ptr<LayerScale>& gamma_2() { return gamma_2_; }

private:
    mx::array _sa_block(const mx::array& x);
    mx::array _ff_block(const mx::array& x);
    
    int d_model_;
    int nhead_;
    int dim_feedforward_;
    float dropout_;
    bool gelu_;
    bool norm_first_;
    bool batch_first_;
    
    // Self-attention weights (combined Q, K, V projection)
    mx::array self_attn_in_proj_weight_;  // (3*d_model, d_model)
    mx::array self_attn_in_proj_bias_;    // (3*d_model,)
    mx::array self_attn_out_proj_weight_; // (d_model, d_model)
    mx::array self_attn_out_proj_bias_;   // (d_model,)
    
    // Feed-forward weights
    mx::array linear1_weight_;  // (dim_feedforward, d_model)
    mx::array linear1_bias_;    // (dim_feedforward,)
    mx::array linear2_weight_;  // (d_model, dim_feedforward)
    mx::array linear2_bias_;    // (d_model,)
    
    // Normalization layers
    std::unique_ptr<MyGroupNorm> norm1_;
    std::unique_ptr<MyGroupNorm> norm2_;
    std::unique_ptr<MyGroupNorm> norm_out_;
    
    // Layer scale
    std::unique_ptr<LayerScale> gamma_1_;
    std::unique_ptr<LayerScale> gamma_2_;
};

/**
 * CrossTransformerEncoderLayer - Cross-attention transformer layer
 * Matches CrossTransformerEncoderLayer class from transformer.py
 */
class CrossTransformerEncoderLayer {
public:
    CrossTransformerEncoderLayer(
        int d_model,
        int nhead,
        int dim_feedforward = 2048,
        float dropout = 0.1f,
        bool gelu = true,
        float layer_norm_eps = 1e-5f,
        bool layer_scale = false,
        float init_values = 1e-4f,
        bool norm_first = false,
        int group_norm = 0,
        bool norm_out = false,
        bool batch_first = true
    );
    
    // Forward: q queries k
    mx::array forward(const mx::array& q, const mx::array& k);
    
    // Weight access for loading
    mx::array& cross_attn_q_proj_weight() { return cross_attn_q_proj_weight_; }
    mx::array& cross_attn_q_proj_bias() { return cross_attn_q_proj_bias_; }
    mx::array& cross_attn_k_proj_weight() { return cross_attn_k_proj_weight_; }
    mx::array& cross_attn_k_proj_bias() { return cross_attn_k_proj_bias_; }
    mx::array& cross_attn_v_proj_weight() { return cross_attn_v_proj_weight_; }
    mx::array& cross_attn_v_proj_bias() { return cross_attn_v_proj_bias_; }
    mx::array& cross_attn_out_proj_weight() { return cross_attn_out_proj_weight_; }
    mx::array& cross_attn_out_proj_bias() { return cross_attn_out_proj_bias_; }
    mx::array& linear1_weight() { return linear1_weight_; }
    mx::array& linear1_bias() { return linear1_bias_; }
    mx::array& linear2_weight() { return linear2_weight_; }
    mx::array& linear2_bias() { return linear2_bias_; }
    std::unique_ptr<MyGroupNorm>& norm1() { return norm1_; }
    std::unique_ptr<MyGroupNorm>& norm2() { return norm2_; }
    std::unique_ptr<MyGroupNorm>& norm3() { return norm3_; }
    std::unique_ptr<MyGroupNorm>& norm_out() { return norm_out_; }
    std::unique_ptr<LayerScale>& gamma_1() { return gamma_1_; }
    std::unique_ptr<LayerScale>& gamma_2() { return gamma_2_; }

private:
    mx::array _ca_block(const mx::array& q, const mx::array& k);
    mx::array _ff_block(const mx::array& x);
    
    int d_model_;
    int nhead_;
    int dim_feedforward_;
    float dropout_;
    bool gelu_;
    bool norm_first_;
    bool batch_first_;
    
    // Cross-attention weights (separate Q, K, V projections)
    mx::array cross_attn_q_proj_weight_;  // (d_model, d_model)
    mx::array cross_attn_q_proj_bias_;    // (d_model,)
    mx::array cross_attn_k_proj_weight_;  // (d_model, d_model)
    mx::array cross_attn_k_proj_bias_;    // (d_model,)
    mx::array cross_attn_v_proj_weight_;  // (d_model, d_model)
    mx::array cross_attn_v_proj_bias_;    // (d_model,)
    mx::array cross_attn_out_proj_weight_; // (d_model, d_model)
    mx::array cross_attn_out_proj_bias_;   // (d_model,)
    
    // Feed-forward weights
    mx::array linear1_weight_;  // (dim_feedforward, d_model)
    mx::array linear1_bias_;    // (dim_feedforward,)
    mx::array linear2_weight_;  // (d_model, dim_feedforward)
    mx::array linear2_bias_;    // (d_model,)
    
    // Normalization layers
    std::unique_ptr<MyGroupNorm> norm1_;  // For query
    std::unique_ptr<MyGroupNorm> norm2_;  // For key/value
    std::unique_ptr<MyGroupNorm> norm3_;  // For FFN input
    std::unique_ptr<MyGroupNorm> norm_out_;
    
    // Layer scale
    std::unique_ptr<LayerScale> gamma_1_;
    std::unique_ptr<LayerScale> gamma_2_;
};

/**
 * CrossTransformerEncoder - Full cross-transformer with alternating layers
 * Matches CrossTransformerEncoder class from transformer.py
 */
class CrossTransformerEncoder {
public:
    CrossTransformerEncoder(
        int dim,
        std::string emb = "sin",
        float hidden_scale = 4.0f,
        int num_heads = 8,
        int num_layers = 6,
        bool cross_first = false,
        float dropout = 0.0f,
        int max_positions = 1000,
        bool norm_in = true,
        bool norm_in_group = false,
        int group_norm = 0,
        bool norm_first = false,
        bool norm_out = false,
        float max_period = 10000.0f,
        float weight_decay = 0.0f,
        bool layer_scale = false,
        bool gelu = true,
        int sin_random_shift = 0,
        float weight_pos_embed = 1.0f
    );
    
    // Forward pass: process frequency and time branches
    std::pair<mx::array, mx::array> forward(const mx::array& x, const mx::array& xt);
    
    // Access to layers for weight loading
    std::vector<std::unique_ptr<MyTransformerEncoderLayer>>& classic_layers() { return classic_layers_; }
    std::vector<std::unique_ptr<MyTransformerEncoderLayer>>& classic_layers_t() { return classic_layers_t_; }
    std::vector<std::unique_ptr<CrossTransformerEncoderLayer>>& cross_layers() { return cross_layers_; }
    std::vector<std::unique_ptr<CrossTransformerEncoderLayer>>& cross_layers_t() { return cross_layers_t_; }
    
    // Norm_in access
    std::unique_ptr<MyGroupNorm>& norm_in() { return norm_in_; }
    std::unique_ptr<MyGroupNorm>& norm_in_t() { return norm_in_t_; }
    
    // Config access
    int classic_parity() const { return classic_parity_; }
    int num_layers() const { return num_layers_; }
    float max_period() const { return max_period_; }
    float weight_pos_embed() const { return weight_pos_embed_; }

private:
    mx::array _get_pos_embedding(int T, int B, int C);
    
    int num_layers_;
    int classic_parity_;  // 1 if cross_first, else 0
    std::string emb_;
    float max_period_;
    float weight_pos_embed_;
    int sin_random_shift_;
    
    // Normalization before transformer
    std::unique_ptr<MyGroupNorm> norm_in_;
    std::unique_ptr<MyGroupNorm> norm_in_t_;
    
    // Transformer layers (alternating classic and cross)
    std::vector<std::unique_ptr<MyTransformerEncoderLayer>> classic_layers_;
    std::vector<std::unique_ptr<MyTransformerEncoderLayer>> classic_layers_t_;
    std::vector<std::unique_ptr<CrossTransformerEncoderLayer>> cross_layers_;
    std::vector<std::unique_ptr<CrossTransformerEncoderLayer>> cross_layers_t_;
};

} // namespace demucs
