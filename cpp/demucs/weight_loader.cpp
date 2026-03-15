#include "weight_loader.hpp"
#include <iostream>

namespace demucs {

bool WeightLoader::load_dconv(
    DConv& dconv,
    const std::unordered_map<std::string, mx::array>& weights,
    const std::string& prefix
) {
    auto& layers = dconv.layers();
    
    for (size_t i = 0; i < layers.size(); ++i) {
        // PyTorch format: encoder.0.dconv.layers.0.0.weight (conv1)
        //                 encoder.0.dconv.layers.0.1.weight (norm1)
        //                 encoder.0.dconv.layers.0.3.weight (conv2)
        //                 encoder.0.dconv.layers.0.4.weight (norm2)
        //                 encoder.0.dconv.layers.0.6.scale (layer_scale)
        std::string layer_prefix = prefix + "dconv.layers." + std::to_string(i) + ".";
        auto& layer = layers[i];
        
        // Load conv1 weights (index 0)
        auto conv1_w = weights.find(layer_prefix + "0.weight");
        auto conv1_b = weights.find(layer_prefix + "0.bias");
        if (conv1_w != weights.end() && conv1_b != weights.end()) {
            layer.conv1_weight = conv1_w->second;
            layer.conv1_bias = conv1_b->second;
        } else {
            std::cerr << "Missing conv1 weights for " << layer_prefix << std::endl;
            return false;
        }
        
        // Load norm1 weights (index 1)
        auto norm1_w = weights.find(layer_prefix + "1.weight");
        auto norm1_b = weights.find(layer_prefix + "1.bias");
        if (norm1_w != weights.end() && norm1_b != weights.end()) {
            layer.norm1_weight = norm1_w->second;
            layer.norm1_bias = norm1_b->second;
        }
        
        // Load conv2 weights (index 3)
        auto conv2_w = weights.find(layer_prefix + "3.weight");
        auto conv2_b = weights.find(layer_prefix + "3.bias");
        if (conv2_w != weights.end() && conv2_b != weights.end()) {
            layer.conv2_weight = conv2_w->second;
            layer.conv2_bias = conv2_b->second;
        } else {
            std::cerr << "Missing conv2 weights for " << layer_prefix << std::endl;
            return false;
        }
        
        // Load norm2 weights (index 4)
        auto norm2_w = weights.find(layer_prefix + "4.weight");
        auto norm2_b = weights.find(layer_prefix + "4.bias");
        if (norm2_w != weights.end() && norm2_b != weights.end()) {
            layer.norm2_weight = norm2_w->second;
            layer.norm2_bias = norm2_b->second;
        }
        
        // Load layer scale (index 6)
        auto scale = weights.find(layer_prefix + "6.scale");
        if (scale != weights.end()) {
            layer.layer_scale.scale() = scale->second;
        } else {
            std::cerr << "Missing layer scale for " << layer_prefix << std::endl;
            return false;
        }
    }
    
    return true;
}

bool WeightLoader::load_encoder_layer(
    HEncLayer& layer,
    const std::unordered_map<std::string, mx::array>& weights,
    const std::string& prefix
) {
    // Load main conv weights
    auto conv_w = weights.find(prefix + "conv.weight");
    auto conv_b = weights.find(prefix + "conv.bias");
    if (conv_w != weights.end() && conv_b != weights.end()) {
        layer.conv_weight() = conv_w->second;
        layer.conv_bias() = conv_b->second;
    } else {
        std::cerr << "Missing conv weights for " << prefix << std::endl;
        return false;
    }
    
    // Load norm1 weights (may be Identity)
    auto norm1_w = weights.find(prefix + "norm1.weight");
    auto norm1_b = weights.find(prefix + "norm1.bias");
    if (norm1_w != weights.end() && norm1_b != weights.end()) {
        layer.norm1_weight() = norm1_w->second;
        layer.norm1_bias() = norm1_b->second;
    }
    
    // Load rewrite weights
    auto rewrite_w = weights.find(prefix + "rewrite.weight");
    auto rewrite_b = weights.find(prefix + "rewrite.bias");
    if (rewrite_w != weights.end() && rewrite_b != weights.end()) {
        layer.rewrite_weight() = rewrite_w->second;
        layer.rewrite_bias() = rewrite_b->second;
    }
    
    // Load norm2 weights
    auto norm2_w = weights.find(prefix + "norm2.weight");
    auto norm2_b = weights.find(prefix + "norm2.bias");
    if (norm2_w != weights.end() && norm2_b != weights.end()) {
        layer.norm2_weight() = norm2_w->second;
        layer.norm2_bias() = norm2_b->second;
    }
    
    // Load DConv if present
    if (layer.dconv()) {
        if (!load_dconv(*layer.dconv(), weights, prefix)) {
            return false;
        }
    }
    
    return true;
}

bool WeightLoader::load_decoder_layer(
    HDecLayer& layer,
    const std::unordered_map<std::string, mx::array>& weights,
    const std::string& prefix
) {
    // Load conv_tr weights
    auto conv_tr_w = weights.find(prefix + "conv_tr.weight");
    auto conv_tr_b = weights.find(prefix + "conv_tr.bias");
    if (conv_tr_w != weights.end() && conv_tr_b != weights.end()) {
        layer.conv_tr_weight() = conv_tr_w->second;
        layer.conv_tr_bias() = conv_tr_b->second;
    } else {
        std::cerr << "Missing conv_tr weights for " << prefix << std::endl;
        return false;
    }
    
    // Load norm2 weights
    auto norm2_w = weights.find(prefix + "norm2.weight");
    auto norm2_b = weights.find(prefix + "norm2.bias");
    if (norm2_w != weights.end() && norm2_b != weights.end()) {
        layer.norm2_weight() = norm2_w->second;
        layer.norm2_bias() = norm2_b->second;
    }
    
    // Load rewrite weights
    auto rewrite_w = weights.find(prefix + "rewrite.weight");
    auto rewrite_b = weights.find(prefix + "rewrite.bias");
    if (rewrite_w != weights.end() && rewrite_b != weights.end()) {
        layer.rewrite_weight() = rewrite_w->second;
        layer.rewrite_bias() = rewrite_b->second;
    }
    
    // Load norm1 weights
    auto norm1_w = weights.find(prefix + "norm1.weight");
    auto norm1_b = weights.find(prefix + "norm1.bias");
    if (norm1_w != weights.end() && norm1_b != weights.end()) {
        layer.norm1_weight() = norm1_w->second;
        layer.norm1_bias() = norm1_b->second;
    }
    
    // Load DConv if present
    if (layer.dconv()) {
        if (!load_dconv(*layer.dconv(), weights, prefix)) {
            return false;
        }
    }
    
    return true;
}

bool WeightLoader::load_scaled_embedding(
    ScaledEmbedding& emb,
    const std::unordered_map<std::string, mx::array>& weights,
    const std::string& prefix
) {
    // PyTorch format: freq_emb.embedding.weight
    auto emb_w = weights.find(prefix + "embedding.weight");
    if (emb_w != weights.end()) {
        emb.embedding_weight() = emb_w->second;
        return true;
    }
    
    std::cerr << "Missing embedding weight for " << prefix << std::endl;
    return false;
}

bool WeightLoader::load_hdemucs_from_map(
    HDemucs& model,
    const std::unordered_map<std::string, mx::array>& weights
) {
    // Load encoder layers
    for (size_t i = 0; i < model.encoder().size(); ++i) {
        std::string prefix = "encoder." + std::to_string(i) + ".";
        if (!load_encoder_layer(*model.encoder()[i], weights, prefix)) {
            std::cerr << "Failed to load encoder layer " << i << std::endl;
            return false;
        }
    }
    
    // Load decoder layers
    for (size_t i = 0; i < model.decoder().size(); ++i) {
        std::string prefix = "decoder." + std::to_string(i) + ".";
        if (!load_decoder_layer(*model.decoder()[i], weights, prefix)) {
            std::cerr << "Failed to load decoder layer " << i << std::endl;
            return false;
        }
    }
    
    // Load time encoder layers (if hybrid)
    for (size_t i = 0; i < model.tencoder().size(); ++i) {
        std::string prefix = "tencoder." + std::to_string(i) + ".";
        if (!load_encoder_layer(*model.tencoder()[i], weights, prefix)) {
            std::cerr << "Failed to load time encoder layer " << i << std::endl;
            return false;
        }
    }
    
    // Load time decoder layers (if hybrid)
    for (size_t i = 0; i < model.tdecoder().size(); ++i) {
        std::string prefix = "tdecoder." + std::to_string(i) + ".";
        if (!load_decoder_layer(*model.tdecoder()[i], weights, prefix)) {
            std::cerr << "Failed to load time decoder layer " << i << std::endl;
            return false;
        }
    }
    
    // Load frequency embedding (if present)
    if (model.freq_emb()) {
        if (!load_scaled_embedding(*model.freq_emb(), weights, "freq_emb.")) {
            std::cerr << "Failed to load frequency embedding" << std::endl;
            return false;
        }
    }
    
    return true;
}

bool WeightLoader::load_hdemucs_weights(
    HDemucs& model,
    const std::string& safetensors_path
) {
    // Parse SafeTensors file
    auto file = SafeTensorsLoader::parse(safetensors_path);
    if (!file) {
        std::cerr << "Failed to parse SafeTensors file: " << safetensors_path << std::endl;
        return false;
    }
    
    std::cout << "Loading weights from: " << safetensors_path << std::endl;
    SafeTensorsLoader::print_info(*file);
    
    // Load all tensors
    auto weights = SafeTensorsLoader::load_all(*file);
    
    std::cout << "Loaded " << weights.size() << " tensors" << std::endl;
    
    // Load into model
    return load_hdemucs_from_map(model, weights);
}

// ============================================================================
// HTDemucs Weight Loading
// ============================================================================

bool WeightLoader::load_transformer_encoder_layer(
    MyTransformerEncoderLayer& layer,
    const std::unordered_map<std::string, mx::array>& weights,
    const std::string& prefix
) {
    // Load self-attention weights (combined QKV projection)
    std::string in_proj_weight_key = prefix + "self_attn.in_proj_weight";
    std::string in_proj_bias_key = prefix + "self_attn.in_proj_bias";
    std::string out_proj_weight_key = prefix + "self_attn.out_proj.weight";
    std::string out_proj_bias_key = prefix + "self_attn.out_proj.bias";
    
    if (weights.find(in_proj_weight_key) != weights.end()) {
        layer.self_attn_in_proj_weight() = weights.at(in_proj_weight_key);
    }
    if (weights.find(in_proj_bias_key) != weights.end()) {
        layer.self_attn_in_proj_bias() = weights.at(in_proj_bias_key);
    }
    if (weights.find(out_proj_weight_key) != weights.end()) {
        layer.self_attn_out_proj_weight() = weights.at(out_proj_weight_key);
    }
    if (weights.find(out_proj_bias_key) != weights.end()) {
        layer.self_attn_out_proj_bias() = weights.at(out_proj_bias_key);
    }
    
    // Load feed-forward weights
    std::string linear1_weight_key = prefix + "linear1.weight";
    std::string linear1_bias_key = prefix + "linear1.bias";
    std::string linear2_weight_key = prefix + "linear2.weight";
    std::string linear2_bias_key = prefix + "linear2.bias";
    
    if (weights.find(linear1_weight_key) != weights.end()) {
        layer.linear1_weight() = weights.at(linear1_weight_key);
    }
    if (weights.find(linear1_bias_key) != weights.end()) {
        layer.linear1_bias() = weights.at(linear1_bias_key);
    }
    if (weights.find(linear2_weight_key) != weights.end()) {
        layer.linear2_weight() = weights.at(linear2_weight_key);
    }
    if (weights.find(linear2_bias_key) != weights.end()) {
        layer.linear2_bias() = weights.at(linear2_bias_key);
    }
    
    // Load normalization weights
    if (layer.norm1()) {
        std::string norm1_weight_key = prefix + "norm1.weight";
        std::string norm1_bias_key = prefix + "norm1.bias";
        if (weights.find(norm1_weight_key) != weights.end()) {
            layer.norm1()->weight() = weights.at(norm1_weight_key);
        }
        if (weights.find(norm1_bias_key) != weights.end()) {
            layer.norm1()->bias() = weights.at(norm1_bias_key);
        }
    }
    
    if (layer.norm2()) {
        std::string norm2_weight_key = prefix + "norm2.weight";
        std::string norm2_bias_key = prefix + "norm2.bias";
        if (weights.find(norm2_weight_key) != weights.end()) {
            layer.norm2()->weight() = weights.at(norm2_weight_key);
        }
        if (weights.find(norm2_bias_key) != weights.end()) {
            layer.norm2()->bias() = weights.at(norm2_bias_key);
        }
    }
    
    if (layer.norm_out()) {
        std::string norm_out_weight_key = prefix + "norm_out.weight";
        std::string norm_out_bias_key = prefix + "norm_out.bias";
        if (weights.find(norm_out_weight_key) != weights.end()) {
            layer.norm_out()->weight() = weights.at(norm_out_weight_key);
        }
        if (weights.find(norm_out_bias_key) != weights.end()) {
            layer.norm_out()->bias() = weights.at(norm_out_bias_key);
        }
    }
    
    // Load layer scale
    if (layer.gamma_1()) {
        std::string gamma_1_key = prefix + "gamma_1.scale";
        if (weights.find(gamma_1_key) != weights.end()) {
            layer.gamma_1()->scale() = weights.at(gamma_1_key);
        }
    }
    
    if (layer.gamma_2()) {
        std::string gamma_2_key = prefix + "gamma_2.scale";
        if (weights.find(gamma_2_key) != weights.end()) {
            layer.gamma_2()->scale() = weights.at(gamma_2_key);
        }
    }
    
    return true;
}

bool WeightLoader::load_cross_transformer_layer(
    CrossTransformerEncoderLayer& layer,
    const std::unordered_map<std::string, mx::array>& weights,
    const std::string& prefix
) {
    // Load cross-attention weights
    // PyTorch nn.MultiheadAttention stores combined in_proj_weight [3*d, d] and in_proj_bias [3*d]
    // We need to split into separate Q, K, V projections
    std::string in_proj_weight_key = prefix + "cross_attn.in_proj_weight";
    std::string in_proj_bias_key = prefix + "cross_attn.in_proj_bias";
    std::string out_proj_weight_key = prefix + "cross_attn.out_proj.weight";
    std::string out_proj_bias_key = prefix + "cross_attn.out_proj.bias";
    
    if (weights.find(in_proj_weight_key) != weights.end()) {
        auto in_proj_w = weights.at(in_proj_weight_key);  // [3*d_model, d_model]
        int d_model = in_proj_w.shape(1);
        // Split into Q, K, V: each [d_model, d_model]
        layer.cross_attn_q_proj_weight() = mx::slice(in_proj_w, {0, 0}, {d_model, d_model});
        layer.cross_attn_k_proj_weight() = mx::slice(in_proj_w, {d_model, 0}, {2 * d_model, d_model});
        layer.cross_attn_v_proj_weight() = mx::slice(in_proj_w, {2 * d_model, 0}, {3 * d_model, d_model});
    }
    if (weights.find(in_proj_bias_key) != weights.end()) {
        auto in_proj_b = weights.at(in_proj_bias_key);  // [3*d_model]
        int d_model = in_proj_b.shape(0) / 3;
        layer.cross_attn_q_proj_bias() = mx::slice(in_proj_b, {0}, {d_model});
        layer.cross_attn_k_proj_bias() = mx::slice(in_proj_b, {d_model}, {2 * d_model});
        layer.cross_attn_v_proj_bias() = mx::slice(in_proj_b, {2 * d_model}, {3 * d_model});
    }
    if (weights.find(out_proj_weight_key) != weights.end()) {
        layer.cross_attn_out_proj_weight() = weights.at(out_proj_weight_key);
    }
    if (weights.find(out_proj_bias_key) != weights.end()) {
        layer.cross_attn_out_proj_bias() = weights.at(out_proj_bias_key);
    }
    
    // Load feed-forward weights
    std::string linear1_weight_key = prefix + "linear1.weight";
    std::string linear1_bias_key = prefix + "linear1.bias";
    std::string linear2_weight_key = prefix + "linear2.weight";
    std::string linear2_bias_key = prefix + "linear2.bias";
    
    if (weights.find(linear1_weight_key) != weights.end()) {
        layer.linear1_weight() = weights.at(linear1_weight_key);
    }
    if (weights.find(linear1_bias_key) != weights.end()) {
        layer.linear1_bias() = weights.at(linear1_bias_key);
    }
    if (weights.find(linear2_weight_key) != weights.end()) {
        layer.linear2_weight() = weights.at(linear2_weight_key);
    }
    if (weights.find(linear2_bias_key) != weights.end()) {
        layer.linear2_bias() = weights.at(linear2_bias_key);
    }
    
    // Load normalization weights
    if (layer.norm1()) {
        std::string norm1_weight_key = prefix + "norm1.weight";
        std::string norm1_bias_key = prefix + "norm1.bias";
        if (weights.find(norm1_weight_key) != weights.end()) {
            layer.norm1()->weight() = weights.at(norm1_weight_key);
        }
        if (weights.find(norm1_bias_key) != weights.end()) {
            layer.norm1()->bias() = weights.at(norm1_bias_key);
        }
    }
    
    if (layer.norm2()) {
        std::string norm2_weight_key = prefix + "norm2.weight";
        std::string norm2_bias_key = prefix + "norm2.bias";
        if (weights.find(norm2_weight_key) != weights.end()) {
            layer.norm2()->weight() = weights.at(norm2_weight_key);
        }
        if (weights.find(norm2_bias_key) != weights.end()) {
            layer.norm2()->bias() = weights.at(norm2_bias_key);
        }
    }
    
    if (layer.norm3()) {
        std::string norm3_weight_key = prefix + "norm3.weight";
        std::string norm3_bias_key = prefix + "norm3.bias";
        if (weights.find(norm3_weight_key) != weights.end()) {
            layer.norm3()->weight() = weights.at(norm3_weight_key);
        }
        if (weights.find(norm3_bias_key) != weights.end()) {
            layer.norm3()->bias() = weights.at(norm3_bias_key);
        }
    }
    
    if (layer.norm_out()) {
        std::string norm_out_weight_key = prefix + "norm_out.weight";
        std::string norm_out_bias_key = prefix + "norm_out.bias";
        if (weights.find(norm_out_weight_key) != weights.end()) {
            layer.norm_out()->weight() = weights.at(norm_out_weight_key);
        }
        if (weights.find(norm_out_bias_key) != weights.end()) {
            layer.norm_out()->bias() = weights.at(norm_out_bias_key);
        }
    }
    
    // Load layer scale
    if (layer.gamma_1()) {
        std::string gamma_1_key = prefix + "gamma_1.scale";
        if (weights.find(gamma_1_key) != weights.end()) {
            layer.gamma_1()->scale() = weights.at(gamma_1_key);
        }
    }
    
    if (layer.gamma_2()) {
        std::string gamma_2_key = prefix + "gamma_2.scale";
        if (weights.find(gamma_2_key) != weights.end()) {
            layer.gamma_2()->scale() = weights.at(gamma_2_key);
        }
    }
    
    return true;
}

bool WeightLoader::load_cross_transformer(
    CrossTransformerEncoder& transformer,
    const std::unordered_map<std::string, mx::array>& weights,
    const std::string& prefix
) {
    // Load norm_in weights (applied before transformer layers)
    std::string norm_in_weight_key = prefix + "norm_in.weight";
    std::string norm_in_bias_key = prefix + "norm_in.bias";
    std::string norm_in_t_weight_key = prefix + "norm_in_t.weight";
    std::string norm_in_t_bias_key = prefix + "norm_in_t.bias";
    
    if (transformer.norm_in()) {
        if (weights.find(norm_in_weight_key) != weights.end()) {
            transformer.norm_in()->weight() = weights.at(norm_in_weight_key);
        }
        if (weights.find(norm_in_bias_key) != weights.end()) {
            transformer.norm_in()->bias() = weights.at(norm_in_bias_key);
        }
    }
    
    if (transformer.norm_in_t()) {
        if (weights.find(norm_in_t_weight_key) != weights.end()) {
            transformer.norm_in_t()->weight() = weights.at(norm_in_t_weight_key);
        }
        if (weights.find(norm_in_t_bias_key) != weights.end()) {
            transformer.norm_in_t()->bias() = weights.at(norm_in_t_bias_key);
        }
    }
    
    // Load transformer layers (alternating classic and cross)
    size_t classic_idx = 0;
    size_t cross_idx = 0;
    
    // Determine total number of layers by checking which keys exist
    int total_layers = 0;
    for (int i = 0; i < 100; ++i) {  // Max 100 layers
        std::string test_key = prefix + "layers." + std::to_string(i) + ".norm1.weight";
        if (weights.find(test_key) == weights.end()) {
            total_layers = i;
            break;
        }
    }
    
    // Load each layer
    for (int i = 0; i < total_layers; ++i) {
        std::string freq_prefix = prefix + "layers." + std::to_string(i) + ".";
        std::string time_prefix = prefix + "layers_t." + std::to_string(i) + ".";
        
        // Determine if this is a classic or cross layer by checking for cross_attn keys
        std::string cross_attn_test_key = freq_prefix + "cross_attn.in_proj_weight";
        bool is_cross = (weights.find(cross_attn_test_key) != weights.end());
        
        if (is_cross) {
            // Load cross layer
            if (cross_idx < transformer.cross_layers().size()) {
                if (!load_cross_transformer_layer(*transformer.cross_layers()[cross_idx], weights, freq_prefix)) {
                    std::cerr << "Failed to load cross layer " << cross_idx << " (freq)" << std::endl;
                    return false;
                }
            }
            if (cross_idx < transformer.cross_layers_t().size()) {
                if (!load_cross_transformer_layer(*transformer.cross_layers_t()[cross_idx], weights, time_prefix)) {
                    std::cerr << "Failed to load cross layer " << cross_idx << " (time)" << std::endl;
                    return false;
                }
            }
            cross_idx++;
        } else {
            // Load classic layer
            if (classic_idx < transformer.classic_layers().size()) {
                if (!load_transformer_encoder_layer(*transformer.classic_layers()[classic_idx], weights, freq_prefix)) {
                    std::cerr << "Failed to load classic layer " << classic_idx << " (freq)" << std::endl;
                    return false;
                }
            }
            if (classic_idx < transformer.classic_layers_t().size()) {
                if (!load_transformer_encoder_layer(*transformer.classic_layers_t()[classic_idx], weights, time_prefix)) {
                    std::cerr << "Failed to load classic layer " << classic_idx << " (time)" << std::endl;
                    return false;
                }
            }
            classic_idx++;
        }
    }
    
    return true;
}

bool WeightLoader::load_htdemucs_from_map(
    HTDemucs& model,
    const std::unordered_map<std::string, mx::array>& weights
) {
    // Load encoder layers (same as HDemucs)
    for (size_t i = 0; i < model.encoder().size(); ++i) {
        std::string prefix = "encoder." + std::to_string(i) + ".";
        if (!load_encoder_layer(*model.encoder()[i], weights, prefix)) {
            std::cerr << "Failed to load encoder layer " << i << std::endl;
            return false;
        }
    }
    
    // Load decoder layers (same as HDemucs)
    for (size_t i = 0; i < model.decoder().size(); ++i) {
        std::string prefix = "decoder." + std::to_string(i) + ".";
        if (!load_decoder_layer(*model.decoder()[i], weights, prefix)) {
            std::cerr << "Failed to load decoder layer " << i << std::endl;
            return false;
        }
    }
    
    // Load time encoder layers (if hybrid)
    for (size_t i = 0; i < model.tencoder().size(); ++i) {
        std::string prefix = "tencoder." + std::to_string(i) + ".";
        if (!load_encoder_layer(*model.tencoder()[i], weights, prefix)) {
            std::cerr << "Failed to load time encoder layer " << i << std::endl;
            return false;
        }
    }
    
    // Load time decoder layers (if hybrid)
    for (size_t i = 0; i < model.tdecoder().size(); ++i) {
        std::string prefix = "tdecoder." + std::to_string(i) + ".";
        if (!load_decoder_layer(*model.tdecoder()[i], weights, prefix)) {
            std::cerr << "Failed to load time decoder layer " << i << std::endl;
            return false;
        }
    }
    
    // Load frequency embedding (if present)
    if (model.freq_emb()) {
        if (!load_scaled_embedding(*model.freq_emb(), weights, "freq_emb.")) {
            std::cerr << "Failed to load frequency embedding" << std::endl;
            return false;
        }
    }
    
    // Load cross-transformer (if present)
    if (model.crosstransformer()) {
        if (!load_cross_transformer(*model.crosstransformer(), weights, "crosstransformer.")) {
            std::cerr << "Failed to load cross-transformer" << std::endl;
            return false;
        }
    }
    
    // Load channel up/downsampler weights (if bottom_channels > 0)
    auto upsampler_w = weights.find("channel_upsampler.weight");
    auto upsampler_b = weights.find("channel_upsampler.bias");
    auto upsampler_t_w = weights.find("channel_upsampler_t.weight");
    auto upsampler_t_b = weights.find("channel_upsampler_t.bias");
    auto downsampler_w = weights.find("channel_downsampler.weight");
    auto downsampler_b = weights.find("channel_downsampler.bias");
    auto downsampler_t_w = weights.find("channel_downsampler_t.weight");
    auto downsampler_t_b = weights.find("channel_downsampler_t.bias");
    
    if (upsampler_w != weights.end()) {
        model.channel_upsampler_weight() = upsampler_w->second;
    }
    if (upsampler_b != weights.end()) {
        model.channel_upsampler_bias() = upsampler_b->second;
    }
    if (upsampler_t_w != weights.end()) {
        model.channel_upsampler_t_weight() = upsampler_t_w->second;
    }
    if (upsampler_t_b != weights.end()) {
        model.channel_upsampler_t_bias() = upsampler_t_b->second;
    }
    if (downsampler_w != weights.end()) {
        model.channel_downsampler_weight() = downsampler_w->second;
    }
    if (downsampler_b != weights.end()) {
        model.channel_downsampler_bias() = downsampler_b->second;
    }
    if (downsampler_t_w != weights.end()) {
        model.channel_downsampler_t_weight() = downsampler_t_w->second;
    }
    if (downsampler_t_b != weights.end()) {
        model.channel_downsampler_t_bias() = downsampler_t_b->second;
    }
    
    return true;
}

bool WeightLoader::load_htdemucs_weights(
    HTDemucs& model,
    const std::string& safetensors_path
) {
    // Parse SafeTensors file
    auto file = SafeTensorsLoader::parse(safetensors_path);
    if (!file) {
        std::cerr << "Failed to parse SafeTensors file: " << safetensors_path << std::endl;
        return false;
    }
    
    std::cout << "Loading HTDemucs weights from: " << safetensors_path << std::endl;
    SafeTensorsLoader::print_info(*file);
    
    // Load all tensors
    auto weights = SafeTensorsLoader::load_all(*file);
    
    std::cout << "Loaded " << weights.size() << " tensors" << std::endl;
    
    // Load into model
    return load_htdemucs_from_map(model, weights);
}

} // namespace demucs
