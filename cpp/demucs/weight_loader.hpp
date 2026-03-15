#pragma once

#include "hdemucs.hpp"
#include "htdemucs.hpp"
#include "safetensors.hpp"
#include <string>
#include <unordered_map>

namespace demucs {

class WeightLoader {
public:
    // Load weights from SafeTensors file into HDemucs model
    static bool load_hdemucs_weights(
        HDemucs& model,
        const std::string& safetensors_path
    );
    
    // Load weights from SafeTensors file into HTDemucs model
    static bool load_htdemucs_weights(
        HTDemucs& model,
        const std::string& safetensors_path
    );
    
    // Load weights from a map of tensors (useful for testing with NPY files)
    static bool load_hdemucs_from_map(
        HDemucs& model,
        const std::unordered_map<std::string, mx::array>& weights
    );
    
    // Load weights from a map of tensors for HTDemucs
    static bool load_htdemucs_from_map(
        HTDemucs& model,
        const std::unordered_map<std::string, mx::array>& weights
    );

private:
    // Helper to load encoder layer weights
    static bool load_encoder_layer(
        HEncLayer& layer,
        const std::unordered_map<std::string, mx::array>& weights,
        const std::string& prefix
    );
    
    // Helper to load decoder layer weights
    static bool load_decoder_layer(
        HDecLayer& layer,
        const std::unordered_map<std::string, mx::array>& weights,
        const std::string& prefix
    );
    
    // Helper to load DConv weights
    static bool load_dconv(
        DConv& dconv,
        const std::unordered_map<std::string, mx::array>& weights,
        const std::string& prefix
    );
    
    // Helper to load ScaledEmbedding weights
    static bool load_scaled_embedding(
        ScaledEmbedding& emb,
        const std::unordered_map<std::string, mx::array>& weights,
        const std::string& prefix
    );
    
    // Helper to load MyTransformerEncoderLayer weights
    static bool load_transformer_encoder_layer(
        MyTransformerEncoderLayer& layer,
        const std::unordered_map<std::string, mx::array>& weights,
        const std::string& prefix
    );
    
    // Helper to load CrossTransformerEncoderLayer weights
    static bool load_cross_transformer_layer(
        CrossTransformerEncoderLayer& layer,
        const std::unordered_map<std::string, mx::array>& weights,
        const std::string& prefix
    );
    
    // Helper to load CrossTransformerEncoder weights
    static bool load_cross_transformer(
        CrossTransformerEncoder& transformer,
        const std::unordered_map<std::string, mx::array>& weights,
        const std::string& prefix
    );
};

} // namespace demucs
