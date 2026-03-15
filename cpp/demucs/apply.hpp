#pragma once

#include "htdemucs.hpp"
#include <mlx/mlx.h>
#include <string>
#include <vector>

namespace demucs {

namespace mx = mlx::core;

/**
 * TensorChunk - matches TensorChunk class from apply.py
 * Represents a view into a tensor with offset and length.
 */
class TensorChunk {
public:
    TensorChunk(const mx::array& tensor, int offset = 0, int length = -1);
    
    std::vector<int> shape() const;
    
    // Pad the chunk to target_length, centering the chunk content
    mx::array padded(int target_length) const;
    
    const mx::array& tensor() const { return tensor_; }
    int offset() const { return offset_; }
    int length() const { return length_; }

private:
    mx::array tensor_;
    int offset_;
    int length_;
};

/**
 * apply_model - matches apply_model function from apply.py
 * 
 * Apply model to a given mixture with overlap-add chunking.
 *
 * @param model The HTDemucs model
 * @param mix Input audio tensor (batch, channels, length) or TensorChunk
 * @param shifts Number of random shifts for equivariant stabilization
 * @param split Whether to split audio into chunks
 * @param overlap Overlap between chunks (0.0 to 1.0)
 * @param transition_power Power for transition weighting
 * @param segment Override segment length (seconds), -1 to use model default
 * @return Separated sources tensor (batch, sources, channels, length)
 */
mx::array apply_model(
    HTDemucs& model,
    const mx::array& mix,
    int shifts = 1,
    bool split = true,
    float overlap = 0.25f,
    float transition_power = 1.0f,
    float segment = -1.0f
);

mx::array apply_model_chunk(
    HTDemucs& model,
    const TensorChunk& chunk,
    int shifts = 1,
    bool split = true,
    float overlap = 0.25f,
    float transition_power = 1.0f,
    float segment = -1.0f
);

/**
 * center_trim - matches center_trim from utils.py
 * Center trim tensor to reference length along last dimension.
 */
mx::array center_trim(const mx::array& tensor, int reference);

/**
 * prevent_clip - matches prevent_clip from audio.py
 * Rescale audio to prevent clipping.
 */
mx::array prevent_clip(const mx::array& wav, const std::string& mode = "rescale");

} // namespace demucs
