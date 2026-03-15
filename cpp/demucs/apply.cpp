#include "apply.hpp"
#include <cmath>
#include <random>
#include <iostream>
#include <cassert>

namespace demucs {

// ============================================================================
// TensorChunk - matches TensorChunk class from apply.py
// ============================================================================

TensorChunk::TensorChunk(const mx::array& tensor, int offset, int length)
    : tensor_(tensor), offset_(offset), length_(0) {
    int total_length = tensor.shape(-1);
    assert(offset >= 0);
    assert(offset < total_length);
    
    if (length < 0) {
        length_ = total_length - offset;
    } else {
        length_ = std::min(total_length - offset, length);
    }
}

std::vector<int> TensorChunk::shape() const {
    auto s = std::vector<int>(tensor_.shape().begin(), tensor_.shape().end());
    s.back() = length_;
    return s;
}

mx::array TensorChunk::padded(int target_length) const {
    // Python: delta = target_length - self.length
    //         start = self.offset - delta // 2
    //         end = start + target_length
    //         correct_start = max(0, start)
    //         correct_end = min(total_length, end)
    //         pad_left = correct_start - start
    //         pad_right = end - correct_end
    //         out = F.pad(self.tensor[..., correct_start:correct_end], (pad_left, pad_right))
    int delta = target_length - length_;
    int total_length = tensor_.shape(-1);
    assert(delta >= 0);
    
    int start = offset_ - delta / 2;
    int end = start + target_length;
    
    int correct_start = std::max(0, start);
    int correct_end = std::min(total_length, end);
    
    int pad_left = correct_start - start;
    int pad_right = end - correct_end;
    
    // Slice: tensor[..., correct_start:correct_end]
    std::vector<int> start_idx(tensor_.ndim(), 0);
    std::vector<int> end_idx(tensor_.shape().begin(), tensor_.shape().end());
    start_idx.back() = correct_start;
    end_idx.back() = correct_end;
    
    auto out = mx::slice(tensor_,
                         mx::Shape(start_idx.begin(), start_idx.end()),
                         mx::Shape(end_idx.begin(), end_idx.end()));
    
    // Pad last dimension
    if (pad_left > 0 || pad_right > 0) {
        std::vector<std::pair<int, int>> pad_widths(tensor_.ndim(), {0, 0});
        pad_widths.back() = {pad_left, pad_right};
        out = mx::pad(out, pad_widths);
    }
    
    assert(out.shape(-1) == target_length);
    return out;
}

// ============================================================================
// center_trim - matches center_trim from utils.py
// ============================================================================

mx::array center_trim(const mx::array& tensor, int reference) {
    // Python: delta = tensor.size(-1) - ref_size
    //         tensor = tensor[..., delta // 2:-(delta - delta // 2)]
    int delta = tensor.shape(-1) - reference;
    if (delta < 0) {
        throw std::runtime_error("tensor must be larger than reference");
    }
    if (delta == 0) {
        return tensor;
    }
    
    int trim_left = delta / 2;
    int trim_right = delta - trim_left;
    
    std::vector<int> start_idx(tensor.ndim(), 0);
    std::vector<int> end_idx(tensor.shape().begin(), tensor.shape().end());
    start_idx.back() = trim_left;
    end_idx.back() = tensor.shape(-1) - trim_right;
    
    return mx::slice(tensor,
                     mx::Shape(start_idx.begin(), start_idx.end()),
                     mx::Shape(end_idx.begin(), end_idx.end()));
}

// ============================================================================
// prevent_clip - matches prevent_clip from audio.py
// ============================================================================

mx::array prevent_clip(const mx::array& wav, const std::string& mode) {
    // Python: audio.py prevent_clip()
    if (mode == "none") {
        return wav;
    } else if (mode == "rescale") {
        // Python: wav = wav / max(1.01 * wav.abs().max(), 1)
        auto peak = mx::max(mx::abs(wav));
        mx::eval(peak);
        float denom = std::max(1.01f * peak.item<float>(), 1.0f);
        return wav / mx::array(denom);
    } else if (mode == "clamp") {
        // Python: wav = wav.clamp(-0.99, 0.99)
        return mx::clip(wav, mx::array(-0.99f), mx::array(0.99f));
    } else {
        throw std::runtime_error("Invalid clip mode: " + mode);
    }
}


// ============================================================================
// apply_model - matches apply_model from apply.py
// ============================================================================

mx::array apply_model_chunk(
    HTDemucs& model,
    const TensorChunk& chunk,
    int shifts,
    bool split,
    float overlap,
    float transition_power,
    float segment)
{
    int length = chunk.length();
    int samplerate = 44100;  // htdemucs samplerate
    
    // Python: if shifts:
    if (shifts > 0) {
        int max_shift = static_cast<int>(0.5 * samplerate);
        auto padded_mix = chunk.padded(length + 2 * max_shift);
        
        std::mt19937 rng(42);  // Fixed seed for reproducibility
        mx::array out = mx::zeros({1, 4, 2, length});
        
        for (int shift_idx = 0; shift_idx < shifts; ++shift_idx) {
            int offset = std::uniform_int_distribution<int>(0, max_shift)(rng);
            TensorChunk shifted(padded_mix, offset, length + max_shift - offset);
            auto res = apply_model_chunk(model, shifted, 0, split, overlap,
                                         transition_power, segment);
            // Python: out += shifted_out[..., max_shift - offset:]
            int trim_start = max_shift - offset;
            auto trimmed = mx::slice(res,
                {0, 0, 0, trim_start},
                {res.shape(0), res.shape(1), res.shape(2), trim_start + length});
            out = out + trimmed;
            mx::eval(out);
        }
        out = out / mx::array(static_cast<float>(shifts));
        mx::eval(out);
        return out;
    }
    
    // Python: elif split:
    if (split) {
        float seg = segment;
        if (seg < 0) {
            // Use model's segment
            // htdemucs segment is 7.8 seconds
            seg = 7.8f;
        }
        
        int segment_length = static_cast<int>(samplerate * seg);
        int stride = static_cast<int>((1.0f - overlap) * segment_length);
        
        // Output accumulator
        mx::array out = mx::zeros({1, 4, 2, length});
        std::vector<float> sum_weight_data(length, 0.0f);
        
        // Build triangle weight
        // Python: weight = cat([arange(1, seg//2+1), arange(seg - seg//2, 0, -1)])
        std::vector<float> weight_data(segment_length);
        int half = segment_length / 2;
        for (int i = 0; i < half; ++i) {
            weight_data[i] = static_cast<float>(i + 1);
        }
        for (int i = half; i < segment_length; ++i) {
            weight_data[i] = static_cast<float>(segment_length - i);
        }
        // Normalize and apply transition_power
        float max_w = *std::max_element(weight_data.begin(), weight_data.end());
        for (auto& w : weight_data) {
            w = std::pow(w / max_w, transition_power);
        }
        
        // Process each chunk
        auto mix_tensor = chunk.padded(length);
        
        for (int offset = 0; offset < length; offset += stride) {
            int chunk_len = std::min(segment_length, length - offset);
            TensorChunk sub_chunk(mix_tensor, offset, segment_length);
            
            std::cout << "  Processing chunk at " << offset << "/" << length 
                      << " (" << static_cast<int>(100.0f * offset / length) << "%)" << std::endl;
            
            auto chunk_out = apply_model_chunk(model, sub_chunk, 0, false, overlap,
                                                transition_power, segment);
            mx::eval(chunk_out);
            
            int chunk_length = chunk_out.shape(-1);
            
            // Python: out[..., offset:offset+seg] += weight[:chunk_length] * chunk_out
            // We need to do this element-wise since MLX doesn't have easy slice assignment
            // Extract the weight for this chunk
            std::vector<float> chunk_weight(weight_data.begin(),
                                            weight_data.begin() + chunk_length);
            auto weight_arr = mx::array(chunk_weight.data(), {chunk_length});
            // Reshape weight to broadcast: (1, 1, 1, chunk_length)
            weight_arr = mx::reshape(weight_arr, {1, 1, 1, chunk_length});
            
            auto weighted = weight_arr * chunk_out;
            mx::eval(weighted);
            
            // Accumulate into output using pad + add
            int pad_left = offset;
            int pad_right = length - offset - chunk_length;
            if (pad_right < 0) pad_right = 0;
            
            auto padded_weighted = mx::pad(weighted,
                {{0, 0}, {0, 0}, {0, 0}, {pad_left, pad_right}});
            out = out + padded_weighted;
            mx::eval(out);
            
            // Accumulate weight
            for (int i = 0; i < chunk_length && (offset + i) < length; ++i) {
                sum_weight_data[offset + i] += chunk_weight[i];
            }
        }
        
        // Normalize by sum of weights
        auto sum_weight = mx::array(sum_weight_data.data(), {length});
        sum_weight = mx::reshape(sum_weight, {1, 1, 1, length});
        out = out / sum_weight;
        mx::eval(out);
        return out;
    }
    
    // Python: else: (no split, no shifts — direct model call)
    // Python: if isinstance(model, HTDemucs) and segment is not None:
    //             valid_length = int(segment * model.samplerate)
    //         elif hasattr(model, 'valid_length'):
    //             valid_length = model.valid_length(length)
    //         else:
    //             valid_length = length
    //         padded_mix = mix.padded(valid_length)
    //         out = model(padded_mix)
    //         return center_trim(out, length)
    
    int valid_length;
    float seg = segment;
    if (seg < 0) {
        seg = 7.8f;  // model default segment
    }
    valid_length = static_cast<int>(seg * samplerate);
    
    auto padded_mix = chunk.padded(valid_length);
    auto out = model.forward(padded_mix);
    mx::eval(out);
    return center_trim(out, length);
}

mx::array apply_model(
    HTDemucs& model,
    const mx::array& mix,
    int shifts,
    bool split,
    float overlap,
    float transition_power,
    float segment)
{
    TensorChunk chunk(mix);
    return apply_model_chunk(model, chunk, shifts, split, overlap,
                             transition_power, segment);
}

} // namespace demucs
