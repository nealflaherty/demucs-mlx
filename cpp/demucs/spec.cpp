#include "spec.hpp"
#include <mlx/mlx.h>
#include <Accelerate/Accelerate.h>
#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>

namespace demucs {

namespace mx = mlx::core;

// Helper: Compute window normalization (sum of squared windows at each position)
std::vector<float> compute_window_normalization(const std::vector<float>& window, 
                                                  int n_fft, int hop_length, int num_frames) {
    int output_length = (num_frames - 1) * hop_length + n_fft;
    std::vector<float> norm(output_length, 0.0f);
    
    for (int frame_idx = 0; frame_idx < num_frames; ++frame_idx) {
        int start_pos = frame_idx * hop_length;
        for (int i = 0; i < n_fft && (start_pos + i) < output_length; ++i) {
            norm[start_pos + i] += window[i] * window[i];
        }
    }
    
    return norm;
}

mx::array spectro(const mx::array& x, int n_fft, int hop_length, int pad) {
    if (hop_length < 0) {
        hop_length = n_fft / 4;
    }
    
    // Get input shape and convert to CPU
    auto shape = x.shape();
    int length = shape.back();
    
    // Calculate batch size
    int batch_size = 1;
    for (size_t i = 0; i < shape.size() - 1; ++i) {
        batch_size *= shape[i];
    }
    
    // Reshape to 2D and get data
    auto x_2d = mx::reshape(x, {batch_size, length});
    mx::eval(x_2d);
    auto x_data = x_2d.data<float>();
    
    // Apply reflect padding (matching demucs.cpp approach)
    int pad_amount = n_fft / 2;
    int padded_length = length + 2 * pad_amount;
    std::vector<float> padded_data(batch_size * padded_length);
    
    for (int b = 0; b < batch_size; ++b) {
        const float* src = x_data + b * length;
        float* dst = padded_data.data() + b * padded_length;
        
        // Copy original signal to middle
        std::copy(src, src + length, dst + pad_amount);
        
        // Reflect padding (PyTorch semantics):
        // Left: copy x[1:pad+1] and reverse -> [x[pad], x[pad-1], ..., x[1]]
        std::copy(src + 1, src + 1 + pad_amount, dst);
        std::reverse(dst, dst + pad_amount);
        
        // Right: copy x[-(pad+1):-1] and reverse -> [x[-2], x[-3], ..., x[-(pad+1)]]
        std::copy(src + length - pad_amount - 1, src + length - 1, dst + pad_amount + length);
        std::reverse(dst + pad_amount + length, dst + padded_length);
    }
    
    // Calculate number of frames
    int num_frames = 1 + (padded_length - n_fft) / hop_length;
    int freq_bins = n_fft / 2 + 1;
    
    // Create periodic Hann window (matches PyTorch hann_window(periodic=True))
    // Generate N+1 points and use first N (equivalent to dividing by N instead of N-1)
    std::vector<float> window(n_fft);
    float floatN = static_cast<float>(n_fft);
    for (int i = 0; i < n_fft; ++i) {
        window[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * static_cast<float>(i) / floatN));
    }
    
    // Setup FFT using Accelerate
    FFTSetup fft_setup = vDSP_create_fftsetup(std::log2(n_fft), FFT_RADIX2);
    if (!fft_setup) {
        throw std::runtime_error("Failed to create FFT setup");
    }
    
    // Allocate output arrays
    std::vector<float> output_real(batch_size * freq_bins * num_frames);
    std::vector<float> output_imag(batch_size * freq_bins * num_frames);
    
    // Process each batch and frame
    std::vector<float> frame_buffer(n_fft);
    DSPSplitComplex split_complex;
    std::vector<float> real_part(n_fft / 2 + 1);
    std::vector<float> imag_part(n_fft / 2 + 1);
    split_complex.realp = real_part.data();
    split_complex.imagp = imag_part.data();
    
    float normalization = 1.0f / std::sqrt(static_cast<float>(n_fft));
    
    for (int b = 0; b < batch_size; ++b) {
        const float* signal = padded_data.data() + b * padded_length;
        
        for (int frame_idx = 0; frame_idx < num_frames; ++frame_idx) {
            int start_pos = frame_idx * hop_length;
            
            // Extract and window the frame
            for (int i = 0; i < n_fft; ++i) {
                frame_buffer[i] = signal[start_pos + i] * window[i];
            }
            
            // Convert to split complex format (required by vDSP)
            // vDSP expects interleaved real/imag pairs
            vDSP_ctoz((DSPComplex*)frame_buffer.data(), 2, &split_complex, 1, n_fft / 2);
            
            // Perform FFT
            vDSP_fft_zrip(fft_setup, &split_complex, 1, std::log2(n_fft), FFT_FORWARD);
            
            // Scale by 0.5 (vDSP convention)
            float scale = 0.5f;
            vDSP_vsmul(split_complex.realp, 1, &scale, split_complex.realp, 1, n_fft / 2);
            vDSP_vsmul(split_complex.imagp, 1, &scale, split_complex.imagp, 1, n_fft / 2);
            
            // Extract results and normalize
            // The output format from vDSP_fft_zrip is:
            // realp[0] = DC, imagp[0] = Nyquist
            // realp[1..n/2-1] = real parts, imagp[1..n/2-1] = imag parts
            
            for (int f = 0; f < freq_bins; ++f) {
                int out_idx = b * freq_bins * num_frames + f * num_frames + frame_idx;
                
                if (f == 0) {
                    // DC component
                    output_real[out_idx] = split_complex.realp[0] * normalization;
                    output_imag[out_idx] = 0.0f;
                } else if (f == freq_bins - 1) {
                    // Nyquist component
                    output_real[out_idx] = split_complex.imagp[0] * normalization;
                    output_imag[out_idx] = 0.0f;
                } else {
                    // Regular bins
                    output_real[out_idx] = split_complex.realp[f] * normalization;
                    output_imag[out_idx] = split_complex.imagp[f] * normalization;
                }
            }
        }
    }
    
    vDSP_destroy_fftsetup(fft_setup);
    
    // Create MLX arrays from the results
    auto real_array = mx::array(output_real.data(), {batch_size, freq_bins, num_frames});
    auto imag_array = mx::array(output_imag.data(), {batch_size, freq_bins, num_frames});
    
    // Combine into complex array
    auto result = real_array + mx::array(std::complex<float>(0, 1)) * imag_array;
    
    // Reshape back to original batch dimensions
    std::vector<int> output_shape_vec(shape.begin(), shape.end() - 1);
    output_shape_vec.push_back(freq_bins);
    output_shape_vec.push_back(num_frames);
    mx::Shape output_shape(output_shape_vec.begin(), output_shape_vec.end());
    
    result = mx::reshape(result, output_shape);
    mx::eval(result);
    
    return result;
}

mx::array ispectro(const mx::array& z, int hop_length, int length, int pad) {
    auto shape = z.shape();
    int freqs = shape[shape.size() - 2];
    int frames = shape[shape.size() - 1];
    int n_fft = 2 * freqs - 2;
    int win_length = n_fft / (1 + pad);
    
    if (hop_length < 0) {
        hop_length = n_fft / 4;
    }
    
    // Calculate batch size
    int batch_size = 1;
    for (size_t i = 0; i < shape.size() - 2; ++i) {
        batch_size *= shape[i];
    }
    
    // Reshape and get data
    auto z_3d = mx::reshape(z, {batch_size, freqs, frames});
    auto z_real = mx::real(z_3d);
    auto z_imag = mx::imag(z_3d);
    mx::eval(z_real);
    mx::eval(z_imag);
    
    auto real_data = z_real.data<float>();
    auto imag_data = z_imag.data<float>();
    
    // Create periodic Hann window (matches PyTorch hann_window(periodic=True))
    // Window is always n_fft length, regardless of pad parameter
    std::vector<float> window(n_fft);
    float floatN = static_cast<float>(n_fft);
    for (int i = 0; i < n_fft; ++i) {
        window[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * static_cast<float>(i) / floatN));
    }
    
    // Compute window normalization (sum of squared windows)
    auto window_norm = compute_window_normalization(window, n_fft, hop_length, frames);
    
    // Setup FFT
    FFTSetup fft_setup = vDSP_create_fftsetup(std::log2(n_fft), FFT_RADIX2);
    if (!fft_setup) {
        throw std::runtime_error("Failed to create FFT setup");
    }
    
    // Output length before removing padding
    int output_length = (frames - 1) * hop_length + n_fft;
    std::vector<float> output_data(batch_size * output_length, 0.0f);
    
    DSPSplitComplex split_complex;
    std::vector<float> real_part(n_fft / 2 + 1);
    std::vector<float> imag_part(n_fft / 2 + 1);
    split_complex.realp = real_part.data();
    split_complex.imagp = imag_part.data();
    
    std::vector<float> frame_buffer(n_fft);
    float normalization = std::sqrt(static_cast<float>(n_fft));
    
    for (int b = 0; b < batch_size; ++b) {
        float* output = output_data.data() + b * output_length;
        
        for (int frame_idx = 0; frame_idx < frames; ++frame_idx) {
            // Prepare split complex input and undo normalization
            for (int f = 0; f < freqs; ++f) {
                int in_idx = b * freqs * frames + f * frames + frame_idx;
                
                if (f == 0) {
                    // DC component
                    split_complex.realp[0] = real_data[in_idx] * normalization;
                } else if (f == freqs - 1) {
                    // Nyquist component
                    split_complex.imagp[0] = real_data[in_idx] * normalization;
                } else {
                    // Regular bins
                    split_complex.realp[f] = real_data[in_idx] * normalization;
                    split_complex.imagp[f] = imag_data[in_idx] * normalization;
                }
            }
            
            // Perform inverse FFT
            vDSP_fft_zrip(fft_setup, &split_complex, 1, std::log2(n_fft), FFT_INVERSE);
            
            // Scale by 0.5 (vDSP convention)
            float scale = 0.5f;
            vDSP_vsmul(split_complex.realp, 1, &scale, split_complex.realp, 1, n_fft / 2);
            vDSP_vsmul(split_complex.imagp, 1, &scale, split_complex.imagp, 1, n_fft / 2);
            
            // Convert from split complex
            vDSP_ztoc(&split_complex, 1, (DSPComplex*)frame_buffer.data(), 2, n_fft / 2);
            
            // Overlap-add with window and normalization
            int start_pos = frame_idx * hop_length;
            for (int i = 0; i < n_fft; ++i) {
                int pos = start_pos + i;
                if (pos < output_length) {
                    // Apply window, scale by 2/n_fft (vDSP inverse convention), and normalize by sum of squared windows
                    float norm_factor = window_norm[pos] + 1e-8f; // avoid division by zero
                    output[pos] += frame_buffer[i] * window[i] / static_cast<float>(n_fft / 2) / norm_factor;
                }
            }
        }
    }
    
    vDSP_destroy_fftsetup(fft_setup);
    
    // Remove center padding
    int pad_amount = n_fft / 2;
    int final_length = (length > 0) ? length : (output_length - 2 * pad_amount);
    
    std::vector<float> trimmed_data(batch_size * final_length);
    for (int b = 0; b < batch_size; ++b) {
        std::copy(
            output_data.data() + b * output_length + pad_amount,
            output_data.data() + b * output_length + pad_amount + final_length,
            trimmed_data.data() + b * final_length
        );
    }
    
    // Create MLX array
    auto result = mx::array(trimmed_data.data(), {batch_size, final_length});
    
    // Reshape back to original dimensions
    std::vector<int> output_shape_vec(shape.begin(), shape.end() - 2);
    output_shape_vec.push_back(final_length);
    mx::Shape output_shape_final(output_shape_vec.begin(), output_shape_vec.end());
    
    result = mx::reshape(result, output_shape_final);
    mx::eval(result);
    
    return result;
}

} // namespace demucs
