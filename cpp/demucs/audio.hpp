#pragma once

#include <mlx/mlx.h>
#include <string>
#include <optional>

namespace demucs {

namespace mx = mlx::core;

/**
 * Audio I/O utilities for loading and saving audio files.
 * Supports WAV via libnyquist, and MP3/AAC/FLAC/M4A/etc via
 * AudioToolbox (macOS) as a fallback.
 */
class Audio {
public:
    /**
     * Load audio file (WAV, MP3, AAC, FLAC, M4A, etc.)
     * Returns MLX array with shape (channels, samples)
     * Automatically converts to stereo and resamples to target_sample_rate.
     * Uses libnyquist for WAV, AudioToolbox for other formats.
     */
    static std::optional<mx::array> load(
        const std::string& path,
        int target_sample_rate = 44100
    );
    
    /**
     * Save audio file. Format is determined by file extension (or codec override):
     *   .wav  — PCM WAV (bits_per_sample: 16, 24, or 32-float)
     *   .m4a  — AAC in M4A container (via AudioToolbox)
     *   .flac — FLAC lossless (via AudioToolbox)
     *
     * Input should be shape (channels, samples) or (batch, channels, samples).
     * If batch dimension present, only saves first item.
     *
     * @param path Output file path (extension determines format)
     * @param audio Audio data, planar float32
     * @param sample_rate Sample rate in Hz
     * @param bits_per_sample For WAV: 16, 24, or 32. For FLAC: 16 or 24.
     * @param as_float For WAV: if true, save as float32 regardless of bits_per_sample
     * @param bitrate For AAC: target bitrate in kbps (default 256)
     * @param codec Override codec: "" = auto from extension, "aac", "alac", "flac"
     */
    static bool save(
        const std::string& path,
        const mx::array& audio,
        int sample_rate = 44100,
        int bits_per_sample = 16,
        bool as_float = false,
        int bitrate = 256,
        const std::string& codec = ""
    );

private:
    /**
     * Resample audio using linear interpolation
     */
    static mx::array resample(
        const mx::array& audio,
        int from_rate,
        int to_rate
    );

    /**
     * Save as WAV using libnyquist.
     */
    static bool save_wav(
        const std::string& path,
        const mx::array& audio_2d,
        int sample_rate,
        int bits_per_sample,
        bool as_float
    );

#ifdef HAVE_AUDIO_TOOLBOX
    /**
     * Load audio using AudioToolbox (macOS).
     */
    static std::optional<mx::array> load_audio_toolbox(
        const std::string& path,
        int target_sample_rate
    );

    /**
     * Save audio using AudioToolbox (macOS).
     * Supports AAC (.m4a), FLAC (.flac), ALAC (.m4a with codec="alac").
     */
    static bool save_audio_toolbox(
        const std::string& path,
        const mx::array& audio_2d,
        int sample_rate,
        int bits_per_sample,
        int bitrate,
        const std::string& codec
    );
#endif
};

} // namespace demucs
