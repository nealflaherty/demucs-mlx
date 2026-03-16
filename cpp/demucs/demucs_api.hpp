// demucs_api.hpp — Public API for HTDemucs music source separation
//
// This API is designed to feel familiar to users of the Python demucs library.
//
// Basic usage:
//   auto model = demucs::api::get_model("htdemucs");
//   auto sources = demucs::api::apply_model(*model, mix);
//
// Or using the convenience class:
//   demucs::api::Separator separator;
//   separator.load_model("models/htdemucs.safetensors");
//   auto result = separator.separate("track.mp3");

#pragma once

#include <mlx/mlx.h>
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <optional>
#include <stdexcept>

namespace demucs {
namespace api {

namespace mx = mlx::core;

// MARK: - Version & Constants

/// Library version
constexpr const char* version = "1.0.0";

/// Default source names for HTDemucs
inline const std::vector<std::string> default_sources = {"drums", "bass", "other", "vocals"};

/// Default model name
constexpr const char* default_model = "htdemucs";

// MARK: - Progress Callback

/// Progress callback: (fraction 0..1, human-readable message)
using ProgressCallback = std::function<void(float, const std::string&)>;

// MARK: - Error Types

/// Exception for Demucs-related errors
class DemucsError : public std::runtime_error {
public:
    enum class Code {
        FileNotFound,
        WeightLoadFailed,
        AudioLoadFailed,
        ModelNotLoaded,
        ModelNotFound,
        UnsupportedModel
    };
    
    DemucsError(Code code, const std::string& message)
        : std::runtime_error(message), code_(code) {}
    
    Code code() const { return code_; }
    
private:
    Code code_;
};

// MARK: - Model Interface

/// Forward declaration of internal model type
class HTDemucs;

/// Abstract interface for Demucs-family models
class DemucsModel {
public:
    virtual ~DemucsModel() = default;
    
    /// Source names the model separates into
    virtual const std::vector<std::string>& sources() const = 0;
    
    /// Expected sample rate
    virtual int sample_rate() const = 0;
    
    /// Number of audio channels (typically 2 for stereo)
    virtual int audio_channels() const = 0;
    
    /// Segment length in seconds for chunked processing
    virtual float segment() const = 0;
    
    /// Run the model forward pass
    virtual mx::array forward(const mx::array& mix) = 0;
    
    /// Get the underlying implementation (for internal use)
    virtual void* impl() = 0;
};

// MARK: - Types

/// A single separated audio stem
struct StemAudio {
    /// Stem name (e.g. "vocals", "drums", "bass", "other")
    std::string name;
    /// Audio data as MLXArray of shape (channels, samples)
    mx::array audio;
    
    StemAudio(const std::string& n, const mx::array& a) : name(n), audio(a) {}
};

/// Result of a separation operation
struct SeparationResult {
    /// The separated stems in source order
    std::vector<StemAudio> stems;
    /// Source names in model order
    std::vector<std::string> source_names;
    
    /// Get a stem by name. Returns nullptr if not found.
    const StemAudio* stem(const std::string& name) const {
        for (const auto& s : stems) {
            if (s.name == name) return &s;
        }
        return nullptr;
    }
};

/// Options for the separation process
struct SeparationOptions {
    /// Number of random shifts for averaging (0 = no shifts)
    int shifts = 1;
    /// Whether to split audio into overlapping chunks
    bool split = true;
    /// Overlap ratio between chunks (0.0–1.0)
    float overlap = 0.25f;
    /// Transition power for overlap-add weighting
    float transition_power = 1.0f;
    /// Segment length in seconds (-1 = model default)
    float segment = -1.0f;
    /// Clip prevention mode: "rescale", "clamp", or "none"
    std::string clip_mode = "rescale";
    
    /// Sensible defaults
    static SeparationOptions defaults() { return SeparationOptions{}; }
};

// MARK: - Model Loading

/// Load a pretrained model by name.
///
/// Currently only "htdemucs" is supported. The model weights must be
/// available at the default path or specified via `repo`.
///
/// @param name Model name (default: "htdemucs")
/// @param repo Optional path to model repository
/// @return A DemucsModel instance
/// @throws DemucsError if model cannot be loaded
std::unique_ptr<DemucsModel> get_model(
    const std::string& name = default_model,
    const std::string& repo = ""
);

/// Load model weights from a SafeTensors file.
///
/// @param path Path to the .safetensors file
/// @return A DemucsModel instance
/// @throws DemucsError if loading fails
std::unique_ptr<DemucsModel> load_model(const std::string& path);

// MARK: - Apply Model

/// Apply a model to a given mixture, returning separated sources.
///
/// This is the core separation function, matching the Python
/// `demucs.apply.apply_model()` interface.
///
/// @param model A DemucsModel instance
/// @param mix Audio tensor of shape (batch, channels, length)
/// @param shifts Number of random shifts for equivariant stabilization.
///   Increases separation time but improves quality. 10 was used in
///   the original paper. Default is 1.
/// @param split If true, split audio into overlapping chunks. Reduces
///   memory usage for long tracks.
/// @param overlap Overlap ratio between chunks (0.0–1.0)
/// @param transition_power Power for overlap-add weighting. Higher values
///   give sharper transitions between chunks.
/// @param segment Override model segment length in seconds. -1 uses
///   the model default.
/// @return MLXArray of shape (batch, sources, channels, length)
mx::array apply_model(
    DemucsModel& model,
    const mx::array& mix,
    int shifts = 1,
    bool split = true,
    float overlap = 0.25f,
    float transition_power = 1.0f,
    float segment = -1.0f
);

// MARK: - Separator (Convenience)

/// High-level convenience class for audio source separation.
///
/// Wraps model loading and the normalize → separate → denormalize pipeline
/// into a simple interface.
///
/// ```cpp
/// demucs::api::Separator separator;
/// separator.load_model("models/htdemucs.safetensors");
/// auto result = separator.separate("track.mp3");
/// for (const auto& stem : result.stems) {
///     demucs::api::save_audio(stem.name + ".wav", stem.audio);
/// }
/// ```
class Separator {
public:
    Separator();
    ~Separator();
    
    /// Source names the loaded model separates into
    const std::vector<std::string>& sources() const;
    
    /// Sample rate the model expects
    int sample_rate() const;
    
    /// Whether a model has been loaded
    bool is_loaded() const;
    
    /// Load model weights from a SafeTensors file path
    void load_model(const std::string& path);
    
    /// Load a pretrained model by name
    void load_model_by_name(
        const std::string& name = default_model,
        const std::string& repo = ""
    );
    
    /// Separate an audio file into stems.
    ///
    /// Handles the full pipeline: load audio → normalize → apply model →
    /// denormalize → extract stems → prevent clipping.
    ///
    /// @param path Path to the audio file
    /// @param options Separation options (shifts, overlap, etc.)
    /// @param progress Optional callback for progress updates
    /// @return A SeparationResult containing the separated stems
    SeparationResult separate(
        const std::string& path,
        const SeparationOptions& options = SeparationOptions::defaults(),
        ProgressCallback progress = nullptr
    );
    
    /// Separate raw audio into stems.
    ///
    /// @param audio MLXArray of shape (channels, samples) at 44100 Hz
    /// @param options Separation options
    /// @param progress Optional callback for progress updates
    /// @return A SeparationResult containing the separated stems
    SeparationResult separate_audio(
        const mx::array& audio,
        const SeparationOptions& options = SeparationOptions::defaults(),
        ProgressCallback progress = nullptr
    );

private:
    std::unique_ptr<DemucsModel> model_;
};

// MARK: - Audio Utilities

/// Load audio from file.
///
/// @param path Path to audio file (WAV, MP3, AAC, FLAC, M4A, etc.)
/// @param sample_rate Target sample rate (default: 44100)
/// @return Audio tensor of shape (channels, samples), or nullopt on failure
std::optional<mx::array> load_audio(
    const std::string& path,
    int sample_rate = 44100
);

/// Save audio with automatic clip prevention.
///
/// Matches the Python `demucs.audio.save_audio()` function.
///
/// @param path Output file path (extension determines format)
/// @param audio Audio tensor of shape (channels, samples)
/// @param sample_rate Output sample rate
/// @param clip Clip prevention mode: "rescale", "clamp", or "none"
/// @param bits_per_sample Bits per sample for WAV output (16 or 24)
/// @param as_float Save as 32-bit float WAV
/// @param bitrate Bitrate for compressed formats (kbps)
/// @param codec Codec for compressed formats ("aac", "flac", "alac")
/// @return true on success
bool save_audio(
    const std::string& path,
    const mx::array& audio,
    int sample_rate = 44100,
    const std::string& clip = "rescale",
    int bits_per_sample = 16,
    bool as_float = false,
    int bitrate = 256,
    const std::string& codec = ""
);

/// Prevent audio clipping.
///
/// @param audio Audio tensor
/// @param mode "rescale", "clamp", or "none"
/// @return Processed audio tensor
mx::array prevent_clip(const mx::array& audio, const std::string& mode = "rescale");

} // namespace api
} // namespace demucs
