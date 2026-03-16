// demucs_api.cpp — Implementation of the public API facade

#include "demucs_api.hpp"
#include "htdemucs.hpp"
#include "audio.hpp"
#include "weight_loader.hpp"
#include "apply.hpp"
#include <filesystem>
#include <cstdlib>

namespace fs = std::filesystem;

namespace demucs {
namespace api {

// MARK: - HTDemucs Model Wrapper

/// Concrete implementation wrapping the internal HTDemucs class
class HTDemucsModel : public DemucsModel {
public:
    HTDemucsModel() : model_(
        default_sources,
        /*audio_channels=*/2,
        /*channels=*/48,
        /*channels_time=*/-1,
        /*growth=*/2.0f,
        /*nfft=*/4096,
        /*cac=*/true,
        /*depth=*/4,
        /*rewrite=*/true,
        /*freq_emb=*/0.2f,
        /*emb_scale=*/10.0f,
        /*emb_smooth=*/true,
        /*kernel_size=*/8,
        /*time_stride=*/2,
        /*stride=*/4,
        /*context=*/1,
        /*context_enc=*/0,
        /*norm_starts=*/4,
        /*norm_groups=*/4,
        /*dconv_mode=*/3,
        /*dconv_depth=*/2,
        /*dconv_comp=*/8,
        /*dconv_attn=*/4,
        /*dconv_lstm=*/4,
        /*dconv_init=*/1e-3f,
        /*bottom_channels=*/512,
        /*t_layers=*/5,
        /*t_heads=*/8,
        /*t_hidden_scale=*/4.0f,
        /*t_dropout=*/0.0f,
        /*t_norm_in=*/true,
        /*t_norm_out=*/true,
        /*t_cross_first=*/false,
        /*t_layer_scale=*/true,
        /*t_gelu=*/true,
        /*samplerate=*/44100,
        /*segment=*/7.8f
    ), sources_(default_sources) {}
    
    const std::vector<std::string>& sources() const override { return sources_; }
    int sample_rate() const override { return 44100; }
    int audio_channels() const override { return 2; }
    float segment() const override { return 7.8f; }
    
    mx::array forward(const mx::array& mix) override {
        return model_.forward(mix);
    }
    
    void* impl() override { return &model_; }
    
    demucs::HTDemucs& internal_model() { return model_; }

private:
    demucs::HTDemucs model_;
    std::vector<std::string> sources_;
};

// MARK: - Model Loading

std::unique_ptr<DemucsModel> get_model(const std::string& name, const std::string& repo) {
    if (name != "htdemucs") {
        throw DemucsError(DemucsError::Code::UnsupportedModel,
            "Only 'htdemucs' model is currently supported");
    }
    
    // Look for model weights
    std::string model_path;
    if (!repo.empty()) {
        model_path = repo + "/htdemucs.safetensors";
    } else {
        // Default locations to search
        std::vector<std::string> search_paths = {
            "models/htdemucs.safetensors",
            "htdemucs.safetensors"
        };
        
        // Add home directory cache path
        const char* home = std::getenv("HOME");
        if (home) {
            search_paths.push_back(std::string(home) + "/.cache/demucs/htdemucs.safetensors");
        }
        
        for (const auto& path : search_paths) {
            if (fs::exists(path)) {
                model_path = path;
                break;
            }
        }
        
        if (model_path.empty()) {
            std::string searched;
            for (size_t i = 0; i < search_paths.size(); ++i) {
                if (i > 0) searched += ", ";
                searched += search_paths[i];
            }
            throw DemucsError(DemucsError::Code::ModelNotFound,
                "Model weights not found. Searched: " + searched);
        }
    }
    
    return load_model(model_path);
}

std::unique_ptr<DemucsModel> load_model(const std::string& path) {
    if (!fs::exists(path)) {
        throw DemucsError(DemucsError::Code::FileNotFound,
            "File not found: " + path);
    }
    
    auto model = std::make_unique<HTDemucsModel>();
    
    if (!WeightLoader::load_htdemucs_weights(model->internal_model(), path)) {
        throw DemucsError(DemucsError::Code::WeightLoadFailed,
            "Failed to load weights from: " + path);
    }
    
    return model;
}

// MARK: - Apply Model

mx::array apply_model(
    DemucsModel& model,
    const mx::array& mix,
    int shifts,
    bool split,
    float overlap,
    float transition_power,
    float segment
) {
    // Get the internal HTDemucs model
    auto* wrapper = dynamic_cast<HTDemucsModel*>(&model);
    if (!wrapper) {
        throw DemucsError(DemucsError::Code::UnsupportedModel,
            "Only HTDemucs models are currently supported");
    }
    
    return demucs::apply_model(
        wrapper->internal_model(),
        mix,
        shifts,
        split,
        overlap,
        transition_power,
        segment
    );
}

// MARK: - Separator Implementation

Separator::Separator() = default;
Separator::~Separator() = default;

const std::vector<std::string>& Separator::sources() const {
    if (model_) {
        return model_->sources();
    }
    return default_sources;
}

int Separator::sample_rate() const {
    return model_ ? model_->sample_rate() : 44100;
}

bool Separator::is_loaded() const {
    return model_ != nullptr;
}

void Separator::load_model(const std::string& path) {
    model_ = api::load_model(path);
}

void Separator::load_model_by_name(const std::string& name, const std::string& repo) {
    model_ = api::get_model(name, repo);
}

SeparationResult Separator::separate(
    const std::string& path,
    const SeparationOptions& options,
    ProgressCallback progress
) {
    auto audio_opt = load_audio(path);
    if (!audio_opt) {
        throw DemucsError(DemucsError::Code::AudioLoadFailed,
            "Failed to load audio from: " + path);
    }
    return separate_audio(*audio_opt, options, progress);
}

SeparationResult Separator::separate_audio(
    const mx::array& audio,
    const SeparationOptions& options,
    ProgressCallback progress
) {
    if (!model_) {
        throw DemucsError(DemucsError::Code::ModelNotLoaded,
            "Model weights not loaded. Call load_model() first.");
    }
    
    if (progress) progress(0.0f, "Normalizing audio...");
    
    // Normalize — matches Python: ref = wav.mean(0); wav = (wav - ref.mean()) / ref.std()
    auto ref = mx::mean(audio, 0);
    auto ref_mean = mx::mean(ref);
    auto ref_var = mx::var(ref, /*keepdims=*/false, /*ddof=*/1);
    auto ref_std = mx::sqrt(ref_var);
    mx::eval(ref_mean);
    mx::eval(ref_std);
    
    auto audio_norm = (audio - ref_mean) / (ref_std + 1e-8f);
    mx::eval(audio_norm);
    
    auto mix = mx::expand_dims(audio_norm, 0);
    
    if (progress) progress(0.05f, "Separating stems...");
    
    // Run model
    auto result = api::apply_model(
        *model_, mix,
        options.shifts,
        options.split,
        options.overlap,
        options.transition_power,
        options.segment > 0 ? options.segment : -1.0f
    );
    mx::eval(result);
    
    if (progress) progress(0.90f, "Denormalizing...");
    
    // Denormalize
    result = result * (ref_std + 1e-8f) + ref_mean;
    mx::eval(result);
    
    if (progress) progress(0.95f, "Extracting stems...");
    
    // Extract stems with clip prevention
    SeparationResult sep_result;
    sep_result.source_names = model_->sources();
    
    for (size_t idx = 0; idx < model_->sources().size(); ++idx) {
        const auto& name = model_->sources()[idx];
        auto source = mx::slice(result,
            {0, static_cast<int>(idx), 0, 0},
            {1, static_cast<int>(idx) + 1, 2, result.shape(-1)});
        source = mx::squeeze(source, {0, 1});
        auto clipped = prevent_clip(source, options.clip_mode);
        mx::eval(clipped);
        sep_result.stems.emplace_back(name, clipped);
    }
    
    if (progress) progress(1.0f, "Complete");
    
    return sep_result;
}

// MARK: - Audio Utilities

std::optional<mx::array> load_audio(const std::string& path, int sample_rate) {
    return Audio::load(path, sample_rate);
}

bool save_audio(
    const std::string& path,
    const mx::array& audio,
    int sample_rate,
    const std::string& clip,
    int bits_per_sample,
    bool as_float,
    int bitrate,
    const std::string& codec
) {
    auto clipped = prevent_clip(audio, clip);
    mx::eval(clipped);
    return Audio::save(path, clipped, sample_rate, bits_per_sample, as_float, bitrate, codec);
}

mx::array prevent_clip(const mx::array& audio, const std::string& mode) {
    return demucs::prevent_clip(audio, mode);
}

} // namespace api
} // namespace demucs
