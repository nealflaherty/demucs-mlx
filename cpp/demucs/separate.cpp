// separate.cpp - matches separate.py
// CLI for separating audio tracks using HTDemucs
//
// Usage: demucs_separate [options] <track1> [track2] ...

#include "htdemucs.hpp"
#include "audio.hpp"
#include "weight_loader.hpp"
#include "apply.hpp"
#include <mlx/mlx.h>
#include <iostream>
#include <filesystem>
#include <string>
#include <vector>
#include <algorithm>

namespace fs = std::filesystem;
namespace mx = mlx::core;

struct Args {
    std::vector<std::string> tracks;
    std::string model_path = "models/htdemucs.safetensors";
    std::string out_dir = "separated/htdemucs";
    std::string filename = "{track}/{stem}.{ext}";
    int shifts = 1;
    float overlap = 0.25f;
    bool split = true;
    float segment = -1.0f;  // -1 = use model default
    std::string clip_mode = "rescale";
    bool float32 = false;
    int bits_per_sample = 16;
    std::string format = "wav";  // wav, m4a, flac, alac
    int bitrate = 256;           // AAC bitrate in kbps
    std::string two_stems;       // empty = all stems, otherwise stem name
    std::string other_method = "add";  // add, minus, none
};

// Simple {key} template replacement
static std::string format_filename(const std::string& pattern,
                                   const std::string& track,
                                   const std::string& trackext,
                                   const std::string& stem,
                                   const std::string& ext) {
    std::string result = pattern;
    auto replace_all = [&](const std::string& key, const std::string& val) {
        size_t pos = 0;
        while ((pos = result.find(key, pos)) != std::string::npos) {
            result.replace(pos, key.length(), val);
            pos += val.length();
        }
    };
    replace_all("{track}", track);
    replace_all("{trackext}", trackext);
    replace_all("{stem}", stem);
    replace_all("{ext}", ext);
    return result;
}

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options] <track1> [track2] ...\n"
              << "\nOptions:\n"
              << "  -o, --out DIR          Output directory (default: separated/htdemucs)\n"
              << "  --model PATH           Model weights path (default: models/htdemucs_pytorch.safetensors)\n"
              << "  --filename PATTERN     Output filename pattern (default: {track}/{stem}.{ext})\n"
              << "                         Variables: {track}, {trackext}, {stem}, {ext}\n"
              << "  --shifts N             Number of random shifts (default: 1)\n"
              << "  --overlap F            Overlap between chunks (default: 0.25)\n"
              << "  --no-split             Don't split audio into chunks\n"
              << "  --segment N            Segment length in seconds\n"
              << "  --clip-mode MODE       Clipping strategy: rescale, clamp, none (default: rescale)\n"
              << "  --two-stems STEM       Only separate into STEM and no_STEM\n"
              << "  --other-method METHOD  How to compute no_STEM: add, minus, none (default: add)\n"
              << "\nOutput format (mutually exclusive):\n"
              << "  --wav                  Output as WAV (default)\n"
              << "  --m4a                  Output as AAC in M4A container\n"
              << "  --flac                 Output as FLAC lossless\n"
              << "  --alac                 Output as ALAC lossless in M4A\n"
              << "\nWAV options:\n"
              << "  --int24                Save WAV as 24-bit integer PCM\n"
              << "  --float32              Save WAV as 32-bit float PCM\n"
              << "\nAAC options:\n"
              << "  --bitrate N            AAC bitrate in kbps (default: 256)\n"
              << "\n  -h, --help             Show this help\n";
}

Args parse_args(int argc, char* argv[]) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            exit(0);
        } else if (arg == "-o" || arg == "--out") {
            if (++i < argc) args.out_dir = argv[i];
        } else if (arg == "--model") {
            if (++i < argc) args.model_path = argv[i];
        } else if (arg == "--filename") {
            if (++i < argc) args.filename = argv[i];
        } else if (arg == "--shifts") {
            if (++i < argc) args.shifts = std::stoi(argv[i]);
        } else if (arg == "--overlap") {
            if (++i < argc) args.overlap = std::stof(argv[i]);
        } else if (arg == "--no-split") {
            args.split = false;
        } else if (arg == "--segment") {
            if (++i < argc) args.segment = std::stof(argv[i]);
        } else if (arg == "--clip-mode") {
            if (++i < argc) args.clip_mode = argv[i];
        } else if (arg == "--two-stems") {
            if (++i < argc) args.two_stems = argv[i];
        } else if (arg == "--other-method") {
            if (++i < argc) args.other_method = argv[i];
        } else if (arg == "--wav") {
            args.format = "wav";
        } else if (arg == "--m4a") {
            args.format = "m4a";
        } else if (arg == "--flac") {
            args.format = "flac";
        } else if (arg == "--alac") {
            args.format = "alac";
        } else if (arg == "--int24") {
            args.bits_per_sample = 24;
        } else if (arg == "--float32") {
            args.float32 = true;
        } else if (arg == "--bitrate") {
            if (++i < argc) args.bitrate = std::stoi(argv[i]);
        } else if (arg[0] != '-') {
            args.tracks.push_back(arg);
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            print_usage(argv[0]);
            exit(1);
        }
    }
    return args;
}

static void save_stem(const mx::array& source, const std::string& path,
                      const Args& args) {
    auto clipped = demucs::prevent_clip(source, args.clip_mode);
    mx::eval(clipped);
    demucs::Audio::save(path, clipped, 44100, args.bits_per_sample,
                        args.float32, args.bitrate, args.format);
}

int main(int argc, char* argv[]) {
    Args args = parse_args(argc, argv);
    
    if (args.tracks.empty()) {
        std::cerr << "Error: no tracks specified" << std::endl;
        print_usage(argv[0]);
        return 1;
    }
    
    // Validate --two-stems
    std::vector<std::string> sources = {"drums", "bass", "other", "vocals"};
    if (!args.two_stems.empty()) {
        auto it = std::find(sources.begin(), sources.end(), args.two_stems);
        if (it == sources.end()) {
            std::cerr << "Error: stem \"" << args.two_stems
                      << "\" is not in model. Must be one of: ";
            for (size_t i = 0; i < sources.size(); ++i) {
                if (i > 0) std::cerr << ", ";
                std::cerr << sources[i];
            }
            std::cerr << std::endl;
            return 1;
        }
    }
    
    // Create model with htdemucs defaults
    demucs::HTDemucs model(
        sources,
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
    );
    
    // Load weights
    if (!fs::exists(args.model_path)) {
        std::cerr << "Error: model file not found: " << args.model_path << std::endl;
        std::cerr << std::endl;
        std::cerr << "Download the model first:" << std::endl;
        std::cerr << "  ./tools/download_model.sh" << std::endl;
        std::cerr << std::endl;
        std::cerr << "Or specify a path with --model <path>" << std::endl;
        return 1;
    }
    std::cout << "Loading model weights from " << args.model_path << "..." << std::endl;
    if (!demucs::WeightLoader::load_htdemucs_weights(model, args.model_path)) {
        std::cerr << "Failed to load weights!" << std::endl;
        return 1;
    }
    std::cout << "Model loaded." << std::endl;
    
    // Create output directory
    fs::create_directories(args.out_dir);
    std::cout << "Separated tracks will be stored in " << fs::absolute(args.out_dir) << std::endl;
    
    std::string ext = args.format;
    // For ALAC, the file extension is .m4a (ALAC in M4A container)
    std::string file_ext = (ext == "alac") ? "m4a" : ext;

    for (const auto& track_path : args.tracks) {
        if (!fs::exists(track_path)) {
            std::cerr << "File " << track_path << " does not exist." << std::endl;
            continue;
        }
        
        std::cout << "Separating track " << track_path << std::endl;
        
        // Load audio
        auto audio_opt = demucs::Audio::load(track_path, 44100);
        if (!audio_opt) {
            std::cerr << "Failed to load audio: " << track_path << std::endl;
            continue;
        }
        auto wav = *audio_opt;  // shape: (2, samples)
        
        // Python: ref = wav.mean(0); wav -= ref.mean(); wav /= ref.std() + 1e-8
        auto ref = mx::mean(wav, 0);
        auto ref_mean = mx::mean(ref);
        auto ref_var = mx::var(ref, /*keepdims=*/false, /*ddof=*/1);
        auto ref_std = mx::sqrt(ref_var);
        mx::eval(ref_mean);
        mx::eval(ref_std);
        
        wav = wav - ref_mean;
        wav = wav / (ref_std + 1e-8f);
        mx::eval(wav);
        
        int length = wav.shape(-1);
        auto mix = mx::expand_dims(wav, 0);
        
        // Apply model
        std::cout << "Running separation..." << std::endl;
        auto out = demucs::apply_model(model, mix, args.shifts, args.split,
                                        args.overlap, 1.0f, args.segment);
        mx::eval(out);
        
        // Denormalize
        out = out * (ref_std + 1e-8f);
        out = out + ref_mean;
        mx::eval(out);
        
        // Also denormalize original wav for --other-method minus
        wav = wav * (ref_std + 1e-8f);
        wav = wav + ref_mean;
        mx::eval(wav);
        
        // Parse track name parts for filename template
        std::string track_name = fs::path(track_path).stem().string();
        std::string track_ext_str = fs::path(track_path).extension().string();
        if (!track_ext_str.empty() && track_ext_str[0] == '.') {
            track_ext_str = track_ext_str.substr(1);
        }
        
        // out shape: (1, 4, 2, length)
        if (args.two_stems.empty()) {
            // Save all stems
            for (size_t s = 0; s < sources.size(); ++s) {
                auto source = mx::slice(out,
                    {0, static_cast<int>(s), 0, 0},
                    {1, static_cast<int>(s) + 1, 2, out.shape(-1)});
                source = mx::squeeze(source, {0, 1});
                
                std::string stem_path = args.out_dir + "/" +
                    format_filename(args.filename, track_name, track_ext_str,
                                    sources[s], file_ext);
                fs::create_directories(fs::path(stem_path).parent_path());
                save_stem(source, stem_path, args);
            }
        } else {
            // Two-stems mode: save selected stem and optionally no_stem
            int stem_idx = -1;
            for (size_t s = 0; s < sources.size(); ++s) {
                if (sources[s] == args.two_stems) { stem_idx = static_cast<int>(s); break; }
            }
            
            // Save the selected stem
            auto selected = mx::slice(out,
                {0, stem_idx, 0, 0},
                {1, stem_idx + 1, 2, out.shape(-1)});
            selected = mx::squeeze(selected, {0, 1});
            
            std::string stem_path = args.out_dir + "/" +
                format_filename(args.filename, track_name, track_ext_str,
                                args.two_stems, file_ext);
            fs::create_directories(fs::path(stem_path).parent_path());
            save_stem(selected, stem_path, args);
            
            // Compute and save the complement
            if (args.other_method == "minus") {
                // no_stem = original - stem
                auto complement = wav - selected;
                std::string comp_path = args.out_dir + "/" +
                    format_filename(args.filename, track_name, track_ext_str,
                                    "no_" + args.two_stems, file_ext);
                fs::create_directories(fs::path(comp_path).parent_path());
                save_stem(complement, comp_path, args);
            } else if (args.other_method == "add") {
                // no_stem = sum of all other stems
                mx::array complement = mx::zeros({2, out.shape(-1)});
                for (size_t s = 0; s < sources.size(); ++s) {
                    if (static_cast<int>(s) == stem_idx) continue;
                    auto other = mx::slice(out,
                        {0, static_cast<int>(s), 0, 0},
                        {1, static_cast<int>(s) + 1, 2, out.shape(-1)});
                    other = mx::squeeze(other, {0, 1});
                    complement = complement + other;
                }
                mx::eval(complement);
                std::string comp_path = args.out_dir + "/" +
                    format_filename(args.filename, track_name, track_ext_str,
                                    "no_" + args.two_stems, file_ext);
                fs::create_directories(fs::path(comp_path).parent_path());
                save_stem(complement, comp_path, args);
            }
            // other_method == "none": don't save complement
        }
        
        std::cout << "Saved stems to " << args.out_dir << "/" << track_name << "/" << std::endl;
    }
    
    return 0;
}
