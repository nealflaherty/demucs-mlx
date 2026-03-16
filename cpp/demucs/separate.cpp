// separate.cpp - matches separate.py
// CLI for separating audio tracks using HTDemucs
//
// Usage: demucs_separate [options] <track1> [track2] ...

#include "demucs_api.hpp"
#include <mlx/mlx.h>
#include <iostream>
#include <filesystem>
#include <string>
#include <vector>
#include <algorithm>

namespace fs = std::filesystem;
namespace mx = mlx::core;
namespace api = demucs::api;

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
              << "  --model PATH           Model weights path (default: models/htdemucs.safetensors)\n"
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

int main(int argc, char* argv[]) {
    Args args = parse_args(argc, argv);
    
    if (args.tracks.empty()) {
        std::cerr << "Error: no tracks specified" << std::endl;
        print_usage(argv[0]);
        return 1;
    }
    
    // Validate --two-stems
    if (!args.two_stems.empty()) {
        auto it = std::find(api::default_sources.begin(), api::default_sources.end(), args.two_stems);
        if (it == api::default_sources.end()) {
            std::cerr << "Error: stem \"" << args.two_stems
                      << "\" is not in model. Must be one of: ";
            for (size_t i = 0; i < api::default_sources.size(); ++i) {
                if (i > 0) std::cerr << ", ";
                std::cerr << api::default_sources[i];
            }
            std::cerr << std::endl;
            return 1;
        }
    }
    
    // Load model via the public API
    std::unique_ptr<api::DemucsModel> model;
    try {
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
        model = api::load_model(args.model_path);
        std::cout << "Model loaded." << std::endl;
    } catch (const api::DemucsError& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    // Create output directory
    fs::create_directories(args.out_dir);
    std::cout << "Separated tracks will be stored in " << fs::absolute(args.out_dir) << std::endl;
    
    std::string ext = args.format;
    // For ALAC, the file extension is .m4a (ALAC in M4A container)
    std::string file_ext = (ext == "alac") ? "m4a" : ext;

    // Build separation options
    api::SeparationOptions options;
    options.shifts = args.shifts;
    options.split = args.split;
    options.overlap = args.overlap;
    options.segment = args.segment;
    options.clip_mode = args.clip_mode;

    for (const auto& track_path : args.tracks) {
        if (!fs::exists(track_path)) {
            std::cerr << "File " << track_path << " does not exist." << std::endl;
            continue;
        }
        
        std::cout << "Separating track " << track_path << std::endl;
        
        // Parse track name parts for filename template
        std::string track_name = fs::path(track_path).stem().string();
        std::string track_ext_str = fs::path(track_path).extension().string();
        if (!track_ext_str.empty() && track_ext_str[0] == '.') {
            track_ext_str = track_ext_str.substr(1);
        }
        
        if (args.two_stems.empty()) {
            // Use the Separator convenience class for the simple case
            try {
                api::Separator separator;
                separator.load_model(args.model_path);
                
                auto result = separator.separate(track_path, options,
                    [](float pct, const std::string& msg) {
                        std::cout << "\r  " << msg << " (" << static_cast<int>(pct * 100) << "%)" << std::flush;
                    });
                std::cout << std::endl;
                
                for (const auto& stem : result.stems) {
                    std::string stem_path = args.out_dir + "/" +
                        format_filename(args.filename, track_name, track_ext_str,
                                        stem.name, file_ext);
                    fs::create_directories(fs::path(stem_path).parent_path());
                    api::save_audio(stem_path, stem.audio, 44100, "none",
                                    args.bits_per_sample, args.float32, args.bitrate, args.format);
                }
            } catch (const api::DemucsError& e) {
                std::cerr << "Error separating " << track_path << ": " << e.what() << std::endl;
                continue;
            }
        } else {
            // Two-stems mode: use lower-level API for more control
            auto audio_opt = api::load_audio(track_path, 44100);
            if (!audio_opt) {
                std::cerr << "Failed to load audio: " << track_path << std::endl;
                continue;
            }
            auto wav = *audio_opt;
            
            // Normalize
            auto ref = mx::mean(wav, 0);
            auto ref_mean = mx::mean(ref);
            auto ref_var = mx::var(ref, /*keepdims=*/false, /*ddof=*/1);
            auto ref_std = mx::sqrt(ref_var);
            mx::eval(ref_mean);
            mx::eval(ref_std);
            
            wav = wav - ref_mean;
            wav = wav / (ref_std + 1e-8f);
            mx::eval(wav);
            
            auto mix = mx::expand_dims(wav, 0);
            
            std::cout << "Running separation..." << std::endl;
            auto out = api::apply_model(*model, mix, args.shifts, args.split,
                                         args.overlap, 1.0f, args.segment);
            mx::eval(out);
            
            // Denormalize
            out = out * (ref_std + 1e-8f) + ref_mean;
            mx::eval(out);
            
            wav = wav * (ref_std + 1e-8f) + ref_mean;
            mx::eval(wav);
            
            const auto& sources = model->sources();
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
            api::save_audio(stem_path, selected, 44100, args.clip_mode,
                            args.bits_per_sample, args.float32, args.bitrate, args.format);
            
            // Compute and save the complement
            if (args.other_method == "minus") {
                auto complement = wav - selected;
                std::string comp_path = args.out_dir + "/" +
                    format_filename(args.filename, track_name, track_ext_str,
                                    "no_" + args.two_stems, file_ext);
                fs::create_directories(fs::path(comp_path).parent_path());
                api::save_audio(comp_path, complement, 44100, args.clip_mode,
                                args.bits_per_sample, args.float32, args.bitrate, args.format);
            } else if (args.other_method == "add") {
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
                api::save_audio(comp_path, complement, 44100, args.clip_mode,
                                args.bits_per_sample, args.float32, args.bitrate, args.format);
            }
        }
        
        std::cout << "Saved stems to " << args.out_dir << "/" << track_name << "/" << std::endl;
    }
    
    return 0;
}
