#include "audio.hpp"
#include <libnyquist/Common.h>
#include <libnyquist/Decoders.h>
#include <libnyquist/Encoders.h>
#include <iostream>
#include <cmath>
#include <filesystem>
#include <algorithm>

#ifdef HAVE_AUDIO_TOOLBOX
#include <AudioToolbox/AudioToolbox.h>
#endif

namespace demucs {

std::optional<mx::array> Audio::load(const std::string& path, int target_sample_rate) {
    // Check if file exists
    if (!std::filesystem::exists(path)) {
        std::cerr << "Audio file not found: " << path << std::endl;
        return std::nullopt;
    }
    
    // Check file extension to decide which loader to use
    std::string ext = std::filesystem::path(path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    
#ifdef HAVE_AUDIO_TOOLBOX
    // Use AudioToolbox for non-WAV formats (mp3, aac, m4a, flac, ogg, etc.)
    // Also use it as fallback for WAV if libnyquist fails
    if (ext != ".wav") {
        auto result = load_audio_toolbox(path, target_sample_rate);
        if (result) return result;
        std::cerr << "AudioToolbox failed, trying libnyquist..." << std::endl;
    }
#endif
    
    // Load using libnyquist (primary path for WAV)
    std::shared_ptr<nqr::AudioData> fileData = std::make_shared<nqr::AudioData>();
    nqr::NyquistIO loader;
    
    try {
        loader.Load(fileData.get(), path);
    } catch (const std::exception& e) {
#ifdef HAVE_AUDIO_TOOLBOX
        // WAV failed with libnyquist, try AudioToolbox as fallback
        if (ext == ".wav") {
            std::cerr << "libnyquist failed (" << e.what() << "), trying AudioToolbox..." << std::endl;
            return load_audio_toolbox(path, target_sample_rate);
        }
#endif
        std::cerr << "Failed to load audio file: " << e.what() << std::endl;
        return std::nullopt;
    }
    
    if (fileData->samples.empty()) {
        std::cerr << "No audio data in file: " << path << std::endl;
        return std::nullopt;
    }
    
    std::cout << "Loaded audio: " << path << std::endl;
    std::cout << "  Sample rate: " << fileData->sampleRate << " Hz" << std::endl;
    std::cout << "  Channels: " << fileData->channelCount << std::endl;
    std::cout << "  Duration: " << fileData->lengthSeconds << " seconds" << std::endl;
    
    int num_channels = fileData->channelCount;
    size_t samples_per_channel = fileData->samples.size() / num_channels;
    
    // Convert to stereo (2, samples) format
    std::vector<float> stereo_data(2 * samples_per_channel);
    
    if (num_channels == 1) {
        // Mono: duplicate to both channels
        for (size_t i = 0; i < samples_per_channel; ++i) {
            stereo_data[i] = fileData->samples[i];  // Left
            stereo_data[samples_per_channel + i] = fileData->samples[i];  // Right
        }
    } else if (num_channels == 2) {
        // Stereo: interleaved to planar
        for (size_t i = 0; i < samples_per_channel; ++i) {
            stereo_data[i] = fileData->samples[2 * i];  // Left
            stereo_data[samples_per_channel + i] = fileData->samples[2 * i + 1];  // Right
        }
    } else {
        // Multi-channel: take first two channels
        std::cout << "  Using first 2 channels from " << num_channels << " channel audio" << std::endl;
        for (size_t i = 0; i < samples_per_channel; ++i) {
            stereo_data[i] = fileData->samples[i * num_channels];  // Left (ch 0)
            stereo_data[samples_per_channel + i] = fileData->samples[i * num_channels + 1];  // Right (ch 1)
        }
    }
    
    // Re-normalize: libnyquist divides int16 by 32767.0, but Python's soundfile
    // (libsndfile) divides by 32768.0. Correct to match Python behavior.
    // This ensures bit-exact matching with the Python reference implementation.
    // Only applies to integer PCM formats (16-bit, 24-bit).
    constexpr float renorm = 32767.0f / 32768.0f;
    for (size_t i = 0; i < stereo_data.size(); ++i) {
        stereo_data[i] *= renorm;
    }
    
    // Create MLX array with shape (2, samples)
    mx::array audio = mx::array(stereo_data.data(), {2, static_cast<int>(samples_per_channel)});
    
    // Resample if needed
    if (fileData->sampleRate != target_sample_rate) {
        std::cout << "  Resampling from " << fileData->sampleRate << " Hz to " 
                  << target_sample_rate << " Hz" << std::endl;
        audio = resample(audio, fileData->sampleRate, target_sample_rate);
    }
    
    return audio;
}

mx::array Audio::resample(const mx::array& audio, int from_rate, int to_rate) {
    if (from_rate == to_rate) {
        return audio;
    }
    
    // Simple linear interpolation resampling
    int num_channels = audio.shape()[0];
    int input_samples = audio.shape()[1];
    int output_samples = static_cast<int>(std::round(static_cast<float>(input_samples) * to_rate / from_rate));
    
    float ratio = static_cast<float>(from_rate) / to_rate;
    
    std::vector<float> output_data(num_channels * output_samples);
    
    // Evaluate input to get data
    mx::eval(audio);
    const float* input_data = audio.data<float>();
    
    for (int ch = 0; ch < num_channels; ++ch) {
        for (int i = 0; i < output_samples; ++i) {
            float src_pos = i * ratio;
            int src_idx = static_cast<int>(src_pos);
            float frac = src_pos - src_idx;
            
            if (src_idx + 1 < input_samples) {
                // Linear interpolation
                float sample1 = input_data[ch * input_samples + src_idx];
                float sample2 = input_data[ch * input_samples + src_idx + 1];
                output_data[ch * output_samples + i] = sample1 + frac * (sample2 - sample1);
            } else {
                // Last sample
                output_data[ch * output_samples + i] = input_data[ch * input_samples + src_idx];
            }
        }
    }
    
    return mx::array(output_data.data(), {num_channels, output_samples});
}

bool Audio::save(const std::string& path, const mx::array& audio,
                 int sample_rate, int bits_per_sample, bool as_float, int bitrate,
                 const std::string& codec) {
    // Handle batch dimension if present
    mx::array audio_2d = audio;
    if (audio.ndim() == 3) {
        audio_2d = mx::squeeze(mx::slice(audio,
            {0, 0, 0},
            {1, audio.shape(1), audio.shape(2)}), 0);
    }
    
    if (audio_2d.ndim() != 2) {
        std::cerr << "Audio must be 2D (channels, samples) or 3D (batch, channels, samples)" << std::endl;
        return false;
    }
    
    // Determine format from extension
    std::string ext = std::filesystem::path(path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    
    if (ext == ".wav") {
        return save_wav(path, audio_2d, sample_rate, bits_per_sample, as_float);
    }
    
#ifdef HAVE_AUDIO_TOOLBOX
    if (ext == ".m4a" || ext == ".flac") {
        return save_audio_toolbox(path, audio_2d, sample_rate, bits_per_sample, bitrate, codec);
    }
#endif
    
    std::cerr << "Unsupported output format: " << ext << std::endl;
    std::cerr << "Supported: .wav, .m4a (AAC), .flac, .alac" << std::endl;
    return false;
}

bool Audio::save_wav(const std::string& path, const mx::array& audio_2d,
                     int sample_rate, int bits_per_sample, bool as_float) {
    int num_channels = audio_2d.shape()[0];
    int samples_per_channel = audio_2d.shape()[1];
    
    mx::eval(audio_2d);
    const float* data = audio_2d.data<float>();
    
    std::shared_ptr<nqr::AudioData> fileData = std::make_shared<nqr::AudioData>();
    fileData->sampleRate = sample_rate;
    fileData->channelCount = num_channels;
    fileData->sourceFormat = nqr::PCM_FLT;
    fileData->samples.resize(num_channels * samples_per_channel);
    
    // Convert from planar to interleaved
    for (int i = 0; i < samples_per_channel; ++i) {
        for (int ch = 0; ch < num_channels; ++ch) {
            fileData->samples[i * num_channels + ch] = data[ch * samples_per_channel + i];
        }
    }
    
    // Determine PCM format
    nqr::PCMFormat pcm_fmt;
    if (as_float || bits_per_sample == 32) {
        pcm_fmt = nqr::PCM_FLT;
    } else if (bits_per_sample == 24) {
        pcm_fmt = nqr::PCM_24;
    } else {
        pcm_fmt = nqr::PCM_16;
    }
    
    try {
        int status = nqr::encode_wav_to_disk(
            {fileData->channelCount, pcm_fmt, nqr::DITHER_TRIANGLE},
            fileData.get(),
            path
        );
        if (status == 0) {
            std::cout << "Saved audio: " << path << std::endl;
            return true;
        } else {
            std::cerr << "Failed to encode WAV file (status: " << status << ")" << std::endl;
            return false;
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception while saving WAV: " << e.what() << std::endl;
        return false;
    }
}

// ============================================================================
// AudioToolbox loader (macOS) — handles MP3, AAC, FLAC, ALAC, M4A, etc.
// ============================================================================

#ifdef HAVE_AUDIO_TOOLBOX

std::optional<mx::array> Audio::load_audio_toolbox(
    const std::string& path, int target_sample_rate)
{
    // Create URL from path
    CFURLRef fileURL = CFURLCreateFromFileSystemRepresentation(
        nullptr,
        reinterpret_cast<const UInt8*>(path.c_str()),
        static_cast<CFIndex>(path.size()),
        false
    );
    if (!fileURL) {
        std::cerr << "AudioToolbox: failed to create URL for " << path << std::endl;
        return std::nullopt;
    }
    
    // Open the audio file
    ExtAudioFileRef audioFile = nullptr;
    OSStatus status = ExtAudioFileOpenURL(fileURL, &audioFile);
    CFRelease(fileURL);
    
    if (status != noErr || !audioFile) {
        std::cerr << "AudioToolbox: failed to open " << path
                  << " (error " << status << ")" << std::endl;
        return std::nullopt;
    }
    
    // Get the file's native format
    AudioStreamBasicDescription fileFormat{};
    UInt32 propSize = sizeof(fileFormat);
    status = ExtAudioFileGetProperty(
        audioFile,
        kExtAudioFileProperty_FileDataFormat,
        &propSize,
        &fileFormat
    );
    if (status != noErr) {
        std::cerr << "AudioToolbox: failed to get file format (error "
                  << status << ")" << std::endl;
        ExtAudioFileDispose(audioFile);
        return std::nullopt;
    }
    
    int src_channels = static_cast<int>(fileFormat.mChannelsPerFrame);
    double src_sample_rate = fileFormat.mSampleRate;
    
    // Set the client (output) format: float32 PCM, stereo, target sample rate
    AudioStreamBasicDescription clientFormat{};
    clientFormat.mSampleRate       = static_cast<Float64>(target_sample_rate);
    clientFormat.mFormatID         = kAudioFormatLinearPCM;
    clientFormat.mFormatFlags      = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked;
    clientFormat.mChannelsPerFrame = 2;
    clientFormat.mBitsPerChannel   = 32;
    clientFormat.mBytesPerFrame    = 2 * sizeof(float);
    clientFormat.mFramesPerPacket  = 1;
    clientFormat.mBytesPerPacket   = 2 * sizeof(float);
    
    status = ExtAudioFileSetProperty(
        audioFile,
        kExtAudioFileProperty_ClientDataFormat,
        sizeof(clientFormat),
        &clientFormat
    );
    if (status != noErr) {
        std::cerr << "AudioToolbox: failed to set client format (error "
                  << status << ")" << std::endl;
        ExtAudioFileDispose(audioFile);
        return std::nullopt;
    }
    
    // Get total frame count (in file frames, not client frames)
    SInt64 fileLengthFrames = 0;
    propSize = sizeof(fileLengthFrames);
    status = ExtAudioFileGetProperty(
        audioFile,
        kExtAudioFileProperty_FileLengthFrames,
        &propSize,
        &fileLengthFrames
    );
    if (status != noErr) {
        std::cerr << "AudioToolbox: failed to get file length (error "
                  << status << ")" << std::endl;
        ExtAudioFileDispose(audioFile);
        return std::nullopt;
    }
    
    // Estimate output frames after sample rate conversion
    SInt64 estimatedFrames = static_cast<SInt64>(
        std::ceil(static_cast<double>(fileLengthFrames) *
                  target_sample_rate / src_sample_rate)
    );
    
    // Read all audio data in chunks
    const UInt32 chunkFrames = 8192;
    std::vector<float> allSamples;
    allSamples.reserve(static_cast<size_t>(estimatedFrames) * 2);
    
    std::vector<float> buffer(chunkFrames * 2);
    AudioBufferList bufList;
    bufList.mNumberBuffers = 1;
    bufList.mBuffers[0].mNumberChannels = 2;
    
    while (true) {
        UInt32 framesToRead = chunkFrames;
        bufList.mBuffers[0].mDataByteSize = chunkFrames * 2 * sizeof(float);
        bufList.mBuffers[0].mData = buffer.data();
        
        status = ExtAudioFileRead(audioFile, &framesToRead, &bufList);
        if (status != noErr) {
            std::cerr << "AudioToolbox: read error (error "
                      << status << ")" << std::endl;
            ExtAudioFileDispose(audioFile);
            return std::nullopt;
        }
        if (framesToRead == 0) break;
        
        allSamples.insert(
            allSamples.end(),
            buffer.begin(),
            buffer.begin() + static_cast<ptrdiff_t>(framesToRead * 2)
        );
    }
    
    ExtAudioFileDispose(audioFile);
    
    size_t totalFrames = allSamples.size() / 2;
    if (totalFrames == 0) {
        std::cerr << "AudioToolbox: no audio data in " << path << std::endl;
        return std::nullopt;
    }
    
    std::cout << "Loaded audio: " << path << std::endl;
    std::cout << "  Sample rate: " << target_sample_rate << " Hz"
              << " (source: " << src_sample_rate << " Hz)" << std::endl;
    std::cout << "  Channels: 2 (source: " << src_channels << ")" << std::endl;
    std::cout << "  Duration: " << totalFrames / target_sample_rate << " seconds" << std::endl;
    
    // Convert from interleaved (L R L R ...) to planar (LL...RR...)
    std::vector<float> planar(totalFrames * 2);
    for (size_t i = 0; i < totalFrames; ++i) {
        planar[i]              = allSamples[2 * i];      // Left
        planar[totalFrames + i] = allSamples[2 * i + 1]; // Right
    }
    
    return mx::array(planar.data(), {2, static_cast<int>(totalFrames)});
}

bool Audio::save_audio_toolbox(
    const std::string& path, const mx::array& audio_2d,
    int sample_rate, int bits_per_sample, int bitrate,
    const std::string& codec)
{
    int num_channels = audio_2d.shape()[0];
    int samples_per_channel = audio_2d.shape()[1];
    
    mx::eval(audio_2d);
    const float* data = audio_2d.data<float>();
    
    // Convert from planar to interleaved
    std::vector<float> interleaved(num_channels * samples_per_channel);
    for (int i = 0; i < samples_per_channel; ++i) {
        for (int ch = 0; ch < num_channels; ++ch) {
            interleaved[i * num_channels + ch] = data[ch * samples_per_channel + i];
        }
    }
    
    // Determine output format from extension
    std::string ext = std::filesystem::path(path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    
    AudioFileTypeID fileType;
    AudioStreamBasicDescription outputFormat = {};
    outputFormat.mSampleRate = static_cast<Float64>(sample_rate);
    outputFormat.mChannelsPerFrame = static_cast<UInt32>(num_channels);
    
    // Use codec override if provided, otherwise infer from extension
    std::string effective_codec = codec;
    if (effective_codec.empty()) {
        if (ext == ".flac") effective_codec = "flac";
        else effective_codec = "aac";  // .m4a defaults to AAC
    }
    
    if (effective_codec == "aac") {
        fileType = kAudioFileM4AType;
        outputFormat.mFormatID = kAudioFormatMPEG4AAC;
    } else if (effective_codec == "flac") {
        fileType = kAudioFileFLACType;
        outputFormat.mFormatID = kAudioFormatFLAC;
        outputFormat.mBitsPerChannel = static_cast<UInt32>(
            bits_per_sample <= 16 ? 16 : 24);
    } else if (effective_codec == "alac") {
        fileType = kAudioFileM4AType;
        outputFormat.mFormatID = kAudioFormatAppleLossless;
    } else {
        std::cerr << "AudioToolbox: unsupported codec " << effective_codec << std::endl;
        return false;
    }
    
    // Create output file
    CFURLRef fileURL = CFURLCreateFromFileSystemRepresentation(
        nullptr, reinterpret_cast<const UInt8*>(path.c_str()),
        static_cast<CFIndex>(path.size()), false);
    if (!fileURL) {
        std::cerr << "AudioToolbox: failed to create URL for " << path << std::endl;
        return false;
    }
    
    ExtAudioFileRef outFile = nullptr;
    OSStatus status = ExtAudioFileCreateWithURL(
        fileURL, fileType, &outputFormat, nullptr,
        kAudioFileFlags_EraseFile, &outFile);
    CFRelease(fileURL);
    
    if (status != noErr || !outFile) {
        std::cerr << "AudioToolbox: failed to create " << path
                  << " (error " << status << ")" << std::endl;
        return false;
    }
    
    // Set AAC bitrate if applicable
    if (outputFormat.mFormatID == kAudioFormatMPEG4AAC) {
        UInt32 aacBitrate = static_cast<UInt32>(bitrate * 1000);
        AudioConverterRef converter = nullptr;
        UInt32 converterSize = sizeof(converter);
        status = ExtAudioFileGetProperty(outFile,
            kExtAudioFileProperty_AudioConverter, &converterSize, &converter);
        if (status == noErr && converter) {
            AudioConverterSetProperty(converter,
                kAudioConverterEncodeBitRate, sizeof(aacBitrate), &aacBitrate);
        }
    }
    
    // Set client format: float32 interleaved
    AudioStreamBasicDescription clientFormat = {};
    clientFormat.mSampleRate = static_cast<Float64>(sample_rate);
    clientFormat.mFormatID = kAudioFormatLinearPCM;
    clientFormat.mFormatFlags = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked;
    clientFormat.mChannelsPerFrame = static_cast<UInt32>(num_channels);
    clientFormat.mBitsPerChannel = 32;
    clientFormat.mBytesPerFrame = static_cast<UInt32>(num_channels) * sizeof(float);
    clientFormat.mFramesPerPacket = 1;
    clientFormat.mBytesPerPacket = clientFormat.mBytesPerFrame;
    
    status = ExtAudioFileSetProperty(outFile,
        kExtAudioFileProperty_ClientDataFormat,
        sizeof(clientFormat), &clientFormat);
    if (status != noErr) {
        std::cerr << "AudioToolbox: failed to set client format (error "
                  << status << ")" << std::endl;
        ExtAudioFileDispose(outFile);
        return false;
    }
    
    // Write in chunks
    const UInt32 chunkFrames = 8192;
    UInt32 framesRemaining = static_cast<UInt32>(samples_per_channel);
    UInt32 offset = 0;
    
    while (framesRemaining > 0) {
        UInt32 framesToWrite = std::min(chunkFrames, framesRemaining);
        
        AudioBufferList bufList;
        bufList.mNumberBuffers = 1;
        bufList.mBuffers[0].mNumberChannels = static_cast<UInt32>(num_channels);
        bufList.mBuffers[0].mDataByteSize = framesToWrite * static_cast<UInt32>(num_channels) * sizeof(float);
        bufList.mBuffers[0].mData = const_cast<float*>(interleaved.data() + offset * num_channels);
        
        status = ExtAudioFileWrite(outFile, framesToWrite, &bufList);
        if (status != noErr) {
            std::cerr << "AudioToolbox: write error (error " << status << ")" << std::endl;
            ExtAudioFileDispose(outFile);
            return false;
        }
        
        offset += framesToWrite;
        framesRemaining -= framesToWrite;
    }
    
    ExtAudioFileDispose(outFile);
    std::cout << "Saved audio: " << path << std::endl;
    return true;
}

#endif // HAVE_AUDIO_TOOLBOX

} // namespace demucs
