#include <gtest/gtest.h>
#include <mlx/mlx.h>
#include "demucs/audio.hpp"
#include <filesystem>
#include <cmath>

namespace mx = mlx::core;
namespace fs = std::filesystem;

class AudioTest : public ::testing::Test {
protected:
    const std::string tmp_dir = ".wip/test_audio_tmp";
    const float tol = 0.01f;  // WAV quantization introduces error

    void SetUp() override {
        fs::create_directories(tmp_dir);
    }

    void TearDown() override {
        fs::remove_all(tmp_dir);
    }
};

TEST_F(AudioTest, SaveLoadWav16Bit) {
    // Create a simple sine wave
    int n = 44100;
    std::vector<float> data(2 * n);
    for (int i = 0; i < n; ++i) {
        float val = 0.5f * std::sin(2.0f * M_PI * 440.0f * i / 44100.0f);
        data[i] = val;          // left
        data[n + i] = val;      // right
    }
    auto audio = mx::array(data.data(), {2, n});

    std::string path = tmp_dir + "/test16.wav";
    bool ok = demucs::Audio::save(path, audio, 44100, 16);
    EXPECT_TRUE(ok);
    EXPECT_TRUE(fs::exists(path));

    auto loaded = demucs::Audio::load(path, 44100);
    ASSERT_TRUE(loaded.has_value());
    mx::eval(*loaded);

    EXPECT_EQ(loaded->shape(0), 2);
    EXPECT_EQ(loaded->shape(1), n);
}

TEST_F(AudioTest, SaveLoadWav24Bit) {
    int n = 22050;
    auto audio = mx::random::normal({2, n}) * 0.5f;
    mx::eval(audio);

    std::string path = tmp_dir + "/test24.wav";
    bool ok = demucs::Audio::save(path, audio, 44100, 24);
    EXPECT_TRUE(ok);

    auto loaded = demucs::Audio::load(path, 44100);
    ASSERT_TRUE(loaded.has_value());
    mx::eval(*loaded);
    EXPECT_EQ(loaded->shape(0), 2);
    EXPECT_EQ(loaded->shape(1), n);
}

TEST_F(AudioTest, SaveLoadWavFloat32) {
    int n = 22050;
    auto audio = mx::random::normal({2, n}) * 0.5f;
    mx::eval(audio);

    std::string path = tmp_dir + "/testf32.wav";
    bool ok = demucs::Audio::save(path, audio, 44100, 32, /*as_float=*/true);
    EXPECT_TRUE(ok);

    auto loaded = demucs::Audio::load(path, 44100);
    ASSERT_TRUE(loaded.has_value());
    mx::eval(*loaded);
    EXPECT_EQ(loaded->shape(0), 2);
}

TEST_F(AudioTest, SaveHandles3DInput) {
    // Batch dimension: (1, 2, samples)
    auto audio = mx::random::normal({1, 2, 11025}) * 0.5f;
    mx::eval(audio);

    std::string path = tmp_dir + "/test_batch.wav";
    bool ok = demucs::Audio::save(path, audio, 44100, 16);
    EXPECT_TRUE(ok);
}

TEST_F(AudioTest, LoadNonexistentFile) {
    auto result = demucs::Audio::load("/nonexistent/path.wav");
    EXPECT_FALSE(result.has_value());
}

TEST_F(AudioTest, SaveUnsupportedFormat) {
    auto audio = mx::random::normal({2, 1000});
    mx::eval(audio);
    bool ok = demucs::Audio::save(tmp_dir + "/test.ogg", audio);
    EXPECT_FALSE(ok);
}

#ifdef HAVE_AUDIO_TOOLBOX
TEST_F(AudioTest, SaveLoadM4a) {
    int n = 44100;
    auto audio = mx::random::normal({2, n}) * 0.5f;
    mx::eval(audio);

    std::string path = tmp_dir + "/test.m4a";
    bool ok = demucs::Audio::save(path, audio, 44100, 16, false, 256, "aac");
    EXPECT_TRUE(ok);
    EXPECT_TRUE(fs::exists(path));

    auto loaded = demucs::Audio::load(path, 44100);
    ASSERT_TRUE(loaded.has_value());
    mx::eval(*loaded);
    EXPECT_EQ(loaded->shape(0), 2);
    // AAC may change length slightly due to encoder padding
    EXPECT_GT(loaded->shape(1), 0);
}

TEST_F(AudioTest, SaveLoadFlac) {
    int n = 44100;
    auto audio = mx::random::normal({2, n}) * 0.5f;
    mx::eval(audio);

    std::string path = tmp_dir + "/test.flac";
    bool ok = demucs::Audio::save(path, audio, 44100, 16, false, 0, "flac");
    EXPECT_TRUE(ok);

    auto loaded = demucs::Audio::load(path, 44100);
    ASSERT_TRUE(loaded.has_value());
    mx::eval(*loaded);
    EXPECT_EQ(loaded->shape(0), 2);
}
#endif
