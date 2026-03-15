#include <gtest/gtest.h>
#include <mlx/mlx.h>
#include "demucs/spec.hpp"
#include <cmath>

namespace mx = mlx::core;

class SpecTest : public ::testing::Test {
protected:
    const float tol = 1e-4f;
};

// --- spectro ---

TEST_F(SpecTest, SpectroOutputShape) {
    // Input: (2, 44100) stereo 1 second
    auto x = mx::random::normal({2, 44100});
    auto z = demucs::spectro(x, 4096, 1024);
    mx::eval(z);
    // Output should be (2, n_fft/2+1, time_frames) complex
    EXPECT_EQ(z.shape(0), 2);
    EXPECT_EQ(z.shape(1), 2049);  // 4096/2 + 1
    EXPECT_GT(z.shape(2), 0);
}

TEST_F(SpecTest, SpectroDefaultHopLength) {
    auto x = mx::random::normal({2, 44100});
    // hop_length = -1 means n_fft / 4
    auto z = demucs::spectro(x, 4096, -1);
    mx::eval(z);
    EXPECT_EQ(z.shape(1), 2049);
}

// --- ispectro (inverse) ---

TEST_F(SpecTest, RoundTrip) {
    // STFT -> iSTFT should reconstruct the signal
    int n_samples = 44100;
    auto x = mx::random::normal({2, n_samples});
    mx::eval(x);

    auto z = demucs::spectro(x, 4096, 1024);
    mx::eval(z);
    auto y = demucs::ispectro(z, 1024, n_samples);
    mx::eval(y);

    EXPECT_EQ(y.shape(0), 2);
    EXPECT_EQ(y.shape(1), n_samples);

    auto diff = mx::abs(x - y);
    mx::eval(diff);
    float max_err = mx::max(diff).item<float>();
    // STFT round-trip should be near-perfect
    EXPECT_LT(max_err, 0.01f) << "Round-trip error too large: " << max_err;
}

TEST_F(SpecTest, RoundTripShortSignal) {
    int n_samples = 8000;
    auto x = mx::random::normal({1, n_samples});
    mx::eval(x);

    auto z = demucs::spectro(x, 4096, 1024);
    mx::eval(z);
    auto y = demucs::ispectro(z, 1024, n_samples);
    mx::eval(y);

    EXPECT_EQ(y.shape(1), n_samples);
    float max_err = mx::max(mx::abs(x - y)).item<float>();
    EXPECT_LT(max_err, 0.01f);
}
