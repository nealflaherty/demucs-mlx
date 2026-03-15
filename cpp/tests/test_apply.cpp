#include <gtest/gtest.h>
#include <mlx/mlx.h>
#include "demucs/apply.hpp"

namespace mx = mlx::core;

class ApplyTest : public ::testing::Test {
protected:
    const float tol = 1e-5f;
};

// --- center_trim ---

TEST_F(ApplyTest, CenterTrimNoOp) {
    auto x = mx::ones({1, 4, 2, 100});
    auto y = demucs::center_trim(x, 100);
    mx::eval(y);
    EXPECT_EQ(y.shape(-1), 100);
}

TEST_F(ApplyTest, CenterTrimEvenDelta) {
    auto x = mx::ones({1, 4, 2, 100});
    auto y = demucs::center_trim(x, 80);
    mx::eval(y);
    EXPECT_EQ(y.shape(-1), 80);
    // Trimmed 20 total: 10 from each side
}

TEST_F(ApplyTest, CenterTrimOddDelta) {
    auto x = mx::ones({1, 4, 2, 101});
    auto y = demucs::center_trim(x, 80);
    mx::eval(y);
    EXPECT_EQ(y.shape(-1), 80);
}

TEST_F(ApplyTest, CenterTrimPreservesOtherDims) {
    auto x = mx::ones({2, 4, 2, 100});
    auto y = demucs::center_trim(x, 60);
    mx::eval(y);
    EXPECT_EQ(y.shape(0), 2);
    EXPECT_EQ(y.shape(1), 4);
    EXPECT_EQ(y.shape(2), 2);
    EXPECT_EQ(y.shape(-1), 60);
}

TEST_F(ApplyTest, CenterTrimThrowsIfTooSmall) {
    auto x = mx::ones({1, 2, 50});
    EXPECT_THROW(demucs::center_trim(x, 100), std::runtime_error);
}

// --- prevent_clip ---

TEST_F(ApplyTest, PreventClipNonePassthrough) {
    auto x = mx::array({-2.0f, 0.0f, 2.0f});
    auto y = demucs::prevent_clip(x, "none");
    mx::eval(y);
    float diff = mx::max(mx::abs(x - y)).item<float>();
    EXPECT_LT(diff, tol);
}

TEST_F(ApplyTest, PreventClipRescaleQuiet) {
    // Signal already within [-1, 1] — should be unchanged
    auto x = mx::array({-0.5f, 0.0f, 0.5f});
    auto y = demucs::prevent_clip(x, "rescale");
    mx::eval(y);
    float diff = mx::max(mx::abs(x - y)).item<float>();
    EXPECT_LT(diff, tol);
}

TEST_F(ApplyTest, PreventClipRescaleLoud) {
    // Signal exceeds [-1, 1] — should be scaled down
    auto x = mx::array({-2.0f, 0.0f, 2.0f});
    auto y = demucs::prevent_clip(x, "rescale");
    mx::eval(y);
    float peak = mx::max(mx::abs(y)).item<float>();
    EXPECT_LT(peak, 1.0f);
}

TEST_F(ApplyTest, PreventClipClamp) {
    auto x = mx::array({-2.0f, 0.0f, 2.0f});
    auto y = demucs::prevent_clip(x, "clamp");
    mx::eval(y);
    float max_val = mx::max(y).item<float>();
    float min_val = mx::min(y).item<float>();
    EXPECT_LE(max_val, 0.99f + tol);
    EXPECT_GE(min_val, -0.99f - tol);
}

TEST_F(ApplyTest, PreventClipInvalidMode) {
    auto x = mx::zeros({4});
    EXPECT_THROW(demucs::prevent_clip(x, "invalid"), std::runtime_error);
}

// --- TensorChunk ---

TEST_F(ApplyTest, TensorChunkFullTensor) {
    auto x = mx::random::normal({1, 2, 1000});
    demucs::TensorChunk chunk(x);
    EXPECT_EQ(chunk.length(), 1000);
    EXPECT_EQ(chunk.offset(), 0);
}

TEST_F(ApplyTest, TensorChunkSubset) {
    auto x = mx::random::normal({1, 2, 1000});
    demucs::TensorChunk chunk(x, 100, 500);
    EXPECT_EQ(chunk.length(), 500);
    EXPECT_EQ(chunk.offset(), 100);
}

TEST_F(ApplyTest, TensorChunkPaddedExact) {
    auto x = mx::random::normal({1, 2, 1000});
    demucs::TensorChunk chunk(x, 0, 1000);
    auto padded = chunk.padded(1000);
    mx::eval(padded);
    EXPECT_EQ(padded.shape(-1), 1000);
}

TEST_F(ApplyTest, TensorChunkPaddedLonger) {
    auto x = mx::random::normal({1, 2, 1000});
    demucs::TensorChunk chunk(x, 0, 1000);
    auto padded = chunk.padded(2000);
    mx::eval(padded);
    EXPECT_EQ(padded.shape(-1), 2000);
}
