#include <gtest/gtest.h>
#include <mlx/mlx.h>
#include "demucs/utils.hpp"
#include <cmath>

namespace mx = mlx::core;

class UtilsTest : public ::testing::Test {
protected:
    const float tol = 1e-4f;
};

// --- GELU ---

TEST_F(UtilsTest, GeluZero) {
    auto x = mx::zeros({1, 4, 8});
    auto y = demucs::utils::gelu(x);
    mx::eval(y);
    float max_val = mx::max(mx::abs(y)).item<float>();
    EXPECT_LT(max_val, tol);
}

TEST_F(UtilsTest, GeluPositiveMonotonic) {
    // GELU is monotonically increasing for x > ~-0.5
    auto x = mx::array({0.0f, 0.5f, 1.0f, 2.0f, 3.0f});
    auto y = demucs::utils::gelu(x);
    mx::eval(y);
    const float* d = y.data<float>();
    for (int i = 1; i < 5; ++i) {
        EXPECT_GT(d[i], d[i - 1]) << "GELU not increasing at index " << i;
    }
}

TEST_F(UtilsTest, GeluKnownValues) {
    // GELU(0) = 0, GELU(1) ≈ 0.8413, GELU(-1) ≈ -0.1587
    auto x = mx::array({0.0f, 1.0f, -1.0f});
    auto y = demucs::utils::gelu(x);
    mx::eval(y);
    const float* d = y.data<float>();
    EXPECT_NEAR(d[0], 0.0f, tol);
    EXPECT_NEAR(d[1], 0.8413f, 0.01f);
    EXPECT_NEAR(d[2], -0.1587f, 0.01f);
}

// --- GLU ---

TEST_F(UtilsTest, GluHalvesChannels) {
    auto x = mx::ones({1, 8, 16});
    auto y = demucs::utils::glu(x, 1);
    mx::eval(y);
    EXPECT_EQ(y.shape(1), 4);
}

TEST_F(UtilsTest, GluSigmoidGating) {
    // If b = 0, sigmoid(0) = 0.5, so GLU(a, 0) = 0.5 * a
    std::vector<float> data(16, 0.0f);
    for (int i = 0; i < 8; ++i) data[i] = 2.0f;  // a = 2
    // b = 0 (already)
    auto x = mx::array(data.data(), {1, 2, 8});
    auto y = demucs::utils::glu(x, 1);
    mx::eval(y);
    // Expect all values ≈ 1.0 (2.0 * 0.5)
    float mean_val = mx::mean(y).item<float>();
    EXPECT_NEAR(mean_val, 1.0f, tol);
}

// --- GroupNorm ---

TEST_F(UtilsTest, GroupNormNormalizesOutput) {
    // With identity affine (weight=1, bias=0), output should be ~zero mean, ~unit var
    auto x = mx::random::normal({1, 4, 64});
    auto w = mx::ones({4});
    auto b = mx::zeros({4});
    auto y = demucs::utils::group_norm(x, w, b, 1);
    mx::eval(y);
    float mean_val = mx::mean(y).item<float>();
    EXPECT_NEAR(mean_val, 0.0f, 0.1f);
}

TEST_F(UtilsTest, GroupNormPreservesShape) {
    auto x = mx::random::normal({2, 8, 32});
    auto w = mx::ones({8});
    auto b = mx::zeros({8});
    auto y = demucs::utils::group_norm(x, w, b, 4);
    mx::eval(y);
    EXPECT_EQ(y.shape(), x.shape());
}

TEST_F(UtilsTest, GroupNorm2D) {
    auto x = mx::random::normal({1, 4, 8, 8});
    auto w = mx::ones({4});
    auto b = mx::zeros({4});
    auto y = demucs::utils::group_norm(x, w, b, 2);
    mx::eval(y);
    EXPECT_EQ(y.shape(), x.shape());
}

// --- Conv1d ---

TEST_F(UtilsTest, Conv1dOutputShape) {
    // Input: (1, 2, 16), Weight: (4, 2, 3), stride=1, pad=1
    auto input = mx::random::normal({1, 2, 16});
    auto weight = mx::random::normal({4, 2, 3});
    auto bias = mx::zeros({4});
    auto y = demucs::utils::conv1d(input, weight, bias, 1, 1);
    mx::eval(y);
    EXPECT_EQ(y.shape(0), 1);
    EXPECT_EQ(y.shape(1), 4);
    EXPECT_EQ(y.shape(2), 16);
}

TEST_F(UtilsTest, Conv1dStride) {
    auto input = mx::random::normal({1, 2, 16});
    auto weight = mx::random::normal({4, 2, 3});
    auto bias = mx::zeros({4});
    auto y = demucs::utils::conv1d(input, weight, bias, 2, 1);
    mx::eval(y);
    EXPECT_EQ(y.shape(2), 8);
}

// --- ConvTranspose1d ---

TEST_F(UtilsTest, ConvTranspose1dOutputShape) {
    // ConvTranspose1d upsamples
    auto input = mx::random::normal({1, 4, 8});
    auto weight = mx::random::normal({4, 2, 4});  // (C_in, C_out, K)
    auto bias = mx::zeros({2});
    auto y = demucs::utils::conv_transpose1d(input, weight, bias, 2, 1);
    mx::eval(y);
    EXPECT_EQ(y.shape(0), 1);
    EXPECT_EQ(y.shape(1), 2);
    // output_length = (input_length - 1) * stride - 2*padding + kernel_size
    // = (8-1)*2 - 2*1 + 4 = 14 - 2 + 4 = 16
    EXPECT_EQ(y.shape(2), 16);
}

// --- Conv2d ---

TEST_F(UtilsTest, Conv2dOutputShape) {
    auto input = mx::random::normal({1, 3, 8, 8});
    auto weight = mx::random::normal({16, 3, 3, 3});
    auto bias = mx::zeros({16});
    auto y = demucs::utils::conv2d(input, weight, bias, {1, 1}, {1, 1});
    mx::eval(y);
    EXPECT_EQ(y.shape(0), 1);
    EXPECT_EQ(y.shape(1), 16);
    EXPECT_EQ(y.shape(2), 8);
    EXPECT_EQ(y.shape(3), 8);
}

// --- ConvTranspose2d ---

TEST_F(UtilsTest, ConvTranspose2dOutputShape) {
    auto input = mx::random::normal({1, 16, 4, 4});
    auto weight = mx::random::normal({16, 8, 4, 4});  // (C_in, C_out, kH, kW)
    auto bias = mx::zeros({8});
    auto y = demucs::utils::conv_transpose2d(input, weight, bias, {2, 2}, {1, 1});
    mx::eval(y);
    EXPECT_EQ(y.shape(0), 1);
    EXPECT_EQ(y.shape(1), 8);
    // (4-1)*2 - 2*1 + 4 = 8
    EXPECT_EQ(y.shape(2), 8);
    EXPECT_EQ(y.shape(3), 8);
}
