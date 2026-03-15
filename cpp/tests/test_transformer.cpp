#include <gtest/gtest.h>
#include <mlx/mlx.h>
#include "demucs/transformer.hpp"

namespace mx = mlx::core;

class TransformerTest : public ::testing::Test {
protected:
    const float tol = 1e-4f;
};

// --- Sinusoidal embeddings ---

TEST_F(TransformerTest, SinEmbeddingShape) {
    auto emb = demucs::create_sin_embedding(100, 64);
    mx::eval(emb);
    // Returns (length, 1, dim)
    EXPECT_EQ(emb.shape(0), 100);
    EXPECT_EQ(emb.shape(1), 1);
    EXPECT_EQ(emb.shape(2), 64);
}

TEST_F(TransformerTest, SinEmbeddingBounded) {
    auto emb = demucs::create_sin_embedding(50, 32);
    mx::eval(emb);
    float max_val = mx::max(mx::abs(emb)).item<float>();
    EXPECT_LE(max_val, 1.0f + tol);
}

TEST_F(TransformerTest, Sin2DEmbeddingShape) {
    auto emb = demucs::create_2d_sin_embedding(64, 8, 16);
    mx::eval(emb);
    // Returns (1, d_model, height, width)
    EXPECT_EQ(emb.shape(0), 1);
    EXPECT_EQ(emb.shape(1), 64);
    EXPECT_EQ(emb.shape(2), 8);
    EXPECT_EQ(emb.shape(3), 16);
}

// --- LayerScale ---

TEST_F(TransformerTest, LayerScaleInitNearZero) {
    demucs::LayerScale ls(64, 1e-4f);
    auto x = mx::ones({1, 10, 64});
    auto y = ls.forward(x);
    mx::eval(y);
    float max_val = mx::max(mx::abs(y)).item<float>();
    // Should be very small (init * 1.0)
    EXPECT_LT(max_val, 0.01f);
}

TEST_F(TransformerTest, LayerScalePreservesShape) {
    demucs::LayerScale ls(32, 0.1f);
    auto x = mx::random::normal({2, 20, 32});
    auto y = ls.forward(x);
    mx::eval(y);
    EXPECT_EQ(y.shape(), x.shape());
}

// --- MyGroupNorm ---

TEST_F(TransformerTest, MyGroupNormShape) {
    demucs::MyGroupNorm gn(1, 64);
    auto x = mx::random::normal({2, 10, 64});
    auto y = gn.forward(x);
    mx::eval(y);
    EXPECT_EQ(y.shape(), x.shape());
}

// --- MyTransformerEncoderLayer ---

TEST_F(TransformerTest, SelfAttentionLayerShape) {
    demucs::MyTransformerEncoderLayer layer(
        /*d_model=*/64, /*nhead=*/4, /*dim_feedforward=*/256,
        /*dropout=*/0.0f, /*gelu=*/true, /*group_norm=*/1,
        /*norm_first=*/true, /*norm_out=*/true,
        /*layer_norm_eps=*/1e-5f, /*layer_scale=*/true,
        /*init_values=*/1e-4f);
    auto x = mx::random::normal({1, 16, 64});
    auto y = layer.forward(x);
    mx::eval(y);
    EXPECT_EQ(y.shape(), x.shape());
}

// --- CrossTransformerEncoderLayer ---

TEST_F(TransformerTest, CrossAttentionLayerShape) {
    demucs::CrossTransformerEncoderLayer layer(
        /*d_model=*/64, /*nhead=*/4, /*dim_feedforward=*/256,
        /*dropout=*/0.0f, /*gelu=*/true, /*layer_norm_eps=*/1e-5f,
        /*layer_scale=*/true, /*init_values=*/1e-4f,
        /*norm_first=*/true, /*group_norm=*/1, /*norm_out=*/true);
    auto q = mx::random::normal({1, 16, 64});
    auto k = mx::random::normal({1, 20, 64});
    auto y = layer.forward(q, k);
    mx::eval(y);
    // Output should match query shape
    EXPECT_EQ(y.shape(0), 1);
    EXPECT_EQ(y.shape(1), 16);
    EXPECT_EQ(y.shape(2), 64);
}

// --- CrossTransformerEncoder ---

TEST_F(TransformerTest, CrossTransformerEncoderShape) {
    demucs::CrossTransformerEncoder encoder(
        /*dim=*/64, /*emb=*/"sin", /*hidden_scale=*/4.0f,
        /*num_heads=*/4, /*num_layers=*/2, /*cross_first=*/false,
        /*dropout=*/0.0f, /*max_positions=*/1000,
        /*norm_in=*/true, /*norm_in_group=*/false,
        /*group_norm=*/1, /*norm_first=*/true, /*norm_out=*/true,
        /*max_period=*/10000.0f, /*weight_decay=*/0.0f,
        /*layer_scale=*/true, /*gelu=*/true);
    // x is 4D: (B, C, Fr, T1) — spectral branch
    // xt is 3D: (B, C, T2) — temporal branch
    auto x = mx::random::normal({1, 64, 4, 8});
    auto xt = mx::random::normal({1, 64, 24});
    auto [y, yt] = encoder.forward(x, xt);
    mx::eval(y);
    mx::eval(yt);
    EXPECT_EQ(y.shape(), x.shape());
    EXPECT_EQ(yt.shape(), xt.shape());
}
