#pragma once

#include <mlx/mlx.h>

namespace demucs {

namespace mx = mlx::core;

/**
 * Compute STFT spectrogram
 * Matches spectro() from spec.py
 */
mx::array spectro(const mx::array& x, int n_fft = 512, 
                  int hop_length = -1, int pad = 0);

/**
 * Compute inverse STFT
 * Matches ispectro() from spec.py
 */
mx::array ispectro(const mx::array& z, int hop_length = -1,
                   int length = -1, int pad = 0);

} // namespace demucs
