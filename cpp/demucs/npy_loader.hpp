#pragma once

#include <mlx/mlx.h>
#include <string>
#include <vector>

namespace demucs {

/**
 * Load a NumPy .npy file into an MLX array
 * 
 * Supports:
 * - float32, float64, int32, int64, complex64, complex128
 * - Any shape/dimensionality
 * - C-contiguous and Fortran-contiguous arrays
 * 
 * @param path Path to .npy file
 * @return MLX array with data from file
 */
mlx::core::array load_npy(const std::string& path);

/**
 * Save an MLX array to a NumPy .npy file
 * 
 * @param path Path to output .npy file
 * @param arr Array to save
 */
void save_npy(const std::string& path, const mlx::core::array& arr);

} // namespace demucs
