#include "npy_loader.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cstring>
#include <vector>
#include <map>

namespace demucs {

namespace mx = mlx::core;

// NPY file format constants
const char NPY_MAGIC[] = "\x93NUMPY";
const size_t NPY_MAGIC_LEN = 6;

struct NpyHeader {
    std::vector<int> shape;
    std::string dtype;
    bool fortran_order;
    size_t header_len;
};

// Parse NPY header dictionary
NpyHeader parse_npy_header(std::ifstream& file) {
    NpyHeader header;
    
    // Read magic string
    char magic[NPY_MAGIC_LEN];
    file.read(magic, NPY_MAGIC_LEN);
    if (std::memcmp(magic, NPY_MAGIC, NPY_MAGIC_LEN) != 0) {
        throw std::runtime_error("Not a valid NPY file (bad magic number)");
    }
    
    // Read version
    uint8_t major_version, minor_version;
    file.read(reinterpret_cast<char*>(&major_version), 1);
    file.read(reinterpret_cast<char*>(&minor_version), 1);
    
    if (major_version != 1 && major_version != 2) {
        throw std::runtime_error("Unsupported NPY version: " + 
                                std::to_string(major_version));
    }
    
    // Read header length
    uint16_t header_len_v1 = 0;
    uint32_t header_len_v2 = 0;
    
    if (major_version == 1) {
        file.read(reinterpret_cast<char*>(&header_len_v1), 2);
        header.header_len = header_len_v1;
    } else {
        file.read(reinterpret_cast<char*>(&header_len_v2), 4);
        header.header_len = header_len_v2;
    }
    
    // Read header dictionary
    std::vector<char> header_str(header.header_len);
    file.read(header_str.data(), header.header_len);
    std::string header_dict(header_str.begin(), header_str.end());
    
    // Parse shape
    size_t shape_start = header_dict.find("'shape': (");
    if (shape_start == std::string::npos) {
        shape_start = header_dict.find("\"shape\": (");
    }
    if (shape_start != std::string::npos) {
        size_t shape_end = header_dict.find(")", shape_start);
        std::string shape_str = header_dict.substr(
            shape_start + 10, shape_end - shape_start - 10);
        
        // Parse comma-separated integers
        std::stringstream ss(shape_str);
        std::string item;
        while (std::getline(ss, item, ',')) {
            // Trim whitespace
            item.erase(0, item.find_first_not_of(" \t"));
            item.erase(item.find_last_not_of(" \t") + 1);
            if (!item.empty()) {
                header.shape.push_back(std::stoi(item));
            }
        }
    }
    
    // Parse dtype
    size_t dtype_start = header_dict.find("'descr': '");
    if (dtype_start == std::string::npos) {
        dtype_start = header_dict.find("\"descr\": \"");
    }
    if (dtype_start != std::string::npos) {
        size_t dtype_end = header_dict.find("'", dtype_start + 10);
        if (dtype_end == std::string::npos) {
            dtype_end = header_dict.find("\"", dtype_start + 10);
        }
        header.dtype = header_dict.substr(
            dtype_start + 10, dtype_end - dtype_start - 10);
    }
    
    // Parse fortran_order
    header.fortran_order = header_dict.find("'fortran_order': True") != std::string::npos ||
                          header_dict.find("\"fortran_order\": True") != std::string::npos;
    
    return header;
}

// Map NumPy dtype to MLX dtype
mx::Dtype numpy_dtype_to_mlx(const std::string& numpy_dtype) {
    // NumPy dtype format: '<f4' = little-endian float32
    // We care about the type character and size
    
    char endian = numpy_dtype[0];  // '<' = little-endian, '>' = big-endian
    char type_char = numpy_dtype[1];
    int size = std::stoi(numpy_dtype.substr(2));
    
    // Check endianness (we assume little-endian system)
    if (endian == '>') {
        throw std::runtime_error("Big-endian arrays not supported");
    }
    
    switch (type_char) {
        case 'f':  // float
            if (size == 4) return mx::float32;
            if (size == 8) return mx::float64;
            break;
        case 'i':  // signed int
            if (size == 4) return mx::int32;
            if (size == 8) return mx::int64;
            break;
        case 'u':  // unsigned int
            if (size == 4) return mx::uint32;
            if (size == 8) return mx::uint64;
            break;
        case 'c':  // complex
            if (size == 8) return mx::complex64;
            // complex128 not supported in this MLX version
            break;
    }
    
    throw std::runtime_error("Unsupported NumPy dtype: " + numpy_dtype);
}

mx::array load_npy(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + path);
    }
    
    // Parse header
    NpyHeader header = parse_npy_header(file);
    
    if (header.shape.empty()) {
        throw std::runtime_error("Could not parse shape from NPY header");
    }
    if (header.dtype.empty()) {
        throw std::runtime_error("Could not parse dtype from NPY header");
    }
    
    // Calculate total number of elements
    size_t num_elements = 1;
    for (int dim : header.shape) {
        num_elements *= dim;
    }
    
    // Get MLX dtype
    mx::Dtype mlx_dtype = numpy_dtype_to_mlx(header.dtype);
    
    // Read data
    size_t element_size = 0;
    switch (mlx_dtype) {
        case mx::float32: element_size = 4; break;
        case mx::float64: element_size = 8; break;
        case mx::int32: element_size = 4; break;
        case mx::int64: element_size = 8; break;
        case mx::uint32: element_size = 4; break;
        case mx::uint64: element_size = 8; break;
        case mx::complex64: element_size = 8; break;
        default:
            throw std::runtime_error("Unsupported dtype for loading");
    }
    
    size_t data_size = num_elements * element_size;
    std::vector<char> data(data_size);
    file.read(data.data(), data_size);
    
    if (!file) {
        throw std::runtime_error("Failed to read data from NPY file");
    }
    
    // Create MLX array from data
    // Note: NumPy uses row-major (C) order by default, MLX also uses row-major
    mx::Shape shape_mlx(header.shape.begin(), header.shape.end());
    
    // For Fortran-order arrays, data is stored column-major.
    // We load with reversed shape (which matches C-order memory layout)
    // then transpose back to the original logical shape.
    mx::Shape load_shape = shape_mlx;
    if (header.fortran_order && header.shape.size() > 1) {
        std::vector<int> reversed_shape(header.shape.rbegin(), header.shape.rend());
        load_shape = mx::Shape(reversed_shape.begin(), reversed_shape.end());
    }

    mx::array arr(0.0f);  // placeholder, will be reassigned
    if (mlx_dtype == mx::float32) {
        arr = mx::array(reinterpret_cast<float*>(data.data()), load_shape, mlx_dtype);
    } else if (mlx_dtype == mx::float64) {
        arr = mx::array(reinterpret_cast<double*>(data.data()), load_shape, mlx_dtype);
    } else if (mlx_dtype == mx::int32) {
        arr = mx::array(reinterpret_cast<int32_t*>(data.data()), load_shape, mlx_dtype);
    } else if (mlx_dtype == mx::int64) {
        arr = mx::array(reinterpret_cast<int64_t*>(data.data()), load_shape, mlx_dtype);
    } else if (mlx_dtype == mx::complex64) {
        arr = mx::array(reinterpret_cast<mx::complex64_t*>(data.data()), load_shape, mlx_dtype);
    } else {
        throw std::runtime_error("Unsupported dtype for array creation");
    }

    // Transpose back to original shape for Fortran-order arrays
    if (header.fortran_order && header.shape.size() > 1) {
        std::vector<int> axes(header.shape.size());
        for (size_t i = 0; i < axes.size(); ++i) {
            axes[i] = axes.size() - 1 - i;
        }
        arr = mx::transpose(arr, axes);
    }

    return arr;
}

void save_npy(const std::string& path, const mx::array& arr) {
    // TODO: Implement NPY saving if needed for debugging
    throw std::runtime_error("save_npy not yet implemented");
}

} // namespace demucs
