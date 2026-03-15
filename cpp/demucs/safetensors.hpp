#pragma once

#include <mlx/mlx.h>
#include <string>
#include <unordered_map>
#include <optional>
#include <vector>

namespace demucs {

namespace mx = mlx::core;

// Weight metadata from SafeTensors header
struct WeightMetadata {
    std::string name;
    std::vector<int> shape;
    std::string dtype;
    size_t data_offset_start;
    size_t data_offset_end;
    
    size_t size_bytes() const {
        return data_offset_end - data_offset_start;
    }
};

// SafeTensors file structure
struct SafeTensorsFile {
    std::unordered_map<std::string, WeightMetadata> tensors;
    std::unordered_map<std::string, std::string> metadata;
    size_t header_size;
    std::string file_path;
};

// SafeTensors loader
class SafeTensorsLoader {
public:
    // Parse SafeTensors file header
    static std::optional<SafeTensorsFile> parse(const std::string& path);
    
    // Load a specific tensor by name
    static std::optional<mx::array> load_tensor(
        const SafeTensorsFile& file,
        const std::string& tensor_name
    );
    
    // Load all tensors into a map
    static std::unordered_map<std::string, mx::array> load_all(
        const SafeTensorsFile& file
    );
    
    // Print file metadata for debugging
    static void print_info(const SafeTensorsFile& file);

private:
    // Convert dtype string to MLX dtype
    static mx::Dtype string_to_dtype(const std::string& dtype_str);
    
    // Read raw tensor data from file
    static std::optional<std::vector<char>> read_tensor_data(
        const std::string& path,
        const WeightMetadata& metadata,
        size_t header_size
    );
};

} // namespace demucs
