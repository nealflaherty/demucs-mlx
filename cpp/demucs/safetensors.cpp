#include "safetensors.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <cstring>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace demucs {

std::optional<SafeTensorsFile> SafeTensorsLoader::parse(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << path << std::endl;
        return std::nullopt;
    }
    
    // Read header size (first 8 bytes, little-endian uint64)
    uint64_t header_size = 0;
    file.read(reinterpret_cast<char*>(&header_size), sizeof(header_size));
    
    if (!file.good() || header_size == 0 || header_size > 100 * 1024 * 1024) {
        std::cerr << "Invalid header size: " << header_size << std::endl;
        return std::nullopt;
    }
    
    // Read JSON header
    std::vector<char> header_data(header_size);
    file.read(header_data.data(), header_size);
    
    if (!file.good()) {
        std::cerr << "Failed to read header data" << std::endl;
        return std::nullopt;
    }
    
    std::string header_json(header_data.begin(), header_data.end());
    
    SafeTensorsFile result;
    result.header_size = header_size;
    result.file_path = path;
    
    // Parse JSON using nlohmann/json
    try {
        json j = json::parse(header_json);
        
        // Iterate through all keys in the JSON
        for (auto& [key, value] : j.items()) {
            // Skip metadata entry
            if (key == "__metadata__") {
                continue;
            }
            
            // Parse tensor metadata
            WeightMetadata metadata;
            metadata.name = key;
            
            if (value.contains("dtype")) {
                metadata.dtype = value["dtype"].get<std::string>();
            }
            
            if (value.contains("shape")) {
                metadata.shape = value["shape"].get<std::vector<int>>();
            }
            
            if (value.contains("data_offsets")) {
                auto offsets = value["data_offsets"].get<std::vector<uint64_t>>();
                if (offsets.size() >= 2) {
                    metadata.data_offset_start = offsets[0];
                    metadata.data_offset_end = offsets[1];
                }
            }
            
            result.tensors[key] = metadata;
        }
    } catch (const json::exception& e) {
        std::cerr << "JSON parsing error: " << e.what() << std::endl;
        return std::nullopt;
    }
    
    return result;
}

mx::Dtype SafeTensorsLoader::string_to_dtype(const std::string& dtype_str) {
    if (dtype_str == "F32" || dtype_str == "float32") return mx::float32;
    if (dtype_str == "F16" || dtype_str == "float16") return mx::float16;
    if (dtype_str == "I32" || dtype_str == "int32") return mx::int32;
    if (dtype_str == "I64" || dtype_str == "int64") return mx::int64;
    if (dtype_str == "BF16" || dtype_str == "bfloat16") return mx::bfloat16;
    
    std::cerr << "Unknown dtype: " << dtype_str << ", defaulting to float32" << std::endl;
    return mx::float32;
}

std::optional<std::vector<char>> SafeTensorsLoader::read_tensor_data(
    const std::string& path,
    const WeightMetadata& metadata,
    size_t header_size
) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return std::nullopt;
    }
    
    // Skip to data location (8 bytes for header size + header + offset)
    size_t file_offset = 8 + header_size + metadata.data_offset_start;
    file.seekg(file_offset);
    
    size_t data_size = metadata.size_bytes();
    std::vector<char> data(data_size);
    file.read(data.data(), data_size);
    
    if (!file.good()) {
        return std::nullopt;
    }
    
    return data;
}

std::optional<mx::array> SafeTensorsLoader::load_tensor(
    const SafeTensorsFile& file,
    const std::string& tensor_name
) {
    auto it = file.tensors.find(tensor_name);
    if (it == file.tensors.end()) {
        std::cerr << "Tensor not found: " << tensor_name << std::endl;
        return std::nullopt;
    }
    
    const auto& metadata = it->second;
    
    auto data = read_tensor_data(file.file_path, metadata, file.header_size);
    if (!data) {
        std::cerr << "Failed to read tensor data: " << tensor_name << std::endl;
        return std::nullopt;
    }
    
    // Convert shape to MLX shape
    std::vector<int> shape_vec = metadata.shape;
    mx::Shape shape(shape_vec.begin(), shape_vec.end());
    
    // Create array from raw data
    mx::Dtype dtype = string_to_dtype(metadata.dtype);
    
    // CRITICAL: Cast the char* pointer to the correct type based on dtype
    // MLX array constructor interprets the pointer type, not just the dtype parameter
    if (dtype == mx::float32) {
        return mx::array(reinterpret_cast<float*>(data->data()), shape, dtype);
    } else if (dtype == mx::float16) {
        return mx::array(reinterpret_cast<uint16_t*>(data->data()), shape, dtype);
    } else if (dtype == mx::bfloat16) {
        return mx::array(reinterpret_cast<uint16_t*>(data->data()), shape, dtype);
    } else if (dtype == mx::int32) {
        return mx::array(reinterpret_cast<int32_t*>(data->data()), shape, dtype);
    } else if (dtype == mx::int64) {
        return mx::array(reinterpret_cast<int64_t*>(data->data()), shape, dtype);
    } else {
        std::cerr << "Unsupported dtype for tensor: " << tensor_name << std::endl;
        return std::nullopt;
    }
}

std::unordered_map<std::string, mx::array> SafeTensorsLoader::load_all(
    const SafeTensorsFile& file
) {
    std::unordered_map<std::string, mx::array> result;
    
    for (const auto& [name, metadata] : file.tensors) {
        auto tensor = load_tensor(file, name);
        if (tensor) {
            result.insert({name, *tensor});
        }
    }
    
    return result;
}

void SafeTensorsLoader::print_info(const SafeTensorsFile& file) {
    std::cout << "SafeTensors file: " << file.file_path << std::endl;
    std::cout << "Header size: " << file.header_size << " bytes" << std::endl;
    std::cout << "Number of tensors: " << file.tensors.size() << std::endl;
    std::cout << "\nTensors:" << std::endl;
    
    for (const auto& [name, metadata] : file.tensors) {
        std::cout << "  " << name << ": ";
        std::cout << metadata.dtype << " [";
        for (size_t i = 0; i < metadata.shape.size(); ++i) {
            std::cout << metadata.shape[i];
            if (i < metadata.shape.size() - 1) std::cout << ", ";
        }
        std::cout << "] (" << metadata.size_bytes() << " bytes)" << std::endl;
    }
}

} // namespace demucs
