cmake_minimum_required(VERSION 3.24)

project(demucs-mlx CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

# Find or build MLX from third_party
if(NOT TARGET mlx)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/mlx)
endif()

# Add nlohmann/json for JSON parsing
include(FetchContent)
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.2
)
FetchContent_MakeAvailable(json)

# Add libnyquist for audio I/O (disable examples to avoid RtAudio dep)
set(LIBNYQUIST_BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    libnyquist
    GIT_REPOSITORY https://github.com/ddiakopoulos/libnyquist.git
    GIT_TAG master
)
FetchContent_MakeAvailable(libnyquist)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/mlx
)

# Source files - matches Python structure
set(SOURCES
    demucs/utils.cpp
    demucs/demucs.cpp
    demucs/hdemucs.cpp
    demucs/htdemucs.cpp
    demucs/spec.cpp
    demucs/transformer.cpp
    demucs/npy_loader.cpp
    demucs/safetensors.cpp
    demucs/weight_loader.cpp
    demucs/audio.cpp
    demucs/apply.cpp
)

# Create library
add_library(demucs_mlx_lib ${SOURCES})
target_link_libraries(demucs_mlx_lib PUBLIC mlx nlohmann_json::nlohmann_json libnyquist)

# AudioToolbox for MP3/AAC/FLAC loading on macOS
if(APPLE)
    find_library(AUDIO_TOOLBOX AudioToolbox REQUIRED)
    find_library(CORE_FOUNDATION CoreFoundation REQUIRED)
    target_link_libraries(demucs_mlx_lib PUBLIC ${AUDIO_TOOLBOX} ${CORE_FOUNDATION})
    target_compile_definitions(demucs_mlx_lib PUBLIC HAVE_AUDIO_TOOLBOX)
endif()

# Main executable - separate CLI (matches separate.py)
add_executable(demucs_separate demucs/separate.cpp)
target_link_libraries(demucs_separate demucs_mlx_lib)

# Testing
option(BUILD_TESTS "Build verification tests" ON)

if(BUILD_TESTS)
    enable_testing()
    
    # Fetch GTest
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    
    # Utils test (gelu, glu, conv wrappers, group_norm)
    add_executable(test_utils tests/test_utils.cpp)
    target_link_libraries(test_utils demucs_mlx_lib gtest gtest_main)
    add_test(NAME UtilsTest COMMAND test_utils)

    # Apply test (center_trim, prevent_clip, TensorChunk)
    add_executable(test_apply tests/test_apply.cpp)
    target_link_libraries(test_apply demucs_mlx_lib gtest gtest_main)
    add_test(NAME ApplyTest COMMAND test_apply)

    # Spec test (spectro, ispectro round-trip)
    add_executable(test_spec tests/test_spec.cpp)
    target_link_libraries(test_spec demucs_mlx_lib gtest gtest_main)
    add_test(NAME SpecTest COMMAND test_spec)

    # Audio I/O test (save/load WAV, M4A, FLAC)
    add_executable(test_audio tests/test_audio.cpp)
    target_link_libraries(test_audio demucs_mlx_lib gtest gtest_main)
    add_test(NAME AudioTest COMMAND test_audio)

    # Transformer test (embeddings, LayerScale, attention layers)
    add_executable(test_transformer_unit tests/test_transformer.cpp)
    target_link_libraries(test_transformer_unit demucs_mlx_lib gtest gtest_main)
    add_test(NAME TransformerTest COMMAND test_transformer_unit)
endif()
