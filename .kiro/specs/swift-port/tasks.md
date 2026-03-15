# Implementation Plan: Swift Port of demucs-mlx

## Overview

Port the C++ demucs-mlx HTDemucs implementation to Swift using MLX Swift, following a bottom-up approach: restructure the repo, build foundational utilities, then layer model components, and finally wire everything together with the CLI. Each task maps directly to a C++ source file and its Swift counterpart.

## Tasks

- [x] 1. Repository restructuring and Swift package setup
  - [x] 1.1 Move existing C++ source files into `cpp/` subdirectory
    - Move all files from `demucs/` to `cpp/demucs/`
    - Move `CMakeLists.txt` into `cpp/` and update paths so the C++ build still works
    - _Requirements: 1.1_

  - [x] 1.2 Create Swift package structure
    - Create `swift/Package.swift` with macOS 14.0 minimum, mlx-swift and swift-argument-parser dependencies
    - Create `swift/Sources/DemucsMLX/` and `swift/Sources/Separator/` directories
    - Create `swift/Tests/DemucsMLXTests/` directory
    - Create placeholder files so `swift build` succeeds (empty lib file + minimal main.swift)
    - _Requirements: 1.2, 1.3, 1.4, 1.5_

- [x] 2. Checkpoint - Verify repo structure
  - Ensure C++ build still works from `cpp/`, ensure `swift build` succeeds from `swift/`, ask the user if questions arise.

- [x] 3. MLX tensor utilities (`Utils.swift`)
  - [x] 3.1 Implement conv1d and convTranspose1d wrappers
    - Port from `utils.cpp`: transpose input (N,C,L)→(N,L,C), transpose weights, call MLX, add bias, transpose back
    - Reference C++ `conv1d()` and `conv_transpose1d()` functions
    - _Requirements: 2.1, 2.2_

  - [x] 3.2 Implement conv2d and convTranspose2d wrappers
    - Port from `utils.cpp`: transpose input (N,C,H,W)→(N,H,W,C), transpose weights, call MLX, add bias, transpose back
    - Reference C++ `conv2d()` and `conv_transpose2d()` functions
    - _Requirements: 2.3, 2.4_

  - [x] 3.3 Implement groupNorm, gelu, and glu
    - `groupNorm`: reshape into groups, compute per-group mean/variance, normalize, apply affine
    - `gelu`: tanh approximation `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`
    - `glu`: split along axis, return `a * sigmoid(b)`
    - Reference C++ `group_norm()`, `gelu()`, `glu()` in `utils.cpp`
    - _Requirements: 2.5, 2.6, 2.7_

  - [ ]\* 3.4 Write property tests for conv wrapper layout correctness
    - **Property 2: Conv Wrapper Layout Correctness**
    - **Validates: Requirements 2.1, 2.2, 2.3, 2.4**

  - [ ]\* 3.5 Write property tests for groupNorm zero-mean unit-variance
    - **Property 3: GroupNorm Zero-Mean Unit-Variance**
    - **Validates: Requirements 2.5**

  - [ ]\* 3.6 Write property tests for activation functions
    - **Property 4: Activation Functions Match Definitions**
    - **Validates: Requirements 2.6, 2.7**

- [x] 4. STFT and inverse STFT (`Spec.swift`)
  - [x] 4.1 Implement spectro function
    - Reflect padding of nFFT/2 on each side
    - Periodic Hann window generation
    - FFT via Accelerate vDSP, 1/sqrt(nFFT) normalization
    - Return complex MLXArray of shape (..., freqBins, numFrames)
    - Reference C++ `spectro()` in `spec.cpp`
    - _Requirements: 3.1, 3.4_

  - [x] 4.2 Implement ispectro function
    - Inverse FFT per frame, Hann window application
    - Overlap-add with squared-window normalization, sqrt(nFFT) denormalization
    - Trim to requested length
    - Reference C++ `ispectro()` in `spec.cpp`
    - _Requirements: 3.2, 3.5_

  - [ ]\* 4.3 Write property test for STFT round-trip reconstruction
    - **Property 1: STFT Round-Trip Reconstruction**
    - **Validates: Requirements 3.3, 12.2**

- [ ] 5. Checkpoint - Verify foundational utilities
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. DConv residual block (`DConv.swift`)
  - [x] 6.1 Implement DConvLayer and DConvBlock structs
    - Each layer: dilated conv1d → GroupNorm → GELU/ReLU → 1×1 conv1d → GroupNorm → GLU → LayerScale → residual add
    - Dilation = 2^layerIndex when dilate mode is on
    - Reference C++ `DConv` in `hdemucs.cpp`
    - _Requirements: 4.1, 4.2, 4.3_

  - [ ]\* 6.2 Write property test for DConv shape preservation
    - **Property 5: DConv Shape Preservation**
    - **Validates: Requirements 4.3**

- [x] 7. Encoder and decoder layers (`HDemucs.swift`)
  - [x] 7.1 Implement HEncLayer struct
    - Conv (1D or 2D depending on freq mode), GroupNorm, optional rewrite conv with GLU, optional DConvBlock
    - Context > 0 uses rewrite conv with kernel size 1+2\*context
    - Implement pad1d utility and ScaledEmbedding struct
    - Reference C++ `HEncLayer` in `hdemucs.cpp`
    - _Requirements: 5.1, 5.3_

  - [x] 7.2 Implement HDecLayer struct
    - Optional DConvBlock, optional rewrite conv with GLU, GroupNorm, transposed conv, skip connection
    - Last layer omits GLU on rewrite output
    - Reference C++ `HDecLayer` in `hdemucs.cpp`
    - _Requirements: 5.2, 5.4_

- [x] 8. Transformer layers (`Transformer.swift`)
  - [x] 8.1 Implement sinusoidal embedding functions
    - `createSinEmbedding(length:dim:shift:maxPeriod:)` for 1D
    - `create2DSinEmbedding(dModel:height:width:maxPeriod:)` for 2D
    - Reference C++ `create_sin_embedding` and `create_2d_sin_embedding` in `transformer.cpp`
    - _Requirements: 6.1_

  - [ ]\* 8.2 Write property test for sinusoidal embedding bounds and shape
    - **Property 6: Sinusoidal Embedding Bounds and Shape**
    - **Validates: Requirements 6.1**

  - [x] 8.3 Implement LayerScale, MyGroupNorm, MyTransformerEncoderLayer
    - Self-attention with combined QKV in_proj, FFN, GroupNorm, LayerScale
    - Reference C++ `MyTransformerEncoderLayer` in `transformer.cpp`
    - _Requirements: 6.2_

  - [x] 8.4 Implement CrossTransformerEncoderLayer and CrossTransformerEncoder
    - Cross-attention with separate Q, K, V projections
    - Alternating self-attention and cross-attention per cross_first flag
    - Optional norm_in GroupNorm on both inputs
    - Reference C++ `CrossTransformerEncoderLayer` and `CrossTransformerEncoder` in `transformer.cpp`
    - _Requirements: 6.3, 6.4, 6.5_

- [ ] 9. Checkpoint - Verify model building blocks
  - Ensure all tests pass, ask the user if questions arise.

- [x] 10. SafeTensors loader (`SafeTensors.swift`)
  - [x] 10.1 Implement SafeTensorsLoader
    - Parse 8-byte header length, JSON header with tensor metadata
    - Load individual tensors by name: read raw bytes, construct MLXArray with correct shape/dtype
    - Reference C++ `safetensors.cpp`
    - _Requirements: 8.1, 8.2_

  - [ ]\* 10.2 Write property test for loaded tensor shape and dtype
    - **Property 8: SafeTensors Loaded Tensor Shape and Dtype**
    - **Validates: Requirements 8.2**

- [x] 11. Weight loader (`WeightLoader.swift`)
  - [x] 11.1 Implement WeightLoader mapping
    - Map SafeTensors key names to HTDemucsModel struct fields
    - Load encoder layers, decoder layers, DConv blocks, transformer layers, channel up/downsamplers, frequency embedding
    - Report error with missing tensor name if a required weight is absent
    - Reference C++ `weight_loader.cpp`
    - _Requirements: 8.3, 8.4_

- [x] 12. HTDemucs model (`HTDemucs.swift`)
  - [x] 12.1 Implement HTDemucsModel struct and forward pass
    - Assemble frequency encoder/decoder stacks, time encoder/decoder stacks, ScaledEmbedding, CrossTransformerEncoder, channel up/downsamplers
    - Forward: STFT → freq encoder → time encoder → channel upsample → cross-transformer → channel downsample → freq decoder → time decoder → iSTFT → combine → output (1, S, 2, L')
    - Default hyperparameters: channels=48, depth=4, nfft=4096, kernel_size=8, stride=4, t_layers=5, t_heads=8, bottom_channels=512, segment=7.8
    - Reference C++ `htdemucs.cpp`
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

  - [ ]\* 12.2 Write property test for HTDemucs output shape invariant
    - **Property 7: HTDemucs Output Shape Invariant**
    - **Validates: Requirements 7.3**

- [ ] 13. Checkpoint - Verify model loads and runs forward pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 14. Audio I/O (`Audio.swift`)
  - [x] 14.1 Implement AudioIO.load
    - Load via AVAudioFile / ExtAudioFile, convert to float32, resample to target sample rate, convert to stereo
    - Support WAV, MP3, AAC, FLAC, M4A, ALAC
    - Return MLXArray of shape (2, samples) or nil on error
    - Reference C++ `audio.cpp`
    - _Requirements: 9.1, 9.2, 9.5_

  - [x] 14.2 Implement AudioIO.save
    - WAV output at 16-bit, 24-bit, float32 PCM
    - M4A (AAC), FLAC, ALAC output via AVFoundation
    - Reference C++ `audio.cpp`
    - _Requirements: 9.3, 9.4_

  - [ ]\* 14.3 Write property test for audio load produces stereo float32
    - **Property 9: Audio Load Produces Stereo Float32**
    - **Validates: Requirements 9.1**

  - [ ]\* 14.4 Write property test for WAV save/load round-trip
    - **Property 10: WAV Save/Load Round-Trip**
    - **Validates: Requirements 9.3**

- [x] 15. Apply engine (`Apply.swift`)
  - [x] 15.1 Implement TensorChunk, centerTrim, preventClip
    - TensorChunk with offset/length and padded() method
    - centerTrim: trim last dimension to reference length
    - preventClip: rescale mode (scale down if max abs > 1.0), clamp mode (clamp to [-0.99, 0.99])
    - Reference C++ `apply.cpp`
    - _Requirements: 10.4, 10.5_

  - [ ]\* 15.2 Write property test for centerTrim output length
    - **Property 11: centerTrim Output Length**
    - **Validates: Requirements 10.4**

  - [ ]\* 15.3 Write property test for preventClip output bounds
    - **Property 12: preventClip Output Bounds**
    - **Validates: Requirements 10.5**

  - [x] 15.4 Implement applyModel with overlap-add chunked inference
    - Split input into overlapping chunks based on segment length and overlap ratio
    - Transition weight blending (raised to transition_power)
    - Random shift averaging: circular shift → inference → reverse shift → average
    - Reference C++ `apply.cpp`
    - _Requirements: 10.1, 10.2, 10.3_

- [ ] 16. Checkpoint - Verify full inference pipeline
  - Ensure all tests pass, ask the user if questions arise.

- [x] 17. CLI executable (`Separator/Separator.swift`)
  - [x] 17.1 Implement CLI argument parsing and main flow
    - Use swift-argument-parser for all flags: --out, --model, --filename, --shifts, --overlap, --no-split, --segment, --clip-mode, --two-stems, --other-method, --wav, --m4a, --flac, --alac, --int24, --float32, --bitrate
    - Filename pattern with {track}, {trackext}, {stem}, {ext} placeholders
    - Two-stems mode: output selected stem + complement via --other-method
    - Error handling: missing model file → print error, exit(1)
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5_

  - [ ]\* 17.2 Write property test for filename pattern formatting
    - **Property 13: Filename Pattern Formatting**
    - **Validates: Requirements 11.4**

- [ ] 18. Final checkpoint - Full integration verification
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Property tests use SwiftCheck and reference design document property numbers
- The C++ source files in `cpp/demucs/` serve as the 1:1 reference for each Swift file
- All Swift code uses PyTorch tensor layout convention (channels-first); conv wrappers handle MLX transposition internally
