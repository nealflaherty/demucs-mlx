# Requirements Document

## Introduction

Swift port of the existing demucs-mlx C++ implementation. The Swift version replicates the full HTDemucs music source separation pipeline using MLX Swift (ml-explore/mlx-swift) for GPU-accelerated inference on Apple Silicon. The port preserves numerical parity with the C++ (and Python reference) implementation while leveraging Swift Package Manager, AVFoundation for audio I/O, and a cleaner language-level type system. The repository is restructured so that C++ code moves into `cpp/`, the new Swift code lives in `swift/`, and shared assets (`tools/`, `models/`) remain at the top level.

## Glossary

- **Separator**: The Swift executable that loads a model, reads audio, runs inference, and writes separated stems
- **HTDemucs_Model**: The top-level Hybrid Transformer Demucs model comprising frequency-branch encoder/decoder, time-branch encoder/decoder, and a cross-transformer
- **Encoder_Layer**: A single encoder stage (HEncLayer) containing convolution, group normalization, optional rewrite convolution, GLU activation, and optional DConv residual branch
- **Decoder_Layer**: A single decoder stage (HDecLayer) containing transposed convolution, group normalization, optional rewrite convolution, GLU activation, and optional DConv residual branch
- **DConv_Block**: Dilated convolution residual branch with LayerScale, used inside encoder and decoder layers
- **Cross_Transformer**: The CrossTransformerEncoder that alternates self-attention and cross-attention layers between frequency and time branches
- **STFT_Module**: The Short-Time Fourier Transform implementation (spectro function) that converts time-domain audio to complex spectrograms
- **ISTFT_Module**: The inverse STFT implementation (ispectro function) that converts complex spectrograms back to time-domain audio via overlap-add with window normalization
- **Audio_IO**: The module responsible for loading audio files into MLX arrays and saving MLX arrays to audio files
- **Weight_Loader**: The module that reads SafeTensors files and maps weight tensors to model parameters
- **Apply_Engine**: The module that implements overlap-add chunked inference, random shift averaging, and stem extraction
- **Conv_Wrapper**: Utility functions that transpose between PyTorch tensor layout (N,C,L) / (N,C,H,W) and MLX layout (N,L,C) / (N,H,W,C) for conv1d, conv_transpose1d, conv2d, and conv_transpose2d
- **SwiftPM_Package**: The Swift Package Manager manifest (Package.swift) that declares targets, dependencies, and platform requirements

## Requirements

### Requirement 1: Repository Restructuring

**User Story:** As a developer, I want the repository restructured so that C++ and Swift implementations coexist cleanly, so that both can be built and tested independently.

#### Acceptance Criteria

1. WHEN the repository is restructured, THE Separator SHALL place all existing C++ source files under a `cpp/` subdirectory with a working CMakeLists.txt
2. WHEN the repository is restructured, THE Separator SHALL place all Swift source files under a `swift/` subdirectory with a Package.swift manifest
3. THE SwiftPM_Package SHALL declare mlx-swift (ml-explore/mlx-swift) as a package dependency
4. THE SwiftPM_Package SHALL declare macOS 14.0 as the minimum deployment target
5. THE SwiftPM_Package SHALL define a library target containing the model and utility code, and an executable target for the CLI

### Requirement 2: MLX Tensor Utilities

**User Story:** As a developer, I want Swift wrappers for convolution and normalization operations that handle PyTorch-to-MLX layout transposition, so that the model code reads naturally.

#### Acceptance Criteria

1. THE Conv_Wrapper SHALL provide a conv1d function that transposes input from (N,C,L) to (N,L,C), transposes weights from (O,I,K) to (O,K,I), calls MLX conv1d, adds bias, and transposes the result back to (N,C,L)
2. THE Conv_Wrapper SHALL provide a conv_transpose1d function that transposes input from (N,C,L) to (N,L,C), transposes weights from (C_in,C_out,K) to (C_out,K,C_in), calls MLX conv_transpose1d, adds bias, and transposes the result back to (N,C,L)
3. THE Conv_Wrapper SHALL provide a conv2d function that transposes input from (N,C,H,W) to (N,H,W,C), transposes weights from (O,I,kH,kW) to (O,kH,kW,I), calls MLX conv2d, adds bias, and transposes the result back to (N,C,H,W)
4. THE Conv_Wrapper SHALL provide a conv_transpose2d function that transposes input from (N,C,H,W) to (N,H,W,C), transposes weights from (C_in,C_out,kH,kW) to (C_out,kH,kW,C_in), calls MLX conv_transpose2d, adds bias, and transposes the result back to (N,C,H,W)
5. THE Conv_Wrapper SHALL provide a group_norm function that reshapes input into groups, computes per-group mean and variance, normalizes, and applies affine weight and bias parameters
6. THE Conv_Wrapper SHALL provide a gelu function using the tanh approximation matching PyTorch's GELU
7. THE Conv_Wrapper SHALL provide a glu function that splits the input tensor in half along a specified axis and returns a \* sigmoid(b)

### Requirement 3: STFT and Inverse STFT

**User Story:** As a developer, I want STFT and iSTFT implementations that produce numerically identical results to the C++ version, so that the model's spectrogram processing is correct.

#### Acceptance Criteria

1. WHEN a time-domain audio tensor is provided, THE STFT_Module SHALL apply reflect padding of n_fft/2 on each side, window with a periodic Hann window, compute the FFT per frame, and return a complex MLX array of shape (..., freq_bins, num_frames)
2. WHEN a complex spectrogram tensor is provided, THE ISTFT_Module SHALL compute the inverse FFT per frame, apply the Hann window, perform overlap-add with squared-window normalization, and return a time-domain tensor trimmed to the requested length
3. FOR ALL valid audio tensors, applying spectro then ispectro with the original length SHALL produce a tensor within 1e-4 absolute tolerance of the original (round-trip property)
4. THE STFT_Module SHALL use the same 1/sqrt(n_fft) normalization factor as the C++ implementation
5. THE ISTFT_Module SHALL use the same sqrt(n_fft) denormalization and window-squared normalization as the C++ implementation

### Requirement 4: DConv Residual Block

**User Story:** As a developer, I want the DConv dilated convolution residual block ported to Swift, so that encoder and decoder layers can use it.

#### Acceptance Criteria

1. THE DConv_Block SHALL contain a configurable number of layers, each consisting of a dilated conv1d, group norm, activation (GELU or ReLU), a 1x1 conv1d, group norm, GLU activation, and LayerScale
2. WHEN dilate mode is enabled, THE DConv_Block SHALL use dilation = 2^layer_index for each successive layer
3. THE DConv_Block SHALL add each layer's output as a residual to the input, preserving the input tensor shape

### Requirement 5: Encoder and Decoder Layers

**User Story:** As a developer, I want HEncLayer and HDecLayer ported to Swift, so that the frequency and time branches of HTDemucs can be constructed.

#### Acceptance Criteria

1. THE Encoder_Layer SHALL apply convolution (1D or 2D depending on freq mode), group normalization, optional rewrite convolution with GLU, optional DConv_Block, and return the encoded tensor
2. THE Decoder_Layer SHALL apply optional DConv_Block, optional rewrite convolution with GLU, group normalization, transposed convolution (1D or 2D depending on freq mode), and add the skip connection from the corresponding encoder layer
3. WHEN context is greater than zero, THE Encoder_Layer SHALL use a rewrite convolution with kernel size 1+2\*context
4. WHEN the decoder is the last layer, THE Decoder_Layer SHALL omit the GLU activation on the rewrite output

### Requirement 6: Transformer Layers

**User Story:** As a developer, I want the self-attention and cross-attention transformer layers ported to Swift, so that the cross-transformer can fuse frequency and time branch features.

#### Acceptance Criteria

1. THE Cross_Transformer SHALL generate sinusoidal positional embeddings (1D and 2D) matching the C++ create_sin_embedding and create_2d_sin_embedding functions
2. THE Cross_Transformer SHALL implement MyTransformerEncoderLayer with multi-head self-attention, feed-forward network, GroupNorm, and optional LayerScale
3. THE Cross_Transformer SHALL implement CrossTransformerEncoderLayer with multi-head cross-attention (query from one branch, key/value from the other), feed-forward network, GroupNorm, and optional LayerScale
4. THE Cross_Transformer SHALL alternate self-attention and cross-attention layers according to the cross_first configuration flag
5. WHEN norm_in is enabled, THE Cross_Transformer SHALL apply GroupNorm to both frequency and time inputs before the transformer layers

### Requirement 7: HTDemucs Model

**User Story:** As a developer, I want the full HTDemucs model assembled in Swift, so that I can run end-to-end source separation inference.

#### Acceptance Criteria

1. THE HTDemucs_Model SHALL construct frequency-branch encoder and decoder layer stacks, time-branch encoder and decoder layer stacks, a ScaledEmbedding for frequency position, and a CrossTransformerEncoder
2. WHEN a stereo audio mix is provided, THE HTDemucs_Model SHALL compute the STFT, run the frequency encoder, run the time encoder, apply channel upsamplers, run the cross-transformer, apply channel downsamplers, run the frequency decoder, run the time decoder, compute the iSTFT, and return separated source tensors
3. THE HTDemucs_Model SHALL produce output with shape (batch, sources, channels, length) where sources defaults to 4 (drums, bass, other, vocals)
4. THE HTDemucs_Model SHALL use the same default hyperparameters as the C++ implementation (channels=48, depth=4, nfft=4096, kernel_size=8, stride=4, t_layers=5, t_heads=8, bottom_channels=512, segment=7.8)

### Requirement 8: SafeTensors Weight Loading

**User Story:** As a developer, I want to load pretrained HTDemucs weights from SafeTensors files, so that the Swift model produces correct separation results.

#### Acceptance Criteria

1. WHEN a SafeTensors file path is provided, THE Weight_Loader SHALL parse the file header to extract tensor names, shapes, dtypes, and data offsets
2. WHEN a tensor name is requested, THE Weight_Loader SHALL read the raw bytes at the correct offset and construct an MLX array with the correct shape and dtype
3. THE Weight_Loader SHALL map SafeTensors key names to HTDemucs model parameters using the same naming convention as the C++ WeightLoader
4. IF a required weight tensor is missing from the SafeTensors file, THEN THE Weight_Loader SHALL report an error identifying the missing tensor name

### Requirement 9: Audio I/O

**User Story:** As a developer, I want to load and save audio files in common formats, so that the CLI can process real music files.

#### Acceptance Criteria

1. WHEN an audio file path is provided, THE Audio_IO SHALL load the file, convert to float32, resample to the target sample rate, convert to stereo, and return an MLX array of shape (2, samples)
2. THE Audio_IO SHALL support loading WAV, MP3, AAC, FLAC, M4A, and ALAC formats using AVFoundation (AVAudioFile or ExtAudioFile)
3. WHEN saving audio, THE Audio_IO SHALL support WAV output at 16-bit, 24-bit, and float32 PCM
4. WHEN saving audio, THE Audio_IO SHALL support M4A (AAC), FLAC, and ALAC output formats using AVFoundation
5. IF an audio file cannot be loaded or decoded, THEN THE Audio_IO SHALL return nil and log a descriptive error message

### Requirement 10: Overlap-Add Chunked Inference

**User Story:** As a developer, I want overlap-add chunked inference with random shift averaging, so that arbitrary-length audio can be separated within bounded memory.

#### Acceptance Criteria

1. WHEN split mode is enabled, THE Apply_Engine SHALL divide the input audio into overlapping chunks based on the model segment length and overlap ratio
2. THE Apply_Engine SHALL apply a transition weight (raised to transition_power) to blend overlapping chunk outputs
3. WHEN shifts is greater than 1, THE Apply_Engine SHALL apply random circular shifts to the input, run inference on each shifted version, reverse the shifts on the outputs, and average the results
4. THE Apply_Engine SHALL center-trim the model output to match the input length along the last dimension
5. THE Apply_Engine SHALL implement prevent_clip with rescale mode (scale down if max absolute value exceeds 1.0) and clamp mode

### Requirement 11: CLI Executable

**User Story:** As a developer, I want a command-line tool that separates audio files into stems, so that I can use the Swift port as a drop-in replacement for the C++ version.

#### Acceptance Criteria

1. THE Separator SHALL accept one or more input audio file paths as positional arguments
2. THE Separator SHALL support the same CLI flags as the C++ version: --out, --model, --filename, --shifts, --overlap, --no-split, --segment, --clip-mode, --two-stems, --other-method, --wav, --m4a, --flac, --alac, --int24, --float32, --bitrate
3. WHEN --two-stems is specified with a valid stem name, THE Separator SHALL output the selected stem and its complement (computed via the --other-method strategy)
4. WHEN separation completes, THE Separator SHALL save each stem to the output directory using the filename pattern with {track}, {trackext}, {stem}, and {ext} placeholders
5. IF the model weights file does not exist at the specified path, THEN THE Separator SHALL print an error message and exit with a non-zero status code

### Requirement 12: Numerical Parity

**User Story:** As a developer, I want the Swift port to produce numerically identical results to the C++ implementation, so that I can trust the separation quality.

#### Acceptance Criteria

1. WHEN the same audio input and model weights are provided, THE HTDemucs_Model in Swift SHALL produce per-sample output within 1e-3 absolute tolerance of the C++ HTDemucs output
2. FOR ALL valid audio tensors, THE STFT_Module round-trip (spectro then ispectro) SHALL produce output within 1e-4 absolute tolerance of the input
3. WHEN the same weights are loaded, THE Conv_Wrapper conv1d output SHALL match the C++ conv1d output within 1e-5 absolute tolerance for identical inputs
4. WHEN the same weights are loaded, THE Cross_Transformer self-attention output SHALL match the C++ MyTransformerEncoderLayer output within 1e-4 absolute tolerance for identical inputs
