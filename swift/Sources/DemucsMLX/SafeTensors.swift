// SafeTensors.swift - SafeTensors file parser and tensor loader
// 1:1 port of cpp/demucs/safetensors.cpp

import MLX
import Foundation

// MARK: - Weight metadata

/// Metadata for a single tensor in a SafeTensors file.
public struct WeightMetadata {
    public let name: String
    public let shape: [Int]
    public let dtype: String
    public let dataOffsetStart: Int
    public let dataOffsetEnd: Int

    public var sizeBytes: Int { dataOffsetEnd - dataOffsetStart }
}

// MARK: - SafeTensors file

/// Parsed SafeTensors file header.
public struct SafeTensorsFile {
    public let tensors: [String: WeightMetadata]
    public let headerSize: Int
    public let filePath: String
}

// MARK: - SafeTensors loader

public enum SafeTensorsLoader {

    /// Parse a SafeTensors file header. Returns nil on error.
    public static func parse(_ path: String) -> SafeTensorsFile? {
        guard let fileHandle = FileHandle(forReadingAtPath: path) else {
            print("SafeTensors: failed to open \(path)")
            return nil
        }
        defer { fileHandle.closeFile() }

        // Read 8-byte little-endian header size
        let sizeData = fileHandle.readData(ofLength: 8)
        guard sizeData.count == 8 else { return nil }
        let headerSize = sizeData.withUnsafeBytes { $0.load(as: UInt64.self) }
        guard headerSize > 0, headerSize < 100 * 1024 * 1024 else {
            print("SafeTensors: invalid header size \(headerSize)")
            return nil
        }

        // Read JSON header
        let headerData = fileHandle.readData(ofLength: Int(headerSize))
        guard headerData.count == Int(headerSize),
              let headerJSON = String(data: headerData, encoding: .utf8) else {
            return nil
        }

        // Parse JSON
        guard let jsonData = headerJSON.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: jsonData) as? [String: Any] else {
            print("SafeTensors: JSON parse error")
            return nil
        }

        var tensors = [String: WeightMetadata]()
        for (key, value) in json {
            if key == "__metadata__" { continue }
            guard let dict = value as? [String: Any] else { continue }

            let dtype = dict["dtype"] as? String ?? "F32"
            let shape = (dict["shape"] as? [Int]) ?? []
            var startOffset = 0, endOffset = 0
            if let offsets = dict["data_offsets"] as? [Int], offsets.count >= 2 {
                startOffset = offsets[0]
                endOffset = offsets[1]
            }

            tensors[key] = WeightMetadata(
                name: key, shape: shape, dtype: dtype,
                dataOffsetStart: startOffset, dataOffsetEnd: endOffset
            )
        }

        return SafeTensorsFile(tensors: tensors, headerSize: Int(headerSize), filePath: path)
    }

    /// Load a specific tensor by name.
    public static func loadTensor(_ file: SafeTensorsFile, name: String) -> MLXArray? {
        guard let meta = file.tensors[name] else {
            print("SafeTensors: tensor not found: \(name)")
            return nil
        }

        guard let data = readTensorData(file.filePath, metadata: meta,
                                         headerSize: file.headerSize) else {
            print("SafeTensors: failed to read data for \(name)")
            return nil
        }

        let dtype = stringToDtype(meta.dtype)
        let shape = meta.shape

        // Create MLXArray from raw bytes
        let result: MLXArray = data.withUnsafeBytes { rawBuffer in
            let ptr = rawBuffer.baseAddress!
            switch dtype {
            case .float32:
                let buf = UnsafeBufferPointer(start: ptr.assumingMemoryBound(to: Float.self),
                                              count: data.count / MemoryLayout<Float>.size)
                return MLXArray(buf, shape)
            case .float16:
                let buf = UnsafeBufferPointer(start: ptr.assumingMemoryBound(to: Float16.self),
                                              count: data.count / MemoryLayout<Float16>.size)
                return MLXArray(buf, shape)
            case .int32:
                let buf = UnsafeBufferPointer(start: ptr.assumingMemoryBound(to: Int32.self),
                                              count: data.count / MemoryLayout<Int32>.size)
                return MLXArray(buf, shape)
            default:
                let buf = UnsafeBufferPointer(start: ptr.assumingMemoryBound(to: Float.self),
                                              count: data.count / MemoryLayout<Float>.size)
                return MLXArray(buf, shape)
            }
        }
        return result
    }

    /// Load all tensors from a SafeTensors file.
    public static func loadAll(_ file: SafeTensorsFile) -> [String: MLXArray] {
        var result = [String: MLXArray]()
        for (name, _) in file.tensors {
            if let tensor = loadTensor(file, name: name) {
                result[name] = tensor
            }
        }
        return result
    }

    // MARK: - Private helpers

    private static func stringToDtype(_ s: String) -> DType {
        switch s {
        case "F32", "float32": return .float32
        case "F16", "float16": return .float16
        case "BF16", "bfloat16": return .bfloat16
        case "I32", "int32": return .int32
        case "I64", "int64": return .int64
        default:
            print("SafeTensors: unknown dtype '\(s)', defaulting to float32")
            return .float32
        }
    }

    private static func readTensorData(_ path: String, metadata: WeightMetadata,
                                        headerSize: Int) -> Data? {
        guard let fh = FileHandle(forReadingAtPath: path) else { return nil }
        defer { fh.closeFile() }

        let fileOffset = UInt64(8 + headerSize + metadata.dataOffsetStart)
        fh.seek(toFileOffset: fileOffset)
        let data = fh.readData(ofLength: metadata.sizeBytes)
        return data.count == metadata.sizeBytes ? data : nil
    }

    /// Print file info for debugging.
    public static func printInfo(_ file: SafeTensorsFile) {
        print("SafeTensors file: \(file.filePath)")
        print("Header size: \(file.headerSize) bytes")
        print("Number of tensors: \(file.tensors.count)")
        for (name, meta) in file.tensors.sorted(by: { $0.key < $1.key }) {
            let shapeStr = meta.shape.map(String.init).joined(separator: ", ")
            print("  \(name): \(meta.dtype) [\(shapeStr)] (\(meta.sizeBytes) bytes)")
        }
    }
}
