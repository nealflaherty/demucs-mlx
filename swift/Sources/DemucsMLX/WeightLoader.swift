// WeightLoader.swift - Load SafeTensors weights into HTDemucsModel
// 1:1 port of cpp/demucs/weight_loader.cpp

import MLX

public enum WeightLoader {

    /// Load weights from a SafeTensors file into an HTDemucsModel.
    public static func loadHTDemucs(
        _ model: inout HTDemucsModel, from path: String
    ) -> Bool {
        guard let file = SafeTensorsLoader.parse(path) else {
            print("WeightLoader: failed to parse \(path)")
            return false
        }
        let weights = SafeTensorsLoader.loadAll(file)
        print("WeightLoader: loaded \(weights.count) tensors")
        return loadHTDemucsFromMap(&model, weights: weights)
    }

    // MARK: - Map-based loading

    public static func loadHTDemucsFromMap(
        _ model: inout HTDemucsModel,
        weights: [String: MLXArray]
    ) -> Bool {
        // Encoder layers
        for i in 0..<model.encoder.count {
            let p = "encoder.\(i)."
            if !loadEncoderLayer(&model.encoder[i], weights: weights, prefix: p) { return false }
        }
        // Decoder layers
        for i in 0..<model.decoder.count {
            let p = "decoder.\(i)."
            if !loadDecoderLayer(&model.decoder[i], weights: weights, prefix: p) { return false }
        }
        // Time encoder
        for i in 0..<model.tencoder.count {
            let p = "tencoder.\(i)."
            if !loadEncoderLayer(&model.tencoder[i], weights: weights, prefix: p) { return false }
        }
        // Time decoder
        for i in 0..<model.tdecoder.count {
            let p = "tdecoder.\(i)."
            if !loadDecoderLayer(&model.tdecoder[i], weights: weights, prefix: p) { return false }
        }

        // Frequency embedding
        if model.freqEmb != nil {
            if let w = weights["freq_emb.embedding.weight"] {
                model.freqEmb!.embeddingWeight = w
            }
        }

        // Cross-transformer
        if model.crossTransformer != nil {
            if !loadCrossTransformer(&model.crossTransformer!, weights: weights,
                                      prefix: "crosstransformer.") { return false }
        }

        // Channel up/downsamplers
        if let w = weights["channel_upsampler.weight"] { model.channelUpsamplerWeight = w }
        if let b = weights["channel_upsampler.bias"] { model.channelUpsamplerBias = b }
        if let w = weights["channel_upsampler_t.weight"] { model.channelUpsamplerTWeight = w }
        if let b = weights["channel_upsampler_t.bias"] { model.channelUpsamplerTBias = b }
        if let w = weights["channel_downsampler.weight"] { model.channelDownsamplerWeight = w }
        if let b = weights["channel_downsampler.bias"] { model.channelDownsamplerBias = b }
        if let w = weights["channel_downsampler_t.weight"] { model.channelDownsamplerTWeight = w }
        if let b = weights["channel_downsampler_t.bias"] { model.channelDownsamplerTBias = b }

        return true
    }

    // MARK: - DConv

    private static func loadDConv(
        _ dconv: inout DConvBlock, weights: [String: MLXArray], prefix: String
    ) -> Bool {
        for i in 0..<dconv.layers.count {
            let lp = "\(prefix)dconv.layers.\(i)."
            guard let c1w = weights[lp + "0.weight"],
                  let c1b = weights[lp + "0.bias"] else { return false }
            dconv.layers[i].conv1Weight = c1w
            dconv.layers[i].conv1Bias = c1b
            if let w = weights[lp + "1.weight"] { dconv.layers[i].norm1Weight = w }
            if let b = weights[lp + "1.bias"] { dconv.layers[i].norm1Bias = b }
            guard let c2w = weights[lp + "3.weight"],
                  let c2b = weights[lp + "3.bias"] else { return false }
            dconv.layers[i].conv2Weight = c2w
            dconv.layers[i].conv2Bias = c2b
            if let w = weights[lp + "4.weight"] { dconv.layers[i].norm2Weight = w }
            if let b = weights[lp + "4.bias"] { dconv.layers[i].norm2Bias = b }
            guard let s = weights[lp + "6.scale"] else { return false }
            dconv.layers[i].layerScale.scale = s
        }
        return true
    }

    // MARK: - Encoder layer

    private static func loadEncoderLayer(
        _ layer: inout HEncLayer, weights: [String: MLXArray], prefix: String
    ) -> Bool {
        guard let cw = weights[prefix + "conv.weight"],
              let cb = weights[prefix + "conv.bias"] else {
            print("WeightLoader: missing conv weights for \(prefix)")
            return false
        }
        layer.convWeight = cw
        layer.convBias = cb
        if let w = weights[prefix + "norm1.weight"] { layer.norm1Weight = w }
        if let b = weights[prefix + "norm1.bias"] { layer.norm1Bias = b }
        if let w = weights[prefix + "rewrite.weight"] { layer.rewriteWeight = w }
        if let b = weights[prefix + "rewrite.bias"] { layer.rewriteBias = b }
        if let w = weights[prefix + "norm2.weight"] { layer.norm2Weight = w }
        if let b = weights[prefix + "norm2.bias"] { layer.norm2Bias = b }
        if layer.dconv != nil {
            if !loadDConv(&layer.dconv!, weights: weights, prefix: prefix) { return false }
        }
        return true
    }

    // MARK: - Decoder layer

    private static func loadDecoderLayer(
        _ layer: inout HDecLayer, weights: [String: MLXArray], prefix: String
    ) -> Bool {
        guard let cw = weights[prefix + "conv_tr.weight"],
              let cb = weights[prefix + "conv_tr.bias"] else {
            print("WeightLoader: missing conv_tr weights for \(prefix)")
            return false
        }
        layer.convTrWeight = cw
        layer.convTrBias = cb
        if let w = weights[prefix + "norm2.weight"] { layer.norm2Weight = w }
        if let b = weights[prefix + "norm2.bias"] { layer.norm2Bias = b }
        if let w = weights[prefix + "rewrite.weight"] { layer.rewriteWeight = w }
        if let b = weights[prefix + "rewrite.bias"] { layer.rewriteBias = b }
        if let w = weights[prefix + "norm1.weight"] { layer.norm1Weight = w }
        if let b = weights[prefix + "norm1.bias"] { layer.norm1Bias = b }
        if layer.dconv != nil {
            if !loadDConv(&layer.dconv!, weights: weights, prefix: prefix) { return false }
        }
        return true
    }

    // MARK: - Transformer encoder layer (self-attention)

    private static func loadTransformerEncoderLayer(
        _ layer: inout MyTransformerEncoderLayer,
        weights: [String: MLXArray], prefix: String
    ) -> Bool {
        if let w = weights[prefix + "self_attn.in_proj_weight"] { layer.selfAttnInProjWeight = w }
        if let b = weights[prefix + "self_attn.in_proj_bias"] { layer.selfAttnInProjBias = b }
        if let w = weights[prefix + "self_attn.out_proj.weight"] { layer.selfAttnOutProjWeight = w }
        if let b = weights[prefix + "self_attn.out_proj.bias"] { layer.selfAttnOutProjBias = b }

        if let w = weights[prefix + "linear1.weight"] { layer.linear1Weight = w }
        if let b = weights[prefix + "linear1.bias"] { layer.linear1Bias = b }
        if let w = weights[prefix + "linear2.weight"] { layer.linear2Weight = w }
        if let b = weights[prefix + "linear2.bias"] { layer.linear2Bias = b }

        if let w = weights[prefix + "norm1.weight"] { layer.norm1.weight = w }
        if let b = weights[prefix + "norm1.bias"] { layer.norm1.bias = b }
        if let w = weights[prefix + "norm2.weight"] { layer.norm2.weight = w }
        if let b = weights[prefix + "norm2.bias"] { layer.norm2.bias = b }

        if layer.normOut != nil {
            if let w = weights[prefix + "norm_out.weight"] { layer.normOut!.weight = w }
            if let b = weights[prefix + "norm_out.bias"] { layer.normOut!.bias = b }
        }
        if layer.gamma1 != nil {
            if let s = weights[prefix + "gamma_1.scale"] { layer.gamma1!.scale = s }
        }
        if layer.gamma2 != nil {
            if let s = weights[prefix + "gamma_2.scale"] { layer.gamma2!.scale = s }
        }
        return true
    }

    // MARK: - Cross-transformer encoder layer (cross-attention)

    private static func loadCrossTransformerLayer(
        _ layer: inout CrossTransformerEncoderLayer,
        weights: [String: MLXArray], prefix: String
    ) -> Bool {
        // PyTorch stores combined in_proj_weight [3*d, d] and in_proj_bias [3*d]
        // Split into separate Q, K, V projections
        if let inProjW = weights[prefix + "cross_attn.in_proj_weight"] {
            let d = inProjW.shape[1]
            layer.crossAttnQProjWeight = sliceAxis(inProjW, axis: 0, start: 0, end: d)
            layer.crossAttnKProjWeight = sliceAxis(inProjW, axis: 0, start: d, end: 2 * d)
            layer.crossAttnVProjWeight = sliceAxis(inProjW, axis: 0, start: 2 * d, end: 3 * d)
        }
        if let inProjB = weights[prefix + "cross_attn.in_proj_bias"] {
            let d = inProjB.shape[0] / 3
            layer.crossAttnQProjBias = sliceAxis(inProjB, axis: 0, start: 0, end: d)
            layer.crossAttnKProjBias = sliceAxis(inProjB, axis: 0, start: d, end: 2 * d)
            layer.crossAttnVProjBias = sliceAxis(inProjB, axis: 0, start: 2 * d, end: 3 * d)
        }
        if let w = weights[prefix + "cross_attn.out_proj.weight"] { layer.crossAttnOutProjWeight = w }
        if let b = weights[prefix + "cross_attn.out_proj.bias"] { layer.crossAttnOutProjBias = b }

        if let w = weights[prefix + "linear1.weight"] { layer.linear1Weight = w }
        if let b = weights[prefix + "linear1.bias"] { layer.linear1Bias = b }
        if let w = weights[prefix + "linear2.weight"] { layer.linear2Weight = w }
        if let b = weights[prefix + "linear2.bias"] { layer.linear2Bias = b }

        if let w = weights[prefix + "norm1.weight"] { layer.norm1.weight = w }
        if let b = weights[prefix + "norm1.bias"] { layer.norm1.bias = b }
        if let w = weights[prefix + "norm2.weight"] { layer.norm2.weight = w }
        if let b = weights[prefix + "norm2.bias"] { layer.norm2.bias = b }
        if let w = weights[prefix + "norm3.weight"] { layer.norm3.weight = w }
        if let b = weights[prefix + "norm3.bias"] { layer.norm3.bias = b }

        if layer.normOut != nil {
            if let w = weights[prefix + "norm_out.weight"] { layer.normOut!.weight = w }
            if let b = weights[prefix + "norm_out.bias"] { layer.normOut!.bias = b }
        }
        if layer.gamma1 != nil {
            if let s = weights[prefix + "gamma_1.scale"] { layer.gamma1!.scale = s }
        }
        if layer.gamma2 != nil {
            if let s = weights[prefix + "gamma_2.scale"] { layer.gamma2!.scale = s }
        }
        return true
    }

    // MARK: - Cross-transformer encoder

    private static func loadCrossTransformer(
        _ transformer: inout CrossTransformerEncoder,
        weights: [String: MLXArray], prefix: String
    ) -> Bool {
        // Load norm_in
        if transformer.normIn != nil {
            if let w = weights[prefix + "norm_in.weight"] { transformer.normIn!.weight = w }
            if let b = weights[prefix + "norm_in.bias"] { transformer.normIn!.bias = b }
        }
        if transformer.normInT != nil {
            if let w = weights[prefix + "norm_in_t.weight"] { transformer.normInT!.weight = w }
            if let b = weights[prefix + "norm_in_t.bias"] { transformer.normInT!.bias = b }
        }

        // Determine total layers by probing keys
        var totalLayers = 0
        for i in 0..<100 {
            if weights[prefix + "layers.\(i).norm1.weight"] != nil {
                totalLayers = i + 1
            } else {
                break
            }
        }

        // Load alternating classic / cross layers
        var classicIdx = 0
        var crossIdx = 0
        for i in 0..<totalLayers {
            let freqPrefix = prefix + "layers.\(i)."
            let timePrefix = prefix + "layers_t.\(i)."

            let isCross = weights[freqPrefix + "cross_attn.in_proj_weight"] != nil

            if isCross {
                if crossIdx < transformer.crossLayers.count {
                    if !loadCrossTransformerLayer(&transformer.crossLayers[crossIdx],
                                                  weights: weights, prefix: freqPrefix) { return false }
                }
                if crossIdx < transformer.crossLayersT.count {
                    if !loadCrossTransformerLayer(&transformer.crossLayersT[crossIdx],
                                                  weights: weights, prefix: timePrefix) { return false }
                }
                crossIdx += 1
            } else {
                if classicIdx < transformer.classicLayers.count {
                    if !loadTransformerEncoderLayer(&transformer.classicLayers[classicIdx],
                                                    weights: weights, prefix: freqPrefix) { return false }
                }
                if classicIdx < transformer.classicLayersT.count {
                    if !loadTransformerEncoderLayer(&transformer.classicLayersT[classicIdx],
                                                    weights: weights, prefix: timePrefix) { return false }
                }
                classicIdx += 1
            }
        }

        return true
    }
}
