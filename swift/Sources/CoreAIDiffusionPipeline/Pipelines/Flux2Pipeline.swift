// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

// Core AI is in the device and Mac SDKs and not in the simulator's, so a
// target that runs a model cannot build there. Guarding the whole file — and
// every file of this target — lets the package build for a simulator with no
// pipeline in it, which is what lets an app have one target rather than one
// per SDK.
#if canImport(CoreAI)
import Accelerate
import CoreAI
import CoreAIShared
import CoreGraphics
import Tokenizers

/// A traced img2img graph: which asset holds it, and under which entrypoint.
public struct Img2ImgRoute: Sendable {
    public let function: CoreAIDiffusionModelFunction
    public let entrypoint: String

    public init(function: CoreAIDiffusionModelFunction, entrypoint: String) {
        self.function = function
        self.entrypoint = entrypoint
    }
}

/// FLUX.2 Klein pipeline using Core AI backend.
///
/// Orchestrates: tokenize → text encode → noise → pack → denoise loop
/// (flow-match Euler) → unpack → BN denorm → unpatchify → VAE decode.
///
/// RoPE is computed inside the transformer graph; this pipeline only supplies
/// position IDs, which depend on grid geometry alone.
public struct Flux2Pipeline: DiffusionPipeline {
    public let descriptor: PipelineDescriptor
    public let mode: DecodeResolution
    /// The pixel size this bundle's assets were traced at, when they say so.
    ///
    /// Non-square resolutions are their own assets, named after themselves,
    /// because the multi-function transformer holds only the two square
    /// entrypoints it was traced with. Nil for those two, where the mode
    /// decides.
    public let pixelSize: (width: Int, height: Int)?

    public let transformer: CoreAIDiffusionModelFunction
    /// How each reference grid reaches a traced graph, resolved at load time.
    ///
    /// img2img arrives two ways and a bundle can contain both, because export directories
    /// accumulate assets across runs. Resolving to (asset, entrypoint) pairs up front
    /// keeps the choice in one place:
    ///
    /// - multi-function: an `img2img_*` entrypoint on `transformer` — preferred, since it
    ///   reuses the already-loaded asset instead of a second ~2 GB weight set
    /// - single-function: a `Transformer[_512]_img2img_<grid>` asset, entrypoint `main`
    ///
    /// A grid absent from both is simply not supported by the bundle.
    public let img2imgRoutes: [ReferenceGrid: Img2ImgRoute]
    /// The same for two reference pictures: `img2img2_<size>_<grid>` entrypoints
    /// of a bundle exported with `--references 2`. Empty otherwise.
    public let img2img2Routes: [ReferenceGrid: Img2ImgRoute]
    public let textEncoder: CoreAIDiffusionModelFunction
    public let decoder: CoreAIDiffusionModelFunction
    public let encoder: CoreAIDiffusionModelFunction?
    /// TAEF2's decoder (`TinyDecoder_<size>` or `TinyDecoder_open`) when the
    /// folder holds one beside the VAE: a preview a step at a time through
    /// `decodePreview`, 3 MB, in the transformer's own latent space. Nil
    /// without one, and nil when the pipeline's decoder *is* TAEF2.
    public let previewDecoder: CoreAIDiffusionModelFunction?
    public let transformerFunctionName: String
    public let tokenizer: any Tokenizer

    public let batchNormMean: [Float]?
    public let batchNormVar: [Float]?
    public let batchNormEps: Float
    /// The decoder and encoder are TAEF2, the tiny autoencoder, whose latents
    /// are already in the transformer's space — so no batch-norm statistics.
    public let usesTinyVAE: Bool

    // MARK: - Architecture Constants

    static let patchSize = 16
    static let latentChannels = 128
    static let textSeqLen = 512
    private static let qwen3PadTokenId = 151643

    /// Reference tokens sit at T=10 on RoPE axis 0, separating them from the noise
    /// grid (T=0) where H/W would otherwise collide. Matches the export-time dummies.
    private static let referenceTokenTimeOffset: Float = 10
    /// A second reference sits at T=20. Matches `SECOND_REFERENCE_TOKEN_TIME_OFFSET`.
    private static let secondReferenceTokenTimeOffset: Float = 20

    /// FLUX.2 flow-matching timestep shift.
    ///
    /// Mirrors diffusers `compute_empirical_mu` (diffusers 0.37.1):
    ///   pipelines/flux2/pipeline_flux2_klein.py:63-78
    ///   (copied from pipelines/flux2/pipeline_flux2.py)
    /// Call site — pipeline_flux2_klein.py:810-811:
    ///   image_seq_len = latents.shape[1]
    ///   mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_inference_steps)
    ///
    /// Reference implementation:
    ///   def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    ///       a1, b1 = 8.73809524e-05, 1.89833333
    ///       a2, b2 = 0.00016927, 0.45666666
    ///       if image_seq_len > 4300:
    ///           mu = a2 * image_seq_len + b2
    ///           return float(mu)
    ///       m_200 = a2 * image_seq_len + b2
    ///       m_10 = a1 * image_seq_len + b1
    ///       a = (m_200 - m_10) / 190.0
    ///       b = m_200 - 200.0 * a
    ///       mu = a * num_steps + b
    ///       return float(mu)
    static func computeEmpiricalMu(imageSeqLen: Int, numSteps: Int) -> Float {
        let a1: Float = 8.73809524e-05
        let b1: Float = 1.89833333
        let a2: Float = 0.00016927
        let b2: Float = 0.45666666
        let seq = Float(imageSeqLen)
        if imageSeqLen > 4300 {
            return a2 * seq + b2
        }
        let m200 = a2 * seq + b2
        let m10 = a1 * seq + b1
        let a = (m200 - m10) / 190.0
        let b = m200 - 200.0 * a
        return a * Float(numSteps) + b
    }

    /// Image size is determined by the mode selected at init.
    ///
    /// A resolution-named export (`Transformer_1024x768`) says its own size, and
    /// `pixelSize` carries it; the square exports fall back to the mode, which
    /// is what they always did.
    public var defaultImageSize: (width: Int, height: Int) {
        if let pixelSize { return pixelSize }
        let full = descriptor.imageSize ?? 1024
        let size = (mode == .half) ? full / 2 : full
        return (size, size)
    }

    public var supportedSchedulers: [SchedulerType] {
        [.discreteFlow]
    }

    public var supportsImageToImage: Bool {
        encoder != nil
    }

    public init(
        descriptor: PipelineDescriptor,
        mode: DecodeResolution = .full,
        transformer: CoreAIDiffusionModelFunction,
        img2imgRoutes: [ReferenceGrid: Img2ImgRoute] = [:],
        img2img2Routes: [ReferenceGrid: Img2ImgRoute] = [:],
        textEncoder: CoreAIDiffusionModelFunction,
        decoder: CoreAIDiffusionModelFunction,
        encoder: CoreAIDiffusionModelFunction?,
        transformerFunctionName: String = "main",
        tokenizer: any Tokenizer,
        batchNormMean: [Float]?,
        batchNormVar: [Float]?,
        batchNormEps: Float,
        pixelSize: (width: Int, height: Int)? = nil,
        usesTinyVAE: Bool = false,
        previewDecoder: CoreAIDiffusionModelFunction? = nil
    ) {
        self.descriptor = descriptor
        self.mode = mode
        self.transformer = transformer
        self.img2imgRoutes = img2imgRoutes
        self.img2img2Routes = img2img2Routes
        self.textEncoder = textEncoder
        self.decoder = decoder
        self.encoder = encoder
        self.transformerFunctionName = transformerFunctionName
        self.tokenizer = tokenizer
        self.batchNormMean = batchNormMean
        self.batchNormVar = batchNormVar
        self.batchNormEps = batchNormEps
        self.pixelSize = pixelSize
        self.usesTinyVAE = usesTinyVAE
        self.previewDecoder = previewDecoder

        if tokenizer.convertTokenToId("<|endoftext|>") == nil {
            CLILogger.log(
                "⚠️ Flux2Pipeline: tokenizer has no <|endoftext|> token, using Qwen3 fallback pad ID",
                component: "Diffusion")
        }
    }

    // MARK: - ResourceManaging

    public func loadResources() async throws {
        try await transformer.loadResources()
        try await textEncoder.loadResources()
        try await decoder.loadResources()
        if let encoder { try await encoder.loadResources() }
        // The img2img transformers are deliberately *not* loaded here. Each is a full
        // weight set (~2 GB resident on GPU) that a txt2img run never touches, and
        // `CoreAIDiffusionModelFunction` loads itself on first use anyway. They are still
        // unloaded below, so a run that did use one releases it.
    }

    public func unloadResources() async {
        await transformer.unloadResources()
        // Distinct assets only: multi-function routes point back at `transformer`.
        for route in img2imgRoutes.values where route.function !== transformer {
            await route.function.unloadResources()
        }
        for route in img2img2Routes.values where route.function !== transformer {
            await route.function.unloadResources()
        }
        await textEncoder.unloadResources()
        await decoder.unloadResources()
        if let encoder { await encoder.unloadResources() }
        if let previewDecoder { await previewDecoder.unloadResources() }
    }

    // MARK: - Generation

    /// The stages (`Flux2Pipeline+Stages.swift`) in their usual order:
    /// encode the prompt, encode any references, denoise, decode. With
    /// `lazyModelLoading` each model is unloaded as soon as its stage is
    /// done, which is what bounds peak memory to the largest one.
    public func generateImages(
        configuration: PipelineConfiguration,
        progressHandler: ((PipelineProgress) -> Bool)?
    ) async throws -> GenerationResult {
        let steps = configuration.stepCount
        let guidanceScale = configuration.guidanceScale

        // 1. Encode text — and the empty prompt for manual CFG while the
        // encoder is loaded, rather than reloading it after the references.
        //
        // At exactly 1.0 the interpolation reduces to the conditional pass (the
        // unconditional term's coefficient is zero) so the second forward pass is
        // wasted. Below 1.0 the blend runs *toward* the unconditional prediction, i.e. 2x
        // the compute to follow the prompt less, so this gate refuses that too.
        //
        // That second half is a deliberate divergence from diffusers, whose
        // `ClassifierFreeGuidance` guider gates on `not isclose(scale, 1.0)` and so honours
        // sub-1.0 scales. Widen this to match if a caller ever has a real use for them.
        let prompt = try await encodePrompt(configuration.prompt)
        var unconditional: TextConditioning?
        if configuration.guidanceMode == .manual {
            if guidanceScale > 1.0 {
                unconditional = try await encodePrompt("")
            } else {
                CLILogger.log(
                    "⚠️ Flux2Pipeline: --guidance-mode manual needs a guidance scale above 1.0 "
                        + "(got \(guidanceScale)); falling back to distilled, which applies no "
                        + "guidance at all. Raise the scale to get the two-pass path.",
                    component: "Diffusion")
            }
        }
        if configuration.lazyModelLoading { await textEncoder.unloadResources() }

        // 2. The size: the one the pipeline was opened at, unless this run
        // names its own — which an open transformer and open VAEs can take.
        let size = configuration.imageSize ?? defaultImageSize
        let isActuallyImg2Img = configuration.isImageToImage && encoder != nil && configuration.startingImage != nil

        // Reference-token img2img is incompatible with tiled decode: tiled uses the
        // half-resolution VAE encoder (traced for 512×512), but the reference is
        // encoded at the full image size — feeding it a 1024 image crashes on a
        // shape mismatch. Fail early with a clear message instead.
        // A size-named pipeline encodes the reference with the size's own or the
        // open encoder and only *decodes* in tiles, so the objection is the square
        // export's alone.
        if isActuallyImg2Img && mode == .tiled && pixelSize == nil {
            throw PipelineLoadError.unsupportedConfiguration(
                "img2img is not supported with tiled decode. Use --decode-resolution full or half.")
        }

        // 3. References: each picture encoded the same way, its tokens after
        // the previous one's, on a graph traced for that many.
        var references: [ReferenceTokens] = []
        if isActuallyImg2Img, let first = configuration.startingImage {
            references.append(try await encodeReference(first, for: size, grid: configuration.referenceGrid))
            if let second = configuration.secondStartingImage {
                references.append(try await encodeReference(second, for: size, grid: configuration.referenceGrid))
            }
            if configuration.lazyModelLoading { await encoder?.unloadResources() }
        }

        // 4. Denoise. The progress handler gets the latent as the VAE takes
        // it, `[1, 32, H/8, W/8]` with the statistics applied — array
        // copies, no model call — which is what the preview coefficients
        // were fitted against.
        let latent = try await denoise(
            size: size, prompt: prompt, references: references, steps: steps,
            guidanceScale: guidanceScale, unconditional: unconditional, seed: configuration.seed,
            onStep: progressHandler.map { handler in
                { step in
                    let (values, shape) = self.vaeLatent(step.latent, denormalised: true)
                    var preview = NDArray(shape: shape, scalarType: .float32)
                    preview.mutableView(as: Float.self).withUnsafeMutablePointer { ptr, _, _ in
                        for i in 0..<values.count { ptr[i] = values[i] }
                    }
                    return handler(PipelineProgress(step: step.step, totalSteps: step.totalSteps, currentLatent: preview))
                }
            })
        if configuration.lazyModelLoading {
            // Release whichever asset ran
            let (ran, _) = try denoiser(references: references.count, grid: configuration.referenceGrid)
            await ran.unloadResources()
        }

        // 5. Decode — the VAE, in tiles when the decoder is the tile decoder.
        let image = try await decode(latent)
        if configuration.lazyModelLoading { await decoder.unloadResources() }

        // The final latent, as the VAE took it.
        let (values, shape) = vaeLatent(latent, denormalised: true)
        var latentsND = NDArray(shape: shape, scalarType: .float32)
        latentsND.mutableView(as: Float.self).withUnsafeMutablePointer { ptr, _, _ in
            for i in 0..<values.count { ptr[i] = values[i] }
        }
        return GenerationResult(images: [image], latents: [latentsND])
    }

    // MARK: - Img2Img

    /// Encode a reference image into packed latent tokens (no noise blending).
    func encodeReferenceImage(
        encoder: CoreAIDiffusionModelFunction,
        srcImage: CGImage,
        imageWidth: Int,
        imageHeight: Int,
        gridW: Int,
        gridH: Int,
        inChannels: Int
    ) async throws -> [Float] {
        let resized = CGImageUtils.resize(srcImage, width: imageWidth, height: imageHeight) ?? srcImage
        let encoderScaleFactor = descriptor.encoderScaleFactor ?? 0.18215

        let imagePixels = try CGImageUtils.toNormalizedPlanarRGB(resized)
        let encodedFloats = try await encoder.run(
            floatInputs: [(imagePixels, [1, 3, imageHeight, imageWidth])])

        let scaledEncoded = encodedFloats.map { $0 * encoderScaleFactor }
        let patchified = Self.patchifyLatents(
            scaledEncoded, inChannels: inChannels, height: gridH, width: gridW)
        let normalized = applyBatchNormNorm(
            patchified, channels: inChannels, height: gridH, width: gridW)
        return packLatentsSpatialFlatten(
            normalized, channels: inChannels, height: gridH, width: gridW)
    }

    // MARK: - Text Encoding

    func encodeText(_ text: String) async throws -> [Float] {
        let seqLen = Self.textSeqLen

        // Tokenize using Qwen3 chat template.
        //
        // Must match diffusers `_get_qwen3_prompt_embeds`
        // (diffusers 0.37.1, pipelines/flux2/pipeline_flux2_klein.py), which builds the
        // input as:
        //     messages = [{"role": "user", "content": single_prompt}]
        //     text = tokenizer.apply_chat_template(
        //         messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        //
        // `enable_thinking=False` is significant for the Qwen3 template: it appends an
        // empty `<think>\n\n</think>\n\n` block after the assistant prompt. Leaving it
        // undefined omits that block, changing the trailing conditioning tokens and
        // hurting prompt adherence. Pass it via additionalContext to match the reference.
        var ids: [Int]
        let messages: [[String: String]] = [["role": "user", "content": text]]
        do {
            ids = try tokenizer.applyChatTemplate(
                messages: messages, chatTemplate: nil,
                addGenerationPrompt: true, truncation: true, maxLength: seqLen, tools: nil,
                additionalContext: ["enable_thinking": false]
            )
        } catch {
            let tokens = tokenizer.tokenize(text: text)
            ids = tokens.compactMap { tokenizer.convertTokenToId($0) }
        }

        if ids.count > seqLen {
            ids = Array(ids.prefix(seqLen))
        }

        let realTokenCount = ids.count
        // diffusers pads with the tokenizer's pad_token, not the eos_token. For
        // FLUX.2 klein's Qwen tokenizer these differ: pad_token is <|endoftext|>
        // (151643) while eos_token is <|im_end|> (151645). The reference builds
        // input_ids via `tokenizer(text, padding="max_length", max_length=512)`
        // (diffusers 0.37.1, pipeline_flux2_klein.py `_get_qwen3_prompt_embeds`),
        // which uses pad_token. These ~490 padding tokens are fed to the DiT
        // UNMASKED, so the id must match the reference exactly.
        let padTokenId = tokenizer.convertTokenToId("<|endoftext|>") ?? Self.qwen3PadTokenId

        while ids.count < seqLen {
            ids.append(padTokenId)
        }

        // input_ids: Int32, attention_mask: Int32
        let int32Ids = ids.map { Int32($0) }
        var maskValues = [Int32](repeating: 0, count: seqLen)
        for i in 0..<realTokenCount { maskValues[i] = 1 }

        let hiddenStates = try await textEncoder.run(intInputs: [
            (int32Ids, [1, seqLen]),
            (maskValues, [1, seqLen]),
        ])

        return hiddenStates
    }

    func hiddenDim(_ embeddings: [Float]) -> Int {
        embeddings.count / Self.textSeqLen
    }

    // MARK: - RoPE Position IDs

    /// `img_ids` for in-graph RoPE: `[1, gridH*gridW, axisCount]` flattened
    /// row-major, one row per image token as [T, H, W, L].
    func buildImageIds(gridW: Int, gridH: Int, axisCount: Int) -> [Float] {
        var ids = [Float](repeating: 0, count: gridW * gridH * axisCount)
        for h in 0..<gridH {
            for w in 0..<gridW {
                let idx = h * gridW + w
                ids[idx * axisCount + 1] = Float(h)
                ids[idx * axisCount + 2] = Float(w)
            }
        }
        return ids
    }

    /// `img_ids` for img2img: noise tokens followed by reference tokens.
    /// Reference rows carry T=10 on axis 0 so in-graph RoPE keeps them positionally
    /// distinct from the noise grid even where H/W coincide.
    func buildImageIdsWithReference(
        noiseW: Int, noiseH: Int, refW: Int, refH: Int, axisCount: Int, references: Int = 1
    ) -> [Float] {
        let noiseSeq = noiseW * noiseH
        let refSeq = refW * refH
        var ids = [Float](repeating: 0, count: (noiseSeq + refSeq * references) * axisCount)

        for h in 0..<noiseH {
            for w in 0..<noiseW {
                let idx = h * noiseW + w
                ids[idx * axisCount + 1] = Float(h)
                ids[idx * axisCount + 2] = Float(w)
            }
        }

        let offsets = [Self.referenceTokenTimeOffset, Self.secondReferenceTokenTimeOffset]
        for reference in 0..<references {
            for h in 0..<refH {
                for w in 0..<refW {
                    let idx = noiseSeq + reference * refSeq + h * refW + w
                    ids[idx * axisCount + 0] = offsets[reference]
                    ids[idx * axisCount + 1] = Float(h)
                    ids[idx * axisCount + 2] = Float(w)
                }
            }
        }

        return ids
    }

    /// `txt_ids` for in-graph RoPE: `[1, textSeqLen, axisCount]` flattened row-major.
    /// Text tokens are [0, 0, 0, s] — sequence index on the last axis, spatial unused.
    func buildTextIds(textSeqLen: Int, axisCount: Int) -> [Float] {
        var ids = [Float](repeating: 0, count: textSeqLen * axisCount)
        for s in 0..<textSeqLen {
            ids[s * axisCount + (axisCount - 1)] = Float(s)
        }
        return ids
    }

    /// Spatially downsample packed tokens to a smaller grid, area-averaging each
    /// `stride`×`stride` block channel-wise.
    ///
    /// Point sampling would keep only 1/stride² of the encoded reference and throw
    /// the rest away; the block mean retains all of it, so structure survives at the
    /// half/quarter grids. Channel index encodes intra-patch position, so averaging
    /// per channel keeps corresponding sub-positions aligned.
    /// Input: [fromH*fromW, channels], Output: [toH*toW, channels]. The two
    /// strides are computed separately, so a non-square grid halves correctly
    /// on both sides.
    static func subsampleTokens(
        _ tokens: [Float], fromW: Int, fromH: Int, toW: Int, toH: Int, channels: Int
    ) -> [Float] {
        let strideW = max(1, fromW / toW)
        let strideH = max(1, fromH / toH)
        let scale = 1.0 / Float(strideW * strideH)
        var result = [Float](repeating: 0, count: toW * toH * channels)
        for h in 0..<toH {
            for w in 0..<toW {
                let dstIdx = (h * toW + w) * channels
                for bh in 0..<strideH {
                    let srcRow = h * strideH + bh
                    for bw in 0..<strideW {
                        let srcIdx = (srcRow * fromW + w * strideW + bw) * channels
                        for c in 0..<channels {
                            result[dstIdx + c] += tokens[srcIdx + c]
                        }
                    }
                }
                for c in 0..<channels {
                    result[dstIdx + c] *= scale
                }
            }
        }
        return result
    }

    // MARK: - Classifier-Free Guidance

    /// `uncond + g*(cond - uncond)`, written into `destination` rather than returned.
    ///
    /// The caller reuses one buffer across denoising steps; at 1024×1024 each result is
    /// ~2 MB, so returning a fresh array would allocate one per step.
    static func applyClassifierFreeGuidance(
        cond: ArraySlice<Float>, uncond: ArraySlice<Float>,
        guidanceScale: Float, into destination: inout [Float]
    ) {
        // Reusing the buffer means a short input would leave the previous step's values
        // in the tail rather than merely producing a short array, so require an exact fit.
        precondition(
            cond.count == destination.count && uncond.count == destination.count,
            "CFG expected \(destination.count) noise values, got "
                + "cond=\(cond.count) uncond=\(uncond.count)")
        for (offset, (u, c)) in zip(uncond, cond).enumerated() {
            destination[offset] = u + guidanceScale * (c - u)
        }
    }

    // MARK: - Latent Packing/Unpacking

    /// (B, C, H, W) → (B, H*W, C) — spatial flatten for patch_size=1
    func packLatentsSpatialFlatten(_ latents: [Float], channels: Int, height: Int, width: Int) -> [Float] {
        let seqLen = height * width
        var packed = [Float](repeating: 0, count: seqLen * channels)
        for c in 0..<channels {
            for h in 0..<height {
                for w in 0..<width {
                    let srcIdx = c * height * width + h * width + w
                    let token = h * width + w
                    let dstIdx = token * channels + c
                    packed[dstIdx] = latents[srcIdx]
                }
            }
        }
        return packed
    }

    /// (B, H*W, C) → (B, C, H, W) — inverse spatial flatten
    func unpackLatentsSpatialFlatten(_ packed: [Float], channels: Int, height: Int, width: Int) -> [Float] {
        var unpacked = [Float](repeating: 0, count: channels * height * width)
        for c in 0..<channels {
            for h in 0..<height {
                for w in 0..<width {
                    let token = h * width + w
                    let srcIdx = token * channels + c
                    let dstIdx = c * height * width + h * width + w
                    unpacked[dstIdx] = packed[srcIdx]
                }
            }
        }
        return unpacked
    }

    // MARK: - Batch Norm Denormalization

    /// latents = latents * sqrt(var + eps) + mean (per-channel in BCHW format)
    func applyBatchNormDenorm(_ latents: [Float], channels: Int, height: Int, width: Int) -> [Float] {
        guard let bnMean = batchNormMean, let bnVar = batchNormVar,
            bnMean.count == channels, bnVar.count == channels
        else {
            return latents
        }

        let spatialSize = height * width
        let std = bnVar.map { sqrtf($0 + batchNormEps) }

        var result = [Float](repeating: 0, count: latents.count)
        for c in 0..<channels {
            let offset = c * spatialSize
            for i in 0..<spatialSize {
                result[offset + i] = latents[offset + i] * std[c] + bnMean[c]
            }
        }
        return result
    }

    // MARK: - Unpatchify

    /// (B, C*4, H, W) → (B, C, H*2, W*2) — reverses 2×2 patchification
    static func unpatchifyLatents(_ latents: [Float], channels: Int, height: Int, width: Int) -> [Float] {
        let outChannels = channels / 4
        let outH = height * 2
        let outW = width * 2

        var result = [Float](repeating: 0, count: outChannels * outH * outW)
        for c in 0..<outChannels {
            for i in 0..<height {
                for j in 0..<width {
                    for dy in 0..<2 {
                        for dx in 0..<2 {
                            let srcC = c * 4 + dy * 2 + dx
                            let srcIdx = srcC * height * width + i * width + j
                            let dstIdx = c * outH * outW + (i * 2 + dy) * outW + (j * 2 + dx)
                            result[dstIdx] = latents[srcIdx]
                        }
                    }
                }
            }
        }
        return result
    }

    // MARK: - Patchify / BN Normalize (img2img forward path)

    /// (B, C, H*2, W*2) → (B, C*4, H, W) — forward 2×2 patchification (inverse of unpatchifyLatents).
    static func patchifyLatents(_ latents: [Float], inChannels: Int, height: Int, width: Int) -> [Float] {
        let inCh = inChannels / 4  // vaeChannels (32)
        let inH = height * 2
        let inW = width * 2

        var result = [Float](repeating: 0, count: inChannels * height * width)
        for c in 0..<inCh {
            for i in 0..<height {
                for j in 0..<width {
                    for dy in 0..<2 {
                        for dx in 0..<2 {
                            let dstC = c * 4 + dy * 2 + dx
                            let srcIdx = c * inH * inW + (i * 2 + dy) * inW + (j * 2 + dx)
                            let dstIdx = dstC * height * width + i * width + j
                            result[dstIdx] = latents[srcIdx]
                        }
                    }
                }
            }
        }
        return result
    }

    /// Inverse of applyBatchNormDenorm: x_norm = (x − mean) / sqrt(var + eps) per channel.
    func applyBatchNormNorm(_ latents: [Float], channels: Int, height: Int, width: Int) -> [Float] {
        guard let bnMean = batchNormMean, let bnVar = batchNormVar,
            bnMean.count == channels, bnVar.count == channels
        else {
            return latents
        }

        let spatialSize = height * width
        let std = bnVar.map { sqrtf($0 + batchNormEps) }

        var result = [Float](repeating: 0, count: latents.count)
        for c in 0..<channels {
            let offset = c * spatialSize
            for i in 0..<spatialSize {
                result[offset + i] = (latents[offset + i] - bnMean[c]) / std[c]
            }
        }
        return result
    }

    // MARK: - Image Conversion

    // MARK: - Half/Tiled Decode Helpers

    /// Area-average downsample BCHW latents by an integer factor using vDSP.
    static func downsampleLatents(
        _ input: [Float], channels: Int, height: Int, width: Int, factor: Int
    ) -> [Float] {
        let outH = height / factor
        let outW = width / factor
        let scale = 1.0 / Float(factor * factor)
        var output = [Float](repeating: 0, count: channels * outH * outW)
        for c in 0..<channels {
            let chIn = c * height * width
            let chOut = c * outH * outW
            for oh in 0..<outH {
                for ow in 0..<outW {
                    var sum: Float = 0
                    for dy in 0..<factor {
                        let rowStart = chIn + (oh * factor + dy) * width + ow * factor
                        for dx in 0..<factor {
                            sum += input[rowStart + dx]
                        }
                    }
                    output[chOut + oh * outW + ow] = sum * scale
                }
            }
        }
        return output
    }

    /// Bicubic 2× upsample planar [C, H, W] image.
    static func bicubicUpsample2x(
        _ input: [Float], channels: Int, height: Int, width: Int
    ) -> [Float] {
        let outH = height * 2
        let outW = width * 2
        var output = [Float](repeating: 0, count: channels * outH * outW)

        for c in 0..<channels {
            let chOffset = c * height * width
            let outChOffset = c * outH * outW
            for oy in 0..<outH {
                let srcY = Float(oy) / 2.0 - 0.25
                for ox in 0..<outW {
                    let srcX = Float(ox) / 2.0 - 0.25
                    output[outChOffset + oy * outW + ox] = bicubicSample(
                        input, offset: chOffset, height: height, width: width, y: srcY, x: srcX)
                }
            }
        }
        return output
    }

    private static func bicubicSample(
        _ data: [Float], offset: Int, height: Int, width: Int, y: Float, x: Float
    ) -> Float {
        let iy = Int(floor(y))
        let ix = Int(floor(x))
        let fy = y - Float(iy)
        let fx = x - Float(ix)

        var result: Float = 0
        for j in -1...2 {
            let wy = cubicWeight(Float(j) - fy)
            for i in -1...2 {
                let wx = cubicWeight(Float(i) - fx)
                let sy = min(max(iy + j, 0), height - 1)
                let sx = min(max(ix + i, 0), width - 1)
                result += wy * wx * data[offset + sy * width + sx]
            }
        }
        return result
    }

    private static func cubicWeight(_ t: Float) -> Float {
        let a: Float = -0.5
        let at = abs(t)
        if at <= 1 {
            return (a + 2) * at * at * at - (a + 3) * at * at + 1
        } else if at < 2 {
            return a * at * at * at - 5 * a * at * at + 8 * a * at - 4 * a
        }
        return 0
    }

    /// Tiled VAE decode: split latents into a grid of tiles, decode each with the half-res VAE, blend overlaps.
    /// The latent side of one decode tile: `VAEDecoder_half`'s input, 512 pixels.
    static let tileLatentSize = 64

    func decodeTiled(
        latents: [Float], channels: Int, height: Int, width: Int,
        decoder: CoreAIDiffusionModelFunction, outputScale: Int
    ) async throws -> [Float] {
        // The tile is the decoder's fixed input — `VAEDecoder_half` takes a
        // 64 × 64 latent, a 512-pixel square — not a fraction of the picture,
        // so any size decodes through it: a picture smaller than a tile on a
        // side is padded by clamping (`extractTile`), a larger one is covered
        // by tiles with the last one pulled back to the edge (`tileStarts`).
        let tileSize = Self.tileLatentSize
        let overlap = 4
        let stride = tileSize - overlap

        let outTileSize = tileSize * outputScale
        let outOverlap = overlap * outputScale
        let outH = height * outputScale
        let outW = width * outputScale
        let outChannels = 3

        var output = [Float](repeating: 0, count: outChannels * outH * outW)
        var weights = [Float](repeating: 0, count: outH * outW)

        let startsY = tileStarts(length: height, tileSize: tileSize, stride: stride)
        let startsX = tileStarts(length: width, tileSize: tileSize, stride: stride)

        for startY in startsY {
            for startX in startsX {
                let tile = extractTile(
                    from: latents, channels: channels, height: height, width: width,
                    startY: startY, startX: startX, tileSize: tileSize)

                let tileShape = [1, channels, tileSize, tileSize]
                let decodedTile = try await decoder.run(floatInputs: [(tile, tileShape)])

                blendTile(
                    decodedTile, into: &output, weights: &weights,
                    outChannels: outChannels, outH: outH, outW: outW,
                    outTileSize: outTileSize, outOverlap: outOverlap,
                    outStartY: startY * outputScale, outStartX: startX * outputScale)
            }
        }

        normalizeByWeights(&output, weights: weights, channels: outChannels, size: outH * outW)
        return output
    }

    private func extractTile(
        from latents: [Float], channels: Int, height: Int, width: Int,
        startY: Int, startX: Int, tileSize: Int
    ) -> [Float] {
        var tile = [Float](repeating: 0, count: channels * tileSize * tileSize)
        for c in 0..<channels {
            for y in 0..<tileSize {
                for x in 0..<tileSize {
                    let srcY = min(startY + y, height - 1)
                    let srcX = min(startX + x, width - 1)
                    tile[c * tileSize * tileSize + y * tileSize + x] =
                        latents[c * height * width + srcY * width + srcX]
                }
            }
        }
        return tile
    }

    private func blendTile(
        _ decodedTile: [Float], into output: inout [Float], weights: inout [Float],
        outChannels: Int, outH: Int, outW: Int,
        outTileSize: Int, outOverlap: Int,
        outStartY: Int, outStartX: Int
    ) {
        for c in 0..<outChannels {
            for y in 0..<outTileSize {
                let outY = outStartY + y
                guard outY < outH else { continue }
                let wy = blendWeight(y, outTileSize, outOverlap)
                for x in 0..<outTileSize {
                    let outX = outStartX + x
                    guard outX < outW else { continue }
                    let w = wy * blendWeight(x, outTileSize, outOverlap)
                    output[c * outH * outW + outY * outW + outX] +=
                        w * decodedTile[c * outTileSize * outTileSize + y * outTileSize + x]
                    if c == 0 { weights[outY * outW + outX] += w }
                }
            }
        }
    }

    private func normalizeByWeights(
        _ output: inout [Float], weights: [Float], channels: Int, size: Int
    ) {
        for c in 0..<channels {
            let offset = c * size
            for i in 0..<size where weights[i] > 0 {
                output[offset + i] /= weights[i]
            }
        }
    }

    /// Generate tile start positions that cover [0, length) with given tile size and stride.
    private func tileStarts(length: Int, tileSize: Int, stride: Int) -> [Int] {
        var starts: [Int] = []
        var pos = 0
        while pos + tileSize <= length {
            starts.append(pos)
            pos += stride
        }
        if starts.isEmpty || starts.last! + tileSize < length {
            starts.append(length - tileSize)
        }
        return starts
    }

    private func blendWeight(_ pos: Int, _ size: Int, _ overlap: Int) -> Float {
        if pos < overlap {
            return Float(pos) / Float(overlap)
        } else if pos >= size - overlap {
            return Float(size - 1 - pos) / Float(overlap)
        }
        return 1.0
    }
}
#endif  // canImport(CoreAI)
