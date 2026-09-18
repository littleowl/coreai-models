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
import CoreAI
import CoreAIShared
import CoreGraphics

// The pipeline as three stages that take and return values — the prompt
// encoded, a picture encoded as reference tokens, the latent denoised, the
// latent decoded — so a caller can run them in its own order, keep what it
// wants between calls (a prompt's conditioning, a loaded transformer) and
// put previews or a node graph over them. `generateImages` is these stages
// in their usual order.
extension Flux2Pipeline {

    /// The token grid a picture is denoised on: its pixel sides divided by
    /// the 16-pixel patch (the VAE's 8 times the transformer's 2 × 2).
    public struct Grid: Hashable, Sendable {
        public let width: Int
        public let height: Int

        public init(width: Int, height: Int) {
            self.width = width
            self.height = height
        }

        /// The grid for a picture size, which must have sides divisible by 16.
        public init(pixels: (width: Int, height: Int)) {
            self.init(width: pixels.width / Flux2Pipeline.patchSize, height: pixels.height / Flux2Pipeline.patchSize)
        }

        public var tokens: Int { width * height }
        public var pixelSize: (width: Int, height: Int) {
            (width * Flux2Pipeline.patchSize, height * Flux2Pipeline.patchSize)
        }
    }

    /// A prompt encoded: what the transformer attends to at every step.
    public struct TextConditioning: Sendable {
        /// `[1, sequenceLength, hiddenDimension]`, flattened.
        public let embeddings: [Float]
        public let sequenceLength: Int
        public let hiddenDimension: Int
    }

    /// A reference picture encoded and packed as tokens the transformer
    /// takes after the noise tokens, on the grid the reference was reduced to.
    public struct ReferenceTokens: Sendable {
        /// `[grid.tokens, 128]`, flattened; normalised the way the
        /// transformer expects (the VAE's statistics, or none for TAEF2).
        public let tokens: [Float]
        public let grid: Grid
        /// The reduction it was made at, which names the traced route.
        public let referenceGrid: ReferenceGrid
    }

    /// A latent in the transformer's own space: 128 channels over the token
    /// grid (`[1, 128, grid.height, grid.width]`, flattened), before the
    /// VAE's statistics are applied. `decode` turns it into the picture
    /// through the VAE and `decodePreview` through TAEF2, which lives in this
    /// space and needs no statistics.
    public struct Latent: Sendable {
        public let values: [Float]
        public let grid: Grid

        public init(values: [Float], grid: Grid) {
            self.values = values
            self.grid = grid
        }

        public var pixelSize: (width: Int, height: Int) { grid.pixelSize }
    }

    /// One denoising step landed.
    public struct DenoiseStep: Sendable {
        /// 1-based.
        public let step: Int
        public let totalSteps: Int
        /// The latent after this step; the last step's is what `denoise` returns.
        public let latent: Latent
    }

    // MARK: - Encode

    /// Encodes a prompt. The text encoder is loaded on first use and stays
    /// loaded until `unloadTextEncoder()`.
    public func encodePrompt(_ text: String) async throws -> TextConditioning {
        let embeddings = try await encodeText(text)
        let hidden = hiddenDim(embeddings)
        return TextConditioning(
            embeddings: embeddings, sequenceLength: embeddings.count / hidden, hiddenDimension: hidden)
    }

    /// Encodes a reference picture for a picture of `size`: resized to it,
    /// through the VAE encoder, patchified and normalised, then reduced to
    /// the reference grid by block means. Throws when the pipeline has no
    /// encoder.
    public func encodeReference(
        _ image: CGImage, for size: (width: Int, height: Int), grid referenceGrid: ReferenceGrid = .half
    ) async throws -> ReferenceTokens {
        guard let encoder else {
            throw PipelineLoadError.missingComponent("VAEEncoder (this pipeline cannot take a picture)")
        }
        let noise = Grid(pixels: size)
        let reduced = Grid(
            width: max(1, noise.width / referenceGrid.divisor), height: max(1, noise.height / referenceGrid.divisor))
        let full = try await encodeReferenceImage(
            encoder: encoder, srcImage: image, imageWidth: size.width, imageHeight: size.height,
            gridW: noise.width, gridH: noise.height, inChannels: Self.latentChannels)
        let tokens = reduced == noise
            ? full
            : Self.subsampleTokens(
                full, fromW: noise.width, fromH: noise.height, toW: reduced.width, toH: reduced.height,
                channels: Self.latentChannels)
        return ReferenceTokens(tokens: tokens, grid: reduced, referenceGrid: referenceGrid)
    }

    // MARK: - Denoise

    /// The transformer function that serves a run: the text-to-image one, or
    /// the route traced for this many references at this grid — which, on an
    /// open or enumerated transformer, is the same asset at a longer count.
    public func denoiser(references: Int, grid: ReferenceGrid) throws -> (function: CoreAIDiffusionModelFunction, entrypoint: String) {
        guard references > 0 else { return (transformer, transformerFunctionName) }
        let routes = references == 2 ? img2img2Routes : img2imgRoutes
        if references == 2, routes.isEmpty {
            throw PipelineLoadError.unsupportedConfiguration(
                "this bundle has no two-reference transformer (img2img2_* entrypoints; "
                    + "export with --bundle … --references 2).")
        }
        guard let route = routes[grid] else {
            let available = routes.keys.map(\.rawValue).sorted()
            throw PipelineLoadError.unsupportedConfiguration(
                available.isEmpty
                    ? "this bundle has no img2img transformer. Export the img2img "
                        + "components, or export without --single-function to get every "
                        + "grid from a single asset."
                    : "this bundle has no img2img transformer for the "
                        + "\(grid) reference grid. Available: "
                        + "\(available.joined(separator: ", ")).")
        }
        return (route.function, route.entrypoint)
    }

    /// Denoises from seeded noise to a latent, at `size`, under a prompt and
    /// any reference tokens (all made at one reference grid). `onStep` is
    /// called after every step with the latent so far and may return false
    /// to stop, which throws `CancellationError`.
    ///
    /// - Parameter unconditional: the empty prompt encoded, for manual
    ///   classifier-free guidance (two passes a step, interpolated by
    ///   `guidanceScale`); nil runs the distilled single pass, where the
    ///   scale is passed to the graph and ignored by this checkpoint.
    public func denoise(
        size: (width: Int, height: Int),
        prompt: TextConditioning,
        references: [ReferenceTokens] = [],
        steps: Int,
        guidanceScale: Float = 1.0,
        unconditional: TextConditioning? = nil,
        seed: UInt32,
        onStep: ((DenoiseStep) async -> Bool)? = nil
    ) async throws -> Latent {
        let grid = Grid(pixels: size)
        let inChannels = Self.latentChannels
        let seqLen = grid.tokens

        // Reference-token img2img uses the full schedule (1.0 → 0): structure
        // comes from the concatenated reference tokens, not from noise
        // blending, so sigmaMax is 1.0 either way.
        let scheduler = DiscreteFlowScheduler(
            stepCount: steps, trainStepCount: 1000, timeStepShift: 1.0,
            mu: Self.computeEmpiricalMu(imageSeqLen: seqLen, numSteps: steps), sigmaMax: 1.0)

        let noise = generateNoise(count: inChannels * seqLen, seed: seed)
        var packedLatents = packLatentsSpatialFlatten(noise, channels: inChannels, height: grid.height, width: grid.width)

        // Every reference sits on the same reduced grid; their tokens follow
        // the noise tokens, marked T=10 and T=20 on RoPE's first axis.
        let referenceTokens: [Float]? = references.isEmpty ? nil : references.flatMap(\.tokens)
        let referenceGrid = references.first?.grid ?? Grid(width: 0, height: 0)
        let refSeqLen = referenceGrid.tokens * references.count

        let axesDims = descriptor.ropeAxesDims ?? [32, 32, 32, 32]
        let axisCount = axesDims.count
        guard axisCount >= 3 else {
            throw PipelineLoadError.missingConfig("rope_axes_dims has \(axisCount) axes; FLUX.2 RoPE needs at least 3")
        }
        let imageIds = references.isEmpty
            ? buildImageIds(gridW: grid.width, gridH: grid.height, axisCount: axisCount)
            : buildImageIdsWithReference(
                noiseW: grid.width, noiseH: grid.height, refW: referenceGrid.width, refH: referenceGrid.height,
                axisCount: axisCount, references: references.count)
        let textIds = buildTextIds(textSeqLen: prompt.sequenceLength, axisCount: axisCount)

        let (denoiser, fnName) = try denoiser(references: references.count, grid: references.first?.referenceGrid ?? .half)

        var cfgBuffer = [Float](repeating: 0, count: seqLen * inChannels)
        for (step, t) in scheduler.timeSteps.enumerated() {
            let timestepValue = Float(t) / 1000.0
            let inputTokens = referenceTokens.map { packedLatents + $0 } ?? packedLatents
            let inputSeqLen = seqLen + refSeqLen

            let output: [Float]
            if let unconditional {
                // Manual CFG: two passes. The guidance input is 0 only because
                // the traced signature wants a value; this checkpoint's
                // `guidance_embeds` is false and the graph drops it.
                let cond = try await denoiser.run(
                    floatInputs: [
                        (inputTokens, [1, inputSeqLen, inChannels]),
                        (prompt.embeddings, [1, prompt.sequenceLength, prompt.hiddenDimension]),
                        ([timestepValue], [1]),
                        ([Float(0)], [1]),
                        (imageIds, [1, inputSeqLen, axisCount]),
                        (textIds, [1, prompt.sequenceLength, axisCount]),
                    ], functionName: fnName)
                let uncond = try await denoiser.run(
                    floatInputs: [
                        (inputTokens, [1, inputSeqLen, inChannels]),
                        (unconditional.embeddings, [1, unconditional.sequenceLength, unconditional.hiddenDimension]),
                        ([timestepValue], [1]),
                        ([Float(0)], [1]),
                        (imageIds, [1, inputSeqLen, axisCount]),
                        (textIds, [1, prompt.sequenceLength, axisCount]),
                    ], functionName: fnName)
                Self.applyClassifierFreeGuidance(
                    cond: cond[0..<(seqLen * inChannels)], uncond: uncond[0..<(seqLen * inChannels)],
                    guidanceScale: guidanceScale, into: &cfgBuffer)
                output = cfgBuffer
            } else {
                let full = try await denoiser.run(
                    floatInputs: [
                        (inputTokens, [1, inputSeqLen, inChannels]),
                        (prompt.embeddings, [1, prompt.sequenceLength, prompt.hiddenDimension]),
                        ([timestepValue], [1]),
                        ([guidanceScale], [1]),
                        (imageIds, [1, inputSeqLen, axisCount]),
                        (textIds, [1, prompt.sequenceLength, axisCount]),
                    ], functionName: fnName)
                output = referenceTokens == nil ? full : Array(full[0..<(seqLen * inChannels)])
            }

            packedLatents = scheduler.step(output: output, timeStep: t, sample: packedLatents)
            try checkLatentsAreFinite(packedLatents, step: step)

            if let onStep {
                let latent = Latent(
                    values: unpackLatentsSpatialFlatten(packedLatents, channels: inChannels, height: grid.height, width: grid.width),
                    grid: grid)
                let goOn = await onStep(DenoiseStep(step: step + 1, totalSteps: steps, latent: latent))
                if !goOn { throw CancellationError() }
            }
        }

        return Latent(
            values: unpackLatentsSpatialFlatten(packedLatents, channels: inChannels, height: grid.height, width: grid.width),
            grid: grid)
    }

    // MARK: - Decode

    /// The latent as the VAE takes it: the statistics applied (none for
    /// TAEF2) and the 2 × 2 patches unfolded to `[1, 32, H/8, W/8]`.
    func vaeLatent(_ latent: Latent, denormalised: Bool) -> (values: [Float], shape: [Int]) {
        let grid = latent.grid
        let spatial = denormalised
            ? applyBatchNormDenorm(latent.values, channels: Self.latentChannels, height: grid.height, width: grid.width)
            : latent.values
        let unpatchified = Self.unpatchifyLatents(spatial, channels: Self.latentChannels, height: grid.height, width: grid.width)
        return (unpatchified, [1, Self.latentChannels / 4, grid.height * 2, grid.width * 2])
    }

    /// The picture, through the pipeline's decoder — the VAE, in tiles when
    /// the decoder is the 512-pixel tile decoder, or TAEF2 when the pipeline
    /// was opened with the tiny autoencoder.
    public func decode(_ latent: Latent) async throws -> CGImage {
        let (values, shape) = vaeLatent(latent, denormalised: true)
        let size = latent.pixelSize
        let pixels: [Float]
        switch mode {
        case .full, .half:
            pixels = try await decoder.run(floatInputs: [(values, shape)])
        case .tiled:
            pixels = try await decodeTiled(
                latents: values, channels: shape[1], height: shape[2], width: shape[3],
                decoder: decoder, outputScale: 8)
        case .auto:
            preconditionFailure("auto resolved at init")
        }
        return try DiffusionUtilities.pixelsToCGImage(pixels, height: size.height, width: size.width)
    }

    /// Whether `decodePreview` has a decoder to answer with.
    public var canPreview: Bool { previewDecoder != nil || usesTinyVAE }

    /// The picture as TAEF2 draws it — a few milliseconds and a few
    /// megabytes, for a step-by-step preview — or nil when the folder had
    /// no tiny decoder. A pipeline opened on the tiny autoencoder previews
    /// through its own decoder.
    public func decodePreview(_ latent: Latent) async throws -> CGImage? {
        if usesTinyVAE { return try await decode(latent) }
        guard let previewDecoder else { return nil }
        // TAEF2 lives in the transformer's latent space: no statistics.
        let (values, shape) = vaeLatent(latent, denormalised: false)
        let size = latent.pixelSize
        let pixels = try await previewDecoder.run(floatInputs: [(values, shape)])
        return try DiffusionUtilities.pixelsToCGImage(pixels, height: size.height, width: size.width)
    }

    // MARK: - Resources, one at a time

    public func unloadTextEncoder() async { await textEncoder.unloadResources() }
    public func unloadEncoder() async { if let encoder { await encoder.unloadResources() } }
    public func unloadDecoder() async { await decoder.unloadResources() }
    public func unloadPreviewDecoder() async { if let previewDecoder { await previewDecoder.unloadResources() } }
    /// Every transformer route, since a single-function bundle keeps
    /// image-to-image in a file of its own.
    public func unloadTransformer() async {
        await transformer.unloadResources()
        for route in img2imgRoutes.values where route.function !== transformer {
            await route.function.unloadResources()
        }
        for route in img2img2Routes.values where route.function !== transformer {
            await route.function.unloadResources()
        }
    }
    public func loadPreviewDecoder() async throws {
        if let previewDecoder { try await previewDecoder.loadResources() }
    }
}

extension ReferenceGrid {
    /// What each side of the noise grid is divided by to make the reference grid.
    var divisor: Int {
        switch self {
        case .full: 1
        case .half: 2
        case .quarter: 4
        }
    }
}
#endif  // canImport(CoreAI)
