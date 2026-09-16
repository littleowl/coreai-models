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

/// Result from image generation — both displayable images and raw latents.
public struct GenerationResult: Sendable {
    /// Decoded images ready for display.
    public let images: [CGImage]
    /// Raw latent tensors [1, C, H, W] — for img2img round-trips or debugging.
    public let latents: [NDArray]

    public init(images: [CGImage], latents: [NDArray]) {
        self.images = images
        self.latents = latents
    }
}

/// Progress callback payload for generation UI.
public struct PipelineProgress: Sendable {
    public let step: Int
    public let totalSteps: Int
    /// Preview latent at current step (if available).
    public let currentLatent: NDArray?

    public init(step: Int, totalSteps: Int, currentLatent: NDArray? = nil) {
        self.step = step
        self.totalSteps = totalSteps
        self.currentLatent = currentLatent
    }
}

/// Orchestrates multi-component diffusion inference (text encode → denoise → decode).
public protocol DiffusionPipeline: ResourceManaging {
    /// Native output resolution for this model.
    var defaultImageSize: (width: Int, height: Int) { get }
    /// Which schedulers this pipeline supports.
    var supportedSchedulers: [SchedulerType] { get }
    /// Whether this pipeline supports image-to-image generation.
    var supportsImageToImage: Bool { get }

    /// Generate images from a configuration.
    /// - Parameters:
    ///   - configuration: Prompt, steps, guidance, seed, etc.
    ///   - progressHandler: Called after each denoising step. Return false to cancel.
    ///     Defaults to a no-op that always continues.
    func generateImages(
        configuration: PipelineConfiguration,
        progressHandler: ((PipelineProgress) -> Bool)?
    ) async throws -> GenerationResult
}

extension DiffusionPipeline {
    /// Convenience: generate without a progress handler.
    public func generateImages(
        configuration: PipelineConfiguration
    ) async throws -> GenerationResult {
        try await generateImages(configuration: configuration, progressHandler: nil)
    }
}
#endif  // canImport(CoreAI)
