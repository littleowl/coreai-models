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
import Foundation

/// Deterministic random source for reproducible noise generation.
///
/// Multiple implementations exist to match the exact random sequences produced by
/// different Python frameworks. Using the matching source guarantees the same seed
/// produces the same image across platforms.
///
/// - ``TorchRandomSource``: Default. Matches PyTorch (`torch.manual_seed`). Used by SDXL, SD3, Flux.
/// - ``NumPyRandomSource``: Matches NumPy (`numpy.random.RandomState`). Used by SD 1.5/2.x.
/// - ``NvRandomSource``: Matches NVIDIA cuRAND (Philox). Used by some ComfyUI/Automatic1111 workflows.
///
/// The implementations are direct ports of MT19937 / Philox; do not refactor the
/// bitwise logic without verifying output against the Python reference for multiple seeds.
public protocol RandomSource {
    mutating func nextNormal(mean: Double, stdev: Double) -> Double
    mutating func normalArray(_ shape: [Int], mean: Double, stdev: Double) -> [Float]
}
#endif  // canImport(CoreAI)
