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
/// Which random number generator to use for noise generation.
public enum RandomSourceType: Sendable {
    case numPy
    case torch
    case nvidia
}

/// Generate Gaussian noise (mean 0, stdev 1) using the specified random source.
public func generateNoise(count: Int, seed: UInt32, sourceType: RandomSourceType = .numPy) -> [Float] {
    switch sourceType {
    case .numPy:
        var rng = NumPyRandomSource(seed: seed)
        return (0..<count).map { _ in Float(rng.nextNormal()) }
    case .torch:
        var rng = TorchRandomSource(seed: seed)
        return (0..<count).map { _ in Float(rng.nextNormal()) }
    case .nvidia:
        var rng = NvRandomSource(seed: seed)
        return (0..<count).map { _ in Float(rng.nextNormal()) }
    }
}
#endif  // canImport(CoreAI)
