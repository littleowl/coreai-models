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

/// Supported scheduler algorithms.
public enum SchedulerType: String, Sendable, CaseIterable {
    case pndm
    case dpmSolverMultistep = "dpmpp"
    case discreteFlow = "flow_match_euler"
}
#endif  // canImport(CoreAI)
