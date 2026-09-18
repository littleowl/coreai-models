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
import Foundation
import Tokenizers

extension Flux2Pipeline {
    /// Load a FLUX.2 pipeline from a directory containing .aimodel files, tokenizer/, and pipeline.json.
    ///
    /// The `mode` parameter selects which components are loaded:
    /// - `.full`: Transformer + VAEDecoder (1024×1024)
    /// - `.half`: Transformer_512 + VAEDecoder_half (512×512, 4× faster)
    /// - `.tiled`: Transformer + VAEDecoder_half (1024×1024 via tiled decode)
    /// - Parameter size: A pixel size the bundle was exported at, like
    ///   `(1024, 768)`. Its assets are named after it — `Transformer_1024x768`,
    ///   `VAEDecoder_1024x768` — because the multi-function transformer carries
    ///   only the two square entrypoints it was traced with, so every other
    ///   resolution is its own set of assets. Nil picks the square ones by
    ///   `mode`, or the bundle's only size when it holds exactly one.
    /// - Parameter tinyVAE: decode and encode with TAEF2 (`TinyDecoder_<size>`
    ///   or `TinyDecoder_open`, and the encoder beside it) instead of the VAE.
    ///   Its latents are the transformer's own, so the batch-norm statistics
    ///   are not applied. A size-named bundle only.
    public init(
        from url: URL,
        config: PipelineDescriptor.ConfigSource = .auto,
        mode: DecodeResolution = .auto,
        size: (width: Int, height: Int)? = nil,
        tinyVAE: Bool = false
    ) async throws {
        let descriptor = try PipelineDescriptor.resolve(at: url, config: config)

        // A bundle that holds one resolution-named set and no square assets
        // needs no telling: it is what it is.
        let wanted = size ?? Self.soleSize(at: url)

        // Resolve .auto → best available mode
        let resolvedMode: DecodeResolution
        if wanted != nil {
            resolvedMode = .full
        } else if mode == .auto {
            resolvedMode = try Self.bestAvailableMode(at: url, descriptor: descriptor)
        } else {
            resolvedMode = mode
        }

        if let wanted {
            try await self.init(
                from: url, descriptor: descriptor, size: wanted, tokenizerAt: url, tinyVAE: tinyVAE)
            return
        }
        if tinyVAE {
            throw PipelineLoadError.unsupportedConfiguration(
                "the tiny autoencoder is resolved by size; open the pipeline with a size.")
        }

        guard let textEncoderPath = descriptor.components.textEncoder else {
            throw PipelineLoadError.missingComponent("text_encoder")
        }

        // Select transformer by mode.
        // Prefer the multi-function Transformer.aimodel. Only fall back to the
        // single-function Transformer_512.aimodel when the multi-function model is
        // absent.
        let transformer: CoreAIDiffusionModelFunction
        let transformerFnName: String
        switch resolvedMode {
        case .full, .tiled:
            guard let path = Self.resolveAsset(at: url, name: "Transformer") else {
                throw PipelineLoadError.missingComponent("Transformer")
            }
            transformer = CoreAIDiffusionModelFunction(modelURL: url.appendingPathComponent(path))
            transformerFnName = "main"
        case .half:
            if let path = Self.resolveAsset(at: url, name: "Transformer") {
                let candidate = CoreAIDiffusionModelFunction(
                    modelURL: url.appendingPathComponent(path))
                if try await candidate.hasFunction(named: "half") {
                    // Multi-function model: use its "half" function (img2img uses
                    // img2img_512_* from the same asset).
                    transformer = candidate
                    transformerFnName = "half"
                } else if let path512 = Self.resolveAsset(at: url, name: "Transformer_512") {
                    transformer = CoreAIDiffusionModelFunction(
                        modelURL: url.appendingPathComponent(path512))
                    transformerFnName = "main"
                } else {
                    // Full-only Transformer without a "half" function and no
                    // Transformer_512 — best effort with "main".
                    transformer = candidate
                    transformerFnName = "main"
                }
            } else if let path512 = Self.resolveAsset(at: url, name: "Transformer_512") {
                transformer = CoreAIDiffusionModelFunction(
                    modelURL: url.appendingPathComponent(path512))
                transformerFnName = "main"
            } else {
                throw PipelineLoadError.missingComponent("Transformer or Transformer_512")
            }
        case .auto:
            preconditionFailure("auto resolved above")
        }

        // Resolve how each reference grid reaches a traced graph.
        //
        // Prefer the multi-function transformer when available.
        let entrypointPrefix = resolvedMode == .half ? "img2img_512_" : "img2img_"
        let assetPrefix = resolvedMode == .half ? "Transformer_512_img2img" : "Transformer_img2img"
        var img2imgRoutes: [ReferenceGrid: Img2ImgRoute] = [:]
        for grid in ReferenceGrid.allCases {
            let entrypoint = "\(entrypointPrefix)\(grid.rawValue)"
            if try await transformer.hasFunction(named: entrypoint) {
                img2imgRoutes[grid] = Img2ImgRoute(function: transformer, entrypoint: entrypoint)
            } else if let path = Self.resolveAsset(at: url, name: "\(assetPrefix)_\(grid.rawValue)") {
                // Single-function: its own asset, traced at one concatenated sequence
                // length, always entered through "main".
                img2imgRoutes[grid] = Img2ImgRoute(
                    function: CoreAIDiffusionModelFunction(
                        modelURL: url.appendingPathComponent(path)),
                    entrypoint: "main")
            }
        }

        // Select decoder by mode (explicit name)
        let decoderName: String
        switch resolvedMode {
        case .full:
            guard let path = Self.resolveAsset(at: url, name: "VAEDecoder") else {
                throw PipelineLoadError.missingComponent("VAEDecoder")
            }
            decoderName = path
        case .half, .tiled:
            guard let path = Self.resolveAsset(at: url, name: "VAEDecoder_half") else {
                throw PipelineLoadError.missingComponent("VAEDecoder_half")
            }
            decoderName = path
        case .auto:
            preconditionFailure("auto resolved above")
        }

        let textEncoder = CoreAIDiffusionModelFunction(
            modelURL: url.appendingPathComponent(textEncoderPath))
        let decoder = CoreAIDiffusionModelFunction(
            modelURL: url.appendingPathComponent(decoderName))

        // Encoder for img2img (optional)
        let encoderName: String?
        switch resolvedMode {
        case .full:
            encoderName = descriptor.components.vaeEncoder
        case .half, .tiled:
            encoderName = Self.resolveAsset(at: url, name: "VAEEncoder_half")
        case .auto:
            preconditionFailure("auto resolved above")
        }
        let encoder: CoreAIDiffusionModelFunction?
        if let name = encoderName {
            encoder = CoreAIDiffusionModelFunction(modelURL: url.appendingPathComponent(name))
        } else {
            encoder = nil
        }

        // Load Qwen3 tokenizer
        let tokenizerDir = url.appendingPathComponent("tokenizer")
        let tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerDir)

        // Load VAE batch norm statistics
        let bnMean = Flux2Pipeline.loadNpyFloat32(url.appendingPathComponent("vae_bn_mean.npy"))
        let bnVar = Flux2Pipeline.loadNpyFloat32(url.appendingPathComponent("vae_bn_var.npy"))
        let bnEps = descriptor.batchNormEps ?? 1e-5

        // Delegating rather than assigning, because the resolution-named path
        // above delegates and an initializer cannot do both.
        self.init(
            descriptor: descriptor,
            mode: resolvedMode,
            transformer: transformer,
            img2imgRoutes: img2imgRoutes,
            textEncoder: textEncoder,
            decoder: decoder,
            encoder: encoder,
            transformerFunctionName: transformerFnName,
            tokenizer: tokenizer,
            batchNormMean: bnMean,
            batchNormVar: bnVar,
            batchNormEps: bnEps)
    }

    /// A bundle exported at one resolution, whose assets carry it in their
    /// names.
    ///
    /// Everything is explicit: the transformer, both VAEs and each img2img
    /// grid are their own assets, entered through `main`, so there is no mode
    /// to resolve and no entrypoint to guess.
    private init(
        from url: URL,
        descriptor: PipelineDescriptor,
        size: (width: Int, height: Int),
        tokenizerAt tokenizerRoot: URL,
        tinyVAE: Bool = false
    ) async throws {
        let suffix = "\(size.width)x\(size.height)"
        // A size's own VAEs, or the open ones (`VAEDecoder_open`, height and
        // width dynamic), which serve any size — or TAEF2's, when asked.
        let decoderName = tinyVAE ? "TinyDecoder" : "VAEDecoder"
        let encoderName = tinyVAE ? "TinyEncoder" : "VAEEncoder"
        // …or, failing both, the 512-pixel tile decoder (`VAEDecoder_half`)
        // run over the picture in tiles: the full VAE's output at any size,
        // memory bounded by the tile.
        var decodeMode: DecodeResolution = .full
        var decoderPath: String
        if let own = Self.resolveAsset(at: url, name: "\(decoderName)_\(suffix)")
            ?? Self.resolveAsset(at: url, name: "\(decoderName)_open") {
            decoderPath = own
        } else if !tinyVAE, let tile = Self.resolveAsset(at: url, name: "VAEDecoder_half") {
            decoderPath = tile
            decodeMode = .tiled
        } else {
            throw PipelineLoadError.missingComponent(
                "\(decoderName)_\(suffix), \(decoderName)_open or VAEDecoder_half")
        }
        guard let textEncoderPath = descriptor.components.textEncoder else {
            throw PipelineLoadError.missingComponent("text_encoder")
        }

        // The size's transformer is its own asset, entered through `main`, or
        // one entrypoint of a bundle — `Transformer_768x576+576x768.aimodel`,
        // several sizes' functions over one set of weights (`--bundle` at
        // export) — entered through `txt2img_<size>`, with its img2img grids
        // as `img2img_<size>_<grid>` in the same asset.
        let transformer: CoreAIDiffusionModelFunction
        let transformerEntry: String
        var routes: [ReferenceGrid: Img2ImgRoute] = [:]
        var twoReferenceRoutes: [ReferenceGrid: Img2ImgRoute] = [:]
        if let transformerPath = Self.resolveAsset(at: url, name: "Transformer_\(suffix)") {
            transformer = CoreAIDiffusionModelFunction(
                modelURL: url.appendingPathComponent(transformerPath))
            transformerEntry = "main"
            for grid in ReferenceGrid.allCases {
                if let path = Self.resolveAsset(
                    at: url, name: "Transformer_\(suffix)_img2img_\(grid.rawValue)")
                {
                    routes[grid] = Img2ImgRoute(
                        function: CoreAIDiffusionModelFunction(
                            modelURL: url.appendingPathComponent(path)),
                        entrypoint: "main")
                }
            }
        } else if let bundlePath = Self.bundleAsset(at: url, holding: suffix) {
            transformerEntry = "txt2img_\(suffix)"
            transformer = CoreAIDiffusionModelFunction(
                modelURL: url.appendingPathComponent(bundlePath), entrypoint: transformerEntry)
            let names = try await transformer.functionNames()
            guard names.contains(transformerEntry) else {
                throw PipelineLoadError.missingComponent(
                    "\(bundlePath) has no \(transformerEntry); it has \(names.joined(separator: ", "))")
            }
            for grid in ReferenceGrid.allCases {
                let entry = "img2img_\(suffix)_\(grid.rawValue)"
                if names.contains(entry) {
                    routes[grid] = Img2ImgRoute(function: transformer, entrypoint: entry)
                }
                let two = "img2img2_\(suffix)_\(grid.rawValue)"
                if names.contains(two) {
                    twoReferenceRoutes[grid] = Img2ImgRoute(function: transformer, entrypoint: two)
                }
            }
        } else if let shapedPath = Self.shapesAsset(at: url, holding: suffix) {
            // One trace specialised per token count (`--shapes` at export):
            // `Transformer_768x576+576x768_shapes`, entered as `main_n<tokens>`.
            // The transformer takes its positions as an input, so a size and
            // its other orientation are the same function, and each reference
            // picture is the same function at a longer count. An `_open`
            // asset is the trace with the dimension still open, entered as
            // `main` for every count.
            let grid = (width: size.width / 16, height: size.height / 16)
            let probe = CoreAIDiffusionModelFunction(modelURL: url.appendingPathComponent(shapedPath))
            let names = try await probe.functionNames()
            func entry(references: Int, of refGrid: ReferenceGrid?) -> String? {
                var tokens = grid.width * grid.height
                if let refGrid { tokens += Self.referenceTokens(refGrid, noise: grid) * references }
                if names.contains("main_n\(tokens)") { return "main_n\(tokens)" }
                return names == ["main"] ? "main" : nil
            }
            guard let text = entry(references: 0, of: nil) else {
                throw PipelineLoadError.missingComponent(
                    "\(shapedPath) has no shape for \(grid.width * grid.height) tokens; "
                        + "it has \(names.joined(separator: ", "))")
            }
            transformerEntry = text
            transformer = CoreAIDiffusionModelFunction(
                modelURL: url.appendingPathComponent(shapedPath), entrypoint: text)
            for refGrid in ReferenceGrid.allCases {
                if let one = entry(references: 1, of: refGrid) {
                    routes[refGrid] = Img2ImgRoute(function: transformer, entrypoint: one)
                }
                if let two = entry(references: 2, of: refGrid) {
                    twoReferenceRoutes[refGrid] = Img2ImgRoute(function: transformer, entrypoint: two)
                }
            }
        } else {
            throw PipelineLoadError.missingComponent(
                "Transformer_\(suffix), or a bundle or shapes transformer holding \(suffix)")
        }

        let encoderPath = Self.resolveAsset(at: url, name: "\(encoderName)_\(suffix)")
            ?? Self.resolveAsset(at: url, name: "\(encoderName)_open")
        let tokenizer = try await AutoTokenizer.from(
            modelFolder: tokenizerRoot.appendingPathComponent("tokenizer"))

        self.init(
            descriptor: descriptor,
            mode: decodeMode,
            transformer: transformer,
            img2imgRoutes: routes,
            img2img2Routes: twoReferenceRoutes,
            textEncoder: CoreAIDiffusionModelFunction(
                modelURL: url.appendingPathComponent(textEncoderPath)),
            decoder: CoreAIDiffusionModelFunction(
                modelURL: url.appendingPathComponent(decoderPath)),
            encoder: encoderPath.map {
                CoreAIDiffusionModelFunction(modelURL: url.appendingPathComponent($0))
            },
            transformerFunctionName: transformerEntry,
            tokenizer: tokenizer,
            // TAEF2 lives in the transformer's latent space: no statistics.
            batchNormMean: tinyVAE ? nil : Flux2Pipeline.loadNpyFloat32(
                url.appendingPathComponent("vae_bn_mean.npy")),
            batchNormVar: tinyVAE ? nil : Flux2Pipeline.loadNpyFloat32(
                url.appendingPathComponent("vae_bn_var.npy")),
            batchNormEps: descriptor.batchNormEps ?? 1e-5,
            pixelSize: size,
            usesTinyVAE: tinyVAE)
    }

    /// A bundle transformer that holds this size: `Transformer_<a>+<b>+….aimodel`
    /// whose `+`-separated sizes include `suffix`. Nil when there is none.
    static func bundleAsset(at url: URL, holding suffix: String) -> String? {
        multiSizeAsset(at: url, holding: suffix, kind: nil)
    }

    /// A shapes transformer that holds this size: `Transformer_<a>+<b>_shapes`
    /// (specialised per token count) or `Transformer_<a>+<b>_open` (the token
    /// dimension left open). Nil when there is none.
    static func shapesAsset(at url: URL, holding suffix: String) -> String? {
        multiSizeAsset(at: url, holding: suffix, kind: "shapes")
            ?? multiSizeAsset(at: url, holding: suffix, kind: "open")
            ?? resolveAsset(at: url, name: "Transformer_open")  // named after no size at all
    }

    /// A transformer named after several sizes joined with `+`, with the
    /// given `_kind` suffix (none for a bundle), whose sizes include `suffix`.
    private static func multiSizeAsset(at url: URL, holding suffix: String, kind: String?) -> String? {
        let names = (try? FileManager.default.contentsOfDirectory(atPath: url.path)) ?? []
        for name in names.sorted() {
            guard name.hasPrefix("Transformer_") else { continue }
            var middle = Substring(name.dropFirst("Transformer_".count))
            if middle.hasSuffix(".aimodelc") { middle = middle.dropLast(".aimodelc".count) }
            else if middle.hasSuffix(".aimodel") { middle = middle.dropLast(".aimodel".count) }
            else { continue }
            guard middle.contains("+") else { continue }
            // `768x576+576x768_shapes`: the kind rides after the last size.
            var sizes = middle.split(separator: "+")
            let last = sizes.removeLast()
            let lastParts = last.split(separator: "_", maxSplits: 1)
            let lastKind = lastParts.count == 2 ? String(lastParts[1]) : nil
            guard lastKind == kind else { continue }
            // An open trace runs at any token count, whatever sizes it was
            // named after.
            if kind == "open" { return name }
            sizes.append(lastParts[0])
            if sizes.contains(where: { $0 == suffix }) { return name }
        }
        return nil
    }

    /// How many tokens a reference picture adds at a grid, the way the run
    /// computes it: each side of the noise grid divided, never below one.
    static func referenceTokens(_ grid: ReferenceGrid, noise: (width: Int, height: Int)) -> Int {
        let divisor: Int
        switch grid {
        case .full: divisor = 1
        case .half: divisor = 2
        case .quarter: divisor = 4
        }
        return max(1, noise.width / divisor) * max(1, noise.height / divisor)
    }

    /// The one resolution a bundle was exported at, when it holds exactly one
    /// and no square assets to be ambiguous with.
    static func soleSize(at url: URL) -> (width: Int, height: Int)? {
        let names = (try? FileManager.default.contentsOfDirectory(atPath: url.path)) ?? []
        var sizes = Set<[Int]>()
        for name in names {
            // `Transformer_1024x768.aimodel`, but not
            // `Transformer_1024x768_img2img_half.aimodel`, which is the same size.
            guard name.hasPrefix("Transformer_"), name.hasSuffix(".aimodel") else { continue }
            let middle = name.dropFirst("Transformer_".count).dropLast(".aimodel".count)
            let parts = middle.split(separator: "x")
            guard parts.count == 2, let width = Int(parts[0]), let height = Int(parts[1]) else {
                continue
            }
            sizes.insert([width, height])
        }
        guard sizes.count == 1, let only = sizes.first else { return nil }
        return (width: only[0], height: only[1])
    }

    /// Resolve an asset name to a filename, checking for .aimodel or .aimodelc.
    private static func resolveAsset(at url: URL, name: String) -> String? {
        let resolved = ModelBundle.resolveAssetURL("\(name).aimodel", in: url)
        guard FileManager.default.fileExists(atPath: resolved.path) else { return nil }
        return resolved.lastPathComponent
    }

    /// Probe available assets and pick the highest quality mode.
    /// Priority: .full > .tiled > .half. Throws if no valid combination exists.
    private static func bestAvailableMode(
        at url: URL, descriptor: PipelineDescriptor
    ) throws -> DecodeResolution {
        let hasFullTransformer = descriptor.components.unet != nil
        let hasFullDecoder = descriptor.components.vaeDecoder != nil
        let hasHalfDecoder = resolveAsset(at: url, name: "VAEDecoder_half") != nil
        let hasHalfTransformer = resolveAsset(at: url, name: "Transformer_512") != nil

        if hasFullTransformer && hasFullDecoder { return .full }
        if hasFullTransformer && hasHalfDecoder { return .tiled }
        if hasHalfTransformer && hasHalfDecoder { return .half }
        throw PipelineLoadError.missingComponent(
            "No valid component combination found. Need Transformer+VAEDecoder, "
                + "Transformer+VAEDecoder_half, or Transformer_512+VAEDecoder_half.")
    }

    // MARK: - Npy Reader

    private static func loadNpyFloat32(_ url: URL) -> [Float]? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        guard data.count > 10,
            data[0] == 0x93, data[1] == 0x4E, data[2] == 0x55,
            data[3] == 0x4D, data[4] == 0x50, data[5] == 0x59
        else {
            return nil
        }
        let majorVersion = data[6]
        let headerLen: Int
        let headerStart: Int
        if majorVersion == 1 {
            headerLen = Int(data[8]) | (Int(data[9]) << 8)
            headerStart = 10
        } else {
            headerLen = Int(data[8]) | (Int(data[9]) << 8) | (Int(data[10]) << 16) | (Int(data[11]) << 24)
            headerStart = 12
        }
        let dataStart = headerStart + headerLen
        let rawData = data[dataStart...]
        return rawData.withUnsafeBytes { ptr in
            Array(ptr.bindMemory(to: Float32.self))
        }
    }
}
#endif  // canImport(CoreAI)
