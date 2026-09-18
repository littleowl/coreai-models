# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""CLI entry point for coreai.diffusion.export."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from coreai_models.diffusion.components import get_valid_components
from coreai_models.diffusion.models import get_pipeline_type
from coreai_models.diffusion.pipeline import DiffusionExportConfig, export_diffusion
from coreai_models.diffusion.presets import DEFAULT_COMPRESSION_PRESET
from coreai_models.model_registry import try_lookup_preset, try_lookup_preset_by_hf_id


def _default_output_dir() -> str:
    """Resolve exports/ relative to the workspace root (where pyproject.toml lives)."""
    d = Path(__file__).resolve().parent
    while d != d.parent:
        if (d / "pyproject.toml").exists() and (d / "python").exists():
            return str(d / "exports")
        d = d.parent
    return "exports"


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the diffusion export CLI."""
    parser = argparse.ArgumentParser(
        prog="coreai.diffusion.export",
        description="Export diffusion models to Core AI format. "
        "Accepts a registry short-name (e.g. flux2-klein-4b) or a HuggingFace model ID.",
    )
    parser.add_argument(
        "model",
        help="Registry short-name (e.g. flux2-klein-4b) or HuggingFace model ID",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for exported assets (default: <repo-root>/exports/)",
    )
    parser.add_argument(
        "--components",
        nargs="+",
        default=None,
        help="Components to export (default: all). "
        "SD 1.x/2.x: text_encoder unet vae_decoder vae_encoder. "
        "SD 3.x: text_encoder text_encoder_2 transformer vae_decoder. "
        "FLUX.2: transformer text_encoder vae_decoder vae_encoder.",
    )
    parser.add_argument(
        "--compute-precision",
        default=None,
        choices=["float16", "bfloat16", "float32"],
        help="Model precision for export. "
        "Required for raw HF IDs; resolved automatically for registry short-names.",
    )
    parser.add_argument(
        "--compression",
        default=None,
        help="Compression preset name, JSON config, or 'none' (see --list-presets)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )
    parser.add_argument(
        "--platform",
        default=None,
        choices=["iOS", "macOS"],
        help="Target platform. iOS defaults to 512 resolution, --single-function, and "
        "the half img2img reference grid; macOS defaults to 1024 and the full grid.",
    )
    parser.add_argument(
        "--resolution",
        default=None,
        type=str,
        help="Output image resolution: 512 or 1024 for the square exports, or WxH for any "
        "other — 1024x768, 768x1024, 1152x864. Both sides must be divisible by 16. "
        "Anything but the two square sizes implies --single-function, since a resolution "
        "that is not one of the multi-function asset's entrypoints has to be its own "
        "asset. Overrides the platform default.",
    )
    parser.add_argument(
        "--bundle",
        default=None,
        help=(
            "Sizes, comma-separated (1152x864,864x1152): one multi-function transformer "
            "holding text-to-image and image-to-image at every size, sharing weights, plus "
            "each size's VAEs and the text encoder. Named Transformer_<sizes joined by +>. "
            "The other components are exported as under --single-function."
        ),
    )
    parser.add_argument(
        "--shapes",
        default=None,
        help=(
            "Sizes, comma-separated, like --bundle, but as one trace of the transformer with "
            "the token dimension open, specialised per distinct token count with enumerated "
            "shapes (main_n<tokens>) instead of traced once per function. Named "
            "Transformer_<sizes joined by +>_shapes. Same companions as --bundle."
        ),
    )
    parser.add_argument(
        "--open-shape",
        action="store_true",
        help=(
            "Leave the spatial dimension open instead of enumerating shapes. Alone: exports the "
            "size-free set — Transformer_open (token axis open), VAEDecoder_open and "
            "VAEEncoder_open (height and width open) — plus the text encoder; one function "
            "`main` each, any size whose sides are multiples of 16. With --shapes: the "
            "transformer named after those sizes, dimension open (Transformer_…_open)."
        ),
    )
    parser.add_argument(
        "--references",
        type=int,
        default=1,
        choices=[1, 2],
        help=(
            "With --bundle: 2 adds an img2img2_<size>_<grid> entrypoint per size that takes two "
            "reference images, and names the bundle …+2ref. With --shapes: 2 adds the "
            "two-reference token count per size. Default 1."
        ),
    )
    parser.add_argument(
        "--reference-grid",
        default="half",
        choices=["full", "half", "quarter"],
        help="Which img2img reference grid a WxH export includes. The reference grid is "
        "the noise grid divided on both sides, so fewer tokens means faster and lighter "
        "with coarser guidance. Default: half.",
    )
    parser.add_argument(
        "--low-memory",
        action="store_true",
        help="Include half-resolution VAEs for tiled decode. Only applies with "
        "--single-function; the half VAEs are always included otherwise.",
    )
    parser.add_argument(
        "--single-function",
        action="store_true",
        help="Export the FLUX.2 transformer as one asset per resolution/grid instead of a "
        "single multi-function .aimodel. Lower peak memory, but ~2 GB per asset and only "
        "the resolution/grid you export. Default: off (one multi-function asset with 8 "
        "entrypoints sharing weights). Needed to name a per-resolution transformer in "
        "--components; implied by --platform iOS.",
    )
    parser.add_argument(
        "--experimental",
        action="store_true",
        help="Allow exporting models without a registry preset. Requires --compute-precision.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved export config and exit without exporting",
    )
    parser.add_argument(
        "--include-debug-info",
        action="store_true",
        help=(
            "Embed debug information in the exported .aimodel for debugging a conversion. "
            "Default: off, which embeds minimum debug information and makes the "
            "exported asset smaller."
        ),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )
    return parser


def _warn(message: str) -> None:
    """Note a flag whose intent is already met, or met elsewhere, rather than failing.

    Reserved for combinations that are redundant or resolved at runtime. A combination
    that genuinely cannot be honoured uses ``parser.error`` instead.
    """
    logging.getLogger(__name__).warning(message)


def _is_hf_id(model: str) -> bool:
    return "/" in model


def _resolution_size(resolution: str | None, parser: Any) -> tuple[int, int] | None:
    """`--resolution` as a (width, height), or None for the two square exports.

    512 and 1024 are the multi-function asset's own entrypoints and keep every
    name they have always had. Anything else — including a square size the
    model was never traced at — becomes its own set of assets, named after it.
    """
    if resolution is None:
        return None
    text = str(resolution).lower()
    if "x" in text:
        try:
            width, height = (int(part) for part in text.split("x", 1))
        except ValueError:
            parser.error(f"--resolution {resolution} is not a number or WxH.")
        return (width, height)
    try:
        square = int(text)
    except ValueError:
        parser.error(f"--resolution {resolution} is not a number or WxH.")
    return None if square in (512, 1024) else (square, square)


def main() -> None:
    """Main entry point for the diffusion export CLI."""
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # --- Registry resolution ---
    hf_model_id = args.model
    compression = args.compression
    compute_precision = args.compute_precision
    output_dir = args.output_dir or _default_output_dir()

    preset = None
    if not _is_hf_id(args.model):
        preset = try_lookup_preset(args.model, model_type="diffusion")
        if preset is None:
            parser.error(
                f"'{args.model}' is not a registered short-name and doesn't look like a "
                "HuggingFace ID (expected 'org/model'). "
                "Run `uv run coreai.model.registry --list-models --type diffusion` to see options."
            )
    else:
        preset = try_lookup_preset_by_hf_id(args.model, model_type="diffusion")

    if preset is not None:
        hf_model_id = preset.hf_id
        if compression is None and preset.compression:
            compression = preset.compression
        if compute_precision is None and preset.compute_precision:
            compute_precision = preset.compute_precision
    elif _is_hf_id(args.model) and not args.experimental:
        parser.error(
            f"'{args.model}' has no registry preset. "
            "Pass --experimental to try exporting it anyway "
            "(requires --compute-precision).\n"
            "See models/README.md for supported models."
        )

    if compute_precision is None:
        parser.error(
            f"--compute-precision is required for '{args.model}' "
            "(no registry preset found). "
            "Pass --compute-precision float16|bfloat16|float32 explicitly.\n"
            "See models/README.md for more information."
        )

    compression = compression if compression is not None else DEFAULT_COMPRESSION_PRESET

    pipeline_type = get_pipeline_type(hf_model_id)

    # A resolution's components have to exist before --components is checked
    # against the registry, so the size is read and registered first.
    size = _resolution_size(args.resolution, parser)
    if size is not None and pipeline_type == "flux2":
        from coreai_models.diffusion.components import register_flux2_resolution

        try:
            register_flux2_resolution(*size)
        except ValueError as why:
            parser.error(str(why))

    bundle: list[tuple[int, int]] | None = None
    if args.bundle is not None and pipeline_type == "flux2":
        if args.resolution is not None:
            parser.error("--bundle and --resolution are two ways of naming sizes; use one.")
        if args.platform:
            parser.error("--bundle picks its own components; do not combine it with --platform.")
        bundle = []
        for part in str(args.bundle).split(","):
            found = _resolution_size(part.strip(), parser)
            if found is None:
                parser.error(f"--bundle takes WxH sizes; {part!r} is a square preset.")
            bundle.append(found)
        from coreai_models.diffusion.components import register_flux2_bundle

        try:
            bundle_keys = register_flux2_bundle(bundle, grids=(args.reference_grid,), references=args.references)
        except ValueError as why:
            parser.error(str(why))
        args.single_function = True
        if not args.components:
            args.components = [bundle_keys[0], "text_encoder", *bundle_keys[1:]]

    if args.open_shape and args.shapes is None and pipeline_type == "flux2":
        if args.resolution is not None or args.bundle is not None:
            parser.error("--open-shape alone is the size-free set; do not name sizes with it.")
        if args.platform:
            parser.error("--open-shape picks its own components; do not combine it with --platform.")
        from coreai_models.diffusion.components import register_flux2_open

        open_keys = register_flux2_open(grids=(args.reference_grid,))
        args.single_function = True
        if not args.components:
            args.components = [*open_keys, "text_encoder"]

    if args.shapes is not None and pipeline_type == "flux2":
        if args.resolution is not None or args.bundle is not None:
            parser.error("--shapes, --bundle and --resolution are three ways of naming sizes; use one.")
        if args.platform:
            parser.error("--shapes picks its own components; do not combine it with --platform.")
        shaped: list[tuple[int, int]] = []
        for part in str(args.shapes).split(","):
            found = _resolution_size(part.strip(), parser)
            if found is None:
                parser.error(f"--shapes takes WxH sizes; {part!r} is a square preset.")
            shaped.append(found)
        from coreai_models.diffusion.components import register_flux2_shapes

        try:
            shape_keys = register_flux2_shapes(
                shaped,
                grids=(args.reference_grid,),
                references=args.references,
                enumerate_shapes=not args.open_shape,
            )
        except ValueError as why:
            parser.error(str(why))
        args.single_function = True
        if not args.components:
            args.components = [shape_keys[0], "text_encoder", *shape_keys[1:]]

    if args.components and args.platform:
        parser.error("Cannot specify both --components and --platform. Use only one.")

    # iOS forces single-function: multi-function saves disk but raises peak memory.
    # Do not use platform=iOS if you want multi-function.
    multifunction = not args.single_function
    if args.platform == "iOS":
        multifunction = False

    resolution: int | None = None
    if size is None and args.resolution is not None:
        resolution = int(str(args.resolution))

    # Warn of unused flags with multifunction.
    if args.platform is None and size is None:
        if args.resolution is not None:
            _warn(
                "--resolution only applies with --platform; every component is exported without it."
            )
        if args.low_memory:
            _warn(
                "--low-memory only applies with --platform; every component is exported without it."
            )
    elif multifunction and size is None:
        if args.resolution is not None:
            _warn(
                f"--resolution {args.resolution} is not used without --single-function: one "
                "asset holds both resolutions. Select at runtime with the pipeline's "
                "--decode-resolution, or export with --single-function."
            )
        if args.low_memory:
            _warn(
                "--low-memory is redundant without --single-function: the half VAEs are "
                "always included."
            )

    if args.components:
        valid = get_valid_components(pipeline_type, multifunction=multifunction)
        invalid = [c for c in args.components if c not in valid]
        if invalid:
            # Per-resolution names only exist under --single-function.
            hint = (
                " Per-resolution/grid names (e.g. transformer_512, "
                "transformer_img2img_full) require --single-function."
                if multifunction
                else ""
            )
            parser.error(
                f"Invalid components for {pipeline_type}: {invalid}. Valid choices: {valid}.{hint}"
            )

    if size is not None and pipeline_type == "flux2":
        from coreai_models.diffusion.components import flux2_component_names

        width, height = size
        if multifunction:
            parser.error(
                f"--resolution {width}x{height} needs --single-function: the multi-function "
                "asset carries the two square resolutions it was traced at and nothing else."
            )
        if not args.components:
            names = flux2_component_names(width, height)
            grid = args.reference_grid
            args.components = [
                names["transformer"],
                names[f"transformer_img2img_{grid}"],
                "text_encoder",
                names["vae_decoder"],
                names["vae_encoder"],
            ]

    # Platform-based component selection (FLUX.2 only)
    target_components: list[str] | None = None
    if args.platform and pipeline_type == "flux2" and size is None:
        # Resolve effective resolution: --resolution overrides platform default
        if resolution is None:
            resolution = 512 if args.platform == "iOS" else 1024

        # Single-function img2img needs one asset per reference grid. Each grid is a
        # different concatenated sequence length, so a different trace, and separate
        # assets do not share weights (~2 GB each). Export exactly one.
        #
        # iOS takes the cheaper grid: it is memory-constrained, which is the same reason
        # it forces single-function. `half` is a quarter of full's reference tokens
        # (256 vs 1024 at 512px). Override with --components.
        img2img_grid = "half" if args.platform == "iOS" else "full"

        if multifunction:
            # Multi-function mode: single transformer has all variants
            target_components = [
                "transformer",
                "text_encoder",
                "vae_decoder",
                "vae_encoder",
            ]
            # Always include half VAEs for img2img reference encoding
            for half in ["vae_decoder_half", "vae_encoder_half"]:
                if half not in target_components:
                    target_components.append(half)
        elif resolution == 512:
            target_components = [
                "transformer_512",
                f"transformer_512_img2img_{img2img_grid}",
                "text_encoder",
                "vae_decoder_half",
                "vae_encoder_half",
            ]
        else:
            target_components = [
                "transformer",
                f"transformer_img2img_{img2img_grid}",
                "text_encoder",
                "vae_decoder",
                "vae_encoder",
            ]

            # --low-memory adds half VAEs for tiled decode
            if args.low_memory:
                for half in ["vae_decoder_half", "vae_encoder_half"]:
                    if half not in target_components:
                        target_components.append(half)

    config = DiffusionExportConfig(
        hf_model_id=hf_model_id,
        output_dir=output_dir,
        components=args.components or target_components,
        compute_precision=compute_precision,
        compression=compression,
        overwrite=args.overwrite,
        include_debug_info=args.include_debug_info,
        multifunction=multifunction,
    )

    if args.dry_run:
        print("Dry run — resolved export config:")
        print(f"  model:              {config.hf_model_id}")
        print(f"  compression:        {config.compression}")
        print(f"  compute_precision:  {config.compute_precision}")
        print(f"  output_dir:         {config.output_dir}")
        if config.components:
            print(f"  components:         {', '.join(config.components)}")
        else:
            print("  components:        all")
        print(f"  multifunction:     {config.multifunction}")
        print(f"  overwrite:         {config.overwrite}")
        print(f"  include_debug_info: {config.include_debug_info}")
        return

    try:
        results = export_diffusion(config)
        failed = [k for k, v in results.items() if "FAILED" in str(v)]
        if failed:
            logging.getLogger(__name__).error(f"Failed components: {failed}")
            sys.exit(1)
        print(f"Export complete: {config.output_dir}")
    except Exception as e:
        logging.getLogger(__name__).error(f"Export failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
