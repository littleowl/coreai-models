# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
Diffusion component specifications and torch wrappers.

Each diffusion pipeline is made of independent components (text encoder, UNet,
VAE decoder, VAE encoder) that are exported separately.  A ComponentSpec
captures everything needed to export one component: its I/O names, a thin
torch.nn.Module wrapper that normalises the HF output, and a factory for
dummy inputs.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch

from coreai_models.diffusion.flux2 import (
    dummy_flux2_transformer_img2img2_at,
    Flux2TextEncoderWrapper,
    Flux2TransformerWrapper,
    Flux2VAEDecoderWrapper,
    Flux2VAEEncoderWrapper,
    dummy_flux2_text_encoder,
    dummy_flux2_transformer,
    dummy_flux2_transformer_512,
    dummy_flux2_transformer_img2img_512_full,
    dummy_flux2_transformer_img2img_512_half,
    dummy_flux2_transformer_img2img_512_quarter,
    dummy_flux2_transformer_img2img_full,
    dummy_flux2_transformer_img2img_half,
    dummy_flux2_transformer_img2img_quarter,
    dummy_flux2_vae_decoder,
    dummy_flux2_vae_decoder_half,
    dummy_flux2_transformer_at,
    dummy_flux2_transformer_img2img_at,
    dummy_flux2_vae_decoder_at,
    dummy_flux2_vae_encoder_at,
    dummy_flux2_vae_encoder,
    dummy_flux2_vae_encoder_half,
    flux2_token_counts,
    flux2_transformer_dynamic_shapes,
    flux2_vae_decoder_dynamic_shapes,
    flux2_vae_encoder_dynamic_shapes,
    flux2_transformer_static_shapes,
    grid_for,
)
from coreai_models.diffusion.wan import (
    WanTextEncoderWrapper,
    WanTransformerWrapper,
    WanVAEDecoderWrapper,
    dummy_wan_text_encoder,
    dummy_wan_transformer,
    dummy_wan_vae_decoder,
    wan_transformer_dynamic_shapes,
)

# ---------------------------------------------------------------------------
# Torch wrappers — thin adapters that extract the tensor we need from the
# HuggingFace model's rich output objects.
# ---------------------------------------------------------------------------


class TextEncoderWrapper(torch.nn.Module):
    def __init__(self, text_encoder: torch.nn.Module) -> None:
        super().__init__()
        self.model = text_encoder

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.model(input_ids).last_hidden_state)


class TextEncoderWithPooledWrapper(torch.nn.Module):
    """Returns (last_hidden_state, pooled). Used by SD3 CLIP-L and CLIP-G."""

    def __init__(self, text_encoder: torch.nn.Module) -> None:
        super().__init__()
        self.model = text_encoder
        # CLIPTextModelWithProjection emits text_embeds; CLIPTextModel emits pooler_output.
        self._use_text_embeds = "WithProjection" in type(text_encoder).__name__

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.model(input_ids)
        pooled = out.text_embeds if self._use_text_embeds else out.pooler_output
        return out.last_hidden_state, pooled


class SD3TransformerWrapper(torch.nn.Module):
    def __init__(self, transformer: torch.nn.Module) -> None:
        super().__init__()
        self.model: Any = transformer

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor,
    ) -> torch.Tensor:
        return cast(
            torch.Tensor,
            self.model(
                hidden_states=hidden_states.contiguous(),
                encoder_hidden_states=encoder_hidden_states.contiguous(),
                pooled_projections=pooled_projections.contiguous(),
                timestep=timestep,
            ).sample,
        )


class UNetWrapper(torch.nn.Module):
    def __init__(self, unet: torch.nn.Module) -> None:
        super().__init__()
        self.model = unet
        _patch_nearest_upsample(self.model)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return cast(torch.Tensor, self.model(sample, timestep, encoder_hidden_states).sample)


def _patch_nearest_upsample(module: torch.nn.Module) -> None:
    """Replace nearest-neighbor interpolate with repeat_interleave in Upsample2D.

    MPSGraph's segmenter rejects coreai.interpolate with nearest_neighbor mode,
    routing those ops to the BNNS (CPU) backend. This causes two problems:
    1. Mixed-backend execution is unsupported for this op.
    2. Performance: cross-backend data copies (GPU→CPU→GPU) at every upsample
       boundary. Full GPU residency avoids these copies entirely.

    repeat_interleave is mathematically equivalent to nearest-neighbor interpolation
    for integer scale factors and traces to ops that MPSGraph accepts.

    This patch should be kept even after the framework fix ships, because full MPSGraph
    residency is important for inference performance.
    """
    from diffusers.models.upsampling import Upsample2D

    for mod in module.modules():
        if isinstance(mod, Upsample2D):
            original_forward = mod.forward

            def _patched_forward(hidden_states, output_size=None, _orig=original_forward, _mod=mod):
                # Skip the interpolate call — do repeat_interleave instead
                if _mod.use_conv_transpose:
                    return _orig(hidden_states, output_size)

                # Only handles 2× upsample (all current diffusion models use this)
                scale = getattr(_mod, "scale_factor", 2)
                assert scale == 2, (
                    f"_patch_nearest_upsample only supports scale_factor=2, got {scale}"
                )

                dtype = hidden_states.dtype
                if dtype == torch.bfloat16:
                    hidden_states = hidden_states.to(torch.float32)

                if hidden_states.shape[0] >= 64:
                    hidden_states = hidden_states.contiguous()

                # Nearest-neighbor 2x upsample via repeat (only if interpolate is enabled)
                if getattr(_mod, "interpolate", True):
                    hidden_states = hidden_states.repeat_interleave(2, dim=-1).repeat_interleave(
                        2, dim=-2
                    )

                if dtype == torch.bfloat16:
                    hidden_states = hidden_states.to(dtype)

                if _mod.use_conv:
                    if getattr(_mod, "name", "conv") == "conv":
                        hidden_states = _mod.conv(hidden_states)
                    else:
                        hidden_states = _mod.Conv2d_0(hidden_states)

                return hidden_states

            mod.forward = _patched_forward


class VAEDecoderWrapper(torch.nn.Module):
    def __init__(self, vae: torch.nn.Module) -> None:
        super().__init__()
        self.vae: Any = vae
        _patch_nearest_upsample(self.vae.decoder)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.vae.decode(z).sample)


class VAEEncoderWrapper(torch.nn.Module):
    def __init__(self, vae: torch.nn.Module) -> None:
        super().__init__()
        self.vae: Any = vae

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.vae.encode(x).latent_dist.parameters)


# ---------------------------------------------------------------------------
# ComponentSpec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComponentSpec:
    asset_name: str
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    wrapper_fn: Callable
    dummy_fn: Callable
    quantizable: bool = False
    dynamic_shapes_fn: Callable | None = None


@dataclass(frozen=True)
class FunctionVariant:
    """A single named function variant within a multi-function .aimodel."""

    name: str
    dummy_fn: Callable


@dataclass(frozen=True)
class MultiFunctionComponentSpec:
    """A component exported as multiple named functions sharing one set of weights.

    Used when the same model architecture needs different input shapes (e.g.
    txt2img at 1024×1024 vs 512×512, img2img at various reference resolutions).
    All functions share weights; disk size equals one copy.
    """

    asset_name: str
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    wrapper_fn: Callable
    functions: tuple[FunctionVariant, ...]
    quantizable: bool = True


@dataclass(frozen=True)
class EnumeratedComponentSpec:
    """A component traced once with a dimension left open, then specialised
    for a list of static shapes (`AIProgram.set_static_shape_config`).

    The other way to get several shapes into one asset: where a
    `MultiFunctionComponentSpec` traces the model once per shape and names
    each trace, this traces it once and hands the compiler the shapes, which
    it exposes as `main_<label>`. `static_shapes_fn` returns that table given
    the pipeline; None leaves the dimension open in the asset, for a runtime
    that can take it.
    """

    asset_name: str
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    wrapper_fn: Callable
    dummy_fn: Callable
    dynamic_shapes_fn: Callable
    static_shapes_fn: Callable | None
    quantizable: bool = True


# ---------------------------------------------------------------------------
# Dummy-input factories — build reference tensors for torch.export
# ---------------------------------------------------------------------------


def _model_dtype(pipe: Any) -> torch.dtype:
    """Infer the dtype from the pipeline's denoiser weights (UNet or transformer)."""
    denoiser = getattr(pipe, "unet", None) or pipe.transformer
    return cast(torch.dtype, next(denoiser.parameters()).dtype)


def _dummy_text_encoder(pipe: Any, batch_size: int = 2) -> tuple[torch.Tensor, ...]:
    return (torch.zeros(1, 77, dtype=torch.long),)


def _dummy_unet(pipe: Any, batch_size: int = 2) -> tuple[torch.Tensor, ...]:
    cfg = pipe.unet.config
    dtype = _model_dtype(pipe)
    return (
        torch.randn(batch_size, cfg.in_channels, cfg.sample_size, cfg.sample_size, dtype=dtype),
        torch.tensor([999.0] * batch_size, dtype=dtype),
        torch.randn(batch_size, 77, cfg.cross_attention_dim, dtype=dtype),
    )


def _dummy_vae_decoder(pipe: Any, batch_size: int = 2) -> tuple[torch.Tensor, ...]:
    latent_ch = pipe.vae.config.latent_channels
    size = (
        pipe.unet.config.sample_size
        if hasattr(pipe, "unet") and pipe.unet is not None
        else pipe.transformer.config.sample_size
    )
    dtype = next(pipe.vae.parameters()).dtype
    return (torch.randn(1, latent_ch, size, size, dtype=dtype),)


def _dummy_vae_encoder(pipe: Any, batch_size: int = 2) -> tuple[torch.Tensor, ...]:
    size = (
        pipe.unet.config.sample_size
        if hasattr(pipe, "unet") and pipe.unet is not None
        else pipe.transformer.config.sample_size
    )
    dtype = _model_dtype(pipe)
    return (torch.randn(1, 3, size * 8, size * 8, dtype=dtype),)


def _dummy_sd3_transformer(pipe: Any, batch_size: int = 2) -> tuple[torch.Tensor, ...]:
    cfg = pipe.transformer.config
    dtype = _model_dtype(pipe)
    return (
        torch.randn(batch_size, cfg.in_channels, cfg.sample_size, cfg.sample_size, dtype=dtype),
        torch.tensor([999.0] * batch_size, dtype=dtype),
        torch.randn(batch_size, 154, cfg.joint_attention_dim, dtype=dtype),
        torch.randn(batch_size, cfg.pooled_projection_dim, dtype=dtype),
    )


# ---------------------------------------------------------------------------
# Component registries
# ---------------------------------------------------------------------------

SD_COMPONENTS: dict[str, ComponentSpec] = {
    "text_encoder": ComponentSpec(
        asset_name="TextEncoder",
        input_names=("input_ids",),
        output_names=("last_hidden_state",),
        wrapper_fn=lambda p: TextEncoderWrapper(p.text_encoder),
        dummy_fn=_dummy_text_encoder,
        quantizable=True,
    ),
    "unet": ComponentSpec(
        asset_name="Unet",
        input_names=("sample", "timestep", "encoder_hidden_states"),
        output_names=("noise_pred",),
        wrapper_fn=lambda p: UNetWrapper(p.unet),
        dummy_fn=_dummy_unet,
        quantizable=True,
    ),
    "vae_decoder": ComponentSpec(
        asset_name="VAEDecoder",
        input_names=("z",),
        output_names=("image",),
        wrapper_fn=lambda p: VAEDecoderWrapper(p.vae),
        dummy_fn=_dummy_vae_decoder,
    ),
    "vae_encoder": ComponentSpec(
        asset_name="VAEEncoder",
        input_names=("image",),
        output_names=("latent_params",),
        wrapper_fn=lambda p: VAEEncoderWrapper(p.vae),
        dummy_fn=_dummy_vae_encoder,
    ),
}

ALL_SD_COMPONENTS: list[str] = list(SD_COMPONENTS.keys())

FLUX2_COMPONENTS: dict[str, ComponentSpec | MultiFunctionComponentSpec | EnumeratedComponentSpec] = {
    "transformer": ComponentSpec(
        asset_name="Transformer",
        input_names=(
            "hidden_states",
            "encoder_hidden_states",
            "timestep",
            "guidance",
            "img_ids",
            "txt_ids",
        ),
        output_names=("output",),
        wrapper_fn=lambda p: Flux2TransformerWrapper(p.transformer),
        dummy_fn=dummy_flux2_transformer,
        quantizable=True,
    ),
    "transformer_img2img_full": ComponentSpec(
        asset_name="Transformer_img2img_full",
        input_names=(
            "hidden_states",
            "encoder_hidden_states",
            "timestep",
            "guidance",
            "img_ids",
            "txt_ids",
        ),
        output_names=("output",),
        wrapper_fn=lambda p: Flux2TransformerWrapper(p.transformer),
        dummy_fn=dummy_flux2_transformer_img2img_full,
        quantizable=True,
    ),
    "transformer_img2img_half": ComponentSpec(
        asset_name="Transformer_img2img_half",
        input_names=(
            "hidden_states",
            "encoder_hidden_states",
            "timestep",
            "guidance",
            "img_ids",
            "txt_ids",
        ),
        output_names=("output",),
        wrapper_fn=lambda p: Flux2TransformerWrapper(p.transformer),
        dummy_fn=dummy_flux2_transformer_img2img_half,
        quantizable=True,
    ),
    "transformer_img2img_quarter": ComponentSpec(
        asset_name="Transformer_img2img_quarter",
        input_names=(
            "hidden_states",
            "encoder_hidden_states",
            "timestep",
            "guidance",
            "img_ids",
            "txt_ids",
        ),
        output_names=("output",),
        wrapper_fn=lambda p: Flux2TransformerWrapper(p.transformer),
        dummy_fn=dummy_flux2_transformer_img2img_quarter,
        quantizable=True,
    ),
    "transformer_512": ComponentSpec(
        asset_name="Transformer_512",
        input_names=(
            "hidden_states",
            "encoder_hidden_states",
            "timestep",
            "guidance",
            "img_ids",
            "txt_ids",
        ),
        output_names=("output",),
        wrapper_fn=lambda p: Flux2TransformerWrapper(p.transformer),
        dummy_fn=dummy_flux2_transformer_512,
        quantizable=True,
    ),
    "transformer_512_img2img_full": ComponentSpec(
        asset_name="Transformer_512_img2img_full",
        input_names=(
            "hidden_states",
            "encoder_hidden_states",
            "timestep",
            "guidance",
            "img_ids",
            "txt_ids",
        ),
        output_names=("output",),
        wrapper_fn=lambda p: Flux2TransformerWrapper(p.transformer),
        dummy_fn=dummy_flux2_transformer_img2img_512_full,
        quantizable=True,
    ),
    "transformer_512_img2img_half": ComponentSpec(
        asset_name="Transformer_512_img2img_half",
        input_names=(
            "hidden_states",
            "encoder_hidden_states",
            "timestep",
            "guidance",
            "img_ids",
            "txt_ids",
        ),
        output_names=("output",),
        wrapper_fn=lambda p: Flux2TransformerWrapper(p.transformer),
        dummy_fn=dummy_flux2_transformer_img2img_512_half,
        quantizable=True,
    ),
    "transformer_512_img2img_quarter": ComponentSpec(
        asset_name="Transformer_512_img2img_quarter",
        input_names=(
            "hidden_states",
            "encoder_hidden_states",
            "timestep",
            "guidance",
            "img_ids",
            "txt_ids",
        ),
        output_names=("output",),
        wrapper_fn=lambda p: Flux2TransformerWrapper(p.transformer),
        dummy_fn=dummy_flux2_transformer_img2img_512_quarter,
        quantizable=True,
    ),
    "text_encoder": ComponentSpec(
        asset_name="TextEncoder",
        input_names=("input_ids", "attention_mask"),
        output_names=("hidden_states",),
        wrapper_fn=lambda p: Flux2TextEncoderWrapper(p.text_encoder),
        dummy_fn=dummy_flux2_text_encoder,
        quantizable=True,
    ),
    "vae_decoder": ComponentSpec(
        asset_name="VAEDecoder",
        input_names=("z",),
        output_names=("image",),
        wrapper_fn=lambda p: Flux2VAEDecoderWrapper(p.vae),
        dummy_fn=dummy_flux2_vae_decoder,
    ),
    "vae_decoder_half": ComponentSpec(
        asset_name="VAEDecoder_half",
        input_names=("z",),
        output_names=("image",),
        wrapper_fn=lambda p: Flux2VAEDecoderWrapper(p.vae),
        dummy_fn=dummy_flux2_vae_decoder_half,
    ),
    "vae_encoder": ComponentSpec(
        asset_name="VAEEncoder",
        input_names=("image",),
        output_names=("latent_params",),
        wrapper_fn=lambda p: Flux2VAEEncoderWrapper(p.vae),
        dummy_fn=dummy_flux2_vae_encoder,
    ),
    "vae_encoder_half": ComponentSpec(
        asset_name="VAEEncoder_half",
        input_names=("image",),
        output_names=("latent_params",),
        wrapper_fn=lambda p: Flux2VAEEncoderWrapper(p.vae),
        dummy_fn=dummy_flux2_vae_encoder_half,
    ),
}

ALL_FLUX2_COMPONENTS: list[str] = list(FLUX2_COMPONENTS.keys())


# ---------------------------------------------------------------------------
# Components at any resolution
# ---------------------------------------------------------------------------

REFERENCE_GRIDS = ("full", "half", "quarter")


def flux2_component_names(width: int, height: int) -> dict[str, str]:
    """What the components for one pixel size are called.

    A resolution's assets carry it in their names — `Transformer_1024x768`,
    `VAEDecoder_1024x768` — so a bundle can hold several and the pipeline can
    pick one by name. The square 512 and 1024 exports keep the names they
    always had, so nothing that reads an existing bundle changes.
    """
    size = f"{width}x{height}"
    names = {
        "transformer": f"transformer_{size}",
        "vae_decoder": f"vae_decoder_{size}",
        "vae_encoder": f"vae_encoder_{size}",
    }
    for grid in REFERENCE_GRIDS:
        names[f"transformer_img2img_{grid}"] = f"transformer_{size}_img2img_{grid}"
    return names


def register_flux2_resolution(width: int, height: int) -> list[str]:
    """Adds one pixel size's components to the registry; returns their keys.

    Every entry is the same architecture traced at a different sequence
    length, so this costs nothing until something asks for one.
    """
    grid_for(width, height)  # refuses a size the model cannot patch
    size = f"{width}x{height}"
    keys: list[str] = []

    transformer_key = f"transformer_{size}"
    if transformer_key not in FLUX2_COMPONENTS:
        FLUX2_COMPONENTS[transformer_key] = ComponentSpec(
            asset_name=f"Transformer_{size}",
            input_names=_FLUX2_TRANSFORMER_INPUT_NAMES,
            output_names=("output",),
            wrapper_fn=lambda p: Flux2TransformerWrapper(p.transformer),
            dummy_fn=dummy_flux2_transformer_at(width, height),
            quantizable=True,
        )
    keys.append(transformer_key)

    for grid in REFERENCE_GRIDS:
        key = f"transformer_{size}_img2img_{grid}"
        if key not in FLUX2_COMPONENTS:
            FLUX2_COMPONENTS[key] = ComponentSpec(
                asset_name=f"Transformer_{size}_img2img_{grid}",
                input_names=_FLUX2_TRANSFORMER_INPUT_NAMES,
                output_names=("output",),
                wrapper_fn=lambda p: Flux2TransformerWrapper(p.transformer),
                dummy_fn=dummy_flux2_transformer_img2img_at(width, height, grid),
                quantizable=True,
            )
        keys.append(key)

    decoder_key = f"vae_decoder_{size}"
    if decoder_key not in FLUX2_COMPONENTS:
        FLUX2_COMPONENTS[decoder_key] = ComponentSpec(
            asset_name=f"VAEDecoder_{size}",
            input_names=("z",),
            output_names=("image",),
            wrapper_fn=lambda p: Flux2VAEDecoderWrapper(p.vae),
            dummy_fn=dummy_flux2_vae_decoder_at(width, height),
        )
    keys.append(decoder_key)

    encoder_key = f"vae_encoder_{size}"
    if encoder_key not in FLUX2_COMPONENTS:
        FLUX2_COMPONENTS[encoder_key] = ComponentSpec(
            asset_name=f"VAEEncoder_{size}",
            input_names=("image",),
            output_names=("latent_params",),
            wrapper_fn=lambda p: Flux2VAEEncoderWrapper(p.vae),
            dummy_fn=dummy_flux2_vae_encoder_at(width, height),
        )
    keys.append(encoder_key)

    ALL_FLUX2_COMPONENTS[:] = list(FLUX2_COMPONENTS.keys())
    return keys


def flux2_bundle_tag(sizes: Sequence[tuple[int, int]]) -> str:
    """The name a bundle of sizes goes by: `1152x864+864x1152`."""
    return "+".join(f"{w}x{h}" for w, h in sizes)


def register_flux2_bundle(
    sizes: Sequence[tuple[int, int]], grids: Sequence[str] = ("half",), references: int = 1
) -> list[str]:
    """One transformer for several pixel sizes: `Transformer_<w>x<h>+<w>x<h>`.

    A multi-function asset with a text-to-image entrypoint and one
    image-to-image entrypoint per reference grid for every size, all sharing
    one set of weights — so a landscape and its portrait, or a whole tier,
    cost one download instead of ~2 GB per function per size. Each size's
    VAEs are registered beside it (they are small and resolution-bound), and
    the keys of everything a bundle needs are returned, transformer first.

    Entrypoints are `txt2img_<w>x<h>` and `img2img_<w>x<h>_<grid>`. What a
    bundle costs at run time — whether Core AI keeps one resident copy of the
    weights across its entrypoints — is the measurement this exists for.
    """
    if not sizes:
        raise ValueError("A bundle needs at least one size.")
    keys: list[str] = []
    for width, height in sizes:
        keys += [k for k in register_flux2_resolution(width, height) if k.startswith("vae_")]
    for grid in grids:
        if grid not in REFERENCE_GRIDS:
            raise ValueError(f"Unknown reference grid {grid!r}; one of {REFERENCE_GRIDS}.")

    if references not in (1, 2):
        raise ValueError("A bundle carries one or two reference images per function.")
    # A two-reference bundle is its own variant, named so: `…+2ref`.
    tag = flux2_bundle_tag(sizes) + ("+2ref" if references == 2 else "")
    key = f"transformer_bundle_{tag}"
    if key not in FLUX2_COMPONENTS:
        functions: list[FunctionVariant] = []
        for width, height in sizes:
            functions.append(
                FunctionVariant(f"txt2img_{width}x{height}", dummy_flux2_transformer_at(width, height))
            )
            for grid in grids:
                functions.append(
                    FunctionVariant(
                        f"img2img_{width}x{height}_{grid}",
                        dummy_flux2_transformer_img2img_at(width, height, grid),
                    )
                )
                if references == 2:
                    functions.append(
                        FunctionVariant(
                            f"img2img2_{width}x{height}_{grid}",
                            dummy_flux2_transformer_img2img2_at(width, height, grid),
                        )
                    )
        FLUX2_COMPONENTS[key] = MultiFunctionComponentSpec(
            asset_name=f"Transformer_{tag}",
            input_names=_FLUX2_TRANSFORMER_INPUT_NAMES,
            output_names=("output",),
            wrapper_fn=lambda p: Flux2TransformerWrapper(p.transformer),
            functions=tuple(functions),
            quantizable=True,
        )
    ALL_FLUX2_COMPONENTS[:] = list(FLUX2_COMPONENTS.keys())
    return [key] + keys


def register_flux2_shapes(
    sizes: Sequence[tuple[int, int]],
    grids: Sequence[str] = ("half",),
    references: int = 1,
    enumerate_shapes: bool = True,
) -> list[str]:
    """One transformer for several sizes by enumerated shapes:
    `Transformer_<w>x<h>+<w>x<h>_shapes`.

    The same sizes, grids and reference count a bundle takes, but traced once
    with the token dimension open and specialised per distinct token count
    (`flux2_token_counts`) — so two orientations of one size are one shape,
    and a size's image-to-image is one more. The runtime enters a shape as
    `main_n<tokens>`. With `enumerate_shapes` false the dimension is left
    open (`…_open`), for seeing whether the GPU runtime takes it as is.
    Each size's VAEs are registered beside it; keys returned, transformer
    first.
    """
    if not sizes:
        raise ValueError("A shapes transformer needs at least one size.")
    for grid in grids:
        if grid not in REFERENCE_GRIDS:
            raise ValueError(f"Unknown reference grid {grid!r}; one of {REFERENCE_GRIDS}.")
    if references not in (1, 2):
        raise ValueError("One or two reference images per function.")
    keys: list[str] = []
    for width, height in sizes:
        keys += [k for k in register_flux2_resolution(width, height) if k.startswith("vae_")]

    tag = flux2_bundle_tag(sizes) + ("+2ref" if references == 2 else "")
    kind = "shapes" if enumerate_shapes else "open"
    key = f"transformer_{kind}_{tag}"
    if key not in FLUX2_COMPONENTS:
        counts = flux2_token_counts(sizes, grids, references)
        FLUX2_COMPONENTS[key] = EnumeratedComponentSpec(
            asset_name=f"Transformer_{tag}_{kind}",
            input_names=_FLUX2_TRANSFORMER_INPUT_NAMES,
            output_names=("output",),
            wrapper_fn=lambda p: Flux2TransformerWrapper(p.transformer),
            # Traced at the smallest count; the shapes say the rest.
            dummy_fn=dummy_flux2_transformer_at(*sizes[0]),
            dynamic_shapes_fn=flux2_transformer_dynamic_shapes,
            static_shapes_fn=flux2_transformer_static_shapes(counts) if enumerate_shapes else None,
            quantizable=True,
        )
    ALL_FLUX2_COMPONENTS[:] = list(FLUX2_COMPONENTS.keys())
    return [key] + keys


def register_flux2_open(grids: Sequence[str] = ("half",)) -> list[str]:
    """The size-free set: `Transformer_open`, `VAEDecoder_open`, `VAEEncoder_open`.

    Each traced once with its spatial dimension open — the transformer's token
    axis, the VAEs' height and width — and enumerated for nothing, so one
    function `main` serves any size whose sides are multiples of 16, either
    way round, text, one reference or two. What it costs at run time against
    the specialised files is the measurement (`Docs/klein-bundles.md`).
    """
    keys: list[str] = []
    if "transformer_open" not in FLUX2_COMPONENTS:
        FLUX2_COMPONENTS["transformer_open"] = EnumeratedComponentSpec(
            asset_name="Transformer_open",
            input_names=_FLUX2_TRANSFORMER_INPUT_NAMES,
            output_names=("output",),
            wrapper_fn=lambda p: Flux2TransformerWrapper(p.transformer),
            dummy_fn=dummy_flux2_transformer_at(768, 576),
            dynamic_shapes_fn=flux2_transformer_dynamic_shapes,
            static_shapes_fn=None,
            quantizable=True,
        )
    keys.append("transformer_open")
    if "vae_decoder_open" not in FLUX2_COMPONENTS:
        FLUX2_COMPONENTS["vae_decoder_open"] = EnumeratedComponentSpec(
            asset_name="VAEDecoder_open",
            input_names=("z",),
            output_names=("image",),
            wrapper_fn=lambda p: Flux2VAEDecoderWrapper(p.vae),
            dummy_fn=dummy_flux2_vae_decoder_at(768, 576),
            dynamic_shapes_fn=flux2_vae_decoder_dynamic_shapes,
            static_shapes_fn=None,
            quantizable=False,
        )
    keys.append("vae_decoder_open")
    if "vae_encoder_open" not in FLUX2_COMPONENTS:
        FLUX2_COMPONENTS["vae_encoder_open"] = EnumeratedComponentSpec(
            asset_name="VAEEncoder_open",
            input_names=("image",),
            output_names=("latent_params",),
            wrapper_fn=lambda p: Flux2VAEEncoderWrapper(p.vae),
            dummy_fn=dummy_flux2_vae_encoder_at(768, 576),
            dynamic_shapes_fn=flux2_vae_encoder_dynamic_shapes,
            static_shapes_fn=None,
            quantizable=False,
        )
    keys.append("vae_encoder_open")
    ALL_FLUX2_COMPONENTS[:] = list(FLUX2_COMPONENTS.keys())
    return keys


# Multi-function transformer: 8 functions in one .aimodel, shared weights (~2 GB)
_FLUX2_TRANSFORMER_INPUT_NAMES = (
    "hidden_states",
    "encoder_hidden_states",
    "timestep",
    "guidance",
    "img_ids",
    "txt_ids",
)

FLUX2_MULTIFUNCTION_TRANSFORMER = MultiFunctionComponentSpec(
    asset_name="Transformer",
    input_names=_FLUX2_TRANSFORMER_INPUT_NAMES,
    output_names=("output",),
    wrapper_fn=lambda p: Flux2TransformerWrapper(p.transformer),
    functions=(
        FunctionVariant("main", dummy_flux2_transformer),
        FunctionVariant("half", dummy_flux2_transformer_512),
        FunctionVariant("img2img_quarter", dummy_flux2_transformer_img2img_quarter),
        FunctionVariant("img2img_half", dummy_flux2_transformer_img2img_half),
        FunctionVariant("img2img_full", dummy_flux2_transformer_img2img_full),
        FunctionVariant("img2img_512_quarter", dummy_flux2_transformer_img2img_512_quarter),
        FunctionVariant("img2img_512_half", dummy_flux2_transformer_img2img_512_half),
        FunctionVariant("img2img_512_full", dummy_flux2_transformer_img2img_512_full),
    ),
    quantizable=True,
)

# The default (no --single-function): replace the two separate transformer entries
# with the single multi-function spec.
FLUX2_MULTIFUNCTION_COMPONENTS: dict[str, ComponentSpec | MultiFunctionComponentSpec] = {
    "transformer": FLUX2_MULTIFUNCTION_TRANSFORMER,
    "text_encoder": FLUX2_COMPONENTS["text_encoder"],
    "vae_decoder": FLUX2_COMPONENTS["vae_decoder"],
    "vae_decoder_half": FLUX2_COMPONENTS["vae_decoder_half"],
    "vae_encoder": FLUX2_COMPONENTS["vae_encoder"],
    "vae_encoder_half": FLUX2_COMPONENTS["vae_encoder_half"],
}

ALL_FLUX2_MULTIFUNCTION_COMPONENTS: list[str] = list(FLUX2_MULTIFUNCTION_COMPONENTS.keys())


SD3_COMPONENTS: dict[str, ComponentSpec] = {
    "text_encoder": ComponentSpec(
        asset_name="TextEncoder",
        input_names=("input_ids",),
        output_names=("hidden_embeds", "pooled_outputs"),
        wrapper_fn=lambda p: TextEncoderWithPooledWrapper(p.text_encoder),
        dummy_fn=_dummy_text_encoder,
        quantizable=True,
    ),
    "text_encoder_2": ComponentSpec(
        asset_name="TextEncoder2",
        input_names=("input_ids",),
        output_names=("hidden_embeds", "pooled_outputs"),
        wrapper_fn=lambda p: TextEncoderWithPooledWrapper(p.text_encoder_2),
        dummy_fn=_dummy_text_encoder,
        quantizable=True,
    ),
    "transformer": ComponentSpec(
        asset_name="MMDiT",
        input_names=("sample", "timestep", "encoder_hidden_states", "pooled_projections"),
        output_names=("noise_pred",),
        wrapper_fn=lambda p: SD3TransformerWrapper(p.transformer),
        dummy_fn=_dummy_sd3_transformer,
        quantizable=True,
    ),
    "vae_decoder": ComponentSpec(
        asset_name="VAEDecoder",
        input_names=("z",),
        output_names=("image",),
        wrapper_fn=lambda p: VAEDecoderWrapper(p.vae),
        dummy_fn=_dummy_vae_decoder,
    ),
}

ALL_SD3_COMPONENTS: list[str] = list(SD3_COMPONENTS.keys())

WAN_COMPONENTS: dict[str, ComponentSpec] = {
    "transformer": ComponentSpec(
        asset_name="Transformer",
        input_names=(
            "hidden_states",
            "encoder_hidden_states",
            "timestep",
        ),
        output_names=("output",),
        wrapper_fn=lambda p: WanTransformerWrapper(p.transformer),
        dummy_fn=dummy_wan_transformer,
        quantizable=True,
        dynamic_shapes_fn=wan_transformer_dynamic_shapes,
    ),
    "text_encoder": ComponentSpec(
        asset_name="TextEncoder",
        input_names=("input_ids", "attention_mask"),
        output_names=("hidden_states",),
        wrapper_fn=lambda p: WanTextEncoderWrapper(p.text_encoder),
        dummy_fn=dummy_wan_text_encoder,
        quantizable=True,
    ),
    "vae_decoder": ComponentSpec(
        asset_name="VAEDecoder",
        input_names=("latent",),
        output_names=("pixels",),
        wrapper_fn=lambda p: WanVAEDecoderWrapper(p.vae),
        dummy_fn=dummy_wan_vae_decoder,
    ),
}

ALL_WAN_COMPONENTS: list[str] = list(WAN_COMPONENTS.keys())


def get_component_registry(
    hf_pipe: Any,
    pipeline_type: str = "sd",
    multifunction: bool = False,
) -> dict[str, ComponentSpec | MultiFunctionComponentSpec]:
    """Return the component registry for the given pipeline type.

    Args:
        hf_pipe: The loaded HuggingFace pipeline (unused for routing, but
            available for future introspection).
        pipeline_type: One of "sd", "sd3", or "flux2".
        multifunction: If True, use multi-function export for FLUX.2 transformer
            (5 functions in one .aimodel: main, half, img2img_quarter/half/full).
    """
    if pipeline_type == "flux2":
        if multifunction:
            return FLUX2_MULTIFUNCTION_COMPONENTS
        return FLUX2_COMPONENTS
    if pipeline_type == "sd3":
        return SD3_COMPONENTS
    if pipeline_type == "wan":
        return WAN_COMPONENTS
    return SD_COMPONENTS


def get_valid_components(pipeline_type: str, multifunction: bool = False) -> list[str]:
    """Return valid component names for a given pipeline type."""
    if pipeline_type == "flux2":
        if multifunction:
            return ALL_FLUX2_MULTIFUNCTION_COMPONENTS
        return ALL_FLUX2_COMPONENTS
    if pipeline_type == "sd3":
        return ALL_SD3_COMPONENTS
    if pipeline_type == "wan":
        return ALL_WAN_COMPONENTS
    return ALL_SD_COMPONENTS
