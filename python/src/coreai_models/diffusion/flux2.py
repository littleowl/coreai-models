# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
FLUX.2 component specifications and torch wrappers for Core AI export.

FLUX.2 Klein 4B is a DiT (Diffusion Transformer) that uses:
- Qwen3 text encoder (intermediate hidden states from layers 9, 18, 27)
- 25-block double-stream + single-stream transformer with 4D RoPE
- AutoencoderKLFlux2 VAE with batch normalization

The transformer computes RoPE in-graph from position IDs (img_ids, txt_ids),
matching upstream diffusers. Position IDs are cheap to build and depend only on
grid geometry, so the exported graph owns the frequency computation.
"""

from typing import Any, cast

import torch

# ---------------------------------------------------------------------------
# Torch wrappers
# ---------------------------------------------------------------------------


class Flux2TransformerWrapper(torch.nn.Module):
    """Wraps Flux2Transformer2DModel for export with in-graph RoPE.

    Takes position IDs and lets the model compute rotary embeddings internally via
    self.pos_embed(), so the exported graph matches upstream diffusers.
    """

    def __init__(self, transformer: torch.nn.Module) -> None:
        super().__init__()
        self.model = transformer

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        guidance: torch.Tensor,
        img_ids: torch.Tensor,
        txt_ids: torch.Tensor,
    ) -> torch.Tensor:
        return cast(
            torch.Tensor,
            self.model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                guidance=guidance,
                img_ids=img_ids,
                txt_ids=txt_ids,
            ).sample,
        )


class Flux2TextEncoderWrapper(torch.nn.Module):
    """Wraps Qwen3ForCausalLM to extract and concatenate intermediate hidden states.

    FLUX.2 uses hidden states from 3 intermediate layers (default: 9, 18, 27),
    stacked and reshaped from [1, 3, seq_len, 2560] -> [1, seq_len, 7680].
    """

    def __init__(
        self, text_encoder: torch.nn.Module, hidden_states_layers: tuple[int, ...] = (9, 18, 27)
    ) -> None:
        super().__init__()
        self.model = text_encoder
        self.hidden_states_layers = hidden_states_layers

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        stacked = torch.stack([outputs.hidden_states[k] for k in self.hidden_states_layers], dim=1)
        batch_size, num_layers, seq_len, hidden_dim = stacked.shape
        return stacked.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_layers * hidden_dim)


class Flux2VAEDecoderWrapper(torch.nn.Module):
    """Wraps AutoencoderKLFlux2.decode: (latent) -> (image)."""

    def __init__(self, vae: torch.nn.Module) -> None:
        super().__init__()
        self.vae: Any = vae
        # Ensure all parameters + buffers (including BN running stats) share the same dtype
        self.vae = self.vae.to(next(vae.parameters()).dtype)
        from coreai_models.diffusion.components import _patch_nearest_upsample

        _patch_nearest_upsample(self.vae.decoder)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.vae.decode(z).sample)


class Flux2VAEEncoderWrapper(torch.nn.Module):
    """Wraps AutoencoderKLFlux2.encode: (image) -> (latent)."""

    def __init__(self, vae: torch.nn.Module) -> None:
        super().__init__()
        self.vae: Any = vae
        self.vae = self.vae.to(next(vae.parameters()).dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # diffusers encodes img2img reference images with
        # `retrieve_latents(..., sample_mode="argmax")` -> `latent_dist.mode()`,
        # i.e. the distribution MEAN (first `latent_channels` channels), not the
        # raw `parameters` tensor (which is mean concat logvar = 2x channels).
        # Returning `.parameters` would emit 64 channels where the pipeline
        # expects 32, corrupting the img2img latents. `.mode()` is deterministic,
        # so it is also the correct choice for a traced/exported graph.
        return cast(torch.Tensor, self.vae.encode(x).latent_dist.mode())


# ---------------------------------------------------------------------------
# Dummy-input factories
# ---------------------------------------------------------------------------

# Reference tokens carry T=10 on RoPE axis 0 so the in-graph RoPE keeps them
# positionally distinct from the noise grid even where H/W coincide. Mirrors
# Flux2Pipeline.referenceTokenTimeOffset on the Swift side.
REFERENCE_TOKEN_TIME_OFFSET = 10.0


def _grid_position_ids(
    grid_w: int, grid_h: int | None = None, num_axes: int = 4, time_offset: float = 0.0
) -> torch.Tensor:
    """`[grid_h*grid_w, num_axes]` position IDs as [T, H, W, L], row-major over H then W.

    `grid_h` defaults to `grid_w`, which is every square resolution; pass both
    for a portrait or landscape grid. An iPad is not square and a coloring page
    is not either, so the grid is two numbers everywhere below.
    """
    if grid_h is None:
        grid_h = grid_w
    rows = torch.arange(grid_h, dtype=torch.float32)
    cols = torch.arange(grid_w, dtype=torch.float32)
    mesh_h, mesh_w = torch.meshgrid(rows, cols, indexing="ij")
    ids = torch.zeros(grid_h * grid_w, num_axes)
    ids[:, 0] = time_offset
    ids[:, 1] = mesh_h.reshape(-1)
    ids[:, 2] = mesh_w.reshape(-1)
    return ids


def _text_position_ids(text_seq_len: int, num_axes: int) -> torch.Tensor:
    """`[1, text_seq_len, num_axes]` — sequence index on the last axis, spatial unused."""
    ids = torch.zeros(1, text_seq_len, num_axes)
    ids[0, :, num_axes - 1] = torch.arange(text_seq_len, dtype=torch.float32)
    return ids


def _dummy_flux2_transformer_impl(
    pipe: Any, grid_size: int, grid_h: int | None = None
) -> tuple[torch.Tensor, ...]:
    cfg = pipe.transformer.config
    dtype = next(pipe.transformer.parameters()).dtype
    grid_w = grid_size
    if grid_h is None:
        grid_h = grid_w
    image_seq_len = grid_w * grid_h
    text_seq_len = 512
    axes_dim = list(cfg.axes_dims_rope)

    # Position IDs per token: [T, H, W, L]. Image tokens carry the spatial grid on
    # axes 1/2; text tokens carry the sequence index on the last axis.
    num_rope_axes = len(axes_dim)
    img_ids = _grid_position_ids(grid_w, grid_h, num_rope_axes).unsqueeze(0)
    txt_ids = _text_position_ids(text_seq_len, num_rope_axes)

    return (
        torch.randn(1, image_seq_len, cfg.in_channels, dtype=dtype),
        torch.randn(1, text_seq_len, cfg.joint_attention_dim, dtype=dtype),
        torch.tensor([0.5], dtype=dtype),
        torch.tensor([1.0], dtype=dtype),
        img_ids,
        txt_ids,
    )


def dummy_flux2_transformer(pipe: Any) -> tuple[torch.Tensor, ...]:
    """1024×1024 (grid=64, seqLen=4096)."""
    return _dummy_flux2_transformer_impl(pipe, grid_size=64)


def dummy_flux2_text_encoder(pipe: Any) -> tuple[torch.Tensor, ...]:
    text_seq_len = 512
    return (
        torch.zeros(1, text_seq_len, dtype=torch.long),  # input_ids
        torch.ones(1, text_seq_len, dtype=torch.long),  # attention_mask
    )


def dummy_flux2_vae_decoder(pipe: Any) -> tuple[torch.Tensor, ...]:
    latent_channels = pipe.vae.config.latent_channels
    sample_size = 128  # 1024 / 8
    dtype = next(pipe.vae.parameters()).dtype
    return (torch.randn(1, latent_channels, sample_size, sample_size, dtype=dtype),)


# ---------------------------------------------------------------------------
# Any resolution, not only the two square ones
# ---------------------------------------------------------------------------

# The VAE downsamples by 8 and the transformer patches 2x2, so a side has to be
# divisible by 16 and the token grid is (width // 16, height // 16).
PIXELS_PER_TOKEN = 16
PIXELS_PER_LATENT = 8


def grid_for(width: int, height: int) -> tuple[int, int]:
    """The token grid a pixel size implies, refusing one it cannot represent."""
    for side, name in ((width, "width"), (height, "height")):
        if side % PIXELS_PER_TOKEN:
            raise ValueError(
                f"{name} {side} is not divisible by {PIXELS_PER_TOKEN}: FLUX.2 downsamples by "
                f"{PIXELS_PER_LATENT} in the VAE and patches 2x2 in the transformer, so a side "
                f"that is not a multiple of {PIXELS_PER_TOKEN} cannot be patched. "
                f"{side - side % PIXELS_PER_TOKEN} and "
                f"{side + PIXELS_PER_TOKEN - side % PIXELS_PER_TOKEN} both can."
            )
    return width // PIXELS_PER_TOKEN, height // PIXELS_PER_TOKEN


def dummy_flux2_transformer_at(width: int, height: int) -> Any:
    """A txt2img transformer dummy for one pixel size."""
    grid_w, grid_h = grid_for(width, height)

    def dummy(pipe: Any) -> tuple[torch.Tensor, ...]:
        return _dummy_flux2_transformer_impl(pipe, grid_w, grid_h)

    dummy.__doc__ = f"{width}x{height} (grid={grid_w}x{grid_h}, seqLen={grid_w * grid_h})."
    return dummy


def dummy_flux2_transformer_img2img_at(width: int, height: int, grid: str) -> Any:
    """An img2img transformer dummy: noise tokens plus a reference grid.

    `grid` is `full`, `half` or `quarter` — the reference grid relative to the
    noise grid. Halving a grid halves both sides, so a 4:3 reference stays 4:3.
    """
    divisor = {"full": 1, "half": 2, "quarter": 4}[grid]
    grid_w, grid_h = grid_for(width, height)
    ref_w, ref_h = max(1, grid_w // divisor), max(1, grid_h // divisor)

    def dummy(pipe: Any) -> tuple[torch.Tensor, ...]:
        return _dummy_flux2_transformer_img2img(
            pipe, noise_grid=grid_w, ref_grid=ref_w, noise_grid_h=grid_h, ref_grid_h=ref_h
        )

    dummy.__doc__ = (
        f"img2img {grid} ({width}x{height}): {grid_w * grid_h} noise + "
        f"{ref_w * ref_h} reference tokens."
    )
    return dummy


def dummy_flux2_vae_decoder_at(width: int, height: int) -> Any:
    """A VAE decoder dummy: the latent for one pixel size."""
    latent_w, latent_h = width // PIXELS_PER_LATENT, height // PIXELS_PER_LATENT

    def dummy(pipe: Any) -> tuple[torch.Tensor, ...]:
        latent_channels = pipe.vae.config.latent_channels
        dtype = next(pipe.vae.parameters()).dtype
        return (torch.randn(1, latent_channels, latent_h, latent_w, dtype=dtype),)

    return dummy


def dummy_flux2_vae_encoder_at(width: int, height: int) -> Any:
    """A VAE encoder dummy: the picture at one pixel size."""

    def dummy(pipe: Any) -> tuple[torch.Tensor, ...]:
        dtype = next(pipe.vae.parameters()).dtype
        return (torch.randn(1, 3, height, width, dtype=dtype),)

    return dummy


def dummy_flux2_vae_decoder_half(pipe: Any) -> tuple[torch.Tensor, ...]:
    latent_channels = pipe.vae.config.latent_channels
    sample_size = 64  # 512 / 8
    dtype = next(pipe.vae.parameters()).dtype
    return (torch.randn(1, latent_channels, sample_size, sample_size, dtype=dtype),)


def dummy_flux2_vae_encoder(pipe: Any) -> tuple[torch.Tensor, ...]:
    dtype = next(pipe.vae.parameters()).dtype
    return (torch.randn(1, 3, 1024, 1024, dtype=dtype),)


def dummy_flux2_vae_encoder_half(pipe: Any) -> tuple[torch.Tensor, ...]:
    dtype = next(pipe.vae.parameters()).dtype
    return (torch.randn(1, 3, 512, 512, dtype=dtype),)


def dummy_flux2_transformer_512(pipe: Any) -> tuple[torch.Tensor, ...]:
    """512×512 (grid=32, seqLen=1024)."""
    return _dummy_flux2_transformer_impl(pipe, grid_size=32)


# ---------------------------------------------------------------------------
# img2img dummy factories — concatenated noise + reference tokens
# ---------------------------------------------------------------------------


def _dummy_flux2_transformer_img2img(
    pipe: Any,
    noise_grid: int,
    ref_grid: int,
    noise_grid_h: int | None = None,
    ref_grid_h: int | None = None,
) -> tuple[torch.Tensor, ...]:
    """Build dummy inputs for img2img transformer with concatenated reference tokens.

    The img2img approach concatenates noise tokens + reference tokens along the
    sequence dimension. The transformer processes the full sequence and we slice
    off the noise predictions afterward.

    Args:
        pipe: HF pipeline (for config access).
        noise_grid: Grid size for noise tokens (64 = 1024×1024).
        ref_grid: Grid size for reference image tokens (16/32/64 = quarter/half/full).
    """
    cfg = pipe.transformer.config
    dtype = next(pipe.transformer.parameters()).dtype
    noise_grid_h = noise_grid if noise_grid_h is None else noise_grid_h
    ref_grid_h = ref_grid if ref_grid_h is None else ref_grid_h
    noise_seq = noise_grid * noise_grid_h
    ref_seq = ref_grid * ref_grid_h
    total_img_seq = noise_seq + ref_seq
    text_seq = 512
    num_rope_axes = len(cfg.axes_dims_rope)

    # Position IDs: text (T=0) + noise (T=0) + reference (T=10)
    img_ids = torch.cat(
        [
            _grid_position_ids(noise_grid, noise_grid_h, num_rope_axes),
            _grid_position_ids(
                ref_grid, ref_grid_h, num_rope_axes, time_offset=REFERENCE_TOKEN_TIME_OFFSET
            ),
        ]
    ).unsqueeze(0)
    txt_ids = _text_position_ids(text_seq, num_rope_axes)

    return (
        torch.randn(1, total_img_seq, cfg.in_channels, dtype=dtype),
        torch.randn(1, text_seq, cfg.joint_attention_dim, dtype=dtype),
        torch.tensor([0.5], dtype=dtype),
        torch.tensor([1.0], dtype=dtype),
        img_ids,
        txt_ids,
    )


def dummy_flux2_transformer_img2img_quarter(pipe: Any) -> tuple[torch.Tensor, ...]:
    """img2img quarter (1024×1024): 4096 noise + 256 reference = 4352 img tokens."""
    return _dummy_flux2_transformer_img2img(pipe, noise_grid=64, ref_grid=16)


def dummy_flux2_transformer_img2img_half(pipe: Any) -> tuple[torch.Tensor, ...]:
    """img2img half (1024×1024): 4096 noise + 1024 reference = 5120 img tokens."""
    return _dummy_flux2_transformer_img2img(pipe, noise_grid=64, ref_grid=32)


def dummy_flux2_transformer_img2img_full(pipe: Any) -> tuple[torch.Tensor, ...]:
    """img2img full (1024×1024): 4096 noise + 4096 reference = 8192 img tokens."""
    return _dummy_flux2_transformer_img2img(pipe, noise_grid=64, ref_grid=64)


def dummy_flux2_transformer_img2img_512_quarter(pipe: Any) -> tuple[torch.Tensor, ...]:
    """img2img quarter (512×512): 1024 noise + 64 reference = 1088 img tokens."""
    return _dummy_flux2_transformer_img2img(pipe, noise_grid=32, ref_grid=8)


def dummy_flux2_transformer_img2img_512_half(pipe: Any) -> tuple[torch.Tensor, ...]:
    """img2img half (512×512): 1024 noise + 256 reference = 1280 img tokens."""
    return _dummy_flux2_transformer_img2img(pipe, noise_grid=32, ref_grid=16)


def dummy_flux2_transformer_img2img_512_full(pipe: Any) -> tuple[torch.Tensor, ...]:
    """img2img full (512×512): 1024 noise + 1024 reference = 2048 img tokens."""
    return _dummy_flux2_transformer_img2img(pipe, noise_grid=32, ref_grid=32)
