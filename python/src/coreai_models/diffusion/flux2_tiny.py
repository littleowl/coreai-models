# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
TAEF2, the Tiny AutoEncoder for FLUX.2, as Core AI components.

A distilled stand-in for FLUX.2's VAE — 10 MB against the VAE's 160, tens of
times faster — with the same latent API: 32 channels at an eighth of the
picture. It is what a step-by-step preview decodes with, and on a phone what
the final picture may decode with too. Architecture and weights are Ollin
Boer Bohan's (https://github.com/madebyollin/taesd, MIT); the layers are
restated here so the export needs no dependency on that repository, and the
weights come from `madebyollin/taef2` on the Hub at export time.

TAEF2 works in the space the transformer denoises: the FLUX.2 pipeline's
batch-norm statistics are the identity around it (its diffusers wrapper uses
an untrained `BatchNorm2d`), so a runtime feeding it must skip the VAE's
normalisation. The exports keep the VAE components' contract otherwise —
pictures in [-1, 1], the encoder returning `latent_params` with a zero
log-variance half — so a pipeline swaps them in by name.
"""

from typing import Any

import torch
from torch import nn

TAEF2_REPO = "madebyollin/taef2"
TAEF2_FILE = "taef2.safetensors"
TAEF2_LATENT_CHANNELS = 32


def _conv(n_in: int, n_out: int, **kwargs: Any) -> nn.Conv2d:
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class _Clamp(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x / 3) * 3


class _Block(nn.Module):
    def __init__(self, n_in: int, n_out: int, use_midblock_gn: bool = False) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            _conv(n_in, n_out), nn.ReLU(), _conv(n_out, n_out), nn.ReLU(), _conv(n_out, n_out)
        )
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.fuse = nn.ReLU()
        self.pool: nn.Module | None = None
        if use_midblock_gn:
            n_gn = n_in * 4
            self.pool = nn.Sequential(
                nn.Conv2d(n_in, n_gn, 1, bias=False), nn.GroupNorm(4, n_gn), nn.ReLU(),
                nn.Conv2d(n_gn, n_in, 1, bias=False),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pool is not None:
            x = x + self.pool(x)
        return self.fuse(self.conv(x) + self.skip(x))


def _encoder(latent_channels: int) -> nn.Sequential:
    gn = dict(use_midblock_gn=True)
    return nn.Sequential(
        _conv(3, 64), _Block(64, 64),
        _conv(64, 64, stride=2, bias=False), _Block(64, 64), _Block(64, 64), _Block(64, 64),
        _conv(64, 64, stride=2, bias=False), _Block(64, 64), _Block(64, 64), _Block(64, 64),
        _conv(64, 64, stride=2, bias=False), _Block(64, 64, **gn), _Block(64, 64, **gn), _Block(64, 64, **gn),
        _conv(64, latent_channels),
    )


def _decoder(latent_channels: int) -> nn.Sequential:
    gn = dict(use_midblock_gn=True)
    return nn.Sequential(
        _Clamp(), _conv(latent_channels, 64), nn.ReLU(),
        _Block(64, 64, **gn), _Block(64, 64, **gn), _Block(64, 64, **gn), nn.Upsample(scale_factor=2), _conv(64, 64, bias=False),
        _Block(64, 64), _Block(64, 64), _Block(64, 64), nn.Upsample(scale_factor=2), _conv(64, 64, bias=False),
        _Block(64, 64), _Block(64, 64), _Block(64, 64), nn.Upsample(scale_factor=2), _conv(64, 64, bias=False),
        _Block(64, 64), _conv(64, 3),
    )


def load_taef2(dtype: torch.dtype = torch.float16) -> tuple[nn.Sequential, nn.Sequential]:
    """The encoder and decoder with the published weights, from the Hub."""
    import safetensors.torch as stt
    from huggingface_hub import hf_hub_download

    state = stt.load_file(hf_hub_download(TAEF2_REPO, TAEF2_FILE))
    encoder, decoder = _encoder(TAEF2_LATENT_CHANNELS), _decoder(TAEF2_LATENT_CHANNELS)
    enc_state, dec_state = {}, {}
    for key, value in state.items():
        # `decoder.layers.<i>.…` in the file; the decoder's first layer here is
        # the parameterless clamp, so its indices sit one higher.
        which, _layers, index, *rest = key.split(".")
        offset = 1 if which == "decoder" else 0
        (dec_state if which == "decoder" else enc_state)[".".join([str(int(index) + offset), *rest])] = value
    encoder.load_state_dict(enc_state)
    decoder.load_state_dict(dec_state)
    return encoder.to(dtype).eval(), decoder.to(dtype).eval()


class TinyDecoderWrapper(nn.Module):
    """Latent `[1, 32, h, w]` in the transformer's space to a picture in [-1, 1]."""

    def __init__(self, decoder: nn.Module) -> None:
        super().__init__()
        self.decoder = decoder

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return (self.decoder(z) * 2 - 1).clamp(-1, 1)


class TinyEncoderWrapper(nn.Module):
    """Picture in [-1, 1] to `latent_params`: the latent and a zero log-variance,
    which is the VAE encoder's contract with the mode equal to the mean."""

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x * 0.5 + 0.5)
        return torch.cat([latent, torch.zeros_like(latent)], dim=1)


def tiny_decoder_wrapper(pipe: Any) -> nn.Module:
    _, decoder = load_taef2(next(pipe.vae.parameters()).dtype)
    return TinyDecoderWrapper(decoder)


def tiny_encoder_wrapper(pipe: Any) -> nn.Module:
    encoder, _ = load_taef2(next(pipe.vae.parameters()).dtype)
    return TinyEncoderWrapper(encoder)
