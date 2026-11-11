"""Unified AR+diffusion educational model and all modality adapters.

All modal encoders project into width D, but raw inputs are never blindly
concatenated: they first become semantically meaningful token representations.
"""
from dataclasses import dataclass
import torch
from torch import nn
from .config import LabConfig
from .tokenizer import VOCAB_SIZE, PAD
from .encoders import ToyAudioCodec, ToyVisualCodec, VisionUnderstandingEncoder
from .attention import DualStreamMoTBlock
from .positions import text_positions, video_positions


@dataclass
class DiffusionSegment:
    name: str                          # unique identifier, e.g. 'video_future'
    modality: str                      # 'vision', 'audio', or 'action'
    features: torch.Tensor             # [B, N, channel_count], clean or noise-mixed
    positions: torch.Tensor            # [N, 3] physical coordinates
    sigma: torch.Tensor | None = None  # per-example [B] 0=clean; >0=noised
    domain: str = "camera"             # action embodiment-specific projection


@dataclass
class ModelOutput:
    text_logits: torch.Tensor          # [B, L_text, vocab]
    predictions: dict[str, torch.Tensor]  # generated velocity for each DM segment
    shapes: dict[str, tuple[int, ...]]


class MiniCosmos3(nn.Module):
    def __init__(self, config: LabConfig):
        super().__init__()
        self.config = config
        d = config.width
        self.text_embedding = nn.Embedding(VOCAB_SIZE, d)
        self.understanding_vision = VisionUnderstandingEncoder(d, config.visual_patch)
        self.visual_codec = ToyVisualCodec(config.visual_latent)
        self.audio_codec = ToyAudioCodec(config.audio_channels, config.audio_latent, config.audio_hop)
        # Projection matrices implement the neural interface between modalities.
        self.vision_in = nn.Linear(config.visual_latent, d)
        self.vision_out = nn.Linear(d, config.visual_latent)
        self.audio_in = nn.Linear(config.audio_latent, d)
        self.audio_out = nn.Linear(d, config.audio_latent)
        self.action_in = nn.ModuleDict({k: nn.Linear(dim, d) for k, dim in config.action_dims.items()})
        self.action_out = nn.ModuleDict({k: nn.Linear(d, dim) for k, dim in config.action_dims.items()})
        self.types = nn.ParameterDict({m: nn.Parameter(torch.randn(1, 1, d) * .02)
                                       for m in ("ar_vision", "dm_vision", "dm_audio", "dm_action")})
        self.noise_flag = nn.Parameter(torch.randn(1, 1, d) * .02)
        self.time_embedding = nn.Sequential(nn.Linear(1, d), nn.SiLU(), nn.Linear(d, d))
        self.layers = nn.ModuleList(DualStreamMoTBlock(d, config.heads) for _ in range(config.layers))
        self.final_ar_norm = nn.LayerNorm(d)
        self.final_dm_norm = nn.LayerNorm(d)
        self.text_head = nn.Linear(d, VOCAB_SIZE, bias=False)

    def freeze_codecs(self):
        # Unlike NVIDIA, our codecs start at random and require pretraining.
        for codec in (self.visual_codec, self.audio_codec):
            codec.eval()
            for p in codec.parameters():
                p.requires_grad_(False)

    def encode_ar(self, text_ids: torch.Tensor, frames: torch.Tensor | None):
        b, length = text_ids.shape
        dev = text_ids.device
        words = self.text_embedding(text_ids)
        txt_pos = text_positions(length, dev)
        text_valid = text_ids != PAD
        if frames is None:
            return words, txt_pos, 0, text_valid
        visual, hp, wp = self.understanding_vision(frames)
        t = frames.shape[1]
        visual = visual + self.types["ar_vision"]
        # Physical-frame/height/width positions, plus language positions.
        visual_pos = video_positions(t, hp, wp, fps=self.config.video_fps,
                                     temporal_compression=1, base_tps=self.config.base_tps,
                                     gap=0, device=dev)
        # Text 1D coordinates start after the visual prefix so they remain ordered.
        txt_pos = txt_pos + visual.shape[1]
        valid = torch.cat((torch.ones(b, visual.shape[1], device=dev, dtype=torch.bool), text_valid), dim=1)
        return torch.cat((visual, words), dim=1), torch.cat((visual_pos, txt_pos)), visual.shape[1], valid

    def embed_segment(self, seg: DiffusionSegment) -> torch.Tensor:
        if seg.modality == "vision":
            x = self.vision_in(seg.features) + self.types["dm_vision"]
        elif seg.modality == "audio":
            x = self.audio_in(seg.features) + self.types["dm_audio"]
        elif seg.modality == "action":
            if seg.domain not in self.action_in:
                raise ValueError(f"Unsupported action domain {seg.domain!r}")
            x = self.action_in[seg.domain](seg.features) + self.types["dm_action"]
        else:
            raise ValueError(f"Unknown modality {seg.modality!r}")
        if seg.sigma is not None:
            sigma = seg.sigma.to(dtype=x.dtype, device=x.device).reshape(-1, 1)
            if sigma.shape[0] != x.shape[0]:
                raise ValueError("sigma must have one value per batch element")
            # Separate timestep embeddings tell the DM tower the noise level.
            x = x + self.time_embedding(sigma).unsqueeze(1)
            x = x + (sigma > 0).to(x.dtype).unsqueeze(-1) * self.noise_flag
        return x

    def forward(self, text_ids: torch.Tensor,
                ar_frames: torch.Tensor | None = None,
                segments: list[DiffusionSegment] | None = None) -> ModelOutput:
        if text_ids.ndim != 2:
            raise ValueError("text_ids must be [batch, length]")
        segments = segments or []
        ar, ar_pos, n_vision_ar, ar_valid = self.encode_ar(text_ids, ar_frames)
        lengths, chunks, coords = [], [], []
        seen = set()
        for s in segments:
            if s.name in seen:
                raise ValueError("Every DM segment needs a unique name")
            seen.add(s.name)
            if s.features.ndim != 3 or s.features.shape[0] != text_ids.shape[0]:
                raise ValueError(f"{s.name} must contain [batch, tokens, channels]")
            if s.positions.shape != (s.features.shape[1], 3):
                raise ValueError(f"{s.name} positions do not match token length")
            lengths.append(s.features.shape[1])
            chunks.append(self.embed_segment(s))
            coords.append(s.positions.to(text_ids.device))
        dm = torch.cat(chunks, dim=1) if chunks else None
        dm_pos = torch.cat(coords, dim=0) if coords else None
        for layer in self.layers:
            ar, dm = layer(ar, dm, ar_pos, dm_pos, ar_key_valid=ar_valid)
        logits = self.text_head(self.final_ar_norm(ar)[:, n_vision_ar:])
        preds = {}
        if dm is not None:
            dm = self.final_dm_norm(dm)
            offset = 0
            for seg, length in zip(segments, lengths):
                hidden = dm[:, offset:offset + length]
                if seg.modality == "vision":
                    preds[seg.name] = self.vision_out(hidden)
                elif seg.modality == "audio":
                    preds[seg.name] = self.audio_out(hidden)
                else:
                    preds[seg.name] = self.action_out[seg.domain](hidden)
                offset += length
        return ModelOutput(logits, preds,
                           {"ar_total": tuple(ar.shape),
                            "dm_total": tuple(dm.shape) if dm is not None else (),
                            "text_logits": tuple(logits.shape),
                            **{name: tuple(val.shape) for name, val in preds.items()}})
