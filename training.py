"""Codec losses, supervised AR loss, and rectified flow matching for DM."""
import random
import torch
from torch.nn import functional as F
from .model import DiffusionSegment, MiniCosmos3
from .positions import video_positions, temporal_positions
from .tokenizer import CharTokenizer

MODES = ("t2i", "t2v", "i2v", "v2v", "transfer", "forward", "inverse", "policy", "all")


def codec_loss(model: MiniCosmos3, batch: dict) -> tuple[torch.Tensor, dict]:
    cfg = model.config
    z, hp, wp = model.visual_codec.encode(batch["video"])
    reconstructed_video = model.visual_codec.decode(z, frames=cfg.video_frames, height=hp, width=wp)
    az = model.audio_codec.encode(batch["audio"])
    reconstructed_audio = model.audio_codec.decode(az)
    visual = F.mse_loss(reconstructed_video, batch["video"])
    audio = F.mse_loss(reconstructed_audio, batch["audio"])
    return visual + audio, {"video_recon": visual.item(), "audio_recon": audio.item()}


def reasoner_loss(model: MiniCosmos3, batch: dict) -> tuple[torch.Tensor, dict]:
    tok = CharTokenizer()
    device = batch["video"].device
    ids, labels = tok.answer_batch(batch["prompt"], batch["answer"],
                                    model.config.max_text, device)
    output = model(ids, ar_frames=batch["video"])
    loss = F.cross_entropy(output.text_logits.reshape(-1, tok.vocab_size),
                           labels.reshape(-1), ignore_index=-100)
    return loss, output.shapes


def sample_sigma(batch_size: int, device, *, shift: float = 1.) -> torch.Tensor:
    # Logit-normal base noise: sigma=sigmoid(N(0,1)), then optional shift.
    s = torch.sigmoid(torch.randn(batch_size, device=device))
    return shift * s / (1.0 + (shift - 1.0) * s)


def noised(name: str, modality: str, clean: torch.Tensor, coords: torch.Tensor,
           shift: float = 1., domain: str = "camera"):
    """Linear interpolation x_sigma=(1-sigma)x0 + sigma*epsilon.

    Correct target v*=epsilon-x0 is the derivative dx_sigma/dsigma.
    """
    b = clean.shape[0]
    sigma = sample_sigma(b, clean.device, shift=shift)
    eps = torch.randn_like(clean)
    noisy = (1 - sigma[:, None, None]) * clean + sigma[:, None, None] * eps
    target_velocity = eps - clean
    return (DiffusionSegment(name, modality, noisy, coords, sigma, domain), target_velocity)


def clean_segment(name: str, modality: str, values: torch.Tensor, coords: torch.Tensor,
                  domain: str = "camera"):
    sigma = torch.zeros(values.shape[0], device=values.device)
    return DiffusionSegment(name, modality, values, coords, sigma, domain)


def generator_loss(model: MiniCosmos3, batch: dict, mode: str = "all",
                   with_audio: bool = False) -> tuple[torch.Tensor, dict]:
    """One training step can denoise one or several modalities simultaneously.

    No gradients go into frozen codecs. Targets are *latent* features, not RGB.
    Only noisy tokens contribute to loss; clean conditioning tokens are masked.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    cfg, device = model.config, batch["video"].device
    model.freeze_codecs()
    with torch.no_grad():
        vz, hp, wp = model.visual_codec.encode(batch["video"])
        az = model.audio_codec.encode(batch["audio"])
    actions = batch["actions"]
    n_per_frame = hp * wp
    vision_coords = video_positions(cfg.video_frames, hp, wp, fps=cfg.video_fps,
                                    temporal_compression=cfg.temporal_compression,
                                    base_tps=cfg.base_tps, gap=cfg.dm_gap, device=device)
    audio_coords = temporal_positions(az.shape[1],
                                     tokens_per_sec=cfg.audio_sample_rate / cfg.audio_hop,
                                     base_tps=cfg.base_tps, gap=cfg.dm_gap, device=device)
    action_coords = temporal_positions(actions.shape[1], tokens_per_sec=cfg.action_hz,
                                      base_tps=cfg.base_tps, gap=cfg.dm_gap, device=device)
    segments, targets = [], {}
    def add_noisy(name, kind, z, coords, *, domain="camera"):
        segment, velocity = noised(name, kind, z, coords, domain=domain)
        segments.append(segment)
        targets[name] = velocity
    if mode in ("t2i", "t2v", "i2v", "v2v", "transfer", "forward", "policy", "all"):
        n_cond = {"i2v": 1, "v2v": 2, "forward": 1, "policy": 1, "all": 1}.get(mode, 0)
        if mode == "t2i":
            z, vc = vz[:, :n_per_frame], vision_coords[:n_per_frame]
            add_noisy("vision_future", "vision", z, vc)
        else:
            if mode == "transfer":
                # A geometrically aligned low-information control video (grayscale).
                grayscale = batch["video"].mean(2, keepdim=True).repeat(1, 1, 3, 1, 1)
                with torch.no_grad():
                    control_z, _, _ = model.visual_codec.encode(grayscale)
                segments.append(clean_segment("vision_control", "vision", control_z, vision_coords))
            elif n_cond:
                end = n_cond * n_per_frame
                segments.append(clean_segment("vision_context", "vision", vz[:, :end], vision_coords[:end]))
            start = n_cond * n_per_frame
            add_noisy("vision_future", "vision", vz[:, start:], vision_coords[start:])
    if mode in ("forward",):
        segments.append(clean_segment("actions_given", "action", actions, action_coords))
    if mode in ("inverse",):
        segments.append(clean_segment("vision_observed", "vision", vz, vision_coords))
        add_noisy("actions_future", "action", actions, action_coords)
    if mode in ("policy", "all"):
        add_noisy("actions_future", "action", actions, action_coords)
    if mode == "all" or (with_audio and mode in ("t2v", "i2v", "v2v")):
        add_noisy("audio_future", "audio", az, audio_coords)
    # Simulate 10% text-conditioning dropout; needed to train unconditioned CFG.
    prompts = ["" if model.training and random.random() < .1 else text
               for text in batch["prompt"]]
    ids = CharTokenizer().batch_prefix(prompts, device)
    output = model(ids, segments=segments)
    if not targets:
        raise RuntimeError("no diffusion target prepared")
    # Conditioning segments are deliberately absent from targets => no loss.
    parts = {name: F.mse_loss(output.predictions[name], velocity)
             for name, velocity in targets.items()}
    total = torch.stack(list(parts.values())).mean()
    return total, {"mode": mode, "losses": {k: float(v.detach()) for k, v in parts.items()},
                   "shapes": output.shapes}
