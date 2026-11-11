"""CLI: python -m cosmos3_lab.train --stage codec|reasoner|generator."""
import argparse
import os
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .config import LabConfig
from .model import MiniCosmos3
from .synthetic import SyntheticWorld
from .training import MODES, codec_loss, generator_loss, reasoner_loss


def load_model(checkpoint: str, device: torch.device, *, require=False) -> MiniCosmos3:
    if Path(checkpoint).exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        config = LabConfig(**state["config"])
        model = MiniCosmos3(config)
        model.load_state_dict(state["state_dict"])
        print(f"Restored checkpoint: {checkpoint}; previous stage: {state.get('stage')}")
    else:
        if require:
            raise FileNotFoundError(f"No checkpoint: {checkpoint}")
        config = LabConfig()
        model = MiniCosmos3(config)
        print("Created new, randomly initialized learning-scale model")
    return model.to(device)


def run(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=["codec", "reasoner", "generator"])
    parser.add_argument("--mode", default="all", choices=MODES, help="DM task configuration")
    parser.add_argument("--steps", type=int, default=30, help="gradient updates")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--with-audio", action="store_true",
                        help="Also generate sound in t2v/i2v/v2v (all always includes sound)")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--manifest", default=None, help="Optional JSONL real-data manifest (see manifest.py)")
    parser.add_argument("--checkpoint", default="artifacts/cosmos3_toy.pt")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    model = load_model(args.checkpoint, device)
    if args.stage == "generator":
        if not Path(args.checkpoint).exists():
            print("WARNING: codecs are random. Run --stage codec first for meaningful latents.")
        model.freeze_codecs()
    elif args.stage == "codec":
        # Restore codec parameters even if a prior generator stage froze them.
        for p in list(model.visual_codec.parameters()) + list(model.audio_codec.parameters()):
            p.requires_grad_(True)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    if args.manifest:
        from .manifest import ManifestWorld
        dataset = ManifestWorld(args.manifest, model.config)
    else:
        dataset = SyntheticWorld(model.config, count=max(32, args.batch_size * 8), seed=101)
    if len(dataset) < args.batch_size:
        raise ValueError("Dataset must contain at least batch-size samples")
    data = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    stream = iter(data)
    for step in range(1, args.steps + 1):
        try:
            batch = next(stream)
        except StopIteration:
            stream = iter(data)
            batch = next(stream)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        model.train()
        if args.stage == "codec":
            loss, metadata = codec_loss(model, batch)
        elif args.stage == "reasoner":
            loss, metadata = reasoner_loss(model, batch)
        else:
            loss, metadata = generator_loss(model, batch, mode=args.mode, with_audio=args.with_audio)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {loss}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
        optimizer.step()
        if step == 1 or step == args.steps or step % max(1, args.steps // 5) == 0:
            print(f"step={step:04d}/{args.steps}, stage={args.stage}, loss={loss.item():.5f}; details={metadata}")
    ckpt = Path(args.checkpoint)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    tmp = ckpt.with_suffix(ckpt.suffix + ".tmp")
    torch.save({"config": model.config.to_dict(), "state_dict": model.state_dict(),
                "stage": args.stage, "steps_last_stage": args.steps}, tmp)
    os.replace(tmp, ckpt)  # atomic checkpoint replacement on local filesystem
    print(f"Saved {ckpt} | device={device} | trainable parameters="
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


if __name__ == "__main__":
    run()
