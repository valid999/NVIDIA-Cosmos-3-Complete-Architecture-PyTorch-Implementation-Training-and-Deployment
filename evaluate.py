"""Reproducible held-out metrics and shape checks; does not claim physical realism."""
import argparse
import json
import random
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .train import load_model
from .synthetic import SyntheticWorld
from .training import codec_loss, generator_loss, reasoner_loss, MODES


def run():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="artifacts/cosmos3_toy.pt")
    p.add_argument("--stage", choices=["codec", "reasoner", "generator"], default="generator")
    p.add_argument("--mode", choices=MODES, default="all")
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--with-audio", action="store_true")
    p.add_argument("--out", default="artifacts/evaluation.json")
    args = p.parse_args()
    torch.manual_seed(9001)
    random.seed(9001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device, require=True).eval()
    loader = DataLoader(SyntheticWorld(model.config, count=args.samples, seed=9001),
                        batch_size=min(args.samples, 2), shuffle=False)
    # `generator_loss` samples fresh noise; reported MSE is not a video-quality metric.
    values = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            if args.stage == "codec":
                loss, _ = codec_loss(model, batch)
            elif args.stage == "reasoner":
                loss, _ = reasoner_loss(model, batch)
            else:
                loss, _ = generator_loss(model, batch, args.mode, with_audio=args.with_audio)
            values.append(float(loss))
    report = {"stage": args.stage, "mode": args.mode, "seed": 9001,
              "validation_samples": args.samples, "mean_loss": sum(values) / len(values),
              "note": "Held-out synthetic validation only; NOT physics/robot safety evaluation."}
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run()
