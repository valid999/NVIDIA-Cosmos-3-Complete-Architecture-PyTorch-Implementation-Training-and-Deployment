"""Inference demonstration: text reasoning, video latent sampling, action policy."""
import argparse
from pathlib import Path
import torch
from .train import load_model
from .synthetic import SyntheticWorld, PROMPT
from .tokenizer import CharTokenizer
from .model import DiffusionSegment
from .positions import video_positions, temporal_positions
from .sampling import generate_answer, sample_joint


def run():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="artifacts/cosmos3_toy.pt")
    p.add_argument("--task", choices=["reason", "joint", "inverse", "forward"], default="joint")
    p.add_argument("--prompt", default=PROMPT)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--guidance", type=float, default=1.0)
    p.add_argument("--out", default="artifacts/inferred.pt")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device, require=True).eval()
    c = model.config
    example = SyntheticWorld(c)[0]
    video = example["video"].unsqueeze(0).to(device)
    audio = example["audio"].unsqueeze(0).to(device)
    actions = example["actions"].unsqueeze(0).to(device)
    if args.task == "reason":
        print("Predicted answer:", generate_answer(model, args.prompt, video))
        return
    with torch.no_grad():
        vz, h, w = model.visual_codec.encode(video)
        az = model.audio_codec.encode(audio)
    vp = video_positions(c.video_frames, h, w, fps=c.video_fps,
                         temporal_compression=c.temporal_compression,
                         base_tps=c.base_tps, gap=c.dm_gap, device=device)
    ap = temporal_positions(az.shape[1], tokens_per_sec=c.audio_sample_rate/c.audio_hop,
                             base_tps=c.base_tps, gap=c.dm_gap, device=device)
    cp = temporal_positions(actions.shape[1], tokens_per_sec=c.action_hz,
                             base_tps=c.base_tps, gap=c.dm_gap, device=device)
    n = h * w
    known, future = [], []
    if args.task == "inverse":
        known.append(DiffusionSegment("observed", "vision", vz, vp,
                                      torch.zeros(1, device=device)))
        future.append(DiffusionSegment("action", "action", torch.zeros_like(actions), cp, domain="camera"))
    else:
        known.append(DiffusionSegment("start_frame", "vision", vz[:, :n], vp[:n],
                                      torch.zeros(1, device=device)))
        future.append(DiffusionSegment("future_frames", "vision", torch.zeros_like(vz[:, n:]), vp[n:]))
        if args.task == "forward":
            known.append(DiffusionSegment("given_actions", "action", actions, cp,
                                          torch.zeros(1, device=device)))
        else:
            future.append(DiffusionSegment("action", "action", torch.zeros_like(actions), cp))
            future.append(DiffusionSegment("audio", "audio", torch.zeros_like(az), ap))
    ids = CharTokenizer().prefix(args.prompt, device)
    generated = sample_joint(model, text_ids=ids, known=known, targets=future,
                             steps=args.steps, guidance=args.guidance)
    results = {name: tensor.detach().cpu() for name, tensor in generated.items()}
    if "future_frames" in generated:
        rgb_future = model.visual_codec.decode(generated["future_frames"],
                                                frames=c.video_frames - 1, height=h, width=w)
        results["rgb_future"] = rgb_future.detach().cpu()
    if "audio" in generated:
        results["waveform"] = model.audio_codec.decode(generated["audio"]).detach().cpu()
    path = Path(args.out)
    path.parent.mkdir(exist_ok=True, parents=True)
    torch.save(results, path)
    print("Outputs:", {k: tuple(v.shape) for k, v in results.items()})
    print("Saved", path, "(toy model; not a validated physical simulator)")


if __name__ == "__main__":
    run()
