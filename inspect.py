"""Print every tensor shape and check the attention direction experimentally."""
import argparse
import torch
from .config import LabConfig
from .model import MiniCosmos3, DiffusionSegment
from .positions import video_positions, temporal_positions
from .synthetic import SyntheticWorld
from .tokenizer import CharTokenizer


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=2)
    opts = parser.parse_args()
    cfg = LabConfig()
    model = MiniCosmos3(cfg).eval()
    samples = [SyntheticWorld(cfg)[i] for i in range(opts.batch_size)]
    videos = torch.stack([s["video"] for s in samples])
    audios = torch.stack([s["audio"] for s in samples])
    actions = torch.stack([s["actions"] for s in samples])
    with torch.no_grad():
        v, hp, wp = model.visual_codec.encode(videos)
        a = model.audio_codec.encode(audios)
    vpos = video_positions(cfg.video_frames, hp, wp, fps=cfg.video_fps,
                            temporal_compression=cfg.temporal_compression,
                            base_tps=cfg.base_tps, gap=cfg.dm_gap, device=torch.device("cpu"))
    apos = temporal_positions(a.shape[1], tokens_per_sec=cfg.audio_sample_rate/cfg.audio_hop,
                              base_tps=cfg.base_tps, gap=cfg.dm_gap, device=torch.device("cpu"))
    mpos = temporal_positions(actions.shape[1], tokens_per_sec=cfg.action_hz,
                              base_tps=cfg.base_tps, gap=cfg.dm_gap, device=torch.device("cpu"))
    tokens = CharTokenizer().prefix(samples[0]["prompt"], torch.device("cpu")).repeat(opts.batch_size, 1)
    segments = [
        DiffusionSegment("rgb_video", "vision", v, vpos, torch.ones(opts.batch_size)),
        DiffusionSegment("stereo_audio", "audio", a, apos, torch.ones(opts.batch_size)),
        DiffusionSegment("robot_action", "action", actions, mpos, torch.ones(opts.batch_size)),
    ]
    out = model(tokens, ar_frames=videos, segments=segments)
    print("RAW INPUTS: text", tuple(tokens.shape), "video", tuple(videos.shape),
          "audio", tuple(audios.shape), "actions", tuple(actions.shape))
    print("ENCODED: vision", tuple(v.shape), "audio", tuple(a.shape),
          "action", tuple(actions.shape))
    print("SAME FORWARD CALL:")
    for name, shape in out.shapes.items():
        print(f"  {name:20s} {shape}")
    # Strong architectural invariant: changing diffusion tokens cannot alter AR outputs.
    altered = [DiffusionSegment(s.name, s.modality, torch.randn_like(s.features),
                                s.positions, s.sigma) for s in segments]
    out_altered = model(tokens, ar_frames=videos, segments=altered)
    max_difference = (out.text_logits - out_altered.text_logits).abs().max().item()
    print("AR logits difference after replacing every DM input:", max_difference)
    assert max_difference < 1e-5, "DM-to-AR information leakage found"
    print("PASS: AR ignores DM, while DM is jointly conditioned on AR+DM")


if __name__ == "__main__":
    run()
