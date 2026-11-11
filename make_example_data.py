"""Create a complete camera/audio/actions JSONL manifest for the toy lab.

Usage: python examples/make_example_data.py --out examples/toy_manifest --count 8
It has deliberately 16x16 frames for runnable CPU demonstrations only.
"""
import argparse
import json
import wave
from pathlib import Path
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="examples/toy_manifest")
    parser.add_argument("--count", type=int, default=8)
    args = parser.parse_args()
    from cosmos3_lab.config import LabConfig
    from cosmos3_lab.synthetic import SyntheticWorld
    cfg = LabConfig()
    data = SyntheticWorld(cfg, count=args.count)
    root = Path(args.out)
    (root / "frames").mkdir(parents=True, exist_ok=True)
    (root / "audio").mkdir(parents=True, exist_ok=True)
    (root / "actions").mkdir(parents=True, exist_ok=True)
    records = []
    for i in range(args.count):
        sample = data[i]
        frame_files = []
        for t, frame in enumerate(sample["video"]):
            filename = f"frames/{i:04d}_{t:02d}.png"
            image = (frame.permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
            Image.fromarray(image).save(root / filename)
            frame_files.append(filename)
        audio_name = f"audio/{i:04d}.wav"
        pcm = (sample["audio"].T.numpy() * 32767).round().clip(-32768, 32767).astype("<i2")
        with wave.open(str(root / audio_name), "wb") as dest:
            dest.setnchannels(cfg.audio_channels)
            dest.setsampwidth(2)
            dest.setframerate(cfg.audio_sample_rate)
            dest.writeframes(pcm.tobytes())
        actions_name = f"actions/{i:04d}.json"
        (root / actions_name).write_text(json.dumps(sample["actions"].tolist()), encoding="utf-8")
        records.append({"frames": frame_files, "audio": audio_name, "actions": actions_name,
                        "prompt": sample["prompt"], "answer": sample["answer"]})
    (root / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    print("Manifest generated:", root / "manifest.jsonl", "with", len(records), "samples")


if __name__ == "__main__":
    main()
