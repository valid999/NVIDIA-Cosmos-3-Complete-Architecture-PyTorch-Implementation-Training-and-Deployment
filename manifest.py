"""Load small real-world synchronized samples without torch.load or video codecs.

Expected manifest JSONL row fields:
 {"frames": ["frames/0.png", "frames/1.png", "frames/2.png"],
  "audio": "audio/sample.wav", "actions": "actions/sample.json",
  "prompt": "Does the product show a scratch?", "answer": "yes"}

Audio must already be stereo, PCM16, at the toy training sample rate.
Never resample or drop frames silently: alignment is part of the learning task.
"""
import json
from pathlib import Path
import wave
import numpy as np
import torch
from torch.utils.data import Dataset
from .config import LabConfig
from .action_geometry import validate_actions


class ManifestWorld(Dataset):
    def __init__(self, manifest_file: str, config: LabConfig):
        self.file = Path(manifest_file).resolve()
        self.root = self.file.parent
        self.config = config
        self.entries = [json.loads(line) for line in self.file.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not self.entries:
            raise ValueError("Empty manifest")

    def __len__(self):
        return len(self.entries)

    def resolve(self, relative: str) -> Path:
        candidate = (self.root / relative).resolve()
        if not candidate.is_relative_to(self.root):
            raise ValueError("Sample path escapes manifest folder")
        return candidate

    def __getitem__(self, idx: int):
        from PIL import Image
        row, cfg = self.entries[idx], self.config
        if len(row["frames"]) != cfg.video_frames:
            raise ValueError("Frame count mismatch")
        frames = []
        for filename in row["frames"]:
            with Image.open(self.resolve(filename)) as im:
                image = im.convert("RGB").resize((cfg.image_size, cfg.image_size))
                frames.append(torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255.0)
        with wave.open(str(self.resolve(row["audio"])), "rb") as wav:
            if (wav.getnchannels(), wav.getframerate(), wav.getsampwidth()) != (
                    cfg.audio_channels, cfg.audio_sample_rate, 2):
                raise ValueError("WAV must be stereo PCM16 at configured sample rate")
            raw = wav.readframes(wav.getnframes())
            audio = np.frombuffer(raw, dtype="<i2").astype(np.float32).reshape(-1, cfg.audio_channels)
            if len(audio) != cfg.audio_samples:
                raise ValueError("Audio sample count mismatch; align the clip explicitly")
            waveform = torch.from_numpy((audio / 32768.0).T.copy())
        actions = torch.tensor(json.loads(self.resolve(row["actions"]).read_text()), dtype=torch.float32)
        if actions.shape != (cfg.video_frames - 1, cfg.action_dims["camera"]):
            raise ValueError("Actions must be [T-1,9] for camera domain")
        validate_actions(actions.unsqueeze(0), cfg.action_dims["camera"])
        return {"video": torch.stack(frames), "audio": waveform, "actions": actions,
                "prompt": str(row["prompt"]), "answer": str(row["answer"])}
