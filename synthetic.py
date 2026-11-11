"""Reproducible toy trajectories: moving part + occasional surface scratch + stereo sound.

No factory data, licensing ambiguity, or external downloads are required.
"""
import math
import torch
from torch.utils.data import Dataset
from .config import LabConfig


PROMPT = "Does the moving part have a defect? Answer yes or no:"


class SyntheticWorld(Dataset):
    def __init__(self, config: LabConfig, count: int = 256, seed: int = 23):
        self.config, self.count, self.seed = config, count, seed

    def __len__(self):
        return self.count

    def __getitem__(self, idx: int) -> dict:
        c = self.config
        h = w = c.image_size
        generator = torch.Generator().manual_seed(self.seed + idx)
        defect = idx % 2 == 0
        video = torch.zeros(c.video_frames, 3, h, w)
        video[:] = torch.tensor([.20, .36, .42]).view(1, 3, 1, 1)
        x0 = int(torch.randint(2, max(3, w - 5), (1,), generator=generator))
        for t in range(c.video_frames):
            x = min(w - 4, x0 + t)
            y = h // 2 - 2
            video[t, :, y:y + 4, x:x + 4] = torch.tensor([.7, .7, .76]).view(3, 1, 1)
            if defect:
                video[t, 0, y + 1, x:x + 3] = .95  # colored scratch, visible under close inspection
                video[t, 1, y + 1, x:x + 3] = .08
        # 2-channel sound (the channels are intentionally different for testing).
        seconds = torch.arange(c.audio_samples).float() / c.audio_sample_rate
        frequency = 440.0 if defect else 220.0
        wave = .35 * torch.sin(2 * math.pi * frequency * seconds)
        audio = torch.stack((wave, .75 * wave), dim=0)
        # 'camera' embodiment 9D = translation 3D + continuous 6D rotation.
        actions = torch.zeros(c.video_frames - 1, c.action_dims["camera"])
        actions[:, 0] = 1.0 / w
        actions[:, 3] = 1.0
        actions[:, 7] = 1.0
        return {"video": video, "audio": audio, "actions": actions,
                "prompt": PROMPT, "answer": "yes" if defect else "no"}
