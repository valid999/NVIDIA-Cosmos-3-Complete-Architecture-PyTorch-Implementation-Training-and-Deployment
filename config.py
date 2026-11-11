"""Model dimensions and physical sampling rates are configured centrally."""
from dataclasses import asdict, dataclass, field


@dataclass
class LabConfig:
    # Small values are intentional: on CPU they make attention inspectable.
    width: int = 96
    heads: int = 4
    layers: int = 2
    image_size: int = 16
    video_frames: int = 3
    visual_patch: int = 4
    visual_latent: int = 12
    audio_latent: int = 8
    audio_samples: int = 1280
    audio_channels: int = 2
    audio_hop: int = 160
    audio_sample_rate: int = 8000
    video_fps: float = 24.0
    action_hz: float = 10.0
    temporal_compression: int = 1  # Toy codec; NVIDIA uses 4, NOT 1.
    base_tps: float = 6.0  # NVIDIA 24 FPS / 4 latent-frame compression.
    dm_gap: float = 15000.0
    max_text: int = 96
    action_dims: dict[str, int] = field(default_factory=lambda: {
        "camera": 9, "vehicle": 9, "single_arm": 10,
        "dual_arm": 20, "humanoid": 29, "egocentric": 57,
    })

    def __post_init__(self) -> None:
        assert self.width % self.heads == 0
        assert (self.width // self.heads) % 6 == 0, "3D RoPE needs even-sized t/h/w sections"
        assert self.image_size % self.visual_patch == 0
        assert self.image_size % 4 == 0
        assert self.audio_samples % self.audio_hop == 0

    def to_dict(self) -> dict:
        return asdict(self)
