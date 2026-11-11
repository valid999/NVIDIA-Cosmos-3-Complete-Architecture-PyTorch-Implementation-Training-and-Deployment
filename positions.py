"""Physical-time coordinates and an educational 3D multimodal RoPE."""
import torch


def text_positions(n: int, device: torch.device) -> torch.Tensor:
    i = torch.arange(n, device=device, dtype=torch.float32)
    return torch.stack((i, i, i), dim=-1)


def video_positions(frames: int, height: int, width: int, *, fps: float,
                    temporal_compression: int, base_tps: float,
                    gap: float, start_frame: int = 0,
                    device: torch.device) -> torch.Tensor:
    # The toy codec does not compress time. The effective tokens/s is fps / compression.
    dt = base_tps / (fps / temporal_compression)
    time = (torch.arange(frames, device=device, dtype=torch.float32) + start_frame) * dt + gap
    yy, xx = torch.meshgrid(torch.arange(height, device=device),
                             torch.arange(width, device=device), indexing="ij")
    xy = torch.stack((yy.reshape(-1), xx.reshape(-1)), -1).float()
    return torch.cat((time[:, None, None].expand(-1, height * width, 1),
                      xy[None].expand(frames, -1, -1)), -1).reshape(-1, 3)


def temporal_positions(n: int, *, tokens_per_sec: float, base_tps: float,
                       gap: float, device: torch.device) -> torch.Tensor:
    # Action/audio have temporal coordinates only, with h=w=0.
    t = torch.arange(n, device=device, dtype=torch.float32) * (base_tps / tokens_per_sec) + gap
    return torch.stack((t, torch.zeros_like(t), torch.zeros_like(t)), -1)


def apply_mrope(x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Apply rotary t/h/w components to [batch, heads, tokens, head_dim].

    This captures the 3-axis construction; exact NVIDIA frequency scaling,
    interpolation and per-layer implementation are outside this toy project.
    """
    batch, heads, length, dim = x.shape
    if dim % 6:
        raise ValueError("head_dim must be divisible by 6")
    if coords.shape != (length, 3):
        raise ValueError(f"expected coords [{length},3], got {tuple(coords.shape)}")
    axis_width = dim // 3  # even count of channels per physical axis
    pieces = []
    for axis in range(3):
        v = x[..., axis * axis_width:(axis + 1) * axis_width]
        original_dtype = v.dtype
        v = v.float().reshape(batch, heads, length, axis_width // 2, 2)
        inv = 1.0 / (10000.0 ** (torch.arange(0, axis_width, 2, device=x.device).float() / axis_width))
        angles = coords[:, axis].float()[:, None] * inv[None, :]
        cos, sin = angles.cos()[None, None], angles.sin()[None, None]
        real = v[..., 0] * cos - v[..., 1] * sin
        imag = v[..., 0] * sin + v[..., 1] * cos
        pieces.append(torch.stack((real, imag), -1).reshape(batch, heads, length, axis_width).to(original_dtype))
    return torch.cat(pieces, -1)
