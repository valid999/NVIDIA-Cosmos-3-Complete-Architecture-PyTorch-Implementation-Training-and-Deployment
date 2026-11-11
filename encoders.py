"""Intentionally tiny learned codecs: training stage is REQUIRED before meaningful decoding.

Real Cosmos 3 uses a pretrained ViT, Wan 2.2 video VAE and pretrained audio VAE.
Those large pretrained components are NOT contained in this learning lab.
"""
import torch
from torch import nn


class VisionUnderstandingEncoder(nn.Module):
    def __init__(self, width: int, patch: int):
        super().__init__()
        self.patch = patch
        self.stem = nn.Conv2d(3, width, kernel_size=patch, stride=patch)
        self.project = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))

    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        # frames [B,T,3,H,W] -> patch embeddings [B,T*Hp*Wp,D]
        b, t, c, h, w = frames.shape
        patches = self.stem(frames.reshape(b * t, c, h, w))
        hp, wp = patches.shape[-2:]
        embeddings = patches.flatten(2).transpose(1, 2).reshape(b, t * hp * wp, -1)
        return self.project(embeddings), hp, wp


class ToyVisualCodec(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.encoder = nn.Conv2d(3, channels, kernel_size=4, stride=4)
        self.decoder = nn.ConvTranspose2d(channels, 3, kernel_size=4, stride=4)

    def encode(self, frames: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        b, t, c, h, w = frames.shape
        z = self.encoder(frames.reshape(b * t, c, h, w)).tanh()
        hp, wp = z.shape[-2:]
        return z.permute(0, 2, 3, 1).reshape(b, t * hp * wp, -1), hp, wp

    def decode(self, latents: torch.Tensor, *, frames: int, height: int, width: int) -> torch.Tensor:
        # z [B,T*Hp*Wp,C] -> RGB video [B,T,3,4*Hp,4*Wp]
        b, n, c = latents.shape
        if n != frames * height * width:
            raise ValueError("latents length does not match video shape")
        z = latents.reshape(b * frames, height, width, c).permute(0, 3, 1, 2)
        img = torch.sigmoid(self.decoder(z))
        return img.reshape(b, frames, 3, height * 4, width * 4)


class ToyAudioCodec(nn.Module):
    def __init__(self, channels: int, latent: int, hop: int):
        super().__init__()
        self.hop = hop
        self.encoder = nn.Conv1d(channels, latent, kernel_size=hop, stride=hop)
        self.decoder = nn.ConvTranspose1d(latent, channels, kernel_size=hop, stride=hop)

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        # audio [B,2,samples] -> one audio token per hop: [B,N,Da]
        return torch.tanh(self.encoder(audio)).transpose(1, 2)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        # reconstruction [B,2,samples]; tanh keeps amplitude in [-1,+1]
        return torch.tanh(self.decoder(latents.transpose(1, 2)))
