"""Two-branch transformer with explicit asymmetric cross-modal attention."""
import torch
from torch import nn
from torch.nn import functional as F
from .positions import apply_mrope


class DualStreamMoTBlock(nn.Module):
    """The AR branch is causal and cannot see DM; DM may see both streams.

    Each branch has its own QKV and FFN weights, unlike a single shared FFN.
    Note: NVIDIA's diagram calls the attention *operation* shared; projections
    remain tower-specific. This block follows that interpretation.
    """
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.heads, self.head_dim = heads, width // heads
        self.ar_norm = nn.LayerNorm(width)
        self.dm_norm = nn.LayerNorm(width)
        self.ar_qkv = nn.Linear(width, width * 3)
        self.dm_qkv = nn.Linear(width, width * 3)
        self.ar_out = nn.Linear(width, width)
        self.dm_out = nn.Linear(width, width)
        self.ar_ff_norm = nn.LayerNorm(width)
        self.dm_ff_norm = nn.LayerNorm(width)
        self.ar_ff = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))
        self.dm_ff = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def split_qkv(self, y: torch.Tensor, layer: nn.Linear):
        b, length, width = y.shape
        q, k, v = layer(y).split(width, -1)
        def heads(x):
            return x.reshape(b, length, self.heads, self.head_dim).transpose(1, 2)
        return heads(q), heads(k), heads(v)

    @staticmethod
    def merge(x: torch.Tensor) -> torch.Tensor:
        b, h, n, d = x.shape
        return x.transpose(1, 2).reshape(b, n, h * d)

    def forward(self, ar: torch.Tensor, dm: torch.Tensor | None,
                ar_pos: torch.Tensor, dm_pos: torch.Tensor | None,
                ar_key_valid: torch.Tensor | None = None):
        qa, ka, va = self.split_qkv(self.ar_norm(ar), self.ar_qkv)
        qa, ka = apply_mrope(qa, ar_pos), apply_mrope(ka, ar_pos)
        # Causal mask: AR token i can attend only to AR positions <=i.
        if ar_key_valid is None or bool(ar_key_valid.all()):
            ao = F.scaled_dot_product_attention(qa, ka, va, is_causal=True)
        else:
            # A padded token is neither a valid AR attention key nor DM context.
            n = qa.shape[-2]
            causal = torch.ones(n, n, device=qa.device, dtype=torch.bool).tril()
            allowed = causal[None, None] & ar_key_valid[:, None, None, :]
            ao = F.scaled_dot_product_attention(qa, ka, va, attn_mask=allowed)
        ar = ar + self.ar_out(self.merge(ao))
        ar = ar + self.ar_ff(self.ar_ff_norm(ar))
        if dm is None:
            return ar, None
        qd, kd, vd = self.split_qkv(self.dm_norm(dm), self.dm_qkv)
        qd, kd = apply_mrope(qd, dm_pos), apply_mrope(kd, dm_pos)
        # Critically, a DM query sees ALL AR keys and ALL DM keys.
        keys, values = torch.cat((ka, kd), dim=-2), torch.cat((va, vd), dim=-2)
        if ar_key_valid is None or bool(ar_key_valid.all()):
            do = F.scaled_dot_product_attention(qd, keys, values, is_causal=False)
        else:
            # DM queries see the valid AR tokens plus every DM token.
            dm_valid = torch.ones(ar_key_valid.shape[0], kd.shape[-2],
                                  dtype=torch.bool, device=qd.device)
            key_valid = torch.cat((ar_key_valid, dm_valid), dim=-1)
            do = F.scaled_dot_product_attention(qd, keys, values,
                                                 attn_mask=key_valid[:, None, None, :])
        dm = dm + self.dm_out(self.merge(do))
        dm = dm + self.dm_ff(self.dm_ff_norm(dm))
        return ar, dm
