"""Euler rectified-flow sampler, optionally with classifier-free guidance."""
import torch
from .model import DiffusionSegment, MiniCosmos3
from .tokenizer import CharTokenizer


@torch.no_grad()
def sample_joint(model: MiniCosmos3, *, text_ids: torch.Tensor,
                 known: list[DiffusionSegment], targets: list[DiffusionSegment],
                 steps: int = 20, guidance: float = 1.0) -> dict[str, torch.Tensor]:
    """Denoise ALL selected target modalities in the SAME model call per step.

    Latent initial state is Gaussian N(0,I) at sigma=1.
    Model v predicts eps-x0 and we integrate from sigma=1 toward sigma=0.
    Call only after training; random weights produce random outputs.
    """
    if steps <= 0:
        raise ValueError("steps must be positive")
    if not targets:
        raise ValueError("at least one output modality is required")
    model.eval()
    x = {s.name: torch.randn_like(s.features) for s in targets}
    blank = CharTokenizer().prefix("", text_ids.device).repeat(text_ids.shape[0], 1)
    for step in range(steps):
        current = 1.0 - step / steps
        following = 1.0 - (step + 1) / steps
        packed = known + [DiffusionSegment(s.name, s.modality, x[s.name], s.positions,
                             torch.full((x[s.name].shape[0],), current, device=text_ids.device),
                             s.domain) for s in targets]
        guided = model(text_ids, segments=packed).predictions
        if guidance != 1.0:
            unconditional = model(blank, segments=packed).predictions
        for s in targets:
            velocity = guided[s.name]
            if guidance != 1.0:
                velocity = unconditional[s.name] + guidance * (velocity - unconditional[s.name])
            x[s.name] = x[s.name] + (following - current) * velocity
    return x


@torch.no_grad()
def generate_answer(model: MiniCosmos3, text: str, frames: torch.Tensor,
                    *, max_new_tokens: int = 8) -> str:
    """Next-token greedy decoding in the AR tower; DM tower is not activated."""
    from .tokenizer import BOS, SEP, EOS
    tok = CharTokenizer()
    device = frames.device
    seq = [BOS] + tok.encode(text, start=False, end=False) + [SEP]
    emitted = []
    model.eval()
    for _ in range(max_new_tokens):
        ids = torch.tensor([seq], dtype=torch.long, device=device)
        out = model(ids, ar_frames=frames)
        next_token = int(out.text_logits[0, -1].argmax())
        if next_token == EOS:
            break
        emitted.append(next_token)
        seq.append(next_token)
    return tok.decode(emitted)
