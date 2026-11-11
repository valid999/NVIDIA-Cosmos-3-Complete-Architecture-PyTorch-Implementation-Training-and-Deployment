import torch
import pytest
from cosmos3_lab.config import LabConfig
from cosmos3_lab.encoders import VisionUnderstandingEncoder
from cosmos3_lab.model import MiniCosmos3, DiffusionSegment
from cosmos3_lab.positions import video_positions, temporal_positions, apply_mrope
from cosmos3_lab.synthetic import SyntheticWorld
from cosmos3_lab.tokenizer import CharTokenizer
from cosmos3_lab.training import codec_loss, reasoner_loss, generator_loss, MODES, noised
from cosmos3_lab.sampling import sample_joint
from cosmos3_lab.action_geometry import rotation6d_to_matrix_svd


@pytest.fixture(scope="module")
def tensors():
    torch.manual_seed(21)
    cfg = LabConfig(layers=1)
    model = MiniCosmos3(cfg)
    s = SyntheticWorld(cfg)[0]
    batch = {k: v.unsqueeze(0) if torch.is_tensor(v) else [v]
             for k, v in s.items()}
    return cfg, model, batch


def test_codec_shapes(tensors):
    cfg, model, b = tensors
    z, h, w = model.visual_codec.encode(b["video"])
    assert z.shape == (1, cfg.video_frames * h * w, cfg.visual_latent)
    assert model.visual_codec.decode(z, frames=cfg.video_frames, height=h, width=w).shape == b["video"].shape
    a = model.audio_codec.encode(b["audio"])
    assert a.shape == (1, cfg.audio_samples // cfg.audio_hop, cfg.audio_latent)
    assert model.audio_codec.decode(a).shape == b["audio"].shape


def test_mrope_norm_preserved():
    torch.manual_seed(1)
    x = torch.randn(2, 4, 5, 24)
    pos = torch.randn(5, 3)
    y = apply_mrope(x, pos)
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=2e-5)


def test_multi_input_single_forward_and_no_leakage(tensors):
    cfg, model, b = tensors
    model.eval()
    with torch.no_grad():
        z, h, w = model.visual_codec.encode(b["video"])
        a = model.audio_codec.encode(b["audio"])
    vp = video_positions(cfg.video_frames, h, w, fps=cfg.video_fps,
                         temporal_compression=1, base_tps=6, gap=cfg.dm_gap, device=z.device)
    ap = temporal_positions(a.shape[1], tokens_per_sec=50, base_tps=6, gap=cfg.dm_gap, device=z.device)
    cp = temporal_positions(2, tokens_per_sec=10, base_tps=6, gap=cfg.dm_gap, device=z.device)
    ids = CharTokenizer().prefix("test", torch.device("cpu"))
    segments = [DiffusionSegment("vision", "vision", z, vp, torch.ones(1)),
                DiffusionSegment("audio", "audio", a, ap, torch.ones(1)),
                DiffusionSegment("action", "action", b["actions"], cp, torch.ones(1))]
    with torch.no_grad():
        a1 = model(ids, ar_frames=b["video"], segments=segments)
        changed = [DiffusionSegment(s.name, s.modality, torch.randn_like(s.features),
                                    s.positions, s.sigma) for s in segments]
        a2 = model(ids, ar_frames=b["video"], segments=changed)
    assert a1.predictions["vision"].shape == z.shape
    assert a1.predictions["audio"].shape == a.shape
    assert a1.predictions["action"].shape == b["actions"].shape
    assert torch.equal(a1.text_logits, a2.text_logits)


def test_flow_interpolation_target():
    clean = torch.randn(2, 4, 9)
    pos = torch.zeros(4, 3)
    item, target = noised("action", "action", clean, pos)
    sigma = item.sigma[:, None, None]
    epsilon = item.features + (1 - sigma) * target
    assert torch.allclose(target, epsilon - clean, atol=1e-5)


@pytest.mark.parametrize("mode", MODES)
def test_generator_modes_finite(tensors, mode):
    cfg, model, b = tensors
    model.train()
    loss, info = generator_loss(model, b, mode)
    assert torch.isfinite(loss)
    assert info["losses"]


def test_ar_and_codec_losses(tensors):
    _, model, b = tensors
    loss1, _ = codec_loss(model, b)
    loss2, _ = reasoner_loss(model, b)
    assert loss1.item() > 0 and loss2.item() > 0
    loss2.backward()
    assert model.text_embedding.weight.grad is not None


def test_realistic_action_domains_and_so3(tensors):
    cfg, model, b = tensors
    ids = CharTokenizer().prefix("robot", b["video"].device)
    for domain in cfg.action_dims:
        dim = cfg.action_dims[domain]
        spec = DiffusionSegment("act", "action", torch.randn(1, 2, dim),
                                torch.zeros(2, 3), torch.ones(1), domain)
        assert model(ids, segments=[spec]).predictions["act"].shape == (1, 2, dim)
    r = rotation6d_to_matrix_svd(torch.tensor([[1., 0., 0., 0., 1., 0.]]))
    eye = torch.eye(3)
    assert torch.allclose(r @ r.transpose(-2, -1), eye[None], atol=1e-5)
    assert torch.all(torch.linalg.det(r) > 0)


def test_sampling_shapes(tensors):
    _, model, _ = tensors
    ids = CharTokenizer().prefix("robot", torch.device("cpu"))
    target = DiffusionSegment("action", "action", torch.zeros(1, 2, 9),
                              torch.zeros(2, 3), domain="camera")
    answer = sample_joint(model, text_ids=ids, known=[], targets=[target], steps=2)
    assert answer["action"].shape == (1, 2, 9)


def test_audio_optional_on_video_mode(tensors):
    _, model, b = tensors
    _, without = generator_loss(model, b, "t2v", with_audio=False)
    _, with_sound = generator_loss(model, b, "t2v", with_audio=True)
    assert "audio_future" not in without["losses"]
    assert "audio_future" in with_sound["losses"]


def test_batch_padding_does_not_change_first_example(tensors):
    """Different prompt lengths must not change a sample's output via PAD keys."""
    _, model, _ = tensors
    model.eval()
    tok = CharTokenizer()
    device = torch.device("cpu")
    first_alone = tok.prefix("short", device)
    pair = tok.batch_prefix(["short", "a dramatically longer independent prompt"], device)
    pos = torch.zeros(3, 3)
    features = torch.randn(1, 3, 9)
    with torch.no_grad():
        lone = model(first_alone, segments=[DiffusionSegment("a", "action", features, pos)])
        paired = model(pair, segments=[DiffusionSegment("a", "action", features.expand(2, -1, -1), pos)])
    assert torch.allclose(lone.predictions["a"][0], paired.predictions["a"][0], atol=1e-5)
    assert torch.allclose(lone.text_logits[0], paired.text_logits[0, :first_alone.shape[1]], atol=1e-5)


def test_generator_loss_with_distinct_minibatch_prompts(tensors):
    cfg, model, b = tensors
    b2 = {key: tensor.repeat(2, *([1] * (tensor.ndim - 1)))
          if torch.is_tensor(tensor) else ["short", "a substantially longer string"]
          for key, tensor in b.items()}
    # Verifies generator_loss handles *both* sample-specific text conditions.
    model.train()
    loss, detail = generator_loss(model, b2, mode="all")
    assert torch.isfinite(loss)
    assert detail["shapes"]["ar_total"][0] == 2
