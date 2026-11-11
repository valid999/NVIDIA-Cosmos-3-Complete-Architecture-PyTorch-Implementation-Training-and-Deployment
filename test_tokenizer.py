import torch
from cosmos3_lab.tokenizer import CharTokenizer, PAD, BOS, SEP, EOS
from cosmos3_lab.positions import temporal_positions


def test_teacher_forcing_alignment():
    t = CharTokenizer()
    x, y = t.answer_batch(["Q"], ["yes"], 10, torch.device("cpu"))
    assert x.shape == (1, 10) and y.shape == (1, 10)
    # [BOS, 'Q', SEP, 'y', 'e', 's'] predicts y from SEP.
    assert x[0, 2] == SEP and y[0, 2] == ord("y") + 6
    assert (y[0, :2] == -100).all()
    assert (y[0, 6:] == -100).all()


def test_physical_time_alignment():
    coords = temporal_positions(4, tokens_per_sec=25, base_tps=6, gap=15000, device=torch.device("cpu"))
    assert torch.allclose(coords[:, 0].diff(), torch.full((3,), 6 / 25), atol=.001)
