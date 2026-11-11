"""Transparent ASCII character tokenizer (NOT the real Cosmos tokenizer)."""
import torch

PAD, BOS, EOS, SEP, BOG, UNK = range(6)
VOCAB_SIZE = 6 + 128


class CharTokenizer:
    vocab_size = VOCAB_SIZE

    def encode(self, text: str, *, start=True, end=True) -> list[int]:
        ids = [BOS] if start else []
        ids.extend(6 + ord(c) if ord(c) < 128 else UNK for c in text)
        if end:
            ids.append(EOS)
        return ids

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i - 6) for i in ids if 6 <= i < VOCAB_SIZE)

    def prefix(self, text: str, device: torch.device) -> torch.Tensor:
        # AR prefix for generation: the special BOG token closes the prompt.
        ids = self.encode(text, start=True, end=True) + [BOG]
        return torch.tensor([ids], dtype=torch.long, device=device)

    def batch_prefix(self, prompts: list[str], device: torch.device) -> torch.Tensor:
        """Pad independent prompts within a batch (the model masks PAD keys)."""
        encoded = [self.encode(text, start=True, end=True) + [BOG] for text in prompts]
        max_length = max(len(row) for row in encoded)
        return torch.tensor([row + [PAD] * (max_length - len(row)) for row in encoded],
                            dtype=torch.long, device=device)

    def answer_batch(self, prompts: list[str], answers: list[str], max_len: int,
                     device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Teacher forcing: CE only on the answer characters and EOS."""
        encoded, targets = [], []
        for prompt, answer in zip(prompts, answers):
            full = [BOS] + self.encode(prompt, start=False, end=False) + [SEP]
            sep_index = len(full) - 1  # input at SEP predicts first answer char
            full += self.encode(answer, start=False, end=False) + [EOS]
            if len(full) > max_len + 1:
                raise ValueError("Increase max_text or shorten prompt/answer")
            inp, lab = full[:-1], full[1:]
            lab = [-100 if i < sep_index else token for i, token in enumerate(lab)]
            inp += [PAD] * (max_len - len(inp))
            lab += [-100] * (max_len - len(lab))
            encoded.append(inp)
            targets.append(lab)
        return (torch.tensor(encoded, dtype=torch.long, device=device),
                torch.tensor(targets, dtype=torch.long, device=device))
