"""ACTUAL NVIDIA Cosmos3-Nano Reasoner via the official Transformers integration.

Reference: https://docs.nvidia.com/cosmos/latest/cosmos3/quickstart_guide.html
Preconditions: accepted gated checkpoint access, supported NVIDIA GPU/VRAM,
correct Transformers/CUDA versions. This sample is NOT run in CPU unit tests.
"""
import argparse
from pathlib import Path
import torch
from transformers import AutoProcessor, Cosmos3OmniForConditionalGeneration


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--model", default="nvidia/Cosmos3-Nano")
    p.add_argument("--question", default="Describe the industrial scene and identify visible defects.")
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Official Nano needs a suitable NVIDIA GPU; see the model matrix")
    image = Path(args.image).resolve()
    if not image.is_file():
        raise FileNotFoundError(image)
    processor = AutoProcessor.from_pretrained(args.model)
    model = Cosmos3OmniForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto"
    )
    messages = [{"role": "user", "content": [
        {"type": "image", "path": str(image)},
        {"type": "text", "text": args.question},
    ]}]
    tokens = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device, torch.bfloat16)
    with torch.inference_mode():
        ids = model.generate(**tokens, do_sample=False, max_new_tokens=256)
    new_ids = [out[len(source):] for source, out in zip(tokens.input_ids, ids)]
    print(processor.batch_decode(new_ids, skip_special_tokens=True,
                                 clean_up_tokenization_spaces=False)[0])


if __name__ == "__main__":
    main()
