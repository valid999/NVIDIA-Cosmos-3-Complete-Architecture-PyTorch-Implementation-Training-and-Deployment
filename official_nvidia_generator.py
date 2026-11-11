"""ACTUAL NVIDIA Cosmos3-Nano generator, using official Diffusers quickstart.

Linux + supported CUDA GPU + gated nvidia/Cosmos3-Nano and Cosmos-1.0-Guardrail
access required. This example is not tested on this CPU-only environment.
"""
import argparse
import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.utils import export_to_video


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default="A mobile robot navigates a warehouse aisle and stops at a shelf.")
    p.add_argument("--output", default="cosmos3_generated.mp4")
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Cosmos3-Nano requires appropriate NVIDIA GPU memory")
    pipe = Cosmos3OmniPipeline.from_pretrained(
        "nvidia/Cosmos3-Nano", torch_dtype=torch.bfloat16, device_map="cuda"
    )
    result = pipe(prompt=args.prompt)
    export_to_video(result.video, args.output, fps=24)
    print("Saved", args.output)


if __name__ == "__main__":
    main()
