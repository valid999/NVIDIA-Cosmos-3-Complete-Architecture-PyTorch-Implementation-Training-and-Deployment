# Cosmos 3 Learning Lab — runnable PyTorch teaching implementation

**What this is:** an independently authored, ~0.54M-parameter CPU-runnable model that demonstrates the central **Cosmos 3 architecture concepts** (AR/DM tokens, dual towers, multi-input modality adapters, causal/full asymmetric attention, 3D temporal/spatial RoPE, flow matching, clean/noisy conditioning, multimodal outputs, domain-specific action projections, CFG and Euler sampling).

**What this is NOT:** it is not NVIDIA's official Cosmos 3 model, cannot load NVIDIA weights, is not a pre-trained industrial defect detector, has no physics guarantees, and is NOT suitable for direct robot/PLC control.

Primary scientific source: [NVIDIA Cosmos 3 technical report (2026)](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf). Official product code: [github.com/NVIDIA/cosmos](https://github.com/NVIDIA/cosmos). Official training/inference framework: [NVIDIA/cosmos-framework](https://github.com/NVIDIA/cosmos-framework).

## Recommended reading order

1. **README.md** — setup and commands.
2. **ARCHITECTURE_GUIDE.md** — paper-to-tensor translation, equations, scenarios and nuances.
3. **CODE_WALKTHROUGH.md** — function-by-function code review.
4. **PRODUCTION_GUIDE.md** — actual Cosmos 3, datasets, post-training and serving.
5. `src/cosmos3_lab/{positions,encoders,attention,model,training}.py` — study these in that order.
6. `tests/` — verify the invariants while changing the source.

## Directory tree

```text
cosmos3_learning_project/
├── pyproject.toml                     # pip-installable package
├── README.md                          # first-time onboarding
├── ARCHITECTURE_GUIDE.md             # full architecture and mathematics
├── CODE_WALKTHROUGH.md               # annotated source walkthrough
├── PRODUCTION_GUIDE.md               # real checkpoints and deployment
├── Dockerfile                         # example CPU serving (demo only)
├── examples/
│   ├── make_example_data.py          # builds real PNG+WAV+JSON manifest
│   ├── official_nvidia_reasoner.py   # actual NVIDIA Nano via Transformers
│   ├── official_nvidia_generator.py  # actual NVIDIA Nano via Diffusers
│   └── manifest.example.jsonl
├── src/cosmos3_lab/
│   ├── config.py, tokenizer.py
│   ├── encoders.py, positions.py, attention.py, model.py
│   ├── training.py, synthetic.py, manifest.py, action_geometry.py
│   ├── train.py, infer.py, sampling.py, inspect.py, evaluate.py, serve.py
└── tests/
    ├── test_architecture.py
    └── test_tokenizer.py
```

## Quick start (Windows PowerShell or Linux/macOS)

All the *toy learning* commands below run on CPU and do not need CUDA.

**Windows PowerShell:**

```powershell
cd cosmos3_learning_project
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[test,api]"
pytest -q
python -m cosmos3_lab.inspect --batch-size 2
python -m cosmos3_lab.train --stage codec --steps 150 --batch-size 4
python -m cosmos3_lab.train --stage reasoner --steps 300 --batch-size 4
python -m cosmos3_lab.train --stage generator --mode all --steps 300 --batch-size 4
python -m cosmos3_lab.evaluate --stage generator --mode all
python -m cosmos3_lab.infer --task joint --steps 20 --out artifacts/joint.pt
python -m cosmos3_lab.infer --task reason
```

**Linux/macOS:**

```bash
cd cosmos3_learning_project
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[test,api]'
pytest -q
python -m cosmos3_lab.inspect --batch-size 2
python -m cosmos3_lab.train --stage codec --steps 150 --batch-size 4
python -m cosmos3_lab.train --stage reasoner --steps 300 --batch-size 4
python -m cosmos3_lab.train --stage generator --mode all --steps 300 --batch-size 4
python -m cosmos3_lab.evaluate --stage generator --mode all
python -m cosmos3_lab.infer --task joint --steps 20 --out artifacts/joint.pt
```

The 150/300-step counts are **illustrative**, not convergence claims or reliable performance settings. Shorter smoke tests: replace each `--steps` with `1`.

### Read actual PNG/WAV/JSON data

```bash
python examples/make_example_data.py --out examples/toy_manifest --count 8
python -m cosmos3_lab.train --stage reasoner --steps 5 --batch-size 2 \
  --manifest examples/toy_manifest/manifest.jsonl
```

All manifest entries have 3 PNG frames (16×16), 1280 stereo PCM16 samples at 8 kHz, 2 actions of 9 floats, prompt and answer. `manifest.py` validates each input; real factory data needs a **separate high-resolution** preprocessor rather than blindly resizing scratches to 16 pixels.

### Turn on demo serving

```bash
# After training; localhost only. This is NOT a production Cosmos service.
uvicorn cosmos3_lab.serve:app --host 127.0.0.1 --port 8080
# GET http://127.0.0.1:8080/health
# POST /reason expects JSON: {"question":"...", "image_base64":"..."}
```

### Check different model targets

```bash
python -m cosmos3_lab.train --stage generator --mode t2i --steps 5
python -m cosmos3_lab.train --stage generator --mode t2v --with-audio --steps 5
python -m cosmos3_lab.train --stage generator --mode i2v --steps 5
python -m cosmos3_lab.train --stage generator --mode v2v --steps 5
python -m cosmos3_lab.train --stage generator --mode transfer --steps 5
python -m cosmos3_lab.train --stage generator --mode forward --steps 5
python -m cosmos3_lab.train --stage generator --mode inverse --steps 5
python -m cosmos3_lab.train --stage generator --mode policy --steps 5
```

Each command updates the *same checkpoint*: train on mixed modes for a unified model; use separate checkpoints for fair comparisons/ablations.

## Measured CPU smoke test

On this project's creation, PyTorch CPU completed one training update for each of `codec`, `reasoner`, and `generator`, followed by joint latent image/audio/action sampling. The `inspect` CLI printed the exact shapes below for `B=2`:

```text
text          [2, 56]             (integer token IDs)
video         [2, 3, 3, 16, 16]  (RGB frames)
audio         [2, 2, 1280]       (stereo waveform)
actions       [2, 2, 9]          (9D camera-pose actions)
encoded video [2, 48, 12]
encoded audio [2, 8, 8]
AR combined   [2, 104, 96]      (48 vision + 56 text tokens)
DM combined   [2, 58, 96]       (48 vision + 8 audio + 2 action tokens)
output text   [2, 56, 134]      (toy vocabulary logits)
output video  [2, 48, 12]       (velocity per latent token)
output audio  [2, 8, 8]
output action [2, 2, 9]
```

The AR invariance test replaces all DM inputs with random tensors and verifies that the AR output does not change. A second regression test verifies that padded, unequal-length prompts in one minibatch produce the same predictions as processing an example alone. This is important: **DM information must never leak into the AR token generation path**.

## Learning milestones

1. Identify raw dimensions for each modality and normalize input ranges.
2. Understand token projection, tokens-per-second and [t,h,w] coordinates.
3. Test causal AR and bidirectional DM attention.
4. Train codecs and verify reconstruction fidelity.
5. Train the AR tower on masked next-token loss.
6. Train the DM tower with rectified flow and per-task conditioning masks.
7. Generate several outputs in a single denoiser call.
8. Replace synthetic samples with your own aligned manifest and study metrics.
9. Move to **official pretrained Cosmos 3** for actual useful visual world prediction.

## License and provenance

This is original educational code that *describes concepts from* the NVIDIA Cosmos 3 report. It does not package NVIDIA's model weights, datasets, guarded dependencies, or source code. Review NVIDIA's [OpenMDW-1.1 terms](https://openmdw.ai/) before deploying or adapting official assets. 
