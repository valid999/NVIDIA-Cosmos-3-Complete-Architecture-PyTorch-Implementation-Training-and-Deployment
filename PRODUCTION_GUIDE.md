# From educational prototype to an actual NVIDIA Cosmos 3 physical-AI project

## 1. First, choose the correct implementation target

The `src/cosmos3_lab/` code is **a study tool**, not NVIDIA's model. If the objective is publishable results on real video/action generation or practical manufacturing inspection, use the official **pretrained model checkpoints and Cosmos Framework**. Training a credible omnimodal world model from scratch requires extremely large curated datasets and hardware; NVIDIA reports 1024–2048 GB200 GPU configurations for large generator pretraining. The small prototype is most useful for understanding and debugging the algorithms.

Official links checked against the maintained sources as of November 5, 2026:

- [Cosmos 3 paper](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf)
- [Core project](https://github.com/NVIDIA/cosmos)
- [Training/inference framework](https://github.com/NVIDIA/cosmos-framework)
- [Framework inference modes and exact command flags](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/inference.md)
- [Cosmos3 installation](https://docs.nvidia.com/cosmos/latest/cosmos3/installation.html)
- [Quickstart](https://docs.nvidia.com/cosmos/latest/cosmos3/quickstart_guide.html)
- [Model matrix and GPU guidance](https://docs.nvidia.com/cosmos/latest/cosmos3/model_matrix.html)
- [Model collection](https://huggingface.co/collections/nvidia/cosmos3)
- [Guardrail access](https://huggingface.co/nvidia/Cosmos-1.0-Guardrail)

## 2. Hardware model selection

| Family | Total MoT parameters | Approximate expert/backbone scale | Official recommendation | Use case |
|---|---:|---:|---|---|
| Edge | 4B | 2B branch/base | Jetson AGX Orin/Thor; RTX Pro 6000 | edge reasoning and action without sound generation |
| Nano | 16B | 8B branch/base | RTX PRO 6000, H100 or B200 | research, sim, policy adaptation |
| Super | 64B | 32B branch/base | H200, B200 or GB200 nodes | highest-fidelity teacher / generation |

**Important count nuance:** "4B/16B/64B" describes the full dual-path family; some NVIDIA Framework docs refer to the *active/dense backbone* as 2B/8B/32B. These counts should not be compared as if they described identical model scope.

At BF16, weights alone would require about 8 GB / 32 GB / 128 GB for 4B / 16B / 64B **before** accounting for activation buffers, visual/audio encoders, latents, KV states, offloading, optimizer state or framework overhead. Loading/training even Edge on a 4-GB consumer GPU is generally impractical without substantial quantization/offloading and compromises. Use the toy project locally and a supported server/cloud workstation for actual models.

## 3. Actual NVIDIA reasoning inference (Transformers)

The official docs have a verified Nano Reasoner API. On a supported **Linux+NVIDIA GPU** host (and after accepting gated model terms):

```bash
# Create and activate an isolated Python environment.
uv venv --python 3.13 --seed --managed-python
source .venv/bin/activate

# Install compatible CUDA-capable dependencies as in NVIDIA's quickstart.
uv pip install --torch-backend=auto \
  accelerate av pillow 'safetensors>=0.8.0' torch \
  'torchvision==0.25.0' 'transformers>=5.11.0'

# Authenticate through your own Hugging Face account; do not put tokens in source code.
uvx hf@latest auth login

# Run the included independently formatted but NVIDIA-backed example.
python examples/official_nvidia_reasoner.py --image example.jpg \
  --question 'Which surface defects are visible? Describe the evidence.'
```

Under the hood, that script calls `AutoProcessor.from_pretrained`, `Cosmos3OmniForConditionalGeneration.from_pretrained`, `apply_chat_template` with typed image and text content, `generate`, and then removes the prefix tokens to decode only the new answer. Unlike the toy code, it loads actual checkpoint weights and tokenization.

**Note**: Cosmos3-Edge has a **different** Transformers integration (`AutoModelForImageTextToText` / `Cosmos3EdgeForConditionalGeneration`); don't use the Nano class for Edge without checking the current cookbook.

## 4. Actual NVIDIA audiovisual generation (Diffusers)

Per the official Cosmos README:

```bash
uv venv --python 3.13 --seed --managed-python
source .venv/bin/activate
uv pip install --torch-backend=auto \
  'diffusers @ git+https://github.com/huggingface/diffusers.git' \
  accelerate av cosmos_guardrail huggingface_hub imageio imageio-ffmpeg \
  torch torchvision transformers
uvx hf@latest auth login
python examples/official_nvidia_generator.py \
  --prompt 'A robotic gripper transfers a manufactured panel; the camera is stationary.' \
  --output generated_panel.mp4
```

The included `examples/official_nvidia_generator.py` uses `Cosmos3OmniPipeline.from_pretrained('nvidia/Cosmos3-Nano', torch_dtype=torch.bfloat16, device_map='cuda')` and exports the returned video to MP4. Access to gated guardrail assets is a separate prerequisite; login alone does not grant model/guardrail access.

For image-to-video, audiovisual output with synchronized sound, transfer conditioning, or actions, use the maintained corresponding official cookbook; do not assume every option is supported by every generic Diffusers call.

## 5. Actual NVIDIA multimodal action and world-model inference (Cosmos Framework)

Use the [official Framework install guide](https://github.com/NVIDIA/cosmos-framework) for the tested environment and download permissions. From the Framework's configured directory, the official inference entrypoint accepts sample argument JSON/JSONL/YAML, plus the checkpoint name. Supported `model_mode` strings include:

- `text2image`, `text2video`, `image2video`, `video2video`;
- `forward_dynamics`: provide `domain_name`, `vision_path`, `action_path`;
- `inverse_dynamics`: provide `domain_name`, `vision_path`;
- `wam` (world-action/policy mode): provide `domain_name`, `vision_path`.

Example **official** single-GPU Edge CLI, after completing framework installation:

```bash
python -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  -i 'inputs/omni/t2i.json' \
  -o outputs/omni_edge \
  --checkpoint-path Cosmos3-Edge \
  --seed=0
```

Example action-conditioning file references are in NVIDIA's `inputs/omni/action_forward_dynamics_camera.json` and `inputs/omni/action_inverse_dynamics_robot.json`. Run the **existing official sample files first**, then modify a copy to point to your validated data. Required action JSON fields depend on embodiment and must follow the actual model schema.

Example multi-GPU Nano throughput command **from the Framework documentation**:

```bash
torchrun --nproc-per-node=8 -m cosmos_framework.scripts.inference \
  --parallelism-preset=throughput \
  -i 'inputs/omni/*.json' \
  -o outputs/omni_nano \
  --checkpoint-path Cosmos3-Nano \
  --seed=0
```

This is an example distributed deployment, not a statement that Nano always requires 8 GPUs; the framework also documents a latency-mode single-GPU command for sufficiently large GPUs.

### How the Framework maps I/O onto the same architecture

- `vision_path` = conditioning image or video; VAE/ViT processing is selected according to task.
- `action_path` = frame-aligned input action sequence; action dimension governed by `domain_name`.
- `prompt` = language condition; tokenized and added as AR prefix.
- generated `vision.mp4` = video latents denoised and decoded.
- generated `sample_outputs.json` = predicted actions and execution metadata.

For multiview Auto camera/LiDAR examples, NVIDIA documents multiple `multiview.views` with per-camera controls and captions plus LiDAR range data. These are **specialized checkpoints and input schemas**; the generic example above does not automatically grant multi-view fusion in a vanilla 16B checkpoint.

## 6. Proper post-training of a pretrained model on industrial data

### 6.1 Data acquisition (the camera-side contract)

A useful training sample should carry at least:

```text
trajectory_id, product_id, defect_id, camera_id,
frame_timestamp_ns, exposure_us, gain_db, shutter_mode,
frame_file, calibration_id, lighting_state,
action_timestamp_ns, action_domain, action_vector,
inspection_result, defect_bbox_or_mask,
product_pose, annotation_source, annotator_confidence
```

For sensor fusion, add audio sample timestamps, microphone layout, LiDAR calibration and robot-frame transforms as appropriate. Use a single monotonic clock or validated synchronization offsets; define what "action at time t" means relative to adjacent observations.

### 6.2 Curation

Remove blurry/faulty captures, duplicate production runs, inconsistent timestamp ranges and annotations from unintended environments. Maintain train/validation/test splits by **product batch, physical unit, acquisition day and production line** where appropriate; random frame splits often leak nearly identical samples across train and test. Balance rare defects without destroying prevalence information required for calibration.

Cosmos Curator and Evaluator are official components for industrial-scale processing/evaluation. Synthetic samples should be tracked with provenance and kept separate from real held-out inspection data.

### 6.3 Fine-tuning objective

- **Reasoner SFT:** prompt + camera evidence → structured defect answer, location, certainty, action recommendation.
- **Transfer/synthetic generation:** texture/edge/depth control + prompt → conditionally generated frames with preserved defect labels.
- **Forward dynamics:** current scene + validated simulated actions → next scene; test whether the model responds *causally* to intervention.
- **Inverse dynamics:** previous and current scene → likely intervening action.
- **Policy/world-action:** observation history → candidate action and predicted future; start offline, then simulation, then human-supervised lab tests.

Start from the appropriate base/post-trained checkpoint. Use documented framework SFT/LoRA/post-training recipes instead of editing 64B architecture parameters by hand.

### 6.4 What must not be compressed away

A world model's learned video tokenizer may smooth over **small scratches, cracks or pinholes**. For high-volume manufacturing, keep a high-resolution inspection path (patch-level segmentation, instance/defect detection or anomaly detector) alongside a wider-field world model. Evaluate loss of signal after tokenizer reconstruction. A world model can provide contextual simulation and planning even if another architecture remains the best pixel-level inspector.

## 7. Production serving architecture

```text
Cameras / other sensors
    |
    v
Clock synchronization + calibrated acquisition + validation
    |
    v
Secure ingestion / queue / backpressure / frame selection
    |
    +----> High-resolution defect specialist (pixel-level detection)
    |
    +----> Cosmos Reasoner (scene context and grounded explanation)
    |
    +----> Cosmos Generator (offline digital twin / synthetic data / next-state forecast)
    |
    v
Fusion + uncertainty + physical plausibility checks
    |
    v
Advisory decision engine + structured event log
    |
    v
Human-approved policy and bounded deterministic controls
    |
    v
PLC / robot controller via safety-certified independent interfaces
```

Use official vLLM/TensorRT-LLM/NIM serving for supported production surfaces; FastAPI in this repo is **only a demonstration** of input validation/packaging around the tiny toy model. The toy server is **not authenticated, rate-limited, calibrated or independently safety verified**. Do not expose it publicly or connect it to actuators.

### Operating controls for a real service

1. **Version pinning:** record model revision and SHA, tokenizer/VAE versions, transformers/diffusers versions, driver/CUDA version, calibrations and post-training dataset ID.
2. **Input validation:** enforce frames per clip, image limits, camera ID, audio sample rate, action frame/units, prompts and image/video MIME handling.
3. **Latency control:** cap token length, request size and diffusion steps; use request queue and admission control; monitor p50/p95/p99 and memory fragmentation.
4. **Observability:** include trace IDs for camera acquisition, model request, output, safety gate and eventual human decision; avoid storing raw sensitive factory media in generic logs.
5. **Model governance:** explicit versioned evaluation/approval process, shadow deployments, drift monitors, manual fallback and rollback.
6. **Security:** least-privilege containers, authentication, private networking, secrets manager, signed model assets, dependency scans and supply-chain provenance.
7. **Robot safety:** enforce joint/workspace bounds, emergency stop, collision checks, watchdog, operator approval and independent low-level control limits. Never let generative text/actions become unbounded motor commands.

### Minimal demo API

After training the toy reasoner:

```bash
pip install -e '.[api]'
uvicorn cosmos3_lab.serve:app --host 127.0.0.1 --port 8080
```

The endpoint `POST /reason` accepts a short question and base64 image, checks size/format, turns it into a 3-frame **repeated static toy clip**, and greedily generates a text answer. It does not preserve real temporal motion, so it is unsuitable for motion reasoning.

Docker demo (CPU serving):

```bash
docker build -t cosmos3-learning .
docker run --rm -p 127.0.0.1:8080:8080 \
  -v "$(pwd)/artifacts:/app/artifacts:ro" \
  cosmos3-learning
```

The model checkpoint must exist on the host in `artifacts/cosmos3_toy.pt` before starting the server.

## 8. Reproducibility and model-quality gates

A practical project should enforce these release conditions:

| Gate | Expected evidence |
|---|---|
| Data integrity | Correct timestamp/frame/audio/action alignment, licensing, privacy, recorded splits |
| Unit tests | All architecture and format tests pass |
| Loss correctness | Verified velocity sign, sigma endpoints, context masking, no AR leakage |
| Detection metrics | False accept/reject by defect, product, camera and lighting group |
| Generation metrics | Physical consistency and effect of deliberate action changes |
| Action metrics | Translation/rotation errors, trajectories, success/safety violations |
| Runtime metrics | Per-resolution/token-size memory, p95 latency, throughput, OOM/fallback |
| Human validation | Operators review failures and edge-case performance |
| Safety review | Independent interlocks and permissive control envelope |
| Rollback | Pinned working version, artifact manifest and reversible deployment |

## 9. Research roadmap connected to CV + robotics + reinforcement learning

**Phase A — Architecture comprehension:** reproduce embeddings, token arrangements, attention masks, flow objective and physical-time alignment, with unit tests. This is what the included lab covers.

**Phase B — Real-world perception:** use genuine factory scenes; compare pretrained Cosmos Reasoner responses against a high-resolution defect detector, and establish a blinded real-data benchmark.

**Phase C — Causal dynamics:** log/simulate safe interventions, generate counterfactual predictions, compare `p(next_video | video,action)` with ground-truth transitions, assess uncertainty accumulation.

**Phase D — Policy learning:** integrate a simulator world model with offline RL / imitation / planning, then evaluate model exploitation (world model inaccuracies yielding deceptively high simulated rewards). Use a bounded/hierarchical action policy and do not directly transfer learned exploratory controls into real machinery.

**Phase E — Deployment:** optimize bottlenecks with tokenizer and expert caching, appropriate Flash Attention / context parallelism, quantization verified against quality, concurrency and deterministic action validation.

The most publishable unanswered questions are likely around **small-defect fidelity of generative codecs, causal action-conditioned counterfactuals, uncertainty-calibrated world models, and safe low-latency inference** rather than simply generating attractive industrial videos.
