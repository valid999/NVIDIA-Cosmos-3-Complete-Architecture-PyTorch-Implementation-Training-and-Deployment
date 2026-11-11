# Source-code walkthrough — what every network, method and training command does

Read this alongside `ARCHITECTURE_GUIDE.md` and the actual source files. All source modules have docstrings and inline comments; the explanations below explain **why** each operation exists, the shapes that must match, and what to debug.

## 1. The execution order

```text
python -m cosmos3_lab.train
  ├─ train.run()
  │   ├─ parse --stage / --steps / --checkpoint / --manifest
  │   ├─ load_model(): construct LabConfig and MiniCosmos3
  │   ├─ DataLoader(SyntheticWorld or ManifestWorld)
  │   ├─ for each optimizer step
  │   │   ├─ batch tensors moved onto torch.device
  │   │   ├─ codec_loss() OR reasoner_loss() OR generator_loss()
  │   │   ├─ loss.backward() -> compute parameter gradients
  │   │   ├─ clip_grad_norm_() -> avoid unstable gradients
  │   │   └─ optimizer.step() -> update parameters
  │   └─ torch.save() + os.replace() -> checkpoint
  └─ checkpoint loadable by infer.py/evaluate.py/serve.py
```

### 1.1 `config.py::LabConfig`

- `width=96`: **D**, the shared token feature dimension, not the number of pixels or action coordinates.
- `heads=4`: 4 independent attention heads, each `head_dim = 96/4=24`.
- `head_dim % 6 == 0`: this makes 3 equal t/h/w rotary sections, each with an even number of channels (8).
- `layers=2`: two stacked `DualStreamMoTBlock` layers.
- `image_size=16`, `video_frames=3`: a CPU-scale synthetic clip, not a realistic production resolution.
- `visual_patch=4`: patch extraction for understanding (`16/4=4` patches per spatial side).
- `visual_latent=12`: the generation codec creates 12 channels per encoded spatial location.
- `audio_latent=8`, `audio_hop=160`: 8 channels for each contiguous audio window.
- `action_dims`: declared dimension per camera/vehicle/robot domain. This controls real matrix shape and validates against wrong embodiments.
- `temporal_compression=1`: toy codec preserves the number of frames. NVIDIA's generation VAE compresses time by 4, so do not reuse the toy frame accounting for real Cosmos.
- `base_tps=6`, `dm_gap=15000`: mirror the relevant *concepts* of the technical report.

### 1.2 `tokenizer.py::CharTokenizer`

`PAD/BOS/EOS/SEP/BOG/UNK` are six reserved token IDs. Text character `c` is represented as `6+ord(c)` for ASCII. `encode("abc")` returns BOS + character IDs + EOS; `decode` strips all special tokens. The `answer_batch` method creates a sequence containing a question and answer, but sets language labels to `-100` *before the answer starts*. This prevents optimizing the model to merely imitate the user prompt. `prefix()` creates prompt tokens with a BOG marker for DM tasks.

**Independent prompts in one batch:** `batch_prefix([prompt_a, prompt_b, ...])` encodes each text separately, pads to the longest sequence, and returns `[B,L_max]`. `model.encode_ar` derives `ar_key_valid = (text_ids != PAD)`, preserving the always-valid vision prefix. The dual-stream attention applies this mask both to AR causal keys and to the AR portion of DM cross-attention. This is a **real batching requirement**: without it, the first example in a minibatch could attend to other pad embeddings and would give different predictions depending on the length of another sample. `generator_loss` also uses each sample’s own prompt, rather than mistakenly repeating the first prompt. See `test_batch_padding_does_not_change_first_example` and `test_generator_loss_with_distinct_minibatch_prompts`.

**Important:** this tokenizer is deliberately simplistic. The production tokenizer includes learned linguistic units and special vision/audio role formats; its token IDs are not compatible with this toy.

## 2. Sensor-specific neural networks

### 2.1 `VisionUnderstandingEncoder.forward`

```python
b, t, c, h, w = frames.shape
patches = self.stem(frames.reshape(b * t, c, h, w))
hp, wp = patches.shape[-2:]
embeddings = patches.flatten(2).transpose(1, 2).reshape(b, t * hp * wp, -1)
return self.project(embeddings), hp, wp
```

**Line 1:** read the batch size, frame count, RGB channels, height and width; this asserts the *meaning* of each axis.

**Line 2:** flatten only `B` and `T` before Conv2D. The Conv2D layer applies the **same learned patch projection** to every frame. It returns `[B*T,D,H_p,W_p]`.

**Line 3:** query the spatial resolution of the resulting patch lattice. Never hardcode `4×4` when resolution can vary.

**Line 4:** `flatten(2)` flattens only `[H_p,W_p]`; `transpose(1,2)` puts tokens before channels, and `reshape(...)` restores `B` while concatenating the `T` frame-token sequences. Output is `[B,T*H_p*W_p,D]`.

**Line 5:** a small MLP adjusts each token representation. This MLP does not reproduce the real pretrained ViT's merge/DeepStack operation.

### 2.2 `ToyVisualCodec.encode` / `decode`

`encoder=Conv2d(3,12,4,4)` reduces `[3,16,16]` into `[12,4,4]` per frame. `tanh()` bounds the latent magnitude; the network learns these channels during `--stage codec`. The token permutation transforms `[B*T,12,4,4]` into `[B,T*16,12]`. The decoder reverses this and reconstructs `[B,T,3,16,16]` using `ConvTranspose2d`; `sigmoid()` bounds reconstructed RGB to `[0,1]`.

This is **an ordinary autoencoder**, not a pretrained variational autoencoder. A genuine VAE additionally models latent probability, often KL regularization and specialized reconstruction/perceptual training.

### 2.3 `ToyAudioCodec.encode` / `decode`

`Conv1d(2,8,kernel_size=160,stride=160)` operates on two input audio channels and groups the waveform into 160-sample segments. For 1280 samples, it yields 8 latent time steps. `transpose(1,2)` changes `[B,8,8]` from channels-first to time-first for transformer input. The audio decoder uses `ConvTranspose1d` to restore `[B,2,1280]`.

There are **no FFT features or pretrained speech models** here. It is a learnable codec used to expose the same input-contract pattern as Cosmos 3.

### 2.4 `action_geometry.py`

`rotation6d_to_matrix_svd` interprets the first six numbers as two (possibly imperfect) rotation columns, constructs a third via cross product, and finds the nearest **proper orthogonal matrix** using SVD. It checks and corrects the determinant sign so the result lies in SO(3), not a reflected coordinate system. This is useful when interpreting generated pseudo-actions, although actual robot controllers require their own kinematics and validated transforms.

## 3. Why time embeddings and modality IDs are both necessary

`model.py::MiniCosmos3` defines:

```python
self.vision_in = nn.Linear(visual_latent, width)
self.audio_in = nn.Linear(audio_latent, width)
self.action_in = nn.ModuleDict({name: nn.Linear(dim, width) ...})
self.types = nn.ParameterDict({ ... })
self.time_embedding = nn.Sequential(nn.Linear(1, width), nn.SiLU(), nn.Linear(width, width))
```

The three `*_in` projections solve **dimension mismatch**: 12 video channels, 8 audio channels and 9/10/20/... action scalars can all become 96-dimensional tokens. Type embeddings solve **semantic ambiguity**: a 96-D audio feature should not be confused with a 96-D robot action. `time_embedding` encodes how much noise remains (`sigma`), so one DM tower can learn to denoise at different points of a flow trajectory.

**Common bug:** using the same `nn.Linear(9,96)` for every robot. This silently assumes all robots have a 9D control space and produces wrong semantics for 10D, 20D, 29D and 57D embodiments. `ModuleDict` maintains separate matrices under the same backbone.

## 4. 3D rotary embeddings

`positions.py::video_positions`: for every latent frame `f`, build a row of `(t,h,w)` coordinates, with t fixed within that frame and h/w increasing across the latent grid. Frame time is `start_frame+f` multiplied by `base_tps / (fps / temporal_compression)`. The function returns `[T*H_p*W_p,3]` and preserves true frame order. `temporal_positions` does the same for audio/action but assigns zero to h/w.

`apply_mrope` accepts `x=[B,num_heads,N,head_dim]`, splits `head_dim` into three t/h/w partitions, splits pairs of channels into real/imaginary components, rotates each pair by an angle proportional to the physical coordinate and inverse-frequency, and returns a tensor of identical shape. Rotation does **not** change the L2 norm of each vector. `tests/test_architecture.py::test_mrope_norm_preserved` validates this.

If h/w axes are swapped or the frame-rate conversion is mistaken, the attention is trained on wrong geometry. If text position logic is conflated with physical timestamps, speech/actions may become misaligned with video events.

## 5. Exactly what happens in `attention.py::DualStreamMoTBlock.forward`

This is the most important method in the project.

```python
qa, ka, va = self.split_qkv(self.ar_norm(ar), self.ar_qkv)
qa, ka = apply_mrope(qa, ar_pos), apply_mrope(ka, ar_pos)
ao = F.scaled_dot_product_attention(qa, ka, va, is_causal=True)
ar = ar + self.ar_out(self.merge(ao))
ar = ar + self.ar_ff(self.ar_ff_norm(ar))
```

1. Normalize the AR states. LayerNorm reduces sensitivity to representation scale.
2. Apply the **AR-specific** learned Q/K/V projections. `split_qkv` splits the `3*D` output into separate Q, K and V arrays and reshapes from `[B,N,D]` to `[B,heads,N,head_dim]`.
3. Apply 3D RoPE to Q and K, not V.
4. Call PyTorch fused/scaled dot-product attention with **causal masking**.
5. Merge heads, project back to D, add the residual skip, then apply an AR-specific FFN and second residual.

For DM tokens:

```python
qd, kd, vd = self.split_qkv(self.dm_norm(dm), self.dm_qkv)
qd, kd = apply_mrope(qd, dm_pos), apply_mrope(kd, dm_pos)
keys, values = torch.cat((ka, kd), -2), torch.cat((va, vd), -2)
do = F.scaled_dot_product_attention(qd, keys, values, is_causal=False)
dm = dm + self.dm_out(self.merge(do))
dm = dm + self.dm_ff(self.dm_ff_norm(dm))
```

1. Normalize and project through **DM-specific** QKV matrices.
2. Apply the DM's physical-time/spatial RoPE.
3. Concatenate AR and DM **keys and values** along the **token** axis. Never concatenate heads or batch axes here.
4. Use **non-causal attention**, because the DM can attend to every AR/DM token.
5. Apply DM-specific output, residual, LayerNorm and FFN.
6. Crucially, no expression in the AR update uses `dm`, `kd` or `vd`—this is the directional information-flow invariant.

Why two sequential calls rather than a single flattened TransformerEncoder? Because ordinary unrestricted self-attention would allow DM values to influence the AR path and violate the model's intended conditional factorization.

## 6. `MiniCosmos3.forward`: routing, packing and splitting

1. `encode_ar(text_ids, ar_frames)`: embeds text and optional **understanding** camera tokens. If frames are present, creates matching AR video spatial positions and concatenates them before text.
2. Iterate over `segments`. Each `DiffusionSegment` has exactly one **unique name**, a modality type, numeric latent features, `(t,h,w)` positions, a per-batch `sigma`, and (for actions) embodiment/domain.
3. Validate shapes: features must be `[B,N,C]`, positions `[N,3]`, and modalities must be known.
4. `embed_segment` selects the correct input linear layer, adds modality embedding, adds sigma embedding and a learnable noise flag (if sigma>0).
5. `torch.cat(chunks, dim=1)` assembles the **DM token sequence**. Every chunk now has the same D=96, even when original channel counts differed.
6. Feed both sequences through each MoT block with their own 3D coordinates.
7. Apply `final_ar_norm` and `text_head` to the **text** part of AR to obtain `[B,L_text,134]` next-token logits.
8. For each DM segment, use its tracked token length to slice `[B,N,D]` out of the combined DM hidden sequence, then dispatch to `vision_out`, `audio_out` or `action_out[domain]` to predict **velocity in native latent channel dimensions**.
9. Return `ModelOutput(text_logits, predictions, shapes)`.

Returning a named dict (`vision_future`, `audio_future`, `actions_future`) is safer than relying on hardcoded positions when task modes change.

## 7. `training.py::codec_loss`

1. `video -> visual_codec.encode` to obtain tokens.
2. Decode into RGB and compare to the input with `MSE`.
3. `audio -> audio_codec.encode -> decode`.
4. Compare decoded waveform with input waveform (MSE).
5. Add the two losses. The loss weights are only demonstration defaults. Real codec pretraining uses more sophisticated objectives/data and should check signal quality, especially for rare small defects.

## 8. `training.py::reasoner_loss`

1. The minibatch carries `prompt` and `answer` strings and a corresponding `video`.
2. `answer_batch` packs the question + answer and prepares a masked label tensor.
3. The video is turned into **AR ViT tokens** and inserted before the language IDs.
4. The AR tower runs with causal attention; `text_head` predicts next-character logits.
5. `F.cross_entropy(..., ignore_index=-100)` optimizes only the answer/EOS target, not the input question.

This demonstrates VLM-style supervised instruction learning, **not** advanced chain-of-thought or benchmark-level reasoning.

## 9. `training.py::generator_loss`

The core of physical generation:

1. `freeze_codecs()` prevents the codec latents from changing during backbone training. In official Cosmos the VAE encoders are already pretrained and frozen; we train ours first.
2. Encode video and audio to fixed-size latent tokens using `torch.no_grad()`; raw camera actions are already continuous vectors.
3. Compute physical `(t,h,w)` positions at correct video/audio/action rates.
4. For each task `mode`, decide which tokens are **clean conditions** and which are **noisy targets**. This must be correct **before** training the model.
5. For a noisy target, draw `sigma`, draw `eps` of the same shape as target, create `x_sigma=(1-sigma)x0+sigma*eps`, and set true target velocity `eps-x0`. Each noised modality can draw its own sigma.
6. Add a clean `DiffusionSegment` for history/control tokens with `sigma=0`; it may be used as attention context but is excluded from the loss.
7. Occasionally drop the text condition so inference can combine unconditional and conditional predictions.
8. Create a single list of video/audio/action segments. Call `model(...)` **once**, regardless of the number of generated modalities.
9. Obtain the velocity for every noised segment, compute MSE relative to its target velocity and average the modality losses.

**The key point:** `torch.stack([video_loss, audio_loss, action_loss]).mean()` is a joint objective for one model that predicts each modality in the same forward pass. Appropriate production training should normalize/perceptually weight each modality and handle very unequal token counts.

## 10. `sampling.py::sample_joint` step by step

1. Build Gaussian latent tensors with the same shapes as each target segment. Initial `sigma=1` means pure noise.
2. Start the loop `for step in range(steps)`, with `sigma_current=1-step/steps` and `sigma_next=1-(step+1)/steps`.
3. Build current noised DM segments for *all* target modalities and append all immutable clean conditioning segments.
4. Perform **one model call per diffusion step** (or two, if using CFG).
5. Optionally combine conditional and unconditional predictions with guidance scale.
6. Euler step: `x = x + (sigma_next-sigma_current)*velocity`. Since `sigma_next<sigma_current`, this moves from noise toward data.
7. Return a dict of final **latent** values, not pixels or raw audio yet.
8. `infer.py` decodes visual latents with the video decoder and audio latents with the audio decoder, and keeps action predictions as numeric vectors.

Do not confuse the `steps` argument here (number of denoising iterations) with the `--steps` argument of `train.py` (number of optimizer updates).

## 11. `train.py` and checkpoint lifecycle

- Parses flags and selects CPU or CUDA based on `torch.cuda.is_available()`.
- `load_model`: if checkpoint exists, reconstructs the `LabConfig` from its stored dict, initializes exactly the same module structure and loads weights. Otherwise creates a fresh random model for training.
- `DataLoader` collates dict entries into `[B,...]` tensors; prompts and answers remain Python lists.
- `optimizer=AdamW`: gradient-based training of eligible parameters. Frozen codec parameters are excluded during generator stage.
- `loss.backward`: accumulate derivatives via PyTorch autograd.
- `clip_grad_norm`: bound global gradient norm to avoid extreme updates.
- `optimizer.step`: update weights.
- `torch.save` to a temporary file and `os.replace`: reduce the risk of a partially written checkpoint. Not a substitute for durable multi-node checkpointing.

This tiny trainer does **not** checkpoint the AdamW momentum state or dataloader RNG and does not implement a global step across stages. A real repeatable trainer should save optimizer/scheduler/seed/state plus provenance.

## 12. Instrumentation and debugging checklist

**Shape errors:** run `python -m cosmos3_lab.inspect`. Print every token axis; check `B` always matches across modalities, `N` matches positions, and action dimensions match the chosen domain.

**Incorrect causal behavior:** run `pytest tests/test_architecture.py::test_multi_input_single_forward_and_no_leakage -q` and compare AR logits after randomizing DM tokens.

**NaN training:** inspect `sigma`, audio normalization, action units, gradients and learning rate. Confirm every target has nonzero-length tokens and mean loss is finite.

**Artifacts/poor outputs:** first verify codec reconstruction and diffusion target sign. The correct velocity for our sigma convention is `epsilon - x0`, and the integration direction has `delta_sigma < 0`. Sign errors can make a denoiser diverge even when training loss decreases.

**Action instability:** validate domain-specific input/output projection and verify SO(3) after converting rotation representation; clamp/validate actions only at a separate, properly designed controller boundary.

**Temporal desynchronization:** inspect timestamps (fps, audio hop, robot action rate), rather than assuming array index `t` means the same physical time everywhere.

**High-resolution insufficiency:** do not treat a toy network trained on 16×16 frames as evidence about micrometer scratches or high-throughput production cameras. Use industrial-resolution data and dedicated inspection/evaluation.

## 13. Code review questions to answer before any serious reproduction

- Why is image understanding ViT encoded in the AR stream while image generation uses VAE latents in DM?
- Is the loss computed on noised tokens only? Does it ignore context tokens correctly?
- Does a single batch contain independent examples rather than unrelated camera views from different timestamps?
- Are the action representation, orientation convention and reference frames defined?
- Does `sigma` use the same convention in training and sampling?
- Are physical-time coordinates based on real sensor rates?
- Can the AR stream accidentally see DM tokens due to fused mask implementation?
- Are future frames or target actions leaked into clean conditioning?
- Can a model predict the future under **counterfactual actions** or only replay statistical correlations?
- Is the benchmark domain sufficiently close to factory use, and does the model satisfy latency/safety constraints?
