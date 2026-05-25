# Stage 4 — Three-Stream Co-Denoising (latent + DINOv2 + EVA-02)

Adds **EVA-02** as a third co-denoising stream alongside the SD-VAE latent
(image-side) and DINOv2 (semantic). This is pure configuration: the model
(`JiTMultiStream`), denoiser (`Denoiser`), data loader
(`MultiStreamShardDataset`), and engine are already N-stream, so a third stream
is selected entirely through `--streams_config`.

EVA-02 is an **auxiliary** co-denoising target (V-Co framing): it is denoised
and contributes to the loss, but it is **not decoded**. The decoder and FID path
stay on `latent` + `dino` → 256px, so the existing decoder checkpoint and
`fid_stats/jit_in256_stats.npz` apply unchanged. (The engine decodes
`sampled["latent"]` / `sampled["dino"]` *by name* — both stream names must exist,
and the configs below provide them.)

Configs:

- `configs/streams/latent_dino_eva02_joint.yaml` — `fusion: joint` (the research target).
- `configs/streams/latent_dino_eva02.yaml` — `fusion: pairwise` (baseline-comparable).

> **Loss weights are neutral (1.0/1.0/1.0) and untuned.** Per-stream `loss_weight`
> is the first knob to sweep — V-Co found semantic streams help most at low weight
> (~0.1–0.5), while the 4.03 dual baseline used `dino=1.0`. Sweep the semantic
> weights before drawing conclusions. See the comment block in each YAML.

```bash
export REPO_DIR=/u/msalunkhe/DiffusionFusion
export VENV_PATH=/u/msalunkhe/.venv
# Parent dir that holds the imagenet*_* feature-shard dataset directories:
export FEATURE_ROOT=/work/hdd/bgnp/msalunkhe/data/imagenet
```

## 1. Extract EVA-02 features

```bash
sbatch sbatch/eva02_features.sbatch
# or, interactively, matching the world size used for latent/dino (see §2):
NPROC_PER_NODE=2 SPLIT=train sbatch sbatch/eva02_features.sbatch
```

This writes `${FEATURE_ROOT}/imagenet224_eva02_small_features` — note the `224`
prefix (EVA-02 runs at 224px), vs. `imagenet256_*` for latent/dino. That is fine:
the loader aligns streams by `sample_id`, not by directory name. Schema is
identical to dino/vae (`feature` Array3D `(384, 16, 16)`, `label`, `sample_id`),
so `inspect_feature_shards` / `MultiStreamShardDataset` consume it unchanged. At
224px / patch-14 the grid is 16×16 = 256 tokens, matching latent (32/2) and dino
(16/1) — the shared-token-grid constraint holds.

## 2. CRITICAL: sample_id alignment across extractions

Each extractor assigns `sample_id = rank * len(sampler) + local_idx` over a
`DistributedSampler(shuffle=False)`. **The mapping from `sample_id` to the
underlying image therefore depends on the extraction world size.**
`MultiStreamShardDataset` only checks that the `sample_id` *arrays* are equal
across stores (they always will be: `[0..N)`), so equal `sample_id`s do **not**
by themselves guarantee the same image. For EVA-02 to line up with the existing
latent/dino shards, it must be extracted with the **same**:

- world size (`nproc_per_node` / GPU count),
- split (`--split`, default `train`),
- underlying dataset and order (same `--data-path`, `shuffle=False`).

All three extractor sbatch files (`vae_features`, `dinov3_features`,
`eva02_features`) already share the same world-size config — `--gpus-per-node=2`,
`--nnodes=1`, and `NPROC_PER_NODE="${NPROC_PER_NODE:-${SLURM_GPUS_ON_NODE:-2}}"`.
**Extract all three at the same `NPROC_PER_NODE`** (if you override it for one,
override it identically for the others, and re-extract any stream whose world
size differed). No sbatch edit is needed for alignment; just keep the GPU count
consistent.

### The de-facto misalignment detector

The label authority is the image-side stream (`latent`). On the first RAM shard,
`MultiStreamShardDataset` compares each store's labels against the authority and
prints **once**:

```
Warning: labels diverged between stream 'latent' and stream 'eva02' for N samples ...
```

After extraction, do a quick one-shard load (or watch the first training step's
stdout) and confirm this warning does **not** appear for `eva02` (nor `dino`).
A clean first shard is the signal the three streams are aligned. If you see it,
the world size / split / order differed for that stream — re-extract it to match.

## 3. Launch a three-stream run

`--streams_config` is already wired (Stage 3b): it parses the N-stream `streams:`
list and honors the top-level `fusion:` key when `--fusion` is not passed on the
CLI (an explicit `--fusion` overrides the YAML). No sbatch edits are required —
add the flag to the existing JiT training command:

**Joint (research target):**

```bash
module load python/3.11.9 cuda/12.9.0
source "${VENV_PATH}/bin/activate"
cd "${REPO_DIR}/JiT"

OMP_NUM_THREADS=16 torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 \
  main_jit.py \
  --model JiT-Dual-B/2-4C-896 \
  --streams_config ../configs/streams/latent_dino_eva02_joint.yaml \
  --data_path "${FEATURE_ROOT}" \
  --class_num 1000 \
  --batch_size 128 --accum_iter 2 --blr 5e-5 \
  --epochs 80 --warmup_epochs 5 \
  --output_dir /work/.../jit_three_stream_joint \
  --online_eval \
  --fid_stats_path JiT/fid_stats/jit_in256_stats.npz \
  --decoder_checkpoint /work/.../jit_decoder_small_gan_lower_noise/checkpoint-last.pth \
  --decoder_checkpoint_key model_ema \
  --cfg 2.9 --interval_min 0.1 --interval_max 1.0
```

**Pairwise (baseline-comparable):** identical, but point at the pairwise YAML:

```bash
  --streams_config ../configs/streams/latent_dino_eva02.yaml \
```

`fusion:` lives in the YAML, so you do not need `--fusion` on the CLI; pass
`--fusion {pairwise,joint}` only to override the YAML.

## 4. Decoder / FID are unchanged

EVA-02 is auxiliary and never decoded. `engine_jit.evaluate()` calls
`decode_with_decoder(decoder, sampled["latent"], sampled["dino"])` — it reads the
`latent` and `dino` streams by name and ignores `eva02`. Output images are still
256px decoded from latent+dino, so:

- reuse the existing decoder checkpoint and `--decoder_checkpoint_key`,
- reuse `fid_stats/jit_in256_stats.npz` (no new FID stats file),
- FID is directly comparable to the 2-stream baseline.

This assumes a stream literally named `dino` exists — the configs provide it. If
you rename or drop `dino`, the hardcoded decode call in the engine will break;
that is intentional and out of scope for this stage.

## 5. CPU sanity (no GPU required)

```bash
./.venv/bin/python -m pytest tests/test_three_stream.py -v
```

(`./.venv/bin/python` is torch 2.12; the shell-default anaconda python is torch
1.12 and fails on `torch.from_numpy`.) These cover both fusion modes at the model
and denoiser level, the 3-stream structural mask, the joint static-graph property
under that mask, a 3-store dataset-alignment batch, and the YAML parse.
