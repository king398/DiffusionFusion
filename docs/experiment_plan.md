# Experiment plan — joint attention & 3-stream co-denoising

**Question:** does V-Co-style joint attention (and a third semantic stream, EVA-02)
improve multi-representation co-denoising over the current 2-stream pairwise baseline?

**Baseline to beat** (`notes/results.md`, 2026-05-05): `jit_dual_struct_cfg_mask_80ep`
— dual-stream (SD-VAE latent + DINO), **pairwise** cross-fusion, 80 epochs.

| | FID-50K | IS-50K |
|---|---|---|
| Baseline | **4.0297** | 166.14 ± 2.577 |

The Stage-2 N-stream refactor is validated as bit-equivalent to this baseline
(`notes/refactor_validation_results.md`, Δ FID +0.0024), so any change measured
below is the *science*, not a plumbing regression.

The two changes relative to baseline are independent and must be measured
separately: **(a) fusion mechanism** pairwise → joint, and **(b) stream count**
2 → 3. The ladder below changes one at a time so a win can be attributed.

---

## Eval protocol (hold constant across all runs)

So every number is comparable to the 4.0297 baseline:

- Solver **Heun**, **50** steps, **CFG 2.9**, interval **[0.1, 1.0]**, EMA **`model_ema1`**.
- FID/IS against `JiT/fid_stats/jit_in256_stats.npz`, decoder
  `jit_decoder_small_gan_lower_noise` (`model_ema`), 256px.
- **Tracking metric** during training: online **FID-10K** (cheap, biased high — use for *trends* and sweeps only).
- **Gate metric** for any headline claim: offline **FID-50K** (the only number compared to 4.0297).
- Train budget **80 epochs** for headline runs (matches baseline); reduced budget allowed for sweeps (see E2).
- Decoder + FID path stay on **latent+DINO** throughout; extra streams are auxiliary co-denoising targets, never decoded.

---

## The ladder (run top to bottom)

| # | Experiment | Streams | Fusion | Budget | Depends on | Compared against | Decision it drives |
|---|---|---|---|---|---|---|---|
| **E0** | Joint-path smoke | latent+dino | joint | 10 ep, 2 GPU | refactor (done) | — (runs/doesn't) | Does the joint graph train + sample at all? |
| **E1** | Joint 2-stream control | latent+dino | joint | 80 ep | E0 green | 4.0297 baseline | Is the *joint mechanism* worth it, streams held fixed? |
| **E2** | Semantic loss-weight sweep | latent+dino | joint | sweep@40 ep → winner@80 ep | E0 green | E1 (w=1.0) & baseline | Best `dino` loss_weight `w*`. |
| **E3p** | EVA-02 feature extraction | — | — | extraction job | — (parallel) | — | Unblocks E3 (data prereq). |
| **E3** | Joint 3-stream | latent+dino+eva02 | joint | 80 ep | E2 (`w*`), E3p | E1/E2 best 2-stream | Marginal value of a 2nd semantic stream. |
| **E4** | EVA-02 redundancy ablation *(optional)* | latent+eva02 | joint | 80 ep | E3p | E1 (latent+dino joint) | Is EVA-02 orthogonal to DINO or redundant? |

E0–E2 are the critical path and need **no new data**. E3p (extraction) can run
in parallel with E0/E1/E2 so EVA-02 shards are ready by the time E3 starts.

---

## E0 — Joint-path smoke  *(prereq for everything joint)*

The joint graph (`JointStreamBlock`, the `[L,L]` additive CFG bias, per-stream
qkv/norm/mlp) has only run on CPU in unit tests. This run answers three things
and nothing else; **ignore the FID** (it's on barely-trained weights):

1. Trains under `ddp_static_graph=True` + `torch.compile` — no "marked ready
   twice" / no recompile-on-mask errors. (This is *why* the CFG mask is an
   additive bias, never a Python module-skip.)
2. Loss descends, no NaN, over ~10 epochs.
3. The joint **sampling** path runs end-to-end: `generate()→dict` → decode
   latent+dino → FID. An eval fires at **epoch 0** so a broken sampling path
   fails in minutes, not after 10 epochs.

**Launch:** `sbatch sbatch/jit_joint2_smoke.sbatch`
(galoiz-owned defaults: `REPO=/u/galoiz/DiffusionFusion-independent-backup`,
its `.venv`, the cuBLAS `LD_LIBRARY_PATH` fix, account `bgnp-dtai-gh`.)
To isolate a compile failure from a graph failure: `JIT_COMPILE=0 sbatch ...`.

**Pass = three green lights.** Then promote to E1.

---

## E1 — Joint 2-stream control  *(the foundational comparison)*

Identical to the baseline except fusion pairwise → **joint**. Same two streams,
same data, same 80-epoch budget, neutral loss weights (latent 1.0 / dino 1.0).

- Config: `configs/streams/latent_dino_joint.yaml` (carries `fusion: joint`).
- Launch: promote the smoke sbatch — `EPOCHS=80 NUM_IMAGES=10000 EVAL_FREQ=40
  OUTPUT_DIR=<repo>/outputs/jit_joint2_80ep WANDB_RUN_NAME=jit_joint2_80ep
  sbatch sbatch/jit_joint2_smoke.sbatch` (or copy to a dedicated
  `jit_joint2_train.sbatch`).
- **Gate:** offline FID-50K vs 4.0297.

This run is also the **`w=1.0` anchor** of the E2 sweep, so E1 and E2 share it.

**Interpretation**
- E1 ≤ ~4.03: joint mechanism is at least as good as pairwise at equal streams → proceed up the ladder with confidence.
- E1 noticeably worse: the mechanism alone costs quality. Before abandoning joint, check (a) E2 weight tuning (joint may need lower semantic weight), (b) whether 80 ep is enough for a from-scratch joint model. Joint can't warm-start from the pairwise checkpoint by design.

---

## E2 — Semantic loss-weight sweep

V-Co found semantic streams help most at **low** weight; our configs sit at an
untuned 1.0. Sweep the DINO loss weight, image-side latent fixed at 1.0:

`dino_loss_weight ∈ {0.1, 0.25, 0.5, 1.0}`

- **Cheap first:** run the sweep at **40 epochs**, rank by online FID-10K → pick `w*`.
- **Confirm:** run `w*` (and 1.0 if `w* ≠ 1.0`, = E1) at **80 epochs**, gate on FID-50K.
- The sweep is **config-file edits**, one YAML per weight: copy
  `latent_dino_joint.yaml` → set the `dino` stream's `loss_weight`. The legacy
  `--dino_loss_weight` CLI flag does **not** apply once `--streams_config` is
  set — with a streams config, `loss_weight` is read only from the YAML
  (`main_jit.py:294`; denoiser applies it at `denoiser.py:210`).

**Output:** best semantic weight `w*`, carried into E3.

---

## E3p — EVA-02 feature extraction  *(data prereq for E3, run in parallel)*

E3 is **blocked** until `imagenet224_eva02_small_features` exists.

- Launch: `sbatch sbatch/eva02_features.sbatch`
  (`eva02_small_patch14_224.mim_in22k`, 224px → 16×16 grid, 384-d, dir
  `imagenet224_eva02_small_features`).
- **Critical:** extract at the **same `DistributedSampler` world size** as the
  latent/DINO shards, or `sample_id`s misalign across stores. The
  label-mismatch warning at train start is the detector — if you see it, the
  shards don't line up and E3 numbers are meaningless.

---

## E3 — Joint 3-stream (+EVA-02)  *(the research target)*

Add EVA-02 as a third (semantic, auxiliary) stream on top of the best E2 config.

- Config: `configs/streams/latent_dino_eva02_joint.yaml` (`fusion: joint`,
  EVA-02 384-d / 16-sp / patch 1 / linear). Set `dino` weight to `w*`; sweep or
  set `eva02` weight similarly (start `{w*, 0.25, 0.1}`).
- **Gate:** FID-50K vs the **best 2-stream joint** (E1/E2), *and* vs 4.0297.

**Interpretation / risk:** DINO and EVA-02 are both semantic ViT features and may
be **largely redundant** → diminishing returns is a plausible, publishable
negative. The E3 − E2 delta is exactly what measures the marginal value of a
second semantic stream; E4 disentangles redundancy from "EVA-02 is just weaker."

---

## E4 — EVA-02 redundancy ablation  *(optional)*

Joint 2-stream **latent + EVA-02** (drop DINO). Compared to E1 (latent+DINO
joint), this says whether EVA-02 alone is competitive with DINO. Combined with
E3 it separates "the streams are redundant" from "EVA-02 adds nothing on its
own." Only worth running if E3 is ambiguous.

---

## Decision flow

```
E0 smoke ──green──► E1 joint-2 (vs 4.03)
   │ red               │
   ▼                   ├─ ≤4.03 ─► E2 weight sweep ─► E3 joint-3 (+EVA-02, needs E3p)
 fix graph/compile     │                                 │
 (JIT_COMPILE=0 to     └─ worse ─► try E2 low-w / more   ├─ beats E2 ─► 3-stream joint wins; (E4 to confirm orthogonality)
  localize)                        epochs before          └─ ≈ E2    ─► streams redundant; report; E4 to confirm
                                   abandoning joint
```

---

## Logistics (galoiz)

- **Checkout / venv:** `/u/galoiz/DiffusionFusion-independent-backup` + its
  `.venv`. Runs need the cuBLAS `LD_LIBRARY_PATH` prepend (torch's bundled CUDA
  must shadow the Cray cuBLAS, else `CUBLAS_STATUS_INVALID_VALUE` on
  non-contiguous bf16 GEMMs). All baked into `sbatch/jit_joint2_smoke.sbatch`.
- **Account / partition:** `bgnp-dtai-gh` / `ghx4`.
- **Shared data/decoder/baseline** (on `/work`, owned by msalunkhe, readable):
  data `/work/hdd/bgnp/msalunkhe/data/imagenet/`, decoder + 4.03 baseline under
  `/work/nvme/bgnp/msalunkhe/outputs/...`.
- **Cost unit:** one 80-epoch B-model run ≈ the baseline's budget. Use 40-epoch
  online-FID-10K runs for sweeps; spend FID-50K only on headline gates.
