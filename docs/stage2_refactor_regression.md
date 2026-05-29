# Stage 2 Legacy-Checkpoint Regression

This ladder validates that the registry-driven multistream JiT still reproduces
the dual-stream 80-epoch baseline:

- JiT checkpoint: `/work/nvme/bgnp/msalunkhe/outputs/jit_dual_struct_cfg_mask_80ep/checkpoint-last.pth`
- Decoder: `/work/nvme/bgnp/msalunkhe/outputs/jit_decoder_small_gan_lower_noise/checkpoint-last.pth`, key `model_ema`
- Sampling: `model_ema1`, Heun-50, CFG `2.9`, interval `[0.1, 1.0]`
- Full baseline: FID-50K about `4.03`, IS about `166.1`; pass at FID within `0.15`

Set the checkout that Slurm will execute. The wrapper checks this exact directory
for Stage 2 symbols before it submits a job.

```bash
export REPO_DIR=/u/msalunkhe/DiffusionFusion
export VENV_PATH=/u/msalunkhe/.venv
```

## Test A: Full-Scale Forward Parity

In a one-GPU interactive allocation:

```bash
module load python/3.11.9 cuda/13.1.1
source "${VENV_PATH}/bin/activate"
cd "${REPO_DIR}"
python scripts/validate_stage2_checkpoint_parity.py \
  --checkpoint /work/nvme/bgnp/msalunkhe/outputs/jit_dual_struct_cfg_mask_80ep/checkpoint-last.pth \
  --device cuda
```

The script verifies the Stage 2 imports, strictly loads both full-size models,
and compares masked and unmasked fp32 outputs. Do not proceed if the maximum
absolute difference is substantially above `1e-4`.

## Test B: Preview And Smoke

Generate retained images through the remapped EMA and refactored dict output:

```bash
python scripts/preview_stage2_generation.py \
  --checkpoint /work/nvme/bgnp/msalunkhe/outputs/jit_dual_struct_cfg_mask_80ep/checkpoint-last.pth \
  --decoder-checkpoint /work/nvme/bgnp/msalunkhe/outputs/jit_decoder_small_gan_lower_noise/checkpoint-last.pth \
  --output-dir /work/nvme/bgnp/msalunkhe/outputs/jit_dual_struct_cfg_mask_80ep/eval_stage2_check/previews \
  --device cuda
```

The PNGs should be recognizable ImageNet-like samples, not static or noise.
Then submit the 256-image end-to-end evaluator:

```bash
scripts/submit_stage2_regression_eval.sh smoke
```

The Slurm log must contain remap lines for `model`, `model_ema1`, and
`model_ema2`, followed by strict model-load and EMA validation success lines.
It must finish generation, decoder inference, FID, and IS without shape or NaN
errors.

## Tests C And D: FID Gates

Only after the preceding rung passes:

```bash
scripts/submit_stage2_regression_eval.sh 10k
scripts/submit_stage2_regression_eval.sh 50k
```

Treat the 10K run as directional: FID should be sane and single-digit. For
50K, record FID and IS and compare FID to `4.03 +/- 0.15`.

## Report

Record:

```text
Deployed checkout/revision:
Test A unmasked latent/dino max abs diff:
Test A masked latent/dino max abs diff:
Test B remap and decoder smoke:
Test C FID-10K / IS:
Test D FID-50K / IS:
Surprises in logs (key mismatches, NaNs, missing remap output):
Bottom line: pass/fail against FID 4.03 +/- 0.15
```
