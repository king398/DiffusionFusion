# Results

## Our JiT Runs

| Date | Checkpoint / Run | Samples | Solver | Steps | CFG | Interval | Resolution | Inception Score | FID |
|---|---|---:|---|---:|---:|---|---:|---:|---:|
| 2026-05-05 | `jit_dual_struct_cfg_mask_80ep` | 50,000 | Heun | 50 | 2.9 | `[0.1, 1.0]` | 256 | 166.1445 +/- 2.5770 | 4.029688 |

Sample directory:

`/work/nvme/bgnp/msalunkhe/outputs/jit_dual_struct_cfg_mask_80ep/eval/heun-steps50-cfg2.9-interval0.1-1.0-image50000-res32`

## JiT Paper

Source: "Back to Basics: Let Denoising Generative Models Denoise" (Li and He, arXiv:2511.13720v2).

FID-50K / IS-50K. Headline rows use CFG interval; 200-epoch IS is not reported.

| Category | Model / Setting | Resolution | Params | GFLOPs | Epochs | FID | IS | Source |
|---|---|---:|---:|---:|---:|---:|---:|---|
| headline | JiT-B/16 | 256 | 131M | 25 | 200 | 4.37 | - | Table 6 |
| headline | JiT-L/16 | 256 | 459M | 88 | 200 | 2.79 | - | Table 6 |
| headline | JiT-H/16 | 256 | 953M | 182 | 200 | 2.29 | - | Table 6 |
| headline | JiT-G/16 | 256 | 2B | 383 | 200 | 2.15 | - | Table 6 |
| headline | JiT-B/16 | 256 | 131M | 25 | 600 | 3.66 | 275.1 | Table 7 |
| headline | JiT-L/16 | 256 | 459M | 88 | 600 | 2.36 | 298.5 | Table 7 |
| headline | JiT-H/16 | 256 | 953M | 182 | 600 | 1.86 | 303.4 | Table 7 |
| headline | JiT-G/16 | 256 | 2B | 383 | 600 | 1.82 | 292.6 | Table 7 |
| headline | JiT-B/32 | 512 | 133M | 26 | 200 | 4.64 | - | Table 6 |
| headline | JiT-L/32 | 512 | 462M | 89 | 200 | 3.06 | - | Table 6 |
| headline | JiT-H/32 | 512 | 956M | 183 | 200 | 2.51 | - | Table 6 |
| headline | JiT-G/32 | 512 | 2B | 384 | 200 | 2.11 | - | Table 6 |
| headline | JiT-B/32 | 512 | 133M | 26 | 600 | 4.02 | 271.0 | Table 8 |
| headline | JiT-L/32 | 512 | 462M | 89 | 600 | 2.53 | 299.9 | Table 8 |
| headline | JiT-H/32 | 512 | 956M | 183 | 600 | 1.94 | 309.1 | Table 8 |
| headline | JiT-G/32 | 512 | 2B | 384 | 600 | 1.78 | 306.8 | Table 8 |
| pred ablation | JiT-B/16, x-loss / x-pred | 256 | - | - | 200 | 10.14 | - | Table 2(a) |
| pred ablation | JiT-B/16, x-loss / eps-pred | 256 | - | - | 200 | 379.21 | - | Table 2(a) |
| pred ablation | JiT-B/16, x-loss / v-pred | 256 | - | - | 200 | 107.55 | - | Table 2(a) |
| pred ablation | JiT-B/16, eps-loss / x-pred | 256 | - | - | 200 | 10.45 | - | Table 2(a) |
| pred ablation | JiT-B/16, eps-loss / eps-pred | 256 | - | - | 200 | 394.58 | - | Table 2(a) |
| pred ablation | JiT-B/16, eps-loss / v-pred | 256 | - | - | 200 | 126.88 | - | Table 2(a) |
| pred ablation | JiT-B/16, v-loss / x-pred | 256 | - | - | 200 | 8.62 | - | Table 2(a) |
| pred ablation | JiT-B/16, v-loss / eps-pred | 256 | - | - | 200 | 372.38 | - | Table 2(a) |
| pred ablation | JiT-B/16, v-loss / v-pred | 256 | - | - | 200 | 96.53 | - | Table 2(a) |
| architecture | JiT-B/16, SwiGLU + RMSNorm | 256 | - | - | 200 | 7.48 (6.32) | - | Table 4 |
| architecture | JiT-B/16, + RoPE + qk-norm | 256 | - | - | 200 | 6.69 (5.44) | - | Table 4 |
| architecture | JiT-B/16, + in-context class tokens | 256 | - | - | 200 | 5.49 (4.37) | - | Table 4 |
| architecture | JiT-L/16, + in-context class tokens | 256 | - | - | 200 | 3.39 (2.79) | - | Table 4 |

## V-Co Paper

Source: "V-Co: A Closer Look at Visual Representation Alignment via Co-Denoising" (Lin et al., arXiv:2603.16792v1).

### ImageNet-256 Headline Results

FID-50K / IS-50K, Table 5.

| Model | Representation | Params | Epochs | FID | IS |
|---|---|---:|---:|---:|---:|
| V-Co-B/16 | DINOv2 | 260M | 200 | 2.52 | 242.6 |
| V-Co-L/16 | DINOv2 | 918M | 200 | 2.10 | 243.0 |
| V-Co-H/16 | DINOv2 | 1.9B | 200 | 1.85 | 246.5 |
| V-Co-B/16 | DINOv2 | 260M | 600 | 2.33 | 250.1 |
| V-Co-L/16 | DINOv2 | 918M | 500 | 1.72 | 245.3 |
| V-Co-H/16 | DINOv2 | 1.9B | 300 | 1.71 | 263.3 |

### Architecture Ablation

FID-50K unguided, ImageNet-256, JiT-B/16 backbone, Table 1.

| Model | Params | Design | FID | IS |
|---|---:|---|---:|---:|
| JiT-B/16 | 133M | baseline | 32.54 | 49.5 |
| JiT-B/16 widened | 261M | baseline | 22.67 | 69.9 |
| LatentForcing | 156M | single-stream | 13.06 | 102.2 |
| V-Co | 156M | direct addition | 15.15 | 103.4 |
| V-Co | 157M | channel concat | 14.33 | 107.7 |
| V-Co | 156M | token concat | 14.70 | 103.8 |
| V-Co | 260M | fully dual-stream | 8.86 | 132.8 |

### CFG / Loss / Calibration Ablations

FID-50K, ImageNet-256.

| Ablation | Best Setting | Unguided FID | Guided FID |
|---|---|---:|---:|
| CFG design | semantic-to-pixel mask + joint dropout | 5.62 | 3.18 |
| Auxiliary loss | perceptual-drifting hybrid loss | 4.44 | 2.44 |
| Feature calibration | RMS scaling | 5.38 | 2.52 |

## Latent Forcing Paper

Source: "Latent Forcing: Reordering the Diffusion Trajectory for Pixel-Space Image Generation" (Baade et al., arXiv:2602.11401v1).

### Conditional ImageNet-256, 80 Epochs

FID-50K, Table 9.

| Model | Representation | Unguided | Guided |
|---|---|---:|---:|
| JiT | none | 25.18 | 5.64 |
| JiT + REPA | DINO | 18.60 | 4.57 |
| LF-DiT | DINOv2 | 9.76 | 4.18 |
| LF-DiT | Data2Vec2 | 12.46 | 5.45 |

### Unconditional ImageNet-256, 80 Epochs

FID-50K, Table 10.

| Model | Representation | Unguided | Guided |
|---|---|---:|---:|
| JiT | none | 53.26 | 44.80 |
| JiT + REPA | DINO | 35.04 | 24.40 |
| LF-DiT | DINOv2 | 20.44 | 13.36 |
| LF-DiT | Data2Vec2 | 20.99 | 15.56 |

### Pixel-Diffusion System Comparison

FID-50K, Table 11.

| Model | Params | Decoder Params | Epochs | FID-U | FID-G |
|---|---:|---:|---:|---:|---:|
| JiT-L | 459M | 0 | 200 | 16.21 | 2.79 |
| LF-DiT-L | 465M | 0 | 200 | 7.20 | 2.48 |

### Single-Schedule Ordering

FID-10K, Table 8.

| Time Schedule | Unguided | Guided |
|---|---:|---:|
| Cascaded | 12.42 | 6.60 |
| Linear offset, o = 0.1 | 20.98 | 10.73 |
| Variance shift, alpha = 9 | 13.48 | 8.16 |

### Multi-Schedule Ordering

FID-10K unguided, Table 1.

| Latent Model | alpha 1/64 | alpha 1/16 | alpha 1/4 | alpha 1 | alpha 4 | alpha 16 | alpha 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 64x64 pixels | 44.51 | 44.45 | 44.35 | 44.57 | 44.20 | 42.35 | 42.31 |
| Data2Vec2 | 55.19 | 50.24 | 38.24 | 27.69 | 24.26 | 23.61 | 24.44 |
| DINOv2-B + registers | 55.35 | 50.64 | 37.63 | 24.39 | 18.99 | 18.65 | 18.90 |

### Cascaded Ablations

FID-10K unguided.

| p_latent | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 |
|---|---:|---:|---:|---:|---:|---:|
| FID | 12.54 | 12.42 | 12.77 | 13.17 | 13.48 | 16.13 |

| Max latent noise beta | Epochs | FID |
|---:|---:|---:|
| 0% | 80 | 13.97 |
| 0% | 200 | 16.47 |
| 25% | 80 | 13.48 |
| 25% | 200 | 10.93 |
| 50% | 80 | 14.64 |

| Inference noise after beta=25% training | FID |
|---:|---:|
| 0% | 10.93 |
| 5% | 11.07 |
| 10% | 11.46 |
| 15% | 11.81 |
| 25% | 12.55 |

### Config Values

Table 12.

| Setting | Value |
|---|---|
| batch size | 1024 |
| learning rate | 2e-4 |
| LR schedule | constant |
| weight decay | 0 |
| EMA decay | 0.9999 |
| solver | Heun |
| ODE steps | 50 |
| cascaded steps | 25 latent + 25 pixel |
| DINOv2 latent loss weight | 0.333 |
| Data2Vec2 latent loss weight | 0.25 |
| pixel loss weight | 1.0 |
