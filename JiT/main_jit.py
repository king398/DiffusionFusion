import argparse
import datetime
import numpy as np
import os
import sys

import time
from pathlib import Path
import math
from typing import Dict, List, Tuple

import torch
import torch.backends.cudnn as cudnn

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

import JiT.util.misc as misc
from JiT.util.dataset import (
    MultiStreamRamLoadedShardDataset,
    inspect_feature_shards,
)
from JiT.engine_jit import train_one_epoch, evaluate
from JiT.denoiser import Denoiser
from JiT.model_jit import StreamSpec, remap_dual_to_multistream
from JiT.eval.diffusion_decoder import load_decoder_for_eval

import wandb
import yaml

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


_FID_STATS_DIR = Path(__file__).resolve().parent / "fid_stats"


def resolve_default_fid_stats_path(latent_size: int) -> str | None:
    image_size = latent_size * 8
    candidate = _FID_STATS_DIR / f"jit_in{image_size}_stats.npz"
    if candidate.is_file():
        return str(candidate)
    return None


def add_bool_arg(parser, name, default, help=None):
    parser.add_argument(f"--{name}", action="store_true", help=help)
    parser.add_argument(f"--no_{name}", action="store_false", dest=name)
    parser.set_defaults(**{name: default})


def get_args_parser():
    parser = argparse.ArgumentParser('JiT', add_help=False)

    # architecture
    parser.add_argument('--model', default='JiT-Dual-B/2-4C-896', type=str, metavar='MODEL',
                        help='Name of the model to train')
    parser.add_argument('--fusion', default='pairwise', choices=['pairwise', 'joint'],
                        help='Cross-stream fusion mode. "pairwise" (default) keeps the '
                             'per-stream towers with periodic cross-fusion (the dual '
                             'baseline). "joint" uses V-Co-style all-to-all attention at '
                             'every block; it diverges from the pairwise baseline and must '
                             'be trained from scratch. A top-level "fusion:" key in '
                             '--streams_config is honored when --fusion is left unset.')
    parser.add_argument('--latent_size', default=32,
                        type=int, help='Latent size')
    parser.add_argument('--dino_patches', default=16,
                        type=int, help='DINO patch size')
    parser.add_argument('--attn_dropout', type=float,
                        default=0.0, help='Attention dropout rate')
    parser.add_argument('--proj_dropout', type=float,
                        default=0.0, help='Projection dropout rate')
    parser.add_argument('--decoder_checkpoint', type=str, default=None,
                        help='Path to trained decoder checkpoint for eval decoding')
    parser.add_argument('--decoder_checkpoint_key', type=str, default='auto',
                        choices=['auto', 'model', 'model_ema'],
                        help='Decoder checkpoint state dict key to load')
    parser.add_argument("--dino_hidden_size", type=int, default=768,
                        help="Hidden size of DINO features (e.g. 768 for DiT-B/2)")

    # training
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='Epochs to warm up LR')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU before gradient accumulation')
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Number of micro-batches to accumulate before each optimizer step')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='Learning rate (absolute)')
    parser.add_argument('--blr', type=float, default=5e-5, metavar='LR',
                        help='Base learning rate: absolute_lr = base_lr * effective_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='Minimum LR for cyclic schedulers that hit 0')
    parser.add_argument('--lr_schedule', type=str, default='cosine',
                        help='Learning rate schedule')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay (default: 0.0)')
    parser.add_argument('--ema_decay1', type=float, default=0.9999,
                        help='The first ema to track. Use the first ema for sampling by default.')
    parser.add_argument('--ema_decay2', type=float, default=0.9996,
                        help='The second ema to track')
    parser.add_argument('--P_mean', default=-0.8, type=float)
    parser.add_argument('--P_std', default=0.8, type=float)
    parser.add_argument('--noise_scale', default=1.0, type=float)
    parser.add_argument('--t_eps', default=5e-2, type=float)
    parser.add_argument('--inference_t_eps', default=1e-5, type=float,
                        help='Clamp floor used only during inference velocity conversion')
    parser.add_argument('--dino_time_shift', default=0.0, type=float,
                        help='Logit-space shift applied to the DINO denoising time schedule')
    parser.add_argument('--label_drop_prob', default=0.1, type=float)
    parser.add_argument('--latent_loss_weight', default=1.0, type=float,
                        help='Weight applied to the latent denoising loss')
    parser.add_argument('--dino_loss_weight', default=1.0, type=float,
                        help='Weight applied to the DINO denoising loss')

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='Starting epoch')
    add_bool_arg(
        parser, 'pin_mem', True,
        help='Pin CPU memory in DataLoader for faster GPU transfers',
    )
    add_bool_arg(
        parser, 'ram_shard_prefetch', True,
        help='While one RAM-loaded shard is training, preload the next shard in a background thread',
    )
    add_bool_arg(
        parser, 'cuda_prefetch', True,
        help='Prefetch the next training batch to CUDA on a side stream',
    )
    parser.add_argument('--ddp_bucket_cap_mb', default=100, type=int,
                        help='DDP gradient bucket size in MB')
    add_bool_arg(
        parser, 'ddp_broadcast_buffers', False,
        help='Broadcast model buffers from rank 0 each forward',
    )
    add_bool_arg(
        parser, 'ddp_gradient_as_bucket_view', True,
        help='Use DDP bucket views to reduce gradient memory copies',
    )
    add_bool_arg(
        parser, 'ddp_static_graph', True,
        help='Enable DDP static graph optimizations',
    )
    add_bool_arg(
        parser, 'ddp_find_unused_parameters', False,
        help='Enable DDP unused-parameter detection for dynamic structural masks',
    )
    add_bool_arg(
        parser, 'compile_model', True,
        help='Compile the DDP training model with torch.compile',
    )

    # sampling
    parser.add_argument('--sampling_method', default='heun', type=str,
                        help='ODE samping method')
    parser.add_argument('--num_sampling_steps', default=50, type=int,
                        help='Sampling steps')
    parser.add_argument('--cfg', default=1.0, type=float,
                        help='Classifier-free guidance factor')
    parser.add_argument('--interval_min', default=0.0, type=float,
                        help='CFG interval min')
    parser.add_argument('--interval_max', default=1.0, type=float,
                        help='CFG interval max')
    parser.add_argument('--num_images', default=50000, type=int,
                        help='Number of images to generate')
    parser.add_argument('--eval_freq', type=int, default=40,
                        help='Frequency (in epochs) for evaluation')
    parser.add_argument('--online_eval', action='store_true')
    parser.add_argument('--evaluate_gen', action='store_true')
    parser.add_argument('--gen_bsz', type=int, default=256,
                        help='Generation batch size')
    parser.add_argument(
        '--fid_stats_path',
        type=str,
        default=None,
        help='Path to a torch-fidelity FID statistics .npz file used for online evaluation.',
    )

    # dataset
    parser.add_argument('--data_path', default='./data/imagenet', type=str,
                        help='Path to the dataset')
    parser.add_argument('--class_num', default=1000, type=int)
    parser.add_argument('--dino_dir_name', default='imagenet256_dinov3_features', type=str,
                        help='Path to DINO features dataset (HF dataset name or local path)')
    parser.add_argument('--latent_dir_name', default='imagenet256_latents', type=str,
                        help='Name for the output HF dataset containing VAE features')
    parser.add_argument('--streams_config', type=str, default=None,
                        help='Path to a YAML file describing the input streams. When unset, '
                             'a 2-stream (latent + dino) config is synthesized from the legacy '
                             'flat flags below for backward compatibility.')

    # checkpointing
    parser.add_argument('--output_dir', default='./output_dir',
                        help='Directory to save outputs (empty for no saving)')
    parser.add_argument('--resume', default='',
                        help='Folder that contains checkpoint to resume from')
    parser.add_argument('--save_last_freq', type=int, default=5,
                        help='Frequency (in epochs) to save checkpoints')
    parser.add_argument('--log_freq', default=5, type=int)
    add_bool_arg(parser, 'use_wandb', True, help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='jit',
                        help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='Weights & Biases entity/team name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='Optional Weights & Biases run name')
    parser.add_argument('--wandb_mode', type=str, default='online',
                        choices=['online', 'offline', 'disabled'],
                        help='Weights & Biases mode')
    parser.add_argument('--wandb_eval_image_interval', type=int, default=10,
                        help='Log one generated eval image to W&B every N images')
    parser.add_argument('--eval_save_workers', type=int, default=4,
                        help='Background workers per rank for saving online-eval PNGs')
    parser.add_argument('--device', default='cuda',
                        help='Device to use for training/testing')

    # distributed training
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='URL used to set up distributed training')
    parser.add_argument('--dist_timeout_sec', default=7200, type=int,
                        help='Distributed process group timeout in seconds')

    return parser


_LEGACY_LATENT_NAME = "latent"
_LEGACY_DINO_NAME = "dino"


def _legacy_two_stream_specs(args) -> Tuple[List[StreamSpec], Dict[str, str]]:
    latent_spec = StreamSpec(
        name=_LEGACY_LATENT_NAME,
        role="image_side",
        feature_channels=4,
        feature_spatial=args.latent_size,
        patch_size=2,
        tokenizer="latent",
        bottleneck_dim=128,
        time_shift=0.0,
        loss_weight=float(getattr(args, "latent_loss_weight", 1.0)),
    )
    dino_time_shift = getattr(args, "dino_time_shift", 0.0)
    dino_time_shift = 0.0 if dino_time_shift is None else float(dino_time_shift)
    dino_spec = StreamSpec(
        name=_LEGACY_DINO_NAME,
        role="semantic",
        feature_channels=int(args.dino_hidden_size),
        feature_spatial=int(args.dino_patches),
        patch_size=1,
        tokenizer="linear",
        time_shift=dino_time_shift,
        loss_weight=float(getattr(args, "dino_loss_weight", 1.0)),
    )
    dir_names = {
        latent_spec.name: args.latent_dir_name,
        dino_spec.name: args.dino_dir_name,
    }
    return [latent_spec, dino_spec], dir_names


def _spec_from_yaml(entry: Dict[str, object]) -> Tuple[StreamSpec, str]:
    required = (
        "name",
        "role",
        "feature_channels",
        "feature_spatial",
        "patch_size",
        "tokenizer",
        "dir_name",
    )
    missing = [k for k in required if k not in entry]
    if missing:
        raise ValueError(
            f"streams_config entry is missing required keys: {missing} (entry={entry})."
        )
    spec = StreamSpec(
        name=str(entry["name"]),
        role=str(entry["role"]),
        feature_channels=int(entry["feature_channels"]),
        feature_spatial=int(entry["feature_spatial"]),
        patch_size=int(entry["patch_size"]),
        tokenizer=str(entry["tokenizer"]),
        bottleneck_dim=int(entry["bottleneck_dim"]) if entry.get("bottleneck_dim") is not None else None,
        time_shift=float(entry.get("time_shift", 0.0) or 0.0),
        loss_weight=float(entry.get("loss_weight", 1.0)),
    )
    return spec, str(entry["dir_name"])


def load_stream_specs(args) -> Tuple[List[StreamSpec], Dict[str, str]]:
    """Resolve a list of StreamSpecs plus a {name: dir_name} mapping.

    If ``args.streams_config`` is set, parses the YAML file. Otherwise
    synthesizes the canonical 2-stream (latent + dino) config from the
    legacy flat flags so existing sbatch scripts keep working.
    """
    streams_config = getattr(args, "streams_config", None)
    if not streams_config:
        return _legacy_two_stream_specs(args)

    with open(streams_config, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or "streams" not in data:
        raise ValueError(
            f"streams_config {streams_config!r} must define a top-level 'streams' list."
        )
    entries = data["streams"]
    if not isinstance(entries, list) or not entries:
        raise ValueError(
            f"streams_config {streams_config!r} has an empty or malformed 'streams' list."
        )

    specs: List[StreamSpec] = []
    dir_names: Dict[str, str] = {}
    for entry in entries:
        spec, dir_name = _spec_from_yaml(entry)
        if spec.name in dir_names:
            raise ValueError(
                f"Duplicate stream name in streams_config: {spec.name!r}."
            )
        specs.append(spec)
        dir_names[spec.name] = dir_name
    return specs, dir_names


def _streams_config_fusion(args) -> "str | None":
    """Return the top-level ``fusion:`` value from the streams_config YAML, if any.

    The streams_config YAML may carry an optional top-level ``fusion`` key
    alongside its ``streams`` list; ``load_stream_specs`` ignores it (it only
    reads ``streams``), so we peek at it separately here.
    """
    streams_config = getattr(args, "streams_config", None)
    if not streams_config:
        return None
    with open(streams_config, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if isinstance(data, dict) and data.get("fusion") is not None:
        return str(data["fusion"])
    return None


def resolve_fusion(args, argv=None) -> str:
    """Resolve the effective fusion mode.

    Precedence: an explicit ``--fusion`` CLI flag wins; otherwise a top-level
    ``fusion:`` key in ``--streams_config`` is honored; otherwise the CLI
    default (``"pairwise"``) stands.
    """
    tokens = sys.argv[1:] if argv is None else argv
    cli_explicit = any(
        tok == "--fusion" or tok.startswith("--fusion=") for tok in tokens
    )
    if cli_explicit:
        fusion = args.fusion
    else:
        yaml_fusion = _streams_config_fusion(args)
        fusion = yaml_fusion if yaml_fusion is not None else args.fusion
    if fusion not in ("pairwise", "joint"):
        raise ValueError(
            f"fusion must be 'pairwise' or 'joint', got {fusion!r} "
            "(check --fusion or the streams_config 'fusion:' key)."
        )
    return fusion


def init_loggers(args, global_rank):
    log_writer = None
    if global_rank == 0 and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        if SummaryWriter is not None:
            log_writer = SummaryWriter(log_dir=args.output_dir)

    wandb_run = None
    if global_rank == 0 and args.use_wandb:
        
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config=vars(args),
            dir=args.output_dir or None,
            mode=args.wandb_mode,
        )
        misc.configure_wandb_step_metrics(wandb_run)
    return log_writer, wandb_run


def build_train_loader(args, stores, label_authority, num_tasks, global_rank):
    dataset = MultiStreamRamLoadedShardDataset(
        stores=stores,
        label_authority=label_authority,
        batch_size=args.batch_size,
        num_replicas=num_tasks,
        rank=global_rank,
        shuffle_shards=True,
        seed=args.seed,
        preload_next_shard=args.ram_shard_prefetch,
    )
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=None,
        num_workers=0,
        pin_memory=args.pin_mem,
    )
    return dataset, loader


def describe_dataset_plan(dataset, stores, args):
    plan = dataset.describe_current_plan()
    max_shard_samples = max(span.size for span in dataset.logical_shards)
    bytes_per_sample = sum(store.bytes_per_sample for store in stores.values())
    approx_max_ram_bytes = max_shard_samples * bytes_per_sample
    print(
        "RAM shard loading enabled using "
        f"{plan['logical_shard_count']} logical shards from {plan['logical_shard_source']}."
    )
    print(
        "Approx max per-rank shard working set: "
        f"{approx_max_ram_bytes / (1024 ** 3):.2f} GiB."
    )
    if args.ram_shard_prefetch:
        print(
            "RAM shard prefetch enabled: peak per-rank working set can temporarily reach about "
            f"{approx_max_ram_bytes * 2 / (1024 ** 3):.2f} GiB while the next shard is staged."
        )
    print(
        "Epoch 0 steps per rank: "
        f"{plan['num_batches']} "
        f"(samples/rank={plan['num_samples_per_rank']}, "
        f"dropped_tail_per_rank={plan['dropped_samples_per_rank']})."
    )
    print(
        "Gradient accumulation: "
        f"{args.accum_iter} micro-batches/update "
        f"-> {math.ceil(plan['num_batches'] / args.accum_iter)} optimizer updates per rank in epoch 0."
    )


def load_eval_decoder(args, device, global_rank):
    if not (args.online_eval or args.evaluate_gen):
        return None

    fid_stats_path = args.fid_stats_path or resolve_default_fid_stats_path(args.latent_size)
    if fid_stats_path is None:
        raise FileNotFoundError(
            "Evaluation requires --fid_stats_path, and no built-in FID stats file was found "
            f"for latent_size={args.latent_size} under {_FID_STATS_DIR}."
        )
    args.fid_stats_path = str(Path(fid_stats_path).expanduser().resolve())
    if not os.path.isfile(args.fid_stats_path):
        raise FileNotFoundError(f"FID statistics file not found: {args.fid_stats_path}")
    if not args.decoder_checkpoint:
        raise ValueError("Evaluation requires --decoder_checkpoint pointing to a trained decoder.")

    decoder = load_decoder_for_eval(
        args.decoder_checkpoint, device, args.decoder_checkpoint_key,
    )
    print(f"Rank {global_rank}: loaded decoder from {args.decoder_checkpoint}")
    return decoder


_DUAL_STATE_DICT_MARKERS = (
    "latent_blocks.",
    "dino_blocks.",
    "x_embedder.",
    "dino_embedder.",
    "cross_fusion_blocks.",
    "latent_final_layer.",
    "dino_final_layer.",
    "latent_in_context_posemb",
    "dino_in_context_posemb",
)


def _maybe_remap_dual_checkpoint(state_dict, *, label):
    """Detect a dual-stream state_dict and translate it to multistream keys.

    Checkpoints saved by the production training loop nest model parameters
    under a top-level ``net.`` wrapper (since the Denoiser stores its network
    in ``self.net``). We strip that prefix when remapping and reapply it
    afterwards so callers can plug the result straight into ``load_state_dict``.
    """
    if not isinstance(state_dict, dict) or not state_dict:
        return state_dict
    # All keys in a real checkpoint share the same outer prefix.
    sample_key = next(iter(state_dict))
    prefix = "net." if sample_key.startswith("net.") else ""

    is_dual = False
    for key in state_dict:
        bare = key[len(prefix):]
        for marker in _DUAL_STATE_DICT_MARKERS:
            if bare == marker or bare.startswith(marker):
                is_dual = True
                break
        if is_dual:
            break
    if not is_dual:
        return state_dict
    print(
        f"Detected dual-stream {label} checkpoint; applying remap_dual_to_multistream."
    )
    bare = {key[len(prefix):]: value for key, value in state_dict.items()}
    remapped_bare = remap_dual_to_multistream(bare)
    return {prefix + k: v for k, v in remapped_bare.items()}


def _validate_ema_checkpoint_state_dict(state_dict, model_without_ddp, *, label):
    """Validate the subset of checkpoint state used for EMA parameter swaps."""
    if not isinstance(state_dict, dict):
        raise TypeError(f"{label} checkpoint state must be a dict, got {type(state_dict)!r}.")

    model_state_keys = set(model_without_ddp.state_dict())
    named_parameters = dict(model_without_ddp.named_parameters())
    state_keys = set(state_dict)
    missing_parameters = sorted(set(named_parameters) - state_keys)
    unexpected_keys = sorted(state_keys - model_state_keys)
    shape_mismatches = [
        f"{name}: checkpoint={tuple(state_dict[name].shape)}, model={tuple(param.shape)}"
        for name, param in named_parameters.items()
        if name in state_dict and state_dict[name].shape != param.shape
    ]
    if missing_parameters or unexpected_keys or shape_mismatches:
        details = []
        if missing_parameters:
            details.append(f"missing parameters={missing_parameters[:10]}")
        if unexpected_keys:
            details.append(f"unexpected keys={unexpected_keys[:10]}")
        if shape_mismatches:
            details.append(f"shape mismatches={shape_mismatches[:10]}")
        raise RuntimeError(
            f"{label} checkpoint is incompatible with EMA parameter swap after remap: "
            + "; ".join(details)
        )
    print(
        f"Validated {label} checkpoint for EMA swap: "
        f"{len(named_parameters)} parameters found, zero unexpected keys."
    )


def resume_or_init_ema(args, model_without_ddp, optimizer, device):
    checkpoint_path = Path(args.resume) / "checkpoint-last.pth" if args.resume else None
    if checkpoint_path and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint['model'] = _maybe_remap_dual_checkpoint(
            checkpoint['model'], label="model"
        )
        checkpoint['model_ema1'] = _maybe_remap_dual_checkpoint(
            checkpoint['model_ema1'], label="model_ema1"
        )
        checkpoint['model_ema2'] = _maybe_remap_dual_checkpoint(
            checkpoint['model_ema2'], label="model_ema2"
        )
        model_without_ddp.load_state_dict(checkpoint['model'])
        print("Strictly loaded model checkpoint: zero missing or unexpected keys.")
        _validate_ema_checkpoint_state_dict(
            checkpoint['model_ema1'], model_without_ddp, label="model_ema1"
        )
        _validate_ema_checkpoint_state_dict(
            checkpoint['model_ema2'], model_without_ddp, label="model_ema2"
        )
        model_without_ddp.ema_params1 = [
            checkpoint['model_ema1'][name].to(device)
            for name, _ in model_without_ddp.named_parameters()
        ]
        model_without_ddp.ema_params2 = [
            checkpoint['model_ema2'][name].to(device)
            for name, _ in model_without_ddp.named_parameters()
        ]
        print("Resumed checkpoint from", args.resume)

        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            print("Loaded optimizer state")
        return

    model_without_ddp.ema_params1 = [
        param.detach().clone() for param in model_without_ddp.parameters()
    ]
    model_without_ddp.ema_params2 = [
        param.detach().clone() for param in model_without_ddp.parameters()
    ]
    print("Training from scratch")


def main(args):
    misc.init_distributed_mode(args)
    print('Job directory:', os.path.dirname(os.path.realpath(__file__)))
    print("Arguments:\n{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    if args.accum_iter < 1:
        raise ValueError("--accum_iter must be at least 1.")

    # Set seeds for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    log_writer, wandb_run = init_loggers(args, global_rank)

    specs, stream_dirs = load_stream_specs(args)
    args.streams = specs
    args.fusion = resolve_fusion(args)
    image_side_names = [spec.name for spec in specs if spec.role == "image_side"]
    if len(image_side_names) != 1:
        raise ValueError(
            f"Streams config must declare exactly one image_side stream; got {image_side_names}."
        )
    label_authority = image_side_names[0]
    if global_rank == 0:
        print(
            "Stream specs:",
            [
                f"{s.name}(role={s.role}, ch={s.feature_channels}, sp={s.feature_spatial}, p={s.patch_size})"
                for s in specs
            ],
        )

    stores = {
        name: inspect_feature_shards(args.data_path, dir_name)
        for name, dir_name in stream_dirs.items()
    }
    dataset_train, data_loader_train = build_train_loader(
        args, stores, label_authority, num_tasks, global_rank
    )
    initial_steps_per_epoch = len(data_loader_train)
    if initial_steps_per_epoch <= 0:
        raise RuntimeError("Training dataloader has zero steps for this epoch.")
    initial_optimizer_steps_per_epoch = math.ceil(initial_steps_per_epoch / args.accum_iter)

    if global_rank == 0:
        describe_dataset_plan(dataset_train, stores, args)

    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.optimize_ddp = False

    # Create denoiser
    model = Denoiser(args)

    print("Model =", model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {:.6f}M".format(n_params / 1e6))

    model.to(device)

    decoder = load_eval_decoder(args, device, global_rank)

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:  # only base_lr (blr) is specified
        args.lr = args.blr * eff_batch_size / 256

    print("Base lr: {:.2e}".format(args.lr * 256 / eff_batch_size))
    print("Actual lr: {:.2e}".format(args.lr))
    print("Gradient accumulation steps: %d" % args.accum_iter)
    print("Effective batch size: %d" % eff_batch_size)

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.gpu],
        output_device=args.gpu,
        find_unused_parameters=args.ddp_find_unused_parameters,
        bucket_cap_mb=args.ddp_bucket_cap_mb,
        broadcast_buffers=args.ddp_broadcast_buffers,
        gradient_as_bucket_view=args.ddp_gradient_as_bucket_view,
        static_graph=args.ddp_static_graph,
    )
    model_without_ddp = model.module

    # Compile the full DDP model for training only;
    # eval uses model_without_ddp (uncompiled) to avoid dynamic-shape issues.
    if args.compile_model:
        if global_rank == 0:
            print(
                "torch.compile enabled; the first optimizer step may spend several "
                "minutes compiling before normal training logs appear."
            )
        compiled_model = torch.compile(model)
    else:
        compiled_model = model
    if global_rank == 0 and not args.compile_model:
        print("torch.compile disabled; training with eager DDP model.")

    # Set up optimizer with weight decay adjustment for bias and norm layers
    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)

    resume_or_init_ema(args, model_without_ddp, optimizer, device)

    try:
        wandb_epoch_end_step = args.start_epoch * initial_optimizer_steps_per_epoch
        # Evaluate generation
        if args.evaluate_gen:
            print("Evaluating checkpoint at {} epoch".format(args.start_epoch))
            with torch.random.fork_rng():
                torch.manual_seed(seed)
                with torch.no_grad():
                    evaluate(
                        model_without_ddp,
                        args,
                        args.start_epoch,
                        batch_size=args.gen_bsz,
                        log_writer=log_writer,
                        decoder=decoder,
                        wandb_run=wandb_run,
                        wandb_step=wandb_epoch_end_step,
                    )
            return

        # Training loop
        print(f"Start training for {args.epochs} epochs")
        start_time = time.time()
        for epoch in range(args.start_epoch, args.epochs):
            dataset_train.set_epoch(epoch)
            steps_per_epoch = len(data_loader_train)
            if steps_per_epoch <= 0:
                raise RuntimeError("Training dataloader has zero steps for this epoch.")
            optimizer_steps_per_epoch = (
                steps_per_epoch + args.accum_iter - 1
            ) // args.accum_iter
            wandb_epoch_end_step = (epoch + 1) * optimizer_steps_per_epoch

            train_one_epoch(
                compiled_model,
                model_without_ddp,
                data_loader_train,
                optimizer,
                device,
                epoch,
                log_writer=log_writer,
                args=args,
                steps_per_epoch=steps_per_epoch,
                optimizer_steps_per_epoch=optimizer_steps_per_epoch,
                wandb_run=wandb_run,
            )

            # Save checkpoint periodically
            did_save_checkpoint = False
            if epoch % args.save_last_freq == 0 or epoch + 1 == args.epochs:
                misc.save_model(
                    args=args,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    epoch=epoch,
                    epoch_name="last"
                )
                did_save_checkpoint = True

            if epoch % 50 == 0 and epoch > 0:
                misc.save_model(
                    args=args,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    epoch=epoch
                )
                did_save_checkpoint = True

            # Keep ranks in lockstep after rank-0 checkpoint I/O before eval/next epoch.
            if did_save_checkpoint and args.distributed:
                misc.distributed_barrier()

            # Perform online evaluation at specified intervals
            completed_epochs = epoch + 1
            if args.online_eval and (
                completed_epochs % args.eval_freq == 0
                or completed_epochs == args.epochs
            ):
                torch.cuda.empty_cache()
                with torch.no_grad():
                    evaluate(
                        model_without_ddp,
                        args,
                        epoch,
                        batch_size=args.gen_bsz,
                        log_writer=log_writer,
                        decoder=decoder,
                        wandb_run=wandb_run,
                        wandb_step=wandb_epoch_end_step,
                    )
                torch.cuda.empty_cache()

            if misc.is_main_process() and log_writer is not None:
                log_writer.flush()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time:', total_time_str)
        if wandb_run is not None:
            payload = {
                "train/total_time_sec": total_time,
                "train/total_time_hms": total_time_str,
            }
            misc.add_wandb_global_step(payload, wandb_epoch_end_step)
            wandb_run.log(payload)
    finally:
        if log_writer is not None:
            log_writer.close()
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
