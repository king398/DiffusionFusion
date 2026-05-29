"""Stage 4 tests: EVA-02 as a third co-denoising stream.

Everything here is config + abstraction coverage — no model/denoiser/dataset/
engine code changed for the third stream. Tests exercise both fusion modes at
the model and denoiser level, the 3-stream structural mask, the joint
static-graph property under that mask, a 3-store dataset alignment batch, and
the YAML config parse + fusion resolution.

Run with ./.venv/bin/python -m pytest (torch 2.12; the shell-default anaconda
python is torch 1.12 and fails on torch.from_numpy).
"""
import math
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from JiT.denoiser import Denoiser
from JiT.main_jit import load_stream_specs, resolve_fusion
from JiT.model_jit import JiT_models, JiTMultiStream, StreamSpec
from JiT.util.feature_shards import (
    DatasetShardSpan,
    FeatureShardStore,
    MultiStreamShardDataset,
)

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "streams"
_MODEL = "JiT-Dual-B/2-4C-896"
_FUSIONS = ("pairwise", "joint")


def _three_specs():
    """Full-channel 3-stream specs (latent image_side; dino + eva02 semantic)."""
    return [
        StreamSpec(
            name="latent", role="image_side", feature_channels=4,
            feature_spatial=32, patch_size=2, tokenizer="latent", bottleneck_dim=128,
        ),
        StreamSpec(
            name="dino", role="semantic", feature_channels=768,
            feature_spatial=16, patch_size=1, tokenizer="linear",
        ),
        StreamSpec(
            name="eva02", role="semantic", feature_channels=384,
            feature_spatial=16, patch_size=1, tokenizer="linear",
        ),
    ]


def _three_stream_inputs(batch=2):
    return {
        "latent": torch.randn(batch, 4, 32, 32),
        "dino": torch.randn(batch, 768, 16, 16),
        "eva02": torch.randn(batch, 384, 16, 16),
    }


def _make_three_stream_model(fusion_mode):
    return JiTMultiStream(
        specs=_three_specs(),
        input_size=32,
        hidden_size=64,
        depth=4,
        num_heads=4,
        num_classes=8,
        in_context_len=4,
        in_context_start=0,
        cross_every=2,
        cross_start=2,
        fusion_mode=fusion_mode,
    )


def _small_factory(**kwargs):
    """Registry stand-in: tiny JiTMultiStream. specs + fusion_mode flow via kwargs."""
    kwargs.setdefault("hidden_size", 64)
    kwargs.setdefault("depth", 4)
    kwargs.setdefault("num_heads", 4)
    kwargs.setdefault("in_context_len", 4)
    return JiTMultiStream(**kwargs)


def _denoiser_args(specs, *, fusion, latent_size=32, class_num=8,
                   label_drop_prob=0.0, num_sampling_steps=2):
    return SimpleNamespace(
        model=_MODEL,
        streams=specs,
        fusion=fusion,
        latent_size=latent_size,
        class_num=class_num,
        attn_dropout=0.0,
        proj_dropout=0.0,
        label_drop_prob=label_drop_prob,
        P_mean=-0.8,
        P_std=0.8,
        t_eps=0.05,
        inference_t_eps=1e-5,
        noise_scale=1.0,
        ema_decay1=0.9999,
        ema_decay2=0.9996,
        sampling_method="heun",
        num_sampling_steps=num_sampling_steps,
        cfg=1.0,
        interval_min=0.0,
        interval_max=1.0,
    )


def _build_denoiser(args):
    """Build a Denoiser whose net is tiny, via the real construction path."""
    with patch.dict(JiT_models, {args.model: _small_factory}):
        return Denoiser(args)


def _make_store(name, total_size, bytes_per_sample, num_shards=1):
    shard_spans = []
    samples_per = total_size // num_shards
    for i in range(num_shards):
        start = i * samples_per
        end = total_size if i == num_shards - 1 else (i + 1) * samples_per
        shard_spans.append(
            DatasetShardSpan(
                path=f"/tmp/{name}_shard_{i}",
                global_start=start,
                global_end=end,
                first_sample_id=start,
                last_sample_id=end - 1,
            )
        )
    return FeatureShardStore(
        name=name,
        root=f"/tmp/{name}",
        shard_spans=shard_spans,
        total_size=total_size,
        bytes_per_sample=bytes_per_sample,
    )


class _FakeMultiStreamShardDataset(MultiStreamShardDataset):
    """Override _load_logical_shard with fabricated in-memory data."""

    def __init__(self, *args, fake_rows, **kwargs):
        self._fake_rows = fake_rows
        super().__init__(*args, **kwargs)

    def _load_logical_shard(self, shard_span):
        return self._fake_rows


class ThreeStreamModelTests(unittest.TestCase):
    def test_three_stream_model_forward(self):
        for fusion_mode in _FUSIONS:
            with self.subTest(fusion=fusion_mode):
                torch.manual_seed(0)
                model = _make_three_stream_model(fusion_mode).eval()
                streams = _three_stream_inputs(batch=2)
                t = torch.rand(2)
                y = torch.zeros(2, dtype=torch.long)
                with torch.no_grad():
                    out = model(streams, t, y)
                self.assertEqual(set(out), {"latent", "dino", "eva02"})
                self.assertEqual(out["latent"].shape, (2, 4, 32, 32))
                self.assertEqual(out["dino"].shape, (2, 768, 16, 16))
                self.assertEqual(out["eva02"].shape, (2, 384, 16, 16))

    def test_three_stream_static_graph_joint(self):
        torch.manual_seed(0)
        model = _make_three_stream_model("joint")
        streams = _three_stream_inputs(batch=2)
        t = torch.rand(2)
        y = torch.zeros(2, dtype=torch.long)
        # Both semantic streams blocked from the image-side (the structural mask).
        mask = {"dino": {"latent": True}, "eva02": {"latent": True}}
        out = model(streams, t, y, mask=mask)
        loss = sum(v.sum() for v in out.values())
        loss.backward()
        missing = [
            name
            for name, p in model.named_parameters()
            if p.requires_grad and p.grad is None
        ]
        self.assertEqual(missing, [], f"params without grad under 3-stream mask: {missing}")


class ThreeStreamDenoiserTests(unittest.TestCase):
    def test_three_stream_denoiser_forward_and_generate(self):
        for fusion in _FUSIONS:
            with self.subTest(fusion=fusion):
                torch.manual_seed(0)
                args = _denoiser_args(_three_specs(), fusion=fusion)
                model = _build_denoiser(args)
                self.assertEqual(model.net.fusion_mode, fusion)

                model.train()
                streams = _three_stream_inputs(batch=2)
                labels = torch.tensor([3, 7], dtype=torch.long)
                loss = model(streams, labels)
                self.assertEqual(loss.ndim, 0)  # scalar
                self.assertTrue(torch.isfinite(loss))

                model.eval()
                out = model.generate(labels)
                self.assertEqual(set(out), {"latent", "dino", "eva02"})
                self.assertEqual(out["latent"].shape, (2, 4, 32, 32))
                self.assertEqual(out["dino"].shape, (2, 768, 16, 16))
                self.assertEqual(out["eva02"].shape, (2, 384, 16, 16))

    def test_three_stream_structural_mask(self):
        torch.manual_seed(0)
        args = _denoiser_args(_three_specs(), fusion="joint")
        model = _build_denoiser(args)
        mask = model._structural_uncond_mask()
        # Both semantics blocked from the image-side; semantics not blocked from
        # each other (no eva02<->dino entries).
        self.assertEqual(
            mask, {"dino": {"latent": True}, "eva02": {"latent": True}}
        )


class ThreeStreamDatasetTests(unittest.TestCase):
    def test_three_store_dataset_alignment(self):
        total = 16
        stores = {
            "latent": _make_store("latent", total, 4 * 32 * 32 * 2),
            "dino": _make_store("dino", total, 768 * 16 * 16 * 2),
            "eva02": _make_store("eva02", total, 384 * 16 * 16 * 2),
        }
        sample_ids = np.arange(total, dtype=np.int64)
        labels = np.arange(total, dtype=np.int64) % 10
        fake_rows = {
            "latent": np.zeros((total, 4, 32, 32), dtype=np.float16),
            "dino": np.zeros((total, 768, 16, 16), dtype=np.float16),
            "eva02": np.zeros((total, 384, 16, 16), dtype=np.float16),
            "y": labels,
            "sample_id": sample_ids,
        }
        dataset = _FakeMultiStreamShardDataset(
            stores=stores,
            label_authority="latent",
            batch_size=4,
            num_replicas=1,
            rank=0,
            shuffle_shards=False,
            seed=0,
            preload_next_shard=False,
            fake_rows=fake_rows,
        )

        first_batch = next(iter(dataset))
        self.assertEqual(
            set(first_batch.keys()), {"latent", "dino", "eva02", "y", "sample_id"}
        )
        for key in ("latent", "dino", "eva02", "y", "sample_id"):
            self.assertEqual(first_batch[key].shape[0], 4)
        self.assertEqual(first_batch["latent"].shape[1:], (4, 32, 32))
        self.assertEqual(first_batch["dino"].shape[1:], (768, 16, 16))
        self.assertEqual(first_batch["eva02"].shape[1:], (384, 16, 16))


class ThreeStreamConfigTests(unittest.TestCase):
    def test_load_stream_specs_three_stream_yaml(self):
        cfg = str(_CONFIG_DIR / "latent_dino_eva02_joint.yaml")
        args = SimpleNamespace(streams_config=cfg)
        specs, dir_names = load_stream_specs(args)

        self.assertEqual([s.name for s in specs], ["latent", "dino", "eva02"])
        self.assertEqual(
            dir_names,
            {
                "latent": "imagenet256_latents",
                "dino": "imagenet256_dinov3_features",
                "eva02": "imagenet224_eva02_small_features",
            },
        )

        eva = specs[2]
        self.assertEqual(eva.role, "semantic")
        self.assertEqual(eva.feature_channels, 384)
        self.assertEqual(eva.feature_spatial, 16)
        self.assertEqual(eva.patch_size, 1)
        self.assertEqual(eva.tokenizer, "linear")

        # fusion: joint is honored when --fusion is not passed on the CLI.
        args_unset = SimpleNamespace(streams_config=cfg, fusion="pairwise")
        self.assertEqual(resolve_fusion(args_unset, argv=[]), "joint")
        # An explicit CLI --fusion still wins over the YAML key.
        self.assertEqual(
            resolve_fusion(args_unset, argv=["--fusion", "pairwise"]), "pairwise"
        )

    def test_pairwise_yaml_parses_three_streams(self):
        cfg = str(_CONFIG_DIR / "latent_dino_eva02.yaml")
        args = SimpleNamespace(streams_config=cfg, fusion="pairwise")
        specs, dir_names = load_stream_specs(args)
        self.assertEqual([s.name for s in specs], ["latent", "dino", "eva02"])
        self.assertEqual(resolve_fusion(args, argv=[]), "pairwise")

    def test_three_stream_requires_streams_config(self):
        # The legacy flat-arg synthesis path only yields 2 streams (latent+dino),
        # so a 3-stream run must use --streams_config. This documents the constraint.
        args = SimpleNamespace(
            streams_config=None,
            latent_dir_name="imagenet256_latents",
            dino_dir_name="imagenet256_dinov3_features",
            latent_size=32,
            dino_hidden_size=768,
            dino_patches=16,
            latent_loss_weight=1.0,
            dino_loss_weight=1.0,
            dino_time_shift=0.0,
        )
        specs, _ = load_stream_specs(args)
        self.assertEqual([s.name for s in specs], ["latent", "dino"])
        self.assertEqual(len(specs), 2)


if __name__ == "__main__":
    unittest.main()
