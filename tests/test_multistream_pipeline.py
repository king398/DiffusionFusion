"""Stage 2 pipeline tests: loss parity, generate shapes, dataset alignment,
checkpoint auto-remap, legacy spec synthesis."""
import os
import tempfile
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn

from JiT.denoiser import Denoiser
from JiT.main_jit import _maybe_remap_dual_checkpoint, load_stream_specs, resume_or_init_ema
from JiT.model_jit import (
    JiT_Dual_B_2_4C_896,
    StreamSpec,
    remap_dual_to_multistream,
)
from JiT.util.feature_shards import (
    DatasetShardSpan,
    FeatureShardStore,
    MultiStreamShardDataset,
)
from tests._legacy_denoiser import LegacyDenoiser


def _legacy_args(latent_size=32):
    return SimpleNamespace(
        model="JiT-Dual-B/2-4C-896",
        latent_size=latent_size,
        class_num=10,
        attn_dropout=0.0,
        proj_dropout=0.0,
        dino_hidden_size=768,
        dino_patches=16,
        label_drop_prob=0.0,
        P_mean=-0.8,
        P_std=0.8,
        t_eps=0.05,
        inference_t_eps=1e-5,
        noise_scale=1.0,
        latent_loss_weight=1.0,
        dino_loss_weight=1.0,
        dino_time_shift=0.0,
        ema_decay1=0.9999,
        ema_decay2=0.9996,
        sampling_method="heun",
        num_sampling_steps=2,
        cfg=1.0,
        interval_min=0.0,
        interval_max=1.0,
    )


def _multistream_args(specs, latent_size=32, num_sampling_steps=2):
    return SimpleNamespace(
        model="JiT-Dual-B/2-4C-896",
        streams=specs,
        latent_size=latent_size,
        class_num=10,
        attn_dropout=0.0,
        proj_dropout=0.0,
        label_drop_prob=0.0,
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


def _default_specs():
    return [
        StreamSpec(
            name="latent",
            role="image_side",
            feature_channels=4,
            feature_spatial=32,
            patch_size=2,
            tokenizer="latent",
            bottleneck_dim=128,
        ),
        StreamSpec(
            name="dino",
            role="semantic",
            feature_channels=768,
            feature_spatial=16,
            patch_size=1,
            tokenizer="linear",
        ),
    ]


class DenoiserLossParityTests(unittest.TestCase):
    def test_denoiser_loss_parity_n2(self):
        torch.manual_seed(0)
        legacy = LegacyDenoiser(_legacy_args())
        new = Denoiser(_multistream_args(_default_specs()))
        legacy.eval()
        new.eval()
        new.net.load_state_dict(
            remap_dual_to_multistream(legacy.net.state_dict())
        )

        torch.manual_seed(1)
        latent = torch.randn(2, 4, 32, 32)
        dino = torch.randn(2, 768, 16, 16)
        labels = torch.tensor([3, 7], dtype=torch.long)

        # Seed inside the forward call so both denoisers draw the same t,
        # latent noise, then dino noise in identical order.
        torch.manual_seed(42)
        loss_legacy = legacy(latent, dino, labels)
        torch.manual_seed(42)
        loss_new = new({"latent": latent, "dino": dino}, labels)

        torch.testing.assert_close(loss_new, loss_legacy, atol=1e-5, rtol=1e-5)


class DenoiserGenerateTests(unittest.TestCase):
    def test_denoiser_generate_shapes(self):
        torch.manual_seed(0)
        model = Denoiser(_multistream_args(_default_specs(), num_sampling_steps=2))
        model.eval()
        labels = torch.zeros(2, dtype=torch.long)
        out = model.generate(labels)
        self.assertEqual(set(out), {"latent", "dino"})
        self.assertEqual(out["latent"].shape, (2, 4, 32, 32))
        self.assertEqual(out["dino"].shape, (2, 768, 16, 16))


class _FakeMultiStreamShardDataset(MultiStreamShardDataset):
    """Override _load_logical_shard with fabricated in-memory data."""

    def __init__(self, *args, fake_rows, **kwargs):
        self._fake_rows = fake_rows
        super().__init__(*args, **kwargs)

    def _load_logical_shard(self, shard_span):
        return self._fake_rows


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


class MultiStreamShardDatasetTests(unittest.TestCase):
    def test_multistream_shard_dataset_alignment(self):
        total = 16
        latent_store = _make_store("latent", total, 4 * 32 * 32 * 2)
        dino_store = _make_store("dino", total, 768 * 16 * 16 * 2)
        stores = {"latent": latent_store, "dino": dino_store}

        sample_ids = np.arange(total, dtype=np.int64)
        labels = np.arange(total, dtype=np.int64) % 10
        fake_rows = {
            "latent": np.zeros((total, 4, 32, 32), dtype=np.float16),
            "dino": np.zeros((total, 768, 16, 16), dtype=np.float16),
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
            set(first_batch.keys()),
            {"latent", "dino", "y", "sample_id"},
        )
        self.assertEqual(first_batch["latent"].shape[0], 4)
        self.assertEqual(first_batch["dino"].shape[0], 4)
        self.assertEqual(first_batch["y"].shape[0], 4)
        self.assertEqual(first_batch["sample_id"].shape[0], 4)
        self.assertEqual(first_batch["latent"].shape[1:], (4, 32, 32))
        self.assertEqual(first_batch["dino"].shape[1:], (768, 16, 16))

    def test_multistream_shard_dataset_size_mismatch_raises(self):
        latent_store = _make_store("latent", 32, 100)
        dino_store = _make_store("dino", 16, 100)  # different size
        with self.assertRaises(ValueError) as ctx:
            MultiStreamShardDataset(
                stores={"latent": latent_store, "dino": dino_store},
                label_authority="latent",
                batch_size=4,
                num_replicas=1,
                rank=0,
            )
        self.assertIn("total_size", str(ctx.exception))


class EmaCheckpointAutoRemapTests(unittest.TestCase):
    def test_resume_or_init_ema_auto_remaps_dual_checkpoint(self):
        # Use the factory's default in_context_len so the new Denoiser's net
        # (also at the default) has matching shapes when we load.
        legacy_net = JiT_Dual_B_2_4C_896(
            num_classes=10, input_size=8, dino_patches=4
        )

        new_args = _multistream_args(
            [
                StreamSpec(
                    name="latent",
                    role="image_side",
                    feature_channels=4,
                    feature_spatial=8,
                    patch_size=2,
                    tokenizer="latent",
                    bottleneck_dim=128,
                ),
                StreamSpec(
                    name="dino",
                    role="semantic",
                    feature_channels=768,
                    feature_spatial=4,
                    patch_size=1,
                    tokenizer="linear",
                ),
            ],
            latent_size=8,
        )

        new_denoiser = Denoiser(new_args)
        new_denoiser.eval()

        # An actual pre-refactor checkpoint stores legacy names in the model
        # state and both full-state EMA snapshots.
        legacy_keys_model = {
            f"net.{k}": v for k, v in legacy_net.state_dict().items()
        }
        checkpoint = {
            "model": legacy_keys_model,
            "model_ema1": dict(legacy_keys_model),
            "model_ema2": dict(legacy_keys_model),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "checkpoint-last.pth")
            torch.save(checkpoint, checkpoint_path)

            args = SimpleNamespace(resume=tmpdir, start_epoch=0)
            optimizer = torch.optim.AdamW(new_denoiser.parameters(), lr=1e-4)

            with patch("builtins.print") as print_mock:
                resume_or_init_ema(
                    args, new_denoiser, optimizer, device=torch.device("cpu")
                )

        printed = [" ".join(str(value) for value in call.args) for call in print_mock.call_args_list]
        for label in ("model", "model_ema1", "model_ema2"):
            self.assertTrue(
                any(f"Detected dual-stream {label} checkpoint" in line for line in printed)
            )
        self.assertTrue(
            any("Strictly loaded model checkpoint" in line for line in printed)
        )
        self.assertTrue(
            any("Validated model_ema1 checkpoint for EMA swap" in line for line in printed)
        )
        self.assertTrue(
            any("Validated model_ema2 checkpoint for EMA swap" in line for line in printed)
        )
        self.assertIsNotNone(new_denoiser.ema_params1)
        self.assertIsNotNone(new_denoiser.ema_params2)
        param_shapes = [p.shape for p in new_denoiser.parameters()]
        self.assertEqual([p.shape for p in new_denoiser.ema_params1], param_shapes)
        self.assertEqual([p.shape for p in new_denoiser.ema_params2], param_shapes)

    def test_maybe_remap_dual_state_dict_passes_through_multistream(self):
        # A clean multistream state dict should be returned unchanged.
        clean = {"embedders.latent.proj1.weight": torch.zeros(1)}
        self.assertIs(_maybe_remap_dual_checkpoint(clean, label="model"), clean)

    def test_maybe_remap_dual_state_dict_remaps_legacy(self):
        legacy = {"x_embedder.proj1.weight": torch.zeros(1)}
        remapped = _maybe_remap_dual_checkpoint(legacy, label="model")
        self.assertIn("embedders.latent.proj1.weight", remapped)
        self.assertNotIn("x_embedder.proj1.weight", remapped)


class LegacyStreamSpecSynthesisTests(unittest.TestCase):
    def test_legacy_2stream_args_path_synthesizes_specs(self):
        args = SimpleNamespace(
            streams_config=None,
            latent_dir_name="imagenet256_latents",
            dino_dir_name="imagenet256_dinov3_features",
            latent_size=32,
            dino_hidden_size=768,
            dino_patches=16,
            latent_loss_weight=1.5,
            dino_loss_weight=0.5,
            dino_time_shift=0.0,
        )
        specs, dir_names = load_stream_specs(args)

        self.assertEqual([s.name for s in specs], ["latent", "dino"])

        latent_spec, dino_spec = specs
        self.assertEqual(latent_spec.role, "image_side")
        self.assertEqual(latent_spec.feature_channels, 4)
        self.assertEqual(latent_spec.feature_spatial, 32)
        self.assertEqual(latent_spec.patch_size, 2)
        self.assertEqual(latent_spec.tokenizer, "latent")
        self.assertEqual(latent_spec.bottleneck_dim, 128)
        self.assertEqual(latent_spec.loss_weight, 1.5)
        self.assertEqual(latent_spec.time_shift, 0.0)

        self.assertEqual(dino_spec.role, "semantic")
        self.assertEqual(dino_spec.feature_channels, 768)
        self.assertEqual(dino_spec.feature_spatial, 16)
        self.assertEqual(dino_spec.patch_size, 1)
        self.assertEqual(dino_spec.tokenizer, "linear")
        self.assertEqual(dino_spec.loss_weight, 0.5)

        self.assertEqual(
            dir_names,
            {
                "latent": "imagenet256_latents",
                "dino": "imagenet256_dinov3_features",
            },
        )

    def test_legacy_synth_honors_dino_time_shift(self):
        args = SimpleNamespace(
            streams_config=None,
            latent_dir_name="x",
            dino_dir_name="y",
            latent_size=32,
            dino_hidden_size=768,
            dino_patches=16,
            latent_loss_weight=1.0,
            dino_loss_weight=1.0,
            dino_time_shift=0.7,
        )
        specs, _ = load_stream_specs(args)
        self.assertEqual(specs[1].time_shift, 0.7)


if __name__ == "__main__":
    unittest.main()
