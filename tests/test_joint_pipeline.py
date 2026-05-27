"""Stage 3b integration tests: --fusion threaded CLI -> Denoiser -> model.

Covers a joint-mode Denoiser forward+generate, the pairwise default flowing
through the real arg parser, a from-scratch smoke train proving the joint path
learns and backprops, and the structural-mask path firing under label dropout.

The model is shrunk to tiny dims by temporarily swapping the model registry
entry for a small-dim factory; fusion_mode (and specs) still flow through the
real Denoiser construction call, so the end-to-end threading is exercised.
"""
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from JiT.denoiser import Denoiser
from JiT.main_jit import get_args_parser
from JiT.model_jit import JiT_models, JiTMultiStream, StreamSpec

_MODEL = "JiT-Dual-B/2-4C-896"


def _default_specs():
    """Full-channel 2-stream specs so generate yields dino [B,768,16,16]."""
    return [
        StreamSpec(
            name="latent", role="image_side", feature_channels=4,
            feature_spatial=32, patch_size=2, tokenizer="latent", bottleneck_dim=128,
        ),
        StreamSpec(
            name="dino", role="semantic", feature_channels=768,
            feature_spatial=16, patch_size=1, tokenizer="linear",
        ),
    ]


def _small_specs():
    """Tiny token grids for fast forward/backward tests."""
    return [
        StreamSpec(
            name="latent", role="image_side", feature_channels=4,
            feature_spatial=8, patch_size=2, tokenizer="latent", bottleneck_dim=16,
        ),
        StreamSpec(
            name="dino", role="semantic", feature_channels=12,
            feature_spatial=4, patch_size=1, tokenizer="linear",
        ),
    ]


def _small_factory(**kwargs):
    """Registry stand-in: tiny JiTMultiStream. specs + fusion_mode flow via kwargs."""
    kwargs.setdefault("hidden_size", 64)
    kwargs.setdefault("depth", 4)
    kwargs.setdefault("num_heads", 4)
    kwargs.setdefault("in_context_len", 4)
    return JiTMultiStream(**kwargs)


def _pipeline_args(specs, *, fusion="joint", latent_size, class_num=8,
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


class JointDenoiserForwardGenerateTests(unittest.TestCase):
    def test_denoiser_joint_forward_and_generate(self):
        torch.manual_seed(0)
        args = _pipeline_args(_default_specs(), fusion="joint", latent_size=32)
        model = _build_denoiser(args)
        self.assertEqual(model.net.fusion_mode, "joint")

        model.train()
        latent = torch.randn(2, 4, 32, 32)
        dino = torch.randn(2, 768, 16, 16)
        labels = torch.tensor([3, 7], dtype=torch.long)
        loss = model({"latent": latent, "dino": dino}, labels)
        self.assertEqual(loss.ndim, 0)  # scalar
        self.assertTrue(torch.isfinite(loss))

        model.eval()
        out = model.generate(labels)
        self.assertEqual(set(out), {"latent", "dino"})
        self.assertEqual(out["latent"].shape, (2, 4, 32, 32))
        self.assertEqual(out["dino"].shape, (2, 768, 16, 16))


class FusionArgDefaultTests(unittest.TestCase):
    def test_fusion_arg_defaults_pairwise(self):
        args = get_args_parser().parse_args([])
        self.assertEqual(args.fusion, "pairwise")

        # A Denoiser built from these args reflects the pairwise default.
        args.streams = _small_specs()
        args.latent_size = 8
        args.class_num = 8
        model = _build_denoiser(args)
        self.assertEqual(model.net.fusion_mode, "pairwise")
        self.assertFalse(hasattr(model.net, "joint_blocks"))


class JointSmokeTrainTests(unittest.TestCase):
    def test_joint_smoke_train_loss_decreases(self):
        torch.manual_seed(0)
        args = _pipeline_args(_small_specs(), fusion="joint", latent_size=8, class_num=4)
        model = _build_denoiser(args)
        model.train()

        # One fixed tiny batch the model should learn to denoise.
        latent = torch.randn(4, 4, 8, 8)
        dino = torch.randn(4, 12, 4, 4)
        labels = torch.randint(0, 4, (4,))
        batch = {"latent": latent, "dino": dino}

        opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
        losses = []
        for _ in range(60):
            opt.zero_grad()
            loss = model(batch, labels)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        self.assertTrue(all(math.isfinite(v) for v in losses))
        initial = sum(losses[:5]) / 5
        final = sum(losses[-5:]) / 5
        self.assertLess(
            final, 0.8 * initial,
            f"joint smoke-train loss did not decrease enough: "
            f"initial={initial:.4f}, final={final:.4f}",
        )


class JointLabelDropMaskPathTests(unittest.TestCase):
    def test_joint_label_drop_activates_mask_path(self):
        torch.manual_seed(0)
        args = _pipeline_args(
            _small_specs(), fusion="joint", latent_size=8, class_num=4,
            label_drop_prob=1.0,  # forces the structural uncond mask every step
        )
        model = _build_denoiser(args)
        model.train()  # forward() consults drop_labels_for_cfg in training mode

        latent = torch.randn(2, 4, 8, 8)
        dino = torch.randn(2, 12, 4, 4)
        labels = torch.randint(0, 4, (2,))
        loss = model({"latent": latent, "dino": dino}, labels)
        loss.backward()  # full joint additive-bias mask path, fwd + bwd

        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        self.assertTrue(all(g is not None for g in grads))


if __name__ == "__main__":
    unittest.main()
