"""Stage 3a tests: V-Co-style joint attention as an opt-in fusion mode.

Covers joint forward shapes, one-way structural masking, the static-graph
property (every parameter trains regardless of mask), the zeros-bias no-op, and
a regression guard that the default pairwise path is untouched.
"""
import unittest

import torch
import torch.nn as nn

from JiT.model_jit import JiTMultiStream, StreamSpec


def _small_specs():
    return [
        StreamSpec(
            name="latent",
            role="image_side",
            feature_channels=4,
            feature_spatial=8,
            patch_size=2,
            tokenizer="latent",
            bottleneck_dim=16,
        ),
        StreamSpec(
            name="dino",
            role="semantic",
            feature_channels=12,
            feature_spatial=4,
            patch_size=1,
            tokenizer="linear",
        ),
    ]


def _make_small_multistream(
    fusion_mode="joint", depth=4, hidden_size=64, num_heads=4, num_classes=8
):
    return JiTMultiStream(
        specs=_small_specs(),
        input_size=8,
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads,
        num_classes=num_classes,
        in_context_len=4,
        in_context_start=0,
        cross_every=2,
        cross_start=2,
        fusion_mode=fusion_mode,
    )


@torch.no_grad()
def _break_zero_init(model: JiTMultiStream):
    """Randomize the adaLN modulations and final-layer linears.

    initialize_weights() zeros these so the network starts as an identity;
    probing cross-stream signal needs non-zero conditioning. Borrowed from
    tests/test_multistream_model.py, extended to also reach the per-stream adaLN
    nn.Sequentials that JointStreamBlock keeps inside a ModuleDict.
    """
    for module in model.modules():
        adaln = getattr(module, "adaLN_modulation", None)
        if isinstance(adaln, nn.Sequential):
            nn.init.normal_(adaln[-1].weight, std=0.1)
            nn.init.normal_(adaln[-1].bias, std=0.1)
        elif isinstance(adaln, nn.ModuleDict):
            for seq in adaln.values():
                nn.init.normal_(seq[-1].weight, std=0.1)
                nn.init.normal_(seq[-1].bias, std=0.1)
    for layer in model.final_layers.values():
        nn.init.normal_(layer.linear.weight, std=0.02)
        nn.init.normal_(layer.linear.bias, std=0.02)


def _inputs(batch=2, seed=1):
    torch.manual_seed(seed)
    return {
        "latent": torch.randn(batch, 4, 8, 8),
        "dino": torch.randn(batch, 12, 4, 4),
    }, torch.rand(batch), torch.tensor([1, 2], dtype=torch.long)[:batch]


class JointForwardShapeTests(unittest.TestCase):
    def test_joint_forward_shapes(self):
        torch.manual_seed(0)
        model = _make_small_multistream(fusion_mode="joint").eval()
        streams, t, y = _inputs()
        with torch.no_grad():
            out = model(streams, t, y)
        self.assertEqual(set(out), {"latent", "dino"})
        self.assertEqual(out["latent"].shape, streams["latent"].shape)
        self.assertEqual(out["dino"].shape, streams["dino"].shape)


class JointMaskTests(unittest.TestCase):
    def test_joint_mask_blocks_semantic_to_image_oneway(self):
        torch.manual_seed(0)
        model = _make_small_multistream(fusion_mode="joint")
        _break_zero_init(model)
        model.eval()

        streams, t, y = _inputs()
        latent, dino = streams["latent"], streams["dino"]
        torch.manual_seed(7)
        dino_perturbed = dino + torch.randn_like(dino)
        latent_perturbed = latent + torch.randn_like(latent)

        mask = {"dino": {"latent": True}}  # block semantic -> image-side
        with torch.no_grad():
            base = model({"latent": latent, "dino": dino}, t, y, mask=mask)
            # Image-side (latent) is masked from semantic keys: perturbing the
            # dino input must leave the latent output bit-for-bit unchanged.
            dino_changed = model(
                {"latent": latent, "dino": dino_perturbed}, t, y, mask=mask
            )
            # One-way mask: semantic (dino) can still read image-side, so
            # perturbing latent must change the dino output.
            latent_changed = model(
                {"latent": latent_perturbed, "dino": dino}, t, y, mask=mask
            )

        torch.testing.assert_close(
            dino_changed["latent"], base["latent"], atol=1e-6, rtol=1e-6
        )
        self.assertFalse(
            torch.allclose(latent_changed["dino"], base["dino"]),
            "semantic stream should still read the (unmasked) image-side stream",
        )

        # Sanity: with no mask the image-side DOES read semantic.
        with torch.no_grad():
            base_nomask = model({"latent": latent, "dino": dino}, t, y, mask=None)
            dino_changed_nomask = model(
                {"latent": latent, "dino": dino_perturbed}, t, y, mask=None
            )
        self.assertFalse(
            torch.allclose(dino_changed_nomask["latent"], base_nomask["latent"]),
            "unmasked image-side should read the semantic stream",
        )


class JointStaticGraphTests(unittest.TestCase):
    def test_joint_static_graph_all_params_grad(self):
        torch.manual_seed(0)
        model = _make_small_multistream(fusion_mode="joint")
        streams, t, y = _inputs()
        for mask in (None, {"dino": {"latent": True}}):
            for p in model.parameters():
                p.grad = None
            out = model(streams, t, y, mask=mask)
            loss = out["latent"].sum() + out["dino"].sum()
            loss.backward()
            missing = [
                name
                for name, p in model.named_parameters()
                if p.requires_grad and p.grad is None
            ]
            self.assertEqual(
                missing, [], f"params without grad (mask={mask}): {missing}"
            )


class JointZeroBiasTests(unittest.TestCase):
    def test_joint_zero_bias_equals_no_block(self):
        torch.manual_seed(0)
        model = _make_small_multistream(fusion_mode="joint")
        _break_zero_init(model)  # so outputs are not trivially zero
        model.eval()
        streams, t, y = _inputs()

        with torch.no_grad():
            out_none = model(streams, t, y, mask=None)
            out_empty = model(streams, t, y, mask={})
            out_all_false = model(streams, t, y, mask={"dino": {"latent": False}})

        # All three resolve to the internal zeros bias, which is a numeric no-op.
        for key in out_none:
            torch.testing.assert_close(out_empty[key], out_none[key], atol=0, rtol=0)
            torch.testing.assert_close(
                out_all_false[key], out_none[key], atol=0, rtol=0
            )


class PairwiseModeRegressionTests(unittest.TestCase):
    def test_pairwise_mode_unchanged(self):
        torch.manual_seed(0)
        model = _make_small_multistream(fusion_mode="pairwise")  # the default
        self.assertEqual(model.fusion_mode, "pairwise")
        self.assertTrue(hasattr(model, "stream_blocks"))
        self.assertTrue(hasattr(model, "cross_fusion"))
        self.assertFalse(hasattr(model, "joint_blocks"))

        # Default constructor (no fusion_mode kwarg) is also pairwise.
        default_model = JiTMultiStream(
            specs=_small_specs(),
            input_size=8,
            hidden_size=64,
            depth=4,
            num_heads=4,
            num_classes=8,
            in_context_len=4,
            in_context_start=0,
            cross_every=2,
            cross_start=2,
        )
        self.assertEqual(default_model.fusion_mode, "pairwise")
        self.assertFalse(hasattr(default_model, "joint_blocks"))

        streams, t, y = _inputs()
        model.eval()
        with torch.no_grad():
            out = model(streams, t, y)
        self.assertEqual(out["latent"].shape, streams["latent"].shape)
        self.assertEqual(out["dino"].shape, streams["dino"].shape)


class JointConstructorValidationTests(unittest.TestCase):
    def test_invalid_fusion_mode_raises(self):
        with self.assertRaises(ValueError):
            _make_small_multistream(fusion_mode="bogus")

    def test_joint_requires_prefix_from_block_zero(self):
        with self.assertRaises(ValueError):
            JiTMultiStream(
                specs=_small_specs(),
                input_size=8,
                hidden_size=64,
                depth=4,
                num_heads=4,
                num_classes=8,
                in_context_len=4,
                in_context_start=2,  # prefix not present from block 0
                fusion_mode="joint",
            )


if __name__ == "__main__":
    unittest.main()
