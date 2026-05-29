import unittest

import torch
import torch.nn as nn

from JiT.model_jit import (
    CrossFusionBlock,
    JiT_Dual_B_2_4C_896,
    JiTMultiStream,
    PairwiseCrossFusion,
    StreamSpec,
    make_jit_dual_b_2_4c_896_multistream,
    remap_dual_to_multistream,
)
from JiT.util.model_util import VisionRotaryEmbeddingFast


@torch.no_grad()
def _break_zero_init(model: JiTMultiStream):
    """Randomize adaLN modulations and final-layer linears.

    initialize_weights() zeros these so the network starts as an identity.
    Tests that probe conditioning need non-zero values.
    """
    for module in model.modules():
        adaln = getattr(module, "adaLN_modulation", None)
        if isinstance(adaln, nn.Sequential):
            nn.init.normal_(adaln[-1].weight, std=0.1)
            nn.init.normal_(adaln[-1].bias, std=0.1)
    for layer in model.final_layers.values():
        nn.init.normal_(layer.linear.weight, std=0.02)
        nn.init.normal_(layer.linear.bias, std=0.02)


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


def _make_small_multistream(num_classes=8, depth=4, hidden_size=64, num_heads=4):
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
    )


def _sync_cross_fusion_into_pairwise(src_block: CrossFusionBlock, dst: PairwiseCrossFusion):
    dst.directions["dino->latent"].load_state_dict(
        src_block.latent_from_dino.state_dict()
    )
    dst.directions["latent->dino"].load_state_dict(
        src_block.dino_from_latent.state_dict()
    )


class PairwiseCrossFusionParityTests(unittest.TestCase):
    def test_pairwise_fusion_n2_matches_cross_fusion_block(self):
        torch.manual_seed(0)
        hidden_size, num_heads, num_patches, batch = 128, 4, 16, 2

        block = CrossFusionBlock(hidden_size=hidden_size, num_heads=num_heads).eval()
        pairwise = PairwiseCrossFusion(
            specs=_small_specs(),
            hidden_size=hidden_size,
            num_heads=num_heads,
        ).eval()
        _sync_cross_fusion_into_pairwise(block, pairwise)

        feat_rope = VisionRotaryEmbeddingFast(
            dim=(hidden_size // num_heads) // 2,
            pt_seq_len=int(num_patches ** 0.5),
            num_cls_token=0,
        )

        torch.manual_seed(1)
        latent = torch.randn(batch, num_patches, hidden_size)
        dino = torch.randn(batch, num_patches, hidden_size)
        c = torch.randn(batch, hidden_size)
        dino_c = torch.randn(batch, hidden_size)

        for mask_dino_to_latent in (False, True):
            with torch.no_grad():
                latent_ref, dino_ref = block(
                    latent,
                    dino,
                    c,
                    feat_rope=feat_rope,
                    latent_num_patches=num_patches,
                    dino_num_patches=num_patches,
                    dino_c=dino_c,
                    mask_dino_to_latent=mask_dino_to_latent,
                )
                mask = {"dino": {"latent": True}} if mask_dino_to_latent else None
                outputs = pairwise(
                    {"latent": latent, "dino": dino},
                    {"latent": c, "dino": dino_c},
                    feat_rope=feat_rope,
                    num_patches={"latent": num_patches, "dino": num_patches},
                    mask=mask,
                )

            torch.testing.assert_close(
                outputs["latent"], latent_ref, atol=1e-6, rtol=1e-6
            )
            torch.testing.assert_close(
                outputs["dino"], dino_ref, atol=1e-6, rtol=1e-6
            )


class JiTMultiStreamParityTests(unittest.TestCase):
    def test_jit_multistream_n2_matches_dual_stream(self):
        torch.manual_seed(0)
        dual = JiT_Dual_B_2_4C_896(num_classes=10).eval()
        multi = make_jit_dual_b_2_4c_896_multistream(num_classes=10).eval()

        remapped = remap_dual_to_multistream(dual.state_dict())
        incompat = multi.load_state_dict(remapped, strict=False)
        self.assertEqual(list(incompat.missing_keys), [])
        self.assertEqual(list(incompat.unexpected_keys), [])

        torch.manual_seed(1)
        latent = torch.randn(2, 4, 32, 32)
        dino = torch.randn(2, 768, 16, 16)
        t = torch.rand(2)
        y = torch.tensor([3, 7], dtype=torch.long)

        for mask_dino_to_latent in (False, True):
            with torch.no_grad():
                latent_ref, dino_ref = dual(
                    latent, dino, t, y, mask_dino_to_latent=mask_dino_to_latent
                )
                mask = {"dino": {"latent": True}} if mask_dino_to_latent else None
                outputs = multi(
                    {"latent": latent, "dino": dino},
                    t,
                    y,
                    mask=mask,
                )
            torch.testing.assert_close(
                outputs["latent"], latent_ref, atol=1e-5, rtol=1e-5
            )
            torch.testing.assert_close(
                outputs["dino"], dino_ref, atol=1e-5, rtol=1e-5
            )


def _run_forward_backward(model, *, mask=None):
    torch.manual_seed(2)
    batch = 2
    latent = torch.randn(batch, 4, 8, 8)
    dino = torch.randn(batch, 12, 4, 4)
    t = torch.rand(batch)
    y = torch.zeros(batch, dtype=torch.long)
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None
    outputs = model({"latent": latent, "dino": dino}, t, y, mask=mask)
    loss = outputs["latent"].sum() + outputs["dino"].sum()
    loss.backward()


class JiTMultiStreamGradientTests(unittest.TestCase):
    def test_jit_multistream_static_graph(self):
        torch.manual_seed(0)
        model = _make_small_multistream()
        for mask in (None, {"dino": {"latent": True}}):
            _run_forward_backward(model, mask=mask)
            missing = [
                name
                for name, p in model.named_parameters()
                if p.requires_grad and p.grad is None
            ]
            self.assertEqual(
                missing,
                [],
                f"Parameters without gradients (mask={mask}): {missing}",
            )

    def test_pairwise_fusion_gate_is_multiplicative(self):
        torch.manual_seed(0)
        model = _make_small_multistream()
        for mask in (None, {"dino": {"latent": True}}):
            _run_forward_backward(model, mask=mask)
            cross_fusion_params = [
                (name, p)
                for name, p in model.named_parameters()
                if name.startswith("cross_fusion.")
            ]
            self.assertTrue(cross_fusion_params, "expected cross-fusion params")
            for name, p in cross_fusion_params:
                self.assertIsNotNone(
                    p.grad,
                    f"missing grad for {name} (mask={mask})",
                )


class JiTMultiStreamDinoTimeTests(unittest.TestCase):
    def test_jit_multistream_dino_time(self):
        torch.manual_seed(0)
        model = _make_small_multistream()
        _break_zero_init(model)
        model.eval()

        torch.manual_seed(1)
        latent = torch.randn(2, 4, 8, 8)
        dino = torch.randn(2, 12, 4, 4)
        t = torch.rand(2)
        y = torch.tensor([1, 2], dtype=torch.long)

        with torch.no_grad():
            out_none = model({"latent": latent, "dino": dino}, t, y)
            out_uniform = model(
                {"latent": latent, "dino": dino},
                t,
                y,
                stream_t={"latent": t, "dino": t},
            )
            t_dino = t + 0.5
            out_split = model(
                {"latent": latent, "dino": dino},
                t,
                y,
                stream_t={"latent": t, "dino": t_dino},
            )

        torch.testing.assert_close(out_none["latent"], out_uniform["latent"])
        torch.testing.assert_close(out_none["dino"], out_uniform["dino"])

        self.assertFalse(torch.allclose(out_none["dino"], out_split["dino"]))
        self.assertFalse(torch.allclose(out_none["latent"], out_split["latent"]))


if __name__ == "__main__":
    unittest.main()
