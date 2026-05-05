import unittest

import torch

from JiT.model_jit import CrossFusionBlock, JiT_models


class _AddDirection(torch.nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value
        self.calls = 0

    def forward(self, x, context, c, feat_rope=None, num_patches=0, context_num_patches=0):
        del context, c, feat_rope, num_patches, context_num_patches
        self.calls += 1
        return x + self.value


class JiTDualStreamTests(unittest.TestCase):
    def _build_model(self, **overrides):
        kwargs = dict(
            input_size=8,
            dino_patches=4,
            in_context_len=4,
        )
        kwargs.update(overrides)
        return JiT_models["JiT-Dual-B/2-4C-896"](**kwargs)

    def test_dual_variant_exposes_early_context_and_periodic_fusion(self):
        model = self._build_model()

        self.assertEqual(model.in_context_start, 0)
        self.assertEqual(model.cross_fusion_layers, (4, 8))
        self.assertTrue(model.supports_dino_time)
        self.assertEqual(model.latent_in_context_posemb.shape, (1, 4, 896))
        self.assertEqual(model.dino_in_context_posemb.shape, (1, 4, 896))

    def test_active_registry_only_exposes_dual_training_model(self):
        self.assertEqual(set(JiT_models), {"JiT-Dual-B/2-4C-896"})

    def test_dual_variant_keeps_latent_and_dino_towers_separate(self):
        model = self._build_model()

        self.assertNotEqual(
            model.latent_blocks[0].attn.qkv.weight.data_ptr(),
            model.dino_blocks[0].attn.qkv.weight.data_ptr(),
        )
        self.assertNotEqual(
            model.latent_blocks[0].mlp.w12.weight.data_ptr(),
            model.dino_blocks[0].mlp.w12.weight.data_ptr(),
        )

    def test_dual_variant_forward_preserves_external_shapes(self):
        model = self._build_model()
        latent = torch.randn(1, 4, 8, 8)
        dino = torch.randn(1, 768, 4, 4)
        t = torch.tensor([1.0])
        y = torch.tensor([5], dtype=torch.long)
        dino_t = torch.tensor([0.25])

        with torch.no_grad():
            latent_out, dino_out = model(latent, dino, t, y, dino_t=dino_t)

        self.assertEqual(latent_out.shape, latent.shape)
        self.assertEqual(dino_out.shape, dino.shape)

    def test_cross_fusion_mask_skips_only_dino_to_latent_direction(self):
        block = CrossFusionBlock(hidden_size=4, num_heads=2)
        block.latent_from_dino = _AddDirection(10.0)
        block.dino_from_latent = _AddDirection(20.0)
        latent = torch.zeros(1, 2, 4)
        dino = torch.zeros(1, 2, 4)
        c = torch.zeros(1, 4)

        latent_out, dino_out = block(latent, dino, c, mask_dino_to_latent=True)

        self.assertTrue(torch.equal(latent_out, latent))
        self.assertTrue(torch.equal(dino_out, dino + 20.0))
        self.assertEqual(block.latent_from_dino.calls, 1)
        self.assertEqual(block.dino_from_latent.calls, 1)


if __name__ == "__main__":
    unittest.main()
