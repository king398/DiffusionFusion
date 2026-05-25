import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from JiT import denoiser as denoiser_module
from JiT.model_jit import StreamSpec


def _two_stream_specs(latent_size=1, dino_channels=1, dino_spatial=1):
    return [
        StreamSpec(
            name="latent",
            role="image_side",
            feature_channels=4,
            feature_spatial=latent_size,
            patch_size=1,
            tokenizer="latent",
            bottleneck_dim=4,
            time_shift=0.0,
            loss_weight=1.0,
        ),
        StreamSpec(
            name="dino",
            role="semantic",
            feature_channels=dino_channels,
            feature_spatial=dino_spatial,
            patch_size=1,
            tokenizer="linear",
            time_shift=0.0,
            loss_weight=1.0,
        ),
    ]


class _ConstantPredictor(torch.nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, streams, t, labels, stream_t=None, mask=None):
        del t, labels, stream_t, mask
        z_latent = streams["latent"]
        z_dino = streams["dino"]
        return {
            "latent": torch.full_like(z_latent, self.value),
            "dino": torch.full_like(z_dino, -self.value),
        }


class _FinalStepPredictor(torch.nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, streams, t, labels, stream_t=None, mask=None):
        del labels, stream_t, mask
        z_latent = streams["latent"]
        z_dino = streams["dino"]
        final_mask = (t > 0.97).view(-1, 1, 1, 1)
        return {
            "latent": torch.where(final_mask, torch.full_like(z_latent, self.value), z_latent),
            "dino": torch.where(final_mask, torch.full_like(z_dino, -self.value), z_dino),
        }


class _MaskRecordingPredictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, streams, t, labels, stream_t=None, mask=None):
        del t, stream_t
        z_latent = streams["latent"]
        z_dino = streams["dino"]
        self.calls.append(
            {
                "labels": labels.detach().clone(),
                "mask": mask,
            }
        )
        return {
            "latent": torch.zeros_like(z_latent),
            "dino": torch.zeros_like(z_dino),
        }


class _ConstantDinoTimePredictor(torch.nn.Module):
    supports_dino_time = True

    def __init__(self, value: float):
        super().__init__()
        self.value = value
        self.seen_t = None
        self.seen_stream_t = None

    def forward(self, streams, t, labels, stream_t=None, mask=None):
        del labels, mask
        z_latent = streams["latent"]
        z_dino = streams["dino"]
        self.seen_t = t.detach().clone()
        self.seen_stream_t = {k: v.detach().clone() for k, v in (stream_t or {}).items()}
        return {
            "latent": torch.full_like(z_latent, self.value),
            "dino": torch.full_like(z_dino, self.value),
        }


def _structural_mask():
    return {"dino": {"latent": True}}


class DenoiserSamplingTests(unittest.TestCase):
    def _build_args(self, **overrides):
        args = dict(
            model="test-model",
            latent_size=1,
            class_num=1000,
            attn_dropout=0.0,
            proj_dropout=0.0,
            label_drop_prob=0.0,
            P_mean=0.0,
            P_std=1.0,
            t_eps=0.05,
            inference_t_eps=1e-5,
            noise_scale=0.0,
            ema_decay1=0.9999,
            ema_decay2=0.9996,
            sampling_method="heun",
            num_sampling_steps=50,
            cfg=1.0,
            interval_min=0.0,
            interval_max=1.0,
        )
        args.update(overrides)
        latent_size = args.get("latent_size", 1)
        dino_channels = args.pop("dino_hidden_size", 1)
        dino_spatial = args.pop("dino_patches", 1)
        dino_time_shift = args.pop("dino_time_shift", 0.0)
        if dino_time_shift is None:
            dino_time_shift = 0.0
        loss_weights = (
            args.pop("latent_loss_weight", 1.0),
            args.pop("dino_loss_weight", 1.0),
        )
        specs = _two_stream_specs(
            latent_size=latent_size,
            dino_channels=dino_channels,
            dino_spatial=dino_spatial,
        )
        # Override dino time_shift / loss_weights via dataclass replace.
        from dataclasses import replace
        specs = [
            replace(specs[0], loss_weight=loss_weights[0]),
            replace(specs[1], time_shift=dino_time_shift, loss_weight=loss_weights[1]),
        ]
        args["streams"] = specs
        return SimpleNamespace(**args)

    def _dino_spec_time(self, model, t):
        dino_spec = model.specs_by_name["dino"]
        return model.stream_time(dino_spec, t)

    def test_default_dino_time_shift_keeps_streams_synchronized(self):
        with patch.dict(
            denoiser_module.JiT_models,
            {"test-model": lambda **_kwargs: _ConstantPredictor(0.0)},
            clear=False,
        ):
            model = denoiser_module.Denoiser(
                self._build_args(
                    latent_size=32,
                    dino_hidden_size=768,
                    dino_patches=16,
                )
            )

        self.assertEqual(model.specs_by_name["dino"].time_shift, 0.0)
        t = torch.tensor([0.25, 0.5, 0.75]).view(-1, 1, 1, 1)
        self.assertTrue(torch.equal(self._dino_spec_time(model, t), t))

    def test_none_dino_time_shift_keeps_checkpoint_args_synchronized(self):
        with patch.dict(
            denoiser_module.JiT_models,
            {"test-model": lambda **_kwargs: _ConstantPredictor(0.0)},
            clear=False,
        ):
            model = denoiser_module.Denoiser(self._build_args(dino_time_shift=None))

        self.assertEqual(model.specs_by_name["dino"].time_shift, 0.0)

    def test_dino_time_shift_is_logit_space_and_preserves_endpoints(self):
        with patch.dict(
            denoiser_module.JiT_models,
            {"test-model": lambda **_kwargs: _ConstantPredictor(0.0)},
            clear=False,
        ):
            model = denoiser_module.Denoiser(
                self._build_args(dino_time_shift=math.log(3.0))
            )

        t = torch.tensor([0.0, 0.5, 1.0]).view(-1, 1, 1, 1)
        dino_t = self._dino_spec_time(model, t)
        expected = torch.tensor([0.0, 0.75, 1.0]).view(-1, 1, 1, 1)
        self.assertTrue(torch.allclose(dino_t, expected))

    def test_forward_sample_passes_dino_time_to_dual_time_model(self):
        predictor = _ConstantDinoTimePredictor(1.0)
        with patch.dict(
            denoiser_module.JiT_models,
            {"test-model": lambda **_kwargs: predictor},
            clear=False,
        ):
            model = denoiser_module.Denoiser(self._build_args(dino_time_shift=0.0))

        z_dict = {
            "latent": torch.zeros(1, 1, 1, 1),
            "dino": torch.zeros(1, 1, 1, 1),
        }
        t = torch.full((1, 1, 1, 1), 0.25)
        stream_t_dict = {
            "latent": torch.full((1, 1, 1, 1), 0.25),
            "dino": torch.full((1, 1, 1, 1), 0.5),
        }
        labels = torch.zeros(1, dtype=torch.long)

        model._forward_sample_xpred(z_dict, t, labels, stream_t_dict)

        self.assertTrue(torch.equal(predictor.seen_t, torch.tensor([0.25])))
        self.assertTrue(torch.equal(predictor.seen_stream_t["latent"], torch.tensor([0.25])))
        self.assertTrue(torch.equal(predictor.seen_stream_t["dino"], torch.tensor([0.5])))

    def test_forward_sample_masks_dino_to_latent_only_for_unconditional_pass(self):
        predictor = _MaskRecordingPredictor()
        with patch.dict(
            denoiser_module.JiT_models,
            {"test-model": lambda **_kwargs: predictor},
            clear=False,
        ):
            model = denoiser_module.Denoiser(self._build_args(dino_time_shift=0.0))

        z_dict = {
            "latent": torch.zeros(2, 1, 1, 1),
            "dino": torch.zeros(2, 1, 1, 1),
        }
        t = torch.full((2, 1, 1, 1), 0.25)
        labels = torch.tensor([3, 7], dtype=torch.long)

        model._forward_sample_xpred(z_dict, t, labels)

        self.assertEqual(len(predictor.calls), 2)
        # conditional pass: no mask
        self.assertIsNone(predictor.calls[0]["mask"])
        self.assertTrue(torch.equal(predictor.calls[0]["labels"], labels))
        # unconditional pass: structural mask + null labels
        self.assertEqual(predictor.calls[1]["mask"], _structural_mask())
        self.assertTrue(
            torch.equal(
                predictor.calls[1]["labels"],
                torch.full_like(labels, model.num_classes),
            )
        )

    def test_training_label_drop_jointly_activates_structural_mask(self):
        predictor = _MaskRecordingPredictor()
        with patch.dict(
            denoiser_module.JiT_models,
            {"test-model": lambda **_kwargs: predictor},
            clear=False,
        ):
            model = denoiser_module.Denoiser(
                self._build_args(label_drop_prob=1.0, dino_time_shift=0.0)
            )

        model.train()
        latent = torch.zeros(2, 4, 1, 1)
        dino = torch.zeros(2, 1, 1, 1)
        labels = torch.tensor([3, 7], dtype=torch.long)

        model({"latent": latent, "dino": dino}, labels)

        self.assertEqual(len(predictor.calls), 1)
        self.assertEqual(predictor.calls[0]["mask"], _structural_mask())
        self.assertTrue(
            torch.equal(
                predictor.calls[0]["labels"],
                torch.full_like(labels, model.num_classes),
            )
        )

    def test_euler_step_uses_separate_dino_time_delta(self):
        with patch.dict(
            denoiser_module.JiT_models,
            {"test-model": lambda **_kwargs: _ConstantDinoTimePredictor(1.0)},
            clear=False,
        ):
            model = denoiser_module.Denoiser(
                self._build_args(sampling_method="euler", dino_time_shift=0.0)
            )

        z_dict = {
            "latent": torch.zeros(1, 1, 1, 1),
            "dino": torch.zeros(1, 1, 1, 1),
        }
        t = torch.full((1, 1, 1, 1), 0.0)
        t_next = torch.full((1, 1, 1, 1), 0.5)
        stream_t_dict = {
            "latent": torch.full((1, 1, 1, 1), 0.0),
            "dino": torch.full((1, 1, 1, 1), 0.0),
        }
        stream_t_next_dict = {
            "latent": torch.full((1, 1, 1, 1), 0.5),
            "dino": torch.full((1, 1, 1, 1), 0.75),
        }
        labels = torch.zeros(1, dtype=torch.long)

        z_next = model._euler_step(
            z_dict, t, t_next, labels, stream_t_dict, stream_t_next_dict
        )

        self.assertTrue(torch.allclose(z_next["latent"], torch.full_like(z_next["latent"], 0.5)))
        self.assertTrue(torch.allclose(z_next["dino"], torch.full_like(z_next["dino"], 0.75)))

    def test_forward_sample_uses_inference_t_eps_near_terminal_time(self):
        with patch.dict(
            denoiser_module.JiT_models,
            {"test-model": lambda **_kwargs: _ConstantPredictor(10.0)},
            clear=False,
        ):
            model = denoiser_module.Denoiser(self._build_args(dino_time_shift=0.0))

        z_dict = {
            "latent": torch.zeros(1, 1, 1, 1),
            "dino": torch.zeros(1, 1, 1, 1),
        }
        t = torch.full((1, 1, 1, 1), 0.98)
        labels = torch.zeros(1, dtype=torch.long)

        v_dict = model._forward_sample(z_dict, t, labels)

        self.assertAlmostEqual(v_dict["latent"].item(), 500.0, places=3)
        self.assertAlmostEqual(v_dict["dino"].item(), -500.0, places=3)

    def test_generate_uses_guided_x_prediction_for_final_step(self):
        with patch.dict(
            denoiser_module.JiT_models,
            {"test-model": lambda **_kwargs: _FinalStepPredictor(10.0)},
            clear=False,
        ):
            model = denoiser_module.Denoiser(self._build_args(dino_time_shift=0.0))

        labels = torch.zeros(1, dtype=torch.long)
        out = model.generate(labels)

        self.assertTrue(torch.allclose(out["latent"], torch.full_like(out["latent"], 10.0)))
        self.assertTrue(torch.allclose(out["dino"], torch.full_like(out["dino"], -10.0)))


if __name__ == "__main__":
    unittest.main()
