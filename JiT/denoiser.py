import argparse
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from JiT.model_jit import JiT_models, StreamSpec


class Denoiser(nn.Module):
    """N-stream rectified-flow denoiser driven by `args.streams: list[StreamSpec]`.

    Streams are dict-keyed throughout. Exactly one stream must have
    role == "image_side"; semantic streams (DINO, EVA-02, …) co-denoise alongside
    it. The structural CFG mask gates semantic→image-side cross-attention to zero
    on the unconditional pass, matching the dual-stream behavior at N=2.
    """

    def __init__(
        self,
        args: argparse.Namespace,
    ) -> None:
        super().__init__()
        specs = getattr(args, "streams", None)
        if not specs:
            raise ValueError(
                "Denoiser requires args.streams (a non-empty list of StreamSpec)."
            )
        specs = tuple(specs)

        image_side = [s for s in specs if s.role == "image_side"]
        if len(image_side) != 1:
            raise ValueError(
                "Denoiser requires exactly one stream with role == 'image_side'; "
                f"got {len(image_side)} (names={[s.name for s in image_side]})."
            )

        self.specs: Tuple[StreamSpec, ...] = specs
        self.specs_by_name: Dict[str, StreamSpec] = {s.name: s for s in specs}
        self.stream_names: List[str] = [s.name for s in specs]
        self.image_side_names: List[str] = [s.name for s in specs if s.role == "image_side"]
        self.semantic_names: List[str] = [s.name for s in specs if s.role == "semantic"]
        self.image_side_name: str = self.image_side_names[0]

        self.net: nn.Module = JiT_models[args.model](
            specs=list(specs),
            num_classes=args.class_num,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
            input_size=args.latent_size,
        )

        self.latent_size: int = args.latent_size
        self.num_classes: int = args.class_num
        self.label_drop_prob: float = args.label_drop_prob
        self.P_mean: float = args.P_mean
        self.P_std: float = args.P_std
        self.t_eps: float = args.t_eps
        self.inference_t_eps: float = getattr(
            args, "inference_t_eps", min(self.t_eps, 1e-5)
        )
        self.noise_scale: float = args.noise_scale

        # EMA
        self.ema_decay1: float = args.ema_decay1
        self.ema_decay2: float = args.ema_decay2
        self.ema_params1: Optional[List[torch.Tensor]] = None
        self.ema_params2: Optional[List[torch.Tensor]] = None

        # generation hyper params
        self.method: str = args.sampling_method
        self.steps: int = args.num_sampling_steps
        self.cfg_scale: float = args.cfg_scale if hasattr(args, "cfg_scale") else args.cfg
        self.cfg_interval: Tuple[float, float] = (
            args.interval_min,
            args.interval_max,
        )

    # ----- structural mask ---------------------------------------------------

    def _structural_uncond_mask(self) -> Dict[str, Dict[str, bool]]:
        # Block every semantic→image-side direction on the unconditional pass.
        return {
            sem: {img: True for img in self.image_side_names}
            for sem in self.semantic_names
        }

    # ----- label dropout -----------------------------------------------------

    def drop_labels(self, labels: torch.Tensor) -> torch.Tensor:
        labels_dropped, _ = self.drop_labels_for_cfg(labels)
        return labels_dropped

    @torch.compiler.disable
    def drop_labels_for_cfg(
        self, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Dict[str, bool]]]]:
        if self.label_drop_prob <= 0.0:
            return labels, None
        if self.label_drop_prob >= 1.0:
            return torch.full_like(labels, self.num_classes), self._structural_uncond_mask()

        # The structural mask is a Python branch, so sample it on CPU to avoid a
        # CUDA scalar sync on every training microbatch.
        drop = bool(torch.rand(()).item() < self.label_drop_prob)
        if not drop:
            return labels, None
        return torch.full_like(labels, self.num_classes), self._structural_uncond_mask()

    # ----- time sampling -----------------------------------------------------

    def sample_t(self, n: int, device: Optional[torch.device] = None) -> torch.Tensor:
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def stream_time(self, spec: StreamSpec, t: torch.Tensor) -> torch.Tensor:
        if spec.time_shift == 0.0:
            return t
        eps = torch.finfo(t.dtype).eps
        shifted = torch.sigmoid(torch.logit(t.clamp(eps, 1.0 - eps)) + spec.time_shift)
        shifted = torch.where(t <= 0.0, torch.zeros_like(shifted), shifted)
        shifted = torch.where(t >= 1.0, torch.ones_like(shifted), shifted)
        return shifted

    def _per_stream_time(self, t: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            name: self.stream_time(self.specs_by_name[name], t)
            for name in self.stream_names
        }

    @staticmethod
    def _batch_time(
        value: float, batch_size: int, device: torch.device, ref: torch.Tensor
    ) -> torch.Tensor:
        return torch.full((batch_size,), value, device=device).view(
            -1, *([1] * (ref.ndim - 1))
        )

    # ----- network plumbing --------------------------------------------------

    def _net_forward(
        self,
        z_dict: Dict[str, torch.Tensor],
        t: torch.Tensor,
        labels: torch.Tensor,
        stream_t_dict: Optional[Dict[str, torch.Tensor]] = None,
        mask: Optional[Dict[str, Dict[str, bool]]] = None,
    ) -> Dict[str, torch.Tensor]:
        stream_t_flat = None
        if stream_t_dict is not None:
            stream_t_flat = {name: s.flatten() for name, s in stream_t_dict.items()}
        return self.net(
            z_dict,
            t.flatten(),
            labels,
            stream_t=stream_t_flat,
            mask=mask,
        )

    # ----- training forward --------------------------------------------------

    def forward(
        self,
        streams: Dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        if self.training:
            labels_dropped, mask = self.drop_labels_for_cfg(labels)
        else:
            labels_dropped = labels
            mask = None

        ref = streams[self.image_side_name]
        batch_size = ref.size(0)
        device = ref.device

        t = self.sample_t(batch_size, device=device).view(
            -1, *([1] * (ref.ndim - 1))
        )

        stream_t_dict: Dict[str, torch.Tensor] = {}
        z_dict: Dict[str, torch.Tensor] = {}
        v_dict: Dict[str, torch.Tensor] = {}
        for name in self.stream_names:
            x = streams[name]
            spec = self.specs_by_name[name]
            stream_t_x = self.stream_time(spec, t)
            noise = torch.randn_like(x) * self.noise_scale
            z = stream_t_x * x + (1 - stream_t_x) * noise
            v = (x - z) / (1 - stream_t_x).clamp_min(self.t_eps)
            stream_t_dict[name] = stream_t_x
            z_dict[name] = z
            v_dict[name] = v

        pred_dict = self._net_forward(
            z_dict, t, labels_dropped, stream_t_dict, mask
        )

        loss = torch.zeros((), device=device, dtype=ref.dtype)
        for name in self.stream_names:
            spec = self.specs_by_name[name]
            pred = pred_dict[name]
            z = z_dict[name]
            stream_t_x = stream_t_dict[name]
            v_pred = (pred - z) / (1 - stream_t_x).clamp_min(self.t_eps)
            v = v_dict[name]
            stream_loss = ((v - v_pred) ** 2).mean()
            loss = loss + spec.loss_weight * stream_loss
        return loss

    # ----- generation --------------------------------------------------------

    @torch.no_grad()
    def generate(self, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        device = labels.device
        bsz = labels.size(0)

        z_dict: Dict[str, torch.Tensor] = {}
        for name in self.stream_names:
            spec = self.specs_by_name[name]
            z_dict[name] = self.noise_scale * torch.randn(
                bsz,
                spec.feature_channels,
                spec.feature_spatial,
                spec.feature_spatial,
                device=device,
            )

        timesteps = torch.linspace(0.0, 1.0, self.steps + 1, device=device)

        stepper: Callable[..., Dict[str, torch.Tensor]]
        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError

        ref = z_dict[self.image_side_name]

        for i in range(self.steps - 1):
            t = self._batch_time(float(timesteps[i]), bsz, device, ref)
            t_next = self._batch_time(float(timesteps[i + 1]), bsz, device, ref)
            stream_t_dict = self._per_stream_time(t)
            stream_t_next_dict = self._per_stream_time(t_next)
            z_dict = stepper(
                z_dict, t, t_next, labels, stream_t_dict, stream_t_next_dict
            )

        # Land on the model's guided x-prediction directly so the final step
        # is not shortened by the training-time velocity clamp near t=1.
        z_dict = self._forward_sample_xpred(
            z_dict,
            self._batch_time(float(timesteps[-2]), bsz, device, ref),
            labels,
        )
        return z_dict

    @torch.no_grad()
    def _cfg_scale_interval(self, t: torch.Tensor) -> torch.Tensor:
        low, high = self.cfg_interval
        interval_mask = (t < high) & ((low == 0) | (t > low))
        return torch.where(interval_mask, self.cfg_scale, 1.0)

    @torch.no_grad()
    def _forward_sample_xpred(
        self,
        z_dict: Dict[str, torch.Tensor],
        t: torch.Tensor,
        labels: torch.Tensor,
        stream_t_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        if stream_t_dict is None:
            stream_t_dict = self._per_stream_time(t)

        cond = self._net_forward(z_dict, t, labels, stream_t_dict)
        uncond = self._net_forward(
            z_dict,
            t,
            torch.full_like(labels, self.num_classes),
            stream_t_dict,
            mask=self._structural_uncond_mask(),
        )
        cfg = self._cfg_scale_interval(t)
        return {
            name: uncond[name] + cfg * (cond[name] - uncond[name])
            for name in self.stream_names
        }

    @torch.no_grad()
    def _forward_sample(
        self,
        z_dict: Dict[str, torch.Tensor],
        t: torch.Tensor,
        labels: torch.Tensor,
        stream_t_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        if stream_t_dict is None:
            stream_t_dict = self._per_stream_time(t)
        pred = self._forward_sample_xpred(z_dict, t, labels, stream_t_dict)
        v_dict: Dict[str, torch.Tensor] = {}
        for name in self.stream_names:
            stream_t_x = stream_t_dict[name]
            denom = (1.0 - stream_t_x).clamp_min(self.inference_t_eps)
            v_dict[name] = (pred[name] - z_dict[name]) / denom
        return v_dict

    @torch.no_grad()
    def _euler_step(
        self,
        z_dict: Dict[str, torch.Tensor],
        t: torch.Tensor,
        t_next: torch.Tensor,
        labels: torch.Tensor,
        stream_t_dict: Optional[Dict[str, torch.Tensor]] = None,
        stream_t_next_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        if stream_t_dict is None:
            stream_t_dict = self._per_stream_time(t)
        if stream_t_next_dict is None:
            stream_t_next_dict = self._per_stream_time(t_next)
        v_dict = self._forward_sample(z_dict, t, labels, stream_t_dict)
        z_next: Dict[str, torch.Tensor] = {}
        for name in self.stream_names:
            dt = stream_t_next_dict[name] - stream_t_dict[name]
            z_next[name] = z_dict[name] + dt * v_dict[name]
        return z_next

    @torch.no_grad()
    def _heun_step(
        self,
        z_dict: Dict[str, torch.Tensor],
        t: torch.Tensor,
        t_next: torch.Tensor,
        labels: torch.Tensor,
        stream_t_dict: Optional[Dict[str, torch.Tensor]] = None,
        stream_t_next_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        if stream_t_dict is None:
            stream_t_dict = self._per_stream_time(t)
        if stream_t_next_dict is None:
            stream_t_next_dict = self._per_stream_time(t_next)

        v_t = self._forward_sample(z_dict, t, labels, stream_t_dict)
        z_euler: Dict[str, torch.Tensor] = {}
        for name in self.stream_names:
            dt = stream_t_next_dict[name] - stream_t_dict[name]
            z_euler[name] = z_dict[name] + dt * v_t[name]

        v_next = self._forward_sample(z_euler, t_next, labels, stream_t_next_dict)
        z_next: Dict[str, torch.Tensor] = {}
        for name in self.stream_names:
            dt = stream_t_next_dict[name] - stream_t_dict[name]
            v_avg = 0.5 * (v_t[name] + v_next[name])
            z_next[name] = z_dict[name] + dt * v_avg
        return z_next

    # ----- EMA ---------------------------------------------------------------

    @torch.no_grad()
    def update_ema(self) -> None:
        assert self.ema_params1 is not None and self.ema_params2 is not None
        params = [param.detach() for param in self.parameters()]
        ema_params1 = [param.detach() for param in self.ema_params1]
        ema_params2 = [param.detach() for param in self.ema_params2]
        torch._foreach_mul_(ema_params1, self.ema_decay1)
        torch._foreach_add_(ema_params1, params, alpha=1 - self.ema_decay1)
        torch._foreach_mul_(ema_params2, self.ema_decay2)
        torch._foreach_add_(ema_params2, params, alpha=1 - self.ema_decay2)
