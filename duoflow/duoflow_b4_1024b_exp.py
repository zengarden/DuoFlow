# usage: python hf_b4_1024b_nojvp_1mm_exp_5xloss_cosine.py mode=train
import os
from dataclasses import dataclass, field

import torch
from tinyexp import store_and_run_exp

from tiny_meanflow.official_mf_b4_exp import MeanFlowB4Exp as MeanFlowB4ExpBase

# torch.multiprocessing.set_start_method("spawn", force=True)


@dataclass(repr=False)
class DuoFlowB4Exp(MeanFlowB4ExpBase):
    output_root: str = "./output/duo_flow"

    @dataclass
    class WandbCfg(MeanFlowB4ExpBase.WandbCfg):
        project: str = "duoflow_release"

    wandb_cfg: WandbCfg = field(default_factory=WandbCfg)

    @dataclass
    class DataloaderCfg(MeanFlowB4ExpBase.DataloaderCfg):
        train_batch_size_per_device: int = 128

    dataloader_cfg: DataloaderCfg = field(default_factory=DataloaderCfg)

    @dataclass
    class OptimizerCfg(MeanFlowB4ExpBase.OptimizerCfg):
        warmup_epochs: int = 4

        def get_ema_decay_rate(self, global_step, total_step):
            return 0.9999

    optimizer_cfg: OptimizerCfg = field(default_factory=OptimizerCfg)

    @dataclass
    class TrainCfg(MeanFlowB4ExpBase.TrainCfg):
        epoch: int = 84 # add 4 warmup epochs

    train_cfg: TrainCfg = field(default_factory=TrainCfg)

    @dataclass
    class LossCfg(MeanFlowB4ExpBase.LossCfg):
        time_mu: float = -0.4
        time_sigma: float = 1.0
        cfg_omega: float = 3.0
        cfg_max_t: float = 1.0

        def build_loss(self):
            loss_fn = UVLoss(
                time_mu=self.time_mu,
                time_sigma=self.time_sigma,
                cfg_omega=self.cfg_omega,
                cfg_max_t=self.cfg_max_t,
            )
            return loss_fn

    loss_cfg: LossCfg = field(default_factory=LossCfg)


class UVLoss:
    def __init__(
        self,
        # New parameters
        time_mu=-0.4,  # Mean parameter for logit_normal distribution
        time_sigma=1.0,  # Std parameter for logit_normal distribution
        # CFG related params
        cfg_omega=3.0,
        cfg_max_t=1.0,  # Maximum CFG trigger time
        num_classes=1000,
    ):
        # Time sampling config
        self.time_sampler = "logit_normal"
        self.time_mu = time_mu
        self.time_sigma = time_sigma

        # CFG config
        self.cfg_omega = cfg_omega
        self.cfg_max_t = cfg_max_t
        self.num_classes = num_classes

    def _compute_target_kappa(
        self,
        *,
        v_cond: torch.Tensor,
        cfg_target: torch.Tensor,
        cfg_time_mask: torch.Tensor,
    ) -> torch.Tensor:
        v_f = v_cond.detach().float()
        t_f = cfg_target.detach().float()

        mask = cfg_time_mask.view(-1)
        mask_f = mask.to(dtype=v_f.dtype)
        den = mask_f.sum()

        dot = (v_f * t_f).mean(dim=(1, 2, 3))
        v_pow = v_f.square().mean(dim=(1, 2, 3))
        t_pow = t_f.square().mean(dim=(1, 2, 3))

        dot_m = (dot * mask_f).sum() / den.clamp_min(1.0)
        v_m = (v_pow * mask_f).sum() / den.clamp_min(1.0)
        t_m = (t_pow * mask_f).sum() / den.clamp_min(1.0)

        dot_b = torch.where(den > 0, dot_m, dot.mean())
        v_b = torch.where(den > 0, v_m, v_pow.mean())
        t_b = torch.where(den > 0, t_m, t_pow.mean())

        cosine = dot_b / (v_b * t_b).clamp_min(1e-6).sqrt()
        # cosine = cosine.clamp(min=0.0, max=1.0)
        kappa = 0.5 + 0.4999 * cosine
        return kappa.view(1, 1, 1, 1)

    def sample_time_steps(self, batch_size, device, dtype):
        """Sample time steps (r, t) according to the configured sampler"""
        # Step1: Sample two time points
        if self.time_sampler == "uniform":
            time_samples = torch.rand(batch_size, 2, device=device)
        elif self.time_sampler == "logit_normal":
            normal_samples = torch.randn(batch_size, 2, device=device)
            normal_samples = normal_samples * self.time_sigma + self.time_mu
            time_samples = torch.sigmoid(normal_samples)
        else:
            raise ValueError(f"Unknown time sampler: {self.time_sampler}")

        # Step2: Ensure t > r by sorting
        sorted_samples, _ = torch.sort(time_samples, dim=1)
        r, t = sorted_samples[:, 0], sorted_samples[:, 1]

        # Pick a base finite-diff step cap (dt_cap) from dtype.
        # We later build feasible one-sided radii:
        #   h_+ = min(dt_cap, 1 - t) and h_- = min(dt_cap, t - r),
        # then use stochastic signed one-sided differences to estimate u'(t).
        eps = torch.finfo(dtype).eps
        sqrt_eps = eps**0.5
        cbrt_eps = eps ** (1.0 / 3.0)
        dt_unit = sqrt_eps if dtype in (torch.float16, torch.bfloat16) else cbrt_eps
        dt_cap = dt_unit
        dt = torch.full_like(t, dt_cap)
        dt = torch.minimum(dt, torch.full_like(dt, 0.25))
        return r, t, dt

    def __call__(
        self,
        model,
        compiled_model,
        ema_model,
        images,
        accelerator,
        model_kwargs=None,
        logger=None,
        is_print_step=False,
        epoch_id=None,
    ):
        """
        Compute UV loss function
        """
        if model_kwargs is None:
            model_kwargs = {}
        else:
            model_kwargs = model_kwargs.copy()

        bs, device = images.shape[0], images.device

        # Sample time steps
        r, t, dt = self.sample_time_steps(bs, device, dtype=images.dtype)
        r, t, dt = r.view(-1, 1, 1, 1), t.view(-1, 1, 1, 1), dt.view(-1, 1, 1, 1)

        noises = torch.randn_like(images)
        y_cond, y_unc = model_kwargs["y"].clone(), torch.full_like(model_kwargs["y"], self.num_classes)

        z_t, dzdt = (1 - t) * images + t * noises, noises - images

        unconditional_mask = (torch.rand(bs, device=y_cond.device) < 0.1).view(-1, 1, 1, 1)
        y_cond[unconditional_mask.view(-1)] = self.num_classes
        z_t_batched, t_batched, r_batched, y_batched = (
            torch.cat([z_t, z_t], dim=0),
            torch.cat([t, t], dim=0),
            torch.cat([r, t], dim=0),
            torch.cat([y_cond, y_cond], dim=0),
        )
        speed_batched = compiled_model(z_t_batched, r=r_batched, t=t_batched, y=y_batched)
        u_cond, v_cond = speed_batched[:bs], speed_batched[bs:]

        # -------------------------------  Finite-diff dudt (no-jvp) --------------------------- #
        with torch.no_grad():
            # Optional: shrink dt when the z-perturbation (dt * v_cond) would be large relative to z_t.
            z_rms = z_t.float().square().mean(dim=(1, 2, 3), keepdim=True).sqrt()
            v_rms = v_cond.float().square().mean(dim=(1, 2, 3), keepdim=True).sqrt()
            rel_scale = torch.minimum(torch.ones_like(v_rms), (z_rms + 1.0) / (v_rms + 1e-6))
            dt = dt * rel_scale

            eps = torch.finfo(v_cond.dtype).eps
            h_plus = torch.minimum(dt, 1.0 - t)
            h_minus = torch.minimum(dt, t - r)
            dt_min = torch.full_like(dt, eps)

            # Stochastic signed one-sided difference with 1 extra eval:
            # Use a single feasible radius h = min(h_+, h_-) so both directions are feasible, then
            #   u'(t) ≈ (u(t + σh) - u(t)) / (σh),  σ ∈ {+1,-1} uniformly.
            # In expectation over σ, this matches the (2-extra-eval) centered difference up to O(h^2).
            h = torch.minimum(h_plus, h_minus)
            dt_is_valid = h >= dt_min

            rand = torch.rand_like(h, dtype=torch.float32)
            sigma = torch.where(rand < 0.5, torch.ones_like(h), -torch.ones_like(h))
            off = sigma * h  # signed

            off_z = off.to(dtype=z_t.dtype)
            z_shift, t_shift = z_t + v_cond * off_z, t + off
            uv_pred = compiled_model(
                torch.cat([z_shift, z_t], dim=0),
                r=torch.cat([r, t], dim=0),
                t=torch.cat([t_shift, t], dim=0),
                y=torch.cat([y_cond, y_unc], dim=0),
            )
            u_shift, v_unc = uv_pred[:bs], uv_pred[bs:]

            u0 = u_cond.float()
            off_f = off.float()
            off_safe = torch.where(
                off_f.abs() >= eps,
                off_f,
                torch.where(off_f >= 0, torch.full_like(off_f, eps), torch.full_like(off_f, -eps)),
            )
            dudt = (u_shift.float() - u0) * torch.reciprocal(off_safe)
            valid_f = dt_is_valid.to(dtype=dudt.dtype)
            correction = ((t - r).float() * dudt * valid_f).to(dtype=u_cond.dtype)

        cfg_target = self.cfg_omega * dzdt + (1 - self.cfg_omega) * v_unc.detach()
        cfg_time_mask = (t <= self.cfg_max_t) & (~unconditional_mask)
        kappa = self._compute_target_kappa(
            v_cond=v_cond,
            cfg_target=cfg_target,
            cfg_time_mask=cfg_time_mask,
        )
        v_target = kappa * v_cond.detach() + (1.0 - kappa) * cfg_target
        v_target = cfg_time_mask * v_target + dzdt * (~cfg_time_mask)
        u_target = v_target - correction.detach()

        u_loss = torch.sum(((u_cond - u_target) ** 2).reshape(bs, -1), dim=-1)
        v_loss = torch.sum(((v_cond - v_target) ** 2).reshape(bs, -1), dim=-1)

        u_loss_final = 1.0 / (u_loss.detach() + 1e-3) * u_loss
        v_loss_final = 1.0 / (v_loss.detach() + 1e-3) * v_loss

        with torch.no_grad():
            kappa_mean_local = kappa.detach().float().mean()
            one_minus_k = (1.0 - kappa_mean_local).clamp_min(1e-6)
            v_loss_weight = torch.reciprocal(one_minus_k * one_minus_k)

        loss = u_loss_final + v_loss_final * v_loss_weight

        if is_print_step:
            return loss.mean(), {
                "u_loss": u_loss.mean(),
                "v_loss": v_loss.mean(),
            }
        else:
            return loss.mean(), {}
        # return total_loss, {"u_cond": u_loss_ref, "v_cond": v_loss_ref}


if __name__ == "__main__":
    store_and_run_exp(DuoFlowB4Exp)
