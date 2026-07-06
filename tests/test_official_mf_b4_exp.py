import logging
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tiny_meanflow.official_mf_b4_exp as mf_exp


class DummyAccelerator:
    def __init__(self):
        self.world_size = 1
        self.rank = 0
        self.device = torch.device("cpu")
        self.is_main_process = True
        self.sync_gradients = True
        self.unwrapped_module = None

    def prepare(self, module, optimizer):
        return module, optimizer

    def backward(self, loss):
        loss.backward()

    def unwrap_model(self, module):
        return self.unwrapped_module if self.unwrapped_module is not None else module

    def clip_grad_norm_(self, params, max_norm):
        torch.nn.utils.clip_grad_norm_(list(params), max_norm)

    def reduce(self, value):
        return value


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.fused_attn_enabled = True

    def forward(self, x, r=None, t=None, y=None):
        del r, t, y
        return torch.zeros_like(x) + self.weight.view(1, 1, 1, 1)

    def disable_fused_attn(self):
        self.fused_attn_enabled = False

    def enable_fused_attn(self):
        self.fused_attn_enabled = True


class DummyPosterior:
    def __init__(self, moments):
        self.moments = moments

    def sample(self):
        return self.moments


class DummyLossFn:
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
        del compiled_model, ema_model, accelerator, model_kwargs, logger, is_print_step, epoch_id
        loss = model.weight.sum() * 0 + images.sum() * 0 + torch.tensor(1.0, requires_grad=True)
        return loss, {}


class DummyLoader:
    def __init__(self, batch):
        self._batch = batch
        self.batch_size = batch[0].shape[0]

    def __iter__(self):
        return iter([self._batch])

    def __len__(self):
        return 1


@pytest.fixture
def logger():
    return logging.getLogger("tiny-meanflow-tests")


def test_loss_sample_time_steps_respects_order_and_equal_ratio():
    loss = mf_exp.Loss(ratio_r_not_equal_t=0.0)
    r, t, equal_mask = loss.sample_time_steps(batch_size=8, device=torch.device("cpu"))

    assert r.shape == (8,)
    assert t.shape == (8,)
    assert equal_mask.shape == (8,)
    assert torch.all(t >= r)
    assert torch.all(equal_mask)
    assert torch.allclose(r, t)


def test_loss_call_works_without_labels():
    loss = mf_exp.Loss()
    accelerator = DummyAccelerator()
    model = DummyModel()
    images = torch.randn(2, 4, 4, 4)

    value, stats = loss(
        model=model,
        compiled_model=model,
        ema_model=None,
        images=images,
        accelerator=accelerator,
        model_kwargs=None,
        is_print_step=False,
    )

    assert value.ndim == 0
    assert torch.isfinite(value)
    assert stats == {}


def test_optimizer_cfg_decay_rate_hits_endpoints():
    cfg = mf_exp.MeanFlowExp.OptimizerCfg(ema_decay_start=0.9, ema_decay_end=0.99)

    start = cfg.get_ema_decay_rate(global_step=0, total_step=10)
    middle = cfg.get_ema_decay_rate(global_step=5, total_step=10)
    end = cfg.get_ema_decay_rate(global_step=10, total_step=10)

    assert start == pytest.approx(0.9)
    assert end == pytest.approx(0.99)
    assert start <= middle <= end


def test_train_uses_unwrapped_model_for_eval(monkeypatch, logger, tmp_path):
    exp = mf_exp.MeanFlowExp()
    exp.wandb_cfg.enable_wandb = False
    exp.module_cfg.use_ema = False
    exp.val_cfg.use_ema_module = False
    exp.val_cfg.eval_interval = 1
    exp.val_cfg.eval_two_step = False
    exp.train_cfg.epoch = 1
    exp.train_cfg.print_freq = 100

    batch = (torch.randn(2, 4, 4, 4), torch.zeros(2, dtype=torch.long))
    dataloader = DummyLoader(batch)
    accelerator = DummyAccelerator()
    model = DummyModel()
    unwrapped = DummyModel()
    accelerator.unwrapped_module = unwrapped
    seen = {}

    monkeypatch.setattr(torch, "compile", lambda fn: fn)
    monkeypatch.setattr(torch, "set_float32_matmul_precision", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mf_exp, "set_seed", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mf_exp, "DiagonalGaussianDistribution", DummyPosterior)
    monkeypatch.setattr(mf_exp, "update_ema", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(exp.module_cfg, "build_model", lambda *_args, **_kwargs: model)
    monkeypatch.setattr(
        exp.dataloader_cfg,
        "build_train_dataloader",
        lambda *_args, **_kwargs: dataloader,
    )
    monkeypatch.setattr(exp.loss_cfg, "build_loss", lambda: DummyLossFn())

    def fake_eval(accel, output_dir, log, module=None, epoch=-1, num_steps=1):
        del accel, output_dir, log, epoch, num_steps
        seen["module"] = module
        return {
            "frechet_inception_distance": 1.0,
            "inception_score_mean": 2.0,
            "inception_score_std": 0.1,
        }

    monkeypatch.setattr(exp, "_eval", fake_eval)

    exp._train(accelerator, str(tmp_path), logger, seed=0)

    assert seen["module"] is unwrapped


def test_train_without_wandb_does_not_require_logger(monkeypatch, logger, tmp_path):
    exp = mf_exp.MeanFlowExp()
    exp.wandb_cfg.enable_wandb = False
    exp.module_cfg.use_ema = False
    exp.val_cfg.use_ema_module = False
    exp.val_cfg.eval_interval = 1
    exp.val_cfg.eval_two_step = False
    exp.train_cfg.epoch = 1
    exp.train_cfg.print_freq = 100

    batch = (torch.randn(2, 4, 4, 4), torch.zeros(2, dtype=torch.long))
    dataloader = DummyLoader(batch)
    accelerator = DummyAccelerator()
    model = DummyModel()

    monkeypatch.setattr(torch, "compile", lambda fn: fn)
    monkeypatch.setattr(torch, "set_float32_matmul_precision", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mf_exp, "set_seed", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mf_exp, "DiagonalGaussianDistribution", DummyPosterior)
    monkeypatch.setattr(mf_exp, "update_ema", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(exp.module_cfg, "build_model", lambda *_args, **_kwargs: model)
    monkeypatch.setattr(
        exp.dataloader_cfg,
        "build_train_dataloader",
        lambda *_args, **_kwargs: dataloader,
    )
    monkeypatch.setattr(exp.loss_cfg, "build_loss", lambda: DummyLossFn())
    monkeypatch.setattr(
        exp,
        "_eval",
        lambda *_args, **_kwargs: {
            "frechet_inception_distance": 1.0,
            "inception_score_mean": 2.0,
            "inception_score_std": 0.1,
        },
    )

    exp._train(accelerator, str(tmp_path), logger, seed=0)
