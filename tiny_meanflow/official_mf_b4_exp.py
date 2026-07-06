import math
import os
import time
from copy import deepcopy
from dataclasses import dataclass, field

import torch

torch.multiprocessing.set_start_method("spawn", force=True)
import torch_fidelity
from accelerate.utils import set_seed
from diffusers.models import AutoencoderKL
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from omegaconf import OmegaConf
from PIL import Image
from tinyexp import RedisCfgMixin, TinyExp, dataclass, store_and_run_exp
from tinyexp.dataset.sampler import InfiniteSampler
from tinyexp.tiny_engine.accelerator import DDPAccelerator, HFAccelerator
from tinyexp.utils.model_utils import update_ema
from torch.func import jvp
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from tiny_meanflow.dataset import LMDBLatentsDataset, RedisCachedImageFolder
from tiny_meanflow.sit import SiT_models


@dataclass
class NeedPreparedCfg:
    data_dir: str = os.environ.get("MEANFLOW_DATA_DIR", "./data/imagenet/train_vae_latents_lmdb")
    val_fid_statistics_path: str = os.path.join("./data/", "fid_stats/adm_in256_stats.npz")
    # vae_ckpt_name_or_path: str = f"stabilityai/sd-vae-ft-ema"
    vae_ckpt_name_or_path: str = os.environ.get("MEANFLOW_VAE_CKPT", f"stabilityai/sd-vae-ft-ema")
    output_root: str = "./output/meanflow"


class ImagePack(Dataset):
    def __init__(self, image_data):
        self.image_data = image_data

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, item_idx):
        raw_image = self.image_data[item_idx]
        return raw_image


class Loss:
    def __init__(
        self,
        time_mu=-0.4,  # Mean parameter for logit_normal distribution
        time_sigma=1.0,  # Std parameter for logit_normal distribution
        ratio_r_not_equal_t=0.25,
        # CFG related params
        cfg_omega=3.0,
        cfg_kappa=0.0,
        cfg_max_t=1.0,
        num_classes=1000,
    ):
        print(
            f"time_mu: {time_mu}, time_sigma: {time_sigma}, ratio_r_not_equal_t: {ratio_r_not_equal_t}, cfg_omega: {cfg_omega}, cfg_kappa: {cfg_kappa}, cfg_max_t: {cfg_max_t}, num_classes: {num_classes}"
        )
        self.path_type = "linear"

        # Time sampling config
        self.time_sampler = "logit_normal"
        self.time_mu = time_mu
        self.time_sigma = time_sigma
        self.ratio_r_not_equal_t = ratio_r_not_equal_t
        self.label_dropout_prob = 0.1
        # Adaptive weight config

        # CFG config
        self.cfg_omega = cfg_omega
        self.cfg_kappa = cfg_kappa
        self.cfg_max_t = cfg_max_t
        self.num_classes = num_classes

    def sample_time_steps(self, batch_size, device):
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

        # Step3: Control the proportion of r=t samples
        fraction_equal = 1.0 - self.ratio_r_not_equal_t
        # Create a mask for samples where r should equal t
        equal_mask = torch.rand(batch_size, device=device) < fraction_equal
        # Apply the mask: where equal_mask is True, set r=t (replace)
        r = torch.where(equal_mask, t, r)

        return r, t, equal_mask

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
        epoch_id: int = None,
    ):
        """
        Compute MeanFlow loss function with bootstrap mechanism
        """
        if model_kwargs is None:
            model_kwargs = {}
        else:
            model_kwargs = model_kwargs.copy()

        bs, device = images.shape[0], images.device

        unconditional_mask = torch.zeros(bs, dtype=torch.bool, device=device)
        num_classes = self.num_classes

        if model_kwargs.get("y") is not None and self.label_dropout_prob > 0:
            y_cond = model_kwargs["y"].clone()
            unconditional_mask = torch.rand(bs, device=y_cond.device) < self.label_dropout_prob
            y_cond[unconditional_mask] = num_classes
            model_kwargs["y"] = y_cond

        # Sample time steps
        r, t, r_eq_t_mask = self.sample_time_steps(bs, device)
        r, t = r.view(-1, 1, 1, 1), t.view(-1, 1, 1, 1)
        noises = torch.randn_like(images)

        z_t = (1 - t) * images + t * noises
        v_t = noises - images
        v_t_tilde = v_t

        cfg_time_mask = (t.view(-1) <= self.cfg_max_t) & (~unconditional_mask)
        if model_kwargs.get("y") is not None and cfg_time_mask.any():
            # Split samples into CFG and non-CFG
            with torch.no_grad():
                if abs(self.cfg_kappa) > 1e-8:
                    z_t_batch = torch.cat([z_t, z_t], dim=0)
                    t_batch = torch.cat([t, t], dim=0)
                    y_batch = torch.cat([y_cond, torch.full_like(y_cond, num_classes)], dim=0)
                    cfg_combined_u_at_t = compiled_model(z_t_batch, t=t_batch, r=t_batch, y=y_batch)
                    cfg_u_cond_at_t, cfg_u_uncond_at_t = torch.chunk(cfg_combined_u_at_t, 2, dim=0)
                    cfg_v_tilde = (
                        self.cfg_omega * v_t
                        + self.cfg_kappa * cfg_u_cond_at_t
                        + (1 - self.cfg_omega - self.cfg_kappa) * cfg_u_uncond_at_t
                    )
                else:
                    cfg_u_uncond_at_t = compiled_model(z_t, t=t, r=t, y=torch.full_like(y_cond, num_classes))
                    cfg_v_tilde = self.cfg_omega * v_t + (1 - self.cfg_omega) * cfg_u_uncond_at_t

                tmp_mask = cfg_time_mask.view(-1, 1, 1, 1)
                v_t_tilde = cfg_v_tilde * tmp_mask + v_t * (~tmp_mask)
                inner_model = model.module if hasattr(model, "module") else model
                inner_model.disable_fused_attn()

                def fn_current(z, cur_r, cur_t):
                    return inner_model(z, r=cur_r, t=cur_t, **model_kwargs)

                _, dudt = jvp(fn_current, (z_t, r, t), (v_t_tilde, torch.zeros_like(r), torch.ones_like(t)))
                inner_model.enable_fused_attn()
                u_target = v_t_tilde - (t - r) * dudt
        else:
            u_target = v_t_tilde

        u = compiled_model(z_t, r=r, t=t, **model_kwargs)

        error = ((u - u_target.detach()) ** 2).reshape(bs, -1)
        loss_flat = torch.sum(error, dim=-1)
        loss_flat_val = loss_flat.detach()
        loss = loss_flat / (loss_flat_val + 1e-3)

        # Extract instantaneous velocity components where r_eq_t_mask is True
        if is_print_step:
            v_cond_mask = r_eq_t_mask & cfg_time_mask
            v_unc_mask = r_eq_t_mask & torch.logical_not(cfg_time_mask)

            u_cond_mask = torch.logical_not(r_eq_t_mask) & cfg_time_mask
            u_unc_mask = torch.logical_not(r_eq_t_mask) & torch.logical_not(cfg_time_mask)

            error_flat = torch.mean(error, dim=-1).detach()

            zero_tensor = torch.tensor(0.0, device=error_flat.device)
            u_cond = torch.sum(error_flat[u_cond_mask]) if u_cond_mask.any() else zero_tensor
            u_unc = torch.sum(error_flat[u_unc_mask]) if u_unc_mask.any() else zero_tensor
            v_cond = torch.sum(error_flat[v_cond_mask]) if v_cond_mask.any() else zero_tensor
            v_unc = torch.sum(error_flat[v_unc_mask]) if v_unc_mask.any() else zero_tensor

            u_cond = accelerator.reduce(u_cond)
            u_unc = accelerator.reduce(u_unc)
            v_cond = accelerator.reduce(v_cond)
            v_unc = accelerator.reduce(v_unc)

            nr_v_cond = accelerator.reduce(v_cond_mask.sum())
            nr_v_unc = accelerator.reduce(v_unc_mask.sum())
            nr_u_cond = accelerator.reduce(u_cond_mask.sum())
            nr_u_unc = accelerator.reduce(u_unc_mask.sum())

            return loss.mean(), {
                "u_cond": u_cond / (nr_u_cond) if nr_u_cond > 0 else zero_tensor,
                "u_unc": u_unc / (nr_u_unc) if nr_u_unc > 0 else zero_tensor,
                "v_cond": v_cond / (nr_v_cond) if nr_v_cond > 0 else zero_tensor,
                "v_unc": v_unc / (nr_v_unc) if nr_v_unc > 0 else zero_tensor,
            }
        else:
            return loss.mean(), {}


@dataclass(repr=False)
class MeanFlowExp(TinyExp, RedisCfgMixin):
    output_root: str = NeedPreparedCfg.output_root
    mode: str = "train"  # or "val"
    num_worker: int = torch.cuda.device_count()  # Number of workers for the experiment
    suffix: str = ""  # suffix for the experiment name

    # ------------------------ override config ------------------------ #
    @dataclass
    class RedisCacheCfg(RedisCfgMixin.RedisCacheCfg):
        redis_cache_max_memory: int = 500

    @dataclass
    class AcceleratorCfg:
        def build_accelerator(self) -> HFAccelerator:
            return HFAccelerator(mixed_precision="bf16")

    @dataclass
    class DataloaderCfg:
        # data_dir: str = "/data/aipack/data/imagenet/train_vae_latents_lmdb"
        data_dir: str = NeedPreparedCfg.data_dir
        val_fid_statistics_path: str = NeedPreparedCfg.val_fid_statistics_path

        num_classes: int = 1000
        train_batch_size_per_device: int = 32
        train_data_worker_per_gpu: int = 8
        val_data_worker_per_gpu: int = 0
        val_batch_size_per_device: int = 128
        val_fid_samples: int = 50000

        def build_train_dataloader(self, accelerator, redis_cache_cfg) -> DataLoader:
            if redis_cache_cfg.redis_cache_enabled:
                train_dataset = RedisCachedImageFolder(
                    redis_ports=redis_cache_cfg.redis_cache_shard_ports, root=self.data_dir, flip_prob=0.5
                )
            else:
                train_dataset = LMDBLatentsDataset(self.data_dir, flip_prob=0.5)
            sampler = InfiniteSampler(len(train_dataset), shuffle=True, accelerator=accelerator)
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=self.train_batch_size_per_device,
                num_workers=self.train_data_worker_per_gpu,
                pin_memory=True,
                drop_last=True,
                sampler=sampler,
            )
            return train_dataloader

    @dataclass
    class OptimizerCfg:
        train_lr_per_img: float = 1e-4 / 256.0
        warmup_epochs: int = 0
        ema_decay_start: float = 0.9999
        ema_decay_end: float = 0.9999

        def build_optimizer(self, module, dataloader, accelerator) -> torch.optim.Optimizer:
            return torch.optim.Adam(
                module.parameters(),
                lr=self.train_lr_per_img * dataloader.batch_size * accelerator.world_size,
                betas=(0.9, 0.95),
                weight_decay=0.0,
                eps=1e-08,
            )

        def get_ema_decay_rate(self, global_step, total_step):
            decay_rate = self.ema_decay_end - (self.ema_decay_end - self.ema_decay_start) * (
                (math.cos(math.pi * global_step / total_step) + 1) * 0.5
            )
            return decay_rate

    @dataclass
    class ModuleCfg:
        vae_ckpt_name_or_path: str = NeedPreparedCfg.vae_ckpt_name_or_path
        name: str = "SiT-B/4"
        resolution: int = 256
        cfg_scale: float = 1.0
        ckpt_path: str = ""
        use_swiglu: bool = False
        use_rope: bool = False
        use_rmsnorm: bool = False
        use_ema: bool = True

        def build_model(self, dataloader_cfg, logger) -> torch.nn.Module:
            block_kwargs = {"fused_attn": True, "qk_norm": False}
            model = SiT_models[self.name](
                input_size=self.resolution // 8,
                num_classes=dataloader_cfg.num_classes,
                use_cfg=True,
                use_swiglu=self.use_swiglu,
                use_rope=self.use_rope,
                use_rmsnorm=self.use_rmsnorm,
                **block_kwargs,
            )
            self._load_model(model, self.ckpt_path, logger)

            return model

        def _load_model(self, model, ckpt_path, logger):
            if ckpt_path is not None and ckpt_path != "":
                logger.info(f"==> Loading model from {ckpt_path}")
                state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)["ema"]
                model.load_state_dict(state_dict)
            else:
                logger.info("==> No checkpoint path is provided. Training model from scratch.")

    @dataclass
    class LossCfg:
        path_bridge_k: float = 0.01
        time_mu: float = -0.4
        time_sigma: float = 1.0
        ratio_r_not_equal_t: float = 0.25
        cfg_omega: float = 3.0
        cfg_kappa: float = 0.0
        cfg_max_t: float = 1.0

        def build_loss(self):
            loss_fn = Loss(
                time_mu=self.time_mu,
                time_sigma=self.time_sigma,
                ratio_r_not_equal_t=self.ratio_r_not_equal_t,
                cfg_omega=self.cfg_omega,
                cfg_kappa=self.cfg_kappa,
                cfg_max_t=self.cfg_max_t,
            )
            return loss_fn

    @dataclass
    class TrainCfg:
        epoch: int = 80
        print_freq: int = 20

    @dataclass
    class ValCfg:
        epoch: int = -1
        use_ema_module: bool = True
        eval_interval: int = 5
        eval_two_step: bool = False
        dump_samples: bool = False
        num_steps: int = 1

    @dataclass
    class WandbCfg(TinyExp.WandbCfg):
        enable_wandb: bool = True
        entity: str = "lizeming"
        project: str = "mf_ablation"

    # ------------------------ instantiation of config --------------------------------- #
    redis_cache_cfg: RedisCacheCfg = field(default_factory=RedisCacheCfg)
    wandb_cfg: WandbCfg = field(default_factory=WandbCfg)
    val_cfg: ValCfg = field(default_factory=ValCfg)
    train_cfg: TrainCfg = field(default_factory=TrainCfg)
    module_cfg: ModuleCfg = field(default_factory=ModuleCfg)
    loss_cfg: LossCfg = field(default_factory=LossCfg)
    optimizer_cfg: OptimizerCfg = field(default_factory=OptimizerCfg)
    dataloader_cfg: DataloaderCfg = field(default_factory=DataloaderCfg)
    accelerator_cfg: AcceleratorCfg = field(default_factory=AcceleratorCfg)

    # ------------------------ end of config, begin of execution ------------------------ #

    def _eval(self, accelerator, output_dir, logger, module=None, epoch=-1, num_steps=1):
        torch.cuda.empty_cache()
        # torch.manual_seed(cfg.seed * accelerator.world_size + accelerator.rank)
        dataloader_cfg = self.dataloader_cfg
        module_cfg = self.module_cfg

        if module is None:
            module = self.module_cfg.build_model(dataloader_cfg, logger)
            epoch = self.val_cfg.epoch
            if epoch > -1:
                ckpt_path = os.path.join(os.path.dirname(output_dir), "train", f"checkpoints/{epoch:07d}.pt")
            elif epoch == -1:
                ckpt_path = os.path.join(os.path.dirname(output_dir), "train", f"checkpoints/best.pt")
            else:
                accelerator.print("==> No validation epoch is provided. eval model from best checkpoint.")
                ckpt_path = self.module_cfg.ckpt_path
            module_cfg._load_model(module, ckpt_path, logger=logger)
            module = module.to(accelerator.device)

        module.eval()

        # vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(accelerator.device)
        vae = AutoencoderKL.from_pretrained(module_cfg.vae_ckpt_name_or_path).to(accelerator.device)
        assert module_cfg.cfg_scale >= 1.0, "In almost all cases, cfg_scale should be >= 1.0"

        if accelerator.is_main_process:
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(output_dir + "/img_dir", exist_ok=True)
        accelerator.wait_for_everyone()

        global_batch_size = dataloader_cfg.val_batch_size_per_device * accelerator.world_size
        total_samples = int(math.ceil(dataloader_cfg.val_fid_samples / global_batch_size) * global_batch_size)

        if accelerator.is_main_process:
            logger.info(f"Total number of images that will be sampled: {total_samples}")
            logger.info(f"SiT Parameters: {sum(p.numel() for p in module.parameters()):,}")

        samples_needed_this_gpu = int(total_samples // accelerator.world_size)
        assert samples_needed_this_gpu % dataloader_cfg.val_batch_size_per_device == 0
        iterations = int(samples_needed_this_gpu // dataloader_cfg.val_batch_size_per_device)
        pbar = range(iterations)
        pbar = tqdm(pbar) if accelerator.rank == 0 else pbar
        total = 0

        all_samples = []
        for _ in pbar:
            z = torch.randn(
                dataloader_cfg.val_batch_size_per_device,
                module.in_channels,
                module_cfg.resolution // 8,
                module_cfg.resolution // 8,
            ).to(accelerator.device)

            y = torch.randint(
                0,
                dataloader_cfg.num_classes,
                (dataloader_cfg.val_batch_size_per_device,),
                device=accelerator.device,
            )

            with torch.no_grad():
                batch_size = z.shape[0]
                device = z.device
                time_steps = torch.linspace(1, 0, num_steps + 1, device=device)
                for i in range(num_steps):
                    t = torch.full((batch_size,), time_steps[i], device=device)
                    r = torch.full((batch_size,), time_steps[i + 1], device=device)
                    u = module(z, r, t, y=y)
                    z = z - (time_steps[i] - time_steps[i + 1]) * u
                samples = z.to(torch.float32)

                latents_scale = (
                    torch.tensor([0.18125, 0.18125, 0.18125, 0.18125]).view(1, 4, 1, 1).to(accelerator.device)
                )
                latents_bias = torch.tensor([0.0, 0.0, 0.0, 0.0]).view(1, 4, 1, 1).to(accelerator.device)
                samples = vae.decode((samples - latents_bias) / latents_scale).sample

                samples = (samples + 1) / 2.0
                samples = torch.clamp(255.0 * samples, 0, 255)  # .numpy()
                if self.val_cfg.dump_samples:
                    for i, sample in enumerate(samples):
                        index = i * accelerator.world_size + accelerator.rank + total
                        sample_cpu = sample.to("cpu", dtype=torch.uint8).permute(1, 2, 0).numpy()
                        # Image.fromarray(sample_cpu).save(f"{output_dir}/img_dir/{index:06d}.png")
                        Image.fromarray(sample_cpu).save(f"/data/hf_2dot6_res/{index:06d}.png")
                samples = accelerator.gather(samples)

                if accelerator.is_main_process:
                    all_samples.append(samples.to("cpu", dtype=torch.uint8))

            total += global_batch_size

        accelerator.wait_for_everyone()

        # -------------------------------------- Evaluation --------------------------------------
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        if accelerator.rank == 0:
            logger.info(f"Computing evaluation metrics...")
            metrics_dict = {}
            metrics_args = {
                # "input1": output_dir + "/img_dir",
                "input1": ImagePack(torch.cat(all_samples, dim=0)),
                "cuda": True,
                "isc": True,
                "fid": True,
                "kid": False,
                "prc": False,
                "verbose": True,
            }
            if module_cfg.resolution == 256:
                metrics_args["input2"] = None
                metrics_args["fid_statistics_file"] = dataloader_cfg.val_fid_statistics_path
            else:
                raise NotImplementedError

            metrics_dict = torch_fidelity.calculate_metrics(**metrics_args)

            fid = metrics_dict.get("frechet_inception_distance", None)
            is_mean = metrics_dict.get("inception_score_mean", None)
            is_std = metrics_dict.get("inception_score_std", None)

            log_str = f"==> {self.exp_name + self.suffix} Epoch-{epoch} Eval Results: "
            if fid is not None:
                log_str += f"FID:{fid:.2f}, "
            if is_mean is not None:
                log_str += f"Inception Score:{is_mean:.2f} ± {is_std:.2f}, "
            logger.info(log_str)
            del all_samples
            return metrics_dict

            # metrics_file = os.path.join(output_dir, "metrics.json")
            # with open(metrics_file, "w") as f:
            #     json.dump(metrics_dict, f, indent=4)
            # logger.info(f"Metrics saved to {metrics_file}")

    def _train(self, accelerator, output_dir, logger, seed):
        wandb_cfg = self.wandb_cfg
        module_cfg = self.module_cfg

        cfg_dict = OmegaConf.to_container(OmegaConf.structured(self), resolve=True)
        del cfg_dict["hydra"]

        if wandb_cfg.enable_wandb:
            wandb_logger = self.wandb_cfg.build_wandb(
                accelerator=accelerator,
                name=self.exp_name + self.suffix,
                project=wandb_cfg.project,
                entity=wandb_cfg.entity,
                config=cfg_dict,
            )
        logger.info(OmegaConf.to_yaml(OmegaConf.create(cfg_dict)))
        torch.set_float32_matmul_precision("high")
        set_seed(seed + accelerator.rank)
        ori_module = self.module_cfg.build_model(self.dataloader_cfg, logger)

        if accelerator.is_main_process:
            os.makedirs(output_dir, exist_ok=True)

        train_dataloader = self.dataloader_cfg.build_train_dataloader(accelerator, self.redis_cache_cfg)
        optimizer_cfg = self.optimizer_cfg
        ori_optimizer = optimizer_cfg.build_optimizer(ori_module, train_dataloader, accelerator)
        module, optimizer = accelerator.prepare(ori_module, ori_optimizer)

        if self.module_cfg.use_ema:
            ema_module = deepcopy(ori_module).to(accelerator.device).eval()
            ema_module.disable_fused_attn()
            for p in ema_module.parameters():
                p.requires_grad = False
        else:
            ema_module = None

        # --------------------------- begin of training ---------------------------
        loss_fn = self.loss_cfg.build_loss()
        latents_scale = torch.tensor([0.18125, 0.18125, 0.18125, 0.18125]).view(1, 4, 1, 1).to(accelerator.device)
        latents_bias = torch.tensor([0.0, 0.0, 0.0, 0.0]).view(1, 4, 1, 1).to(accelerator.device)

        train_iter = iter(train_dataloader)

        global_step = 0
        global_epoch = 0

        total_batch_size = train_dataloader.batch_size * accelerator.world_size
        base_lr = optimizer_cfg.train_lr_per_img * total_batch_size
        steps_per_epoch = len(train_dataloader)
        total_step = self.train_cfg.epoch * steps_per_epoch
        warmup_steps = optimizer_cfg.warmup_epochs * steps_per_epoch

        @torch.compile
        def compiled_module(*args, **kwargs):
            return module(*args, **kwargs)

        if ema_module is not None:

            @torch.compile
            def compiled_ema_module(*args, **kwargs):
                return ema_module(*args, **kwargs)

        else:
            compiled_ema_module = None

        best_fid = float("inf")
        for epoch_id in range(self.train_cfg.epoch):
            module.train()

            epoch_start_time = time.time()
            for step_in_epoch in range(len(train_dataloader)):
                it_start_time = time.time()
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_dataloader)
                    batch = next(train_iter)

                moments, labels = (_.to(accelerator.device) for _ in batch)

                with torch.no_grad():
                    posterior = DiagonalGaussianDistribution(moments)
                    x = posterior.sample()
                    x = x * latents_scale + latents_bias

                data_end_time = time.time()
                is_print_step = global_step % self.train_cfg.print_freq == 0

                loss, loss_ref = loss_fn(
                    module,
                    compiled_module,
                    compiled_ema_module,
                    x,
                    accelerator,
                    dict(y=labels),
                    logger,
                    is_print_step,
                    epoch_id=epoch_id,
                )
                loss_end_time = time.time()

                loss = loss.mean()
                if loss_ref is not None:
                    for k, v in loss_ref.items():
                        loss_ref[k] = v.mean()

                # Apply learning rate warmup
                if global_step < warmup_steps:
                    lr = base_lr * (global_step + 1) / warmup_steps
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = lr

                if total_batch_size >= 1024 and global_step == 40 * steps_per_epoch:
                    for param_group in optimizer.param_groups:
                        param_group["lr"] /= 2.0

                optimizer.zero_grad()
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(module.parameters(), 1.0)
                optimizer.step()

                if ema_module is not None:
                    decay_rate = optimizer_cfg.get_ema_decay_rate(global_step, total_step)
                    update_ema(ema_module, module, decay_rate)
                model_end_time = time.time()

                global_step += 1

                if is_print_step and accelerator.is_main_process:
                    epoch_elapsed_time = time.time() - epoch_start_time
                    epoch_elapsed_str = f"{int(epoch_elapsed_time / 60):02d}:{int(epoch_elapsed_time % 60):02d}"

                    epoch_total_seconds = epoch_elapsed_time / ((step_in_epoch + 1) / steps_per_epoch)
                    epoch_total_str = f"{int(epoch_total_seconds / 60):02d}:{int(epoch_total_seconds % 60):02d}"
                    loss_ref_str = ", ".join([f"{k}:{v.item():.4f}" for k, v in loss_ref.items()])

                    logger.info(
                        f"e:{global_epoch},{step_in_epoch + 1}/{steps_per_epoch}, "
                        f"loss:{loss.item():.4f}, "
                        f"lr:{optimizer.param_groups[0]['lr']:.4f}, "
                        f"elapsed:{epoch_elapsed_str},total:{epoch_total_str},"
                        f"data_time:{data_end_time - it_start_time:.2f}s,"
                        f"loss_time:{loss_end_time - data_end_time:.2f}s,"
                        f"model_time:{model_end_time - loss_end_time:.2f}s, "
                        f"loss_ref:{loss_ref_str}"
                    )

                    if wandb_cfg.enable_wandb:
                        log_dict = {}
                        for k, v in loss_ref.items():
                            log_dict[f"loss_ref/{k}"] = v.item()

                        # wandb_logger.log(log_dict)
                        if module_cfg.use_ema:
                            log_dict["metrics/decay_rate"] = decay_rate
                        wandb_logger.log(log_dict, step=global_step)

            global_epoch += 1

            if (
                global_epoch % self.val_cfg.eval_interval == 0
                or global_epoch == 1
                or self.train_cfg.epoch - global_epoch < 5
            ):
                if self.val_cfg.use_ema_module:
                    eval_model = ema_module if ema_module is not None else accelerator.unwrap_model(module)
                else:
                    eval_model = accelerator.unwrap_model(module)
                metrics_dict = self._eval(
                    accelerator, output_dir, logger, module=eval_model, epoch=global_epoch, num_steps=1
                )
                if self.val_cfg.eval_two_step:
                    metrics_dict_2_steps = self._eval(
                        accelerator, output_dir, logger, module=eval_model, epoch=global_epoch, num_steps=2
                    )

                if accelerator.is_main_process:
                    fid = metrics_dict.get("frechet_inception_distance", None)
                    wandb_logger_dict = {
                        "metrics/fid": fid,
                        "metrics/is_mean": metrics_dict.get("inception_score_mean", None),
                        "metrics/is_std": metrics_dict.get("inception_score_std", None),
                    }
                    if self.val_cfg.eval_two_step:
                        wandb_logger_dict["metrics/fid_2_steps"] = metrics_dict_2_steps.get(
                            "frechet_inception_distance", None
                        )
                        wandb_logger_dict["metrics/is_mean_2_steps"] = metrics_dict_2_steps.get(
                            "inception_score_mean", None
                        )
                        wandb_logger_dict["metrics/is_std_2_steps"] = metrics_dict_2_steps.get(
                            "inception_score_std", None
                        )

                    if wandb_cfg.enable_wandb:
                        wandb_logger.log(wandb_logger_dict, step=global_step)

                    # checkpoint_path = os.path.join(output_dir, f"checkpoints/{global_epoch:07d}.pt")
                    checkpoint_path = os.path.join(output_dir, f"checkpoints/best.pt")
                    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                    if fid < best_fid:
                        logger.info(f"=> Saving checkpoint to {checkpoint_path}")
                        checkpoint = {
                            "module": accelerator.unwrap_model(module).state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "global_step": global_step,
                            "global_epoch": global_epoch,
                        }
                        if ema_module is not None:
                            checkpoint["ema"] = ema_module.state_dict()

                        torch.save(checkpoint, checkpoint_path)
                        logger.info(f"=> Saved checkpoint to {checkpoint_path}")
                        best_fid = fid

    def run(self):
        accelerator = self.accelerator_cfg.build_accelerator()
        output_dir = os.path.join(self.output_root, self.exp_name + self.suffix, self.mode)
        # --------------------------- begin of execute ---------------------------
        seed = 0
        logger = self.logger_cfg.build_logger(save_dir=output_dir, distributed_rank=accelerator.rank)

        if self.mode == "train":
            self._train(accelerator, output_dir, logger, seed)
        if self.mode == "val":
            torch.set_float32_matmul_precision("high")
            set_seed(seed + accelerator.rank)
            self._eval(accelerator, output_dir, logger, num_steps=self.val_cfg.num_steps)


MeanFlowB4Exp = MeanFlowExp

if __name__ == "__main__":
    store_and_run_exp(MeanFlowB4Exp)
