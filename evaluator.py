from typing import NamedTuple, List, Any, Optional, Dict
from itertools import chain
from dataclasses import dataclass
import itertools
import os
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import numpy as np
from matplotlib import pyplot as plt

from schedulers import Scheduler, LRSchedule
from models import Prober, build_mlp
from configs import ConfigBase

from dataset import WallDataset
from normalizer import Normalizer


@dataclass
class ProbingConfig(ConfigBase):
    probe_targets: str = "locations"
    lr: float = 0.0002
    epochs: int = 20
    schedule: LRSchedule = LRSchedule.Cosine
    sample_timesteps: int = 30
    prober_arch: str = "256"


class ProbeResult(NamedTuple):
    model: torch.nn.Module
    average_eval_loss: float
    eval_losses_per_step: List[float]
    plots: List[Any]


default_config = ProbingConfig()


def location_losses(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    assert pred.shape == target.shape
    mse = (pred - target).pow(2).mean(dim=0)
    return mse


class ProbingEvaluator:
    def __init__(
            self,
            device: "cuda",
            model: torch.nn.Module,
            probe_train_ds,
            probe_val_ds: dict,
            config: ProbingConfig = default_config,
            quick_debug: bool = False,
    ):
        self.device = device
        self.config = config

        self.model = model
        self.model.eval()

        self.quick_debug = quick_debug

        self.ds = probe_train_ds
        self.val_ds = probe_val_ds

        self.normalizer = Normalizer()

    def train_pred_prober(self):
        """
        使用递归方式（unrolling embeddings）的评估思路训练 prober。
        假设 JEPA 模型可在单步输入下返回对应时间步的embedding。
        """
        repr_dim = self.model.repr_dim
        dataset = self.ds
        model = self.model
        config = self.config
        epochs = config.epochs

        if self.quick_debug:
            epochs = 1
        test_batch = next(iter(dataset))

        prober_output_shape = getattr(test_batch, "locations")[0, 0].shape
        prober = Prober(
            repr_dim,
            config.prober_arch,
            output_shape=prober_output_shape,
        ).to(self.device)

        all_parameters = []
        all_parameters += list(prober.parameters())

        optimizer_pred_prober = torch.optim.Adam(all_parameters, config.lr)

        step = 0

        batch_size = dataset.batch_size
        batch_steps = None

        scheduler = Scheduler(
            schedule=self.config.schedule,
            base_lr=config.lr,
            data_loader=dataset,
            epochs=epochs,
            optimizer=optimizer_pred_prober,
            batch_steps=batch_steps,
            batch_size=batch_size,
        )

        for epoch in tqdm(range(epochs), desc=f"Probe prediction epochs"):
            for batch in tqdm(dataset, desc="Probe prediction step"):

                # 使用递归生成 embeddings 的方式
                # -----------------------------------------------------
                # 初始embedding (根据模型需要修改，确保能从首个state获取embedding)
                pred_encs = []

                # 假设 model 接受单步输入并返回当前时刻 embedding
                # 注意：此处只是示意，需要根据 JEPA 模型实际接口编写
                # 例如：embedding = model.get_embedding_from_state(batch.states[:,0])
                # 或者需要你在 JEPA 中实现一个函数来获取单步embedding。

                # 首先对第 0 个时间步进行编码
                embedding = model.online_encoder(batch.states[:, 0])  # (B, D)
                pred_encs.append(embedding)

                # 对后续的时间步进行递归预测
                # 假设actions维度为(B, T-1, action_dim)
                # 对 t in [1, T-1]：
                # 使用 predictor 根据 embedding 和 actions[:,t-1] 预测下一步 embedding
                # 再对下一步 states 编码对比 (如果需要）
                # 这里根据前embedding和action预测下一时刻的embedding，
                # 或者你可以选择只用 predictor 预测未来embedding，不额外编码下一时刻state。
                # 下方代码只是一个示意，需要你根据 JEPA forward逻辑适配。

                T = batch.states.size(1)
                for t in range(1, T):
                    prev_embedding = pred_encs[-1]  # 上一个时间步的embedding
                    prev_action = batch.actions[:, t - 1]  # 对应的action
                    # 使用 predictor 获得下一步的预测embedding
                    next_embedding = model.predictor(prev_embedding, prev_action)
                    pred_encs.append(next_embedding)
                # -----------------------------------------------------

                pred_encs = torch.stack(pred_encs, dim=0)  # (T, B, D)

                # 与非递归方式类似，后面处理目标和loss计算
                n_steps = pred_encs.shape[0]
                bs = pred_encs.shape[1]

                target = getattr(batch, "locations").cuda()
                target = self.normalizer.normalize_location(target)

                # 随机采样时间步以避免OOM
                if (
                        config.sample_timesteps is not None
                        and config.sample_timesteps < n_steps
                ):
                    sample_shape = (config.sample_timesteps,) + pred_encs.shape[1:]
                    sampled_pred_encs = torch.empty(
                        sample_shape,
                        dtype=pred_encs.dtype,
                        device=pred_encs.device,
                    )

                    sampled_target_locs = torch.empty(bs, config.sample_timesteps, 2, device=pred_encs.device)

                    for i in range(bs):
                        indices = torch.randperm(n_steps)[: config.sample_timesteps]
                        sampled_pred_encs[:, i, :] = pred_encs[indices, i, :]
                        sampled_target_locs[i, :] = target[i, indices]

                    pred_encs = sampled_pred_encs
                    target = sampled_target_locs

                pred_locs = torch.stack([prober(x) for x in pred_encs], dim=1)  # (T_samp, B, 2) -> (B, T_samp, 2)
                losses = location_losses(pred_locs, target)
                per_probe_loss = losses.mean()

                if step % 100 == 0:
                    print(f"normalized pred locations loss {per_probe_loss.item()}")

                optimizer_pred_prober.zero_grad()
                per_probe_loss.backward()
                optimizer_pred_prober.step()

                lr = scheduler.adjust_learning_rate(step)

                step += 1

                if self.quick_debug and step > 2:
                    break

        return prober

    @torch.no_grad()
    def evaluate_all(
            self,
            prober,
    ):
        """
        对所有验证集进行评估
        """
        avg_losses = {}

        for prefix, val_ds in self.val_ds.items():
            avg_losses[prefix] = self.evaluate_pred_prober(
                prober=prober,
                val_ds=val_ds,
                prefix=prefix,
            )

        return avg_losses

    @torch.no_grad()
    def evaluate_pred_prober(
            self,
            prober,
            val_ds,
            prefix="",
    ):
        quick_debug = self.quick_debug
        config = self.config

        model = self.model
        probing_losses = []
        prober.eval()

        for idx, batch in enumerate(tqdm(val_ds, desc="Eval probe pred")):
            # 使用递归生成embeddings进行评估
            pred_encs = []
            embedding = model.online_encoder(batch.states[:, 0])
            pred_encs.append(embedding)

            T = batch.states.size(1)
            for t in range(1, T):
                prev_embedding = pred_encs[-1]
                prev_action = batch.actions[:, t - 1]
                next_embedding = model.predictor(prev_embedding, prev_action)
                pred_encs.append(next_embedding)

            pred_encs = torch.stack(pred_encs, dim=0)  # (T, B, D)

            target = getattr(batch, "locations").cuda()
            target = self.normalizer.normalize_location(target)

            pred_locs = torch.stack([prober(x) for x in pred_encs], dim=1)  # (T, B, 2) -> output shape (B, T, 2)
            losses = location_losses(pred_locs, target)
            probing_losses.append(losses.cpu())

            if quick_debug and idx > 2:
                break

        losses_t = torch.stack(probing_losses, dim=0).mean(dim=0)
        losses_t = self.normalizer.unnormalize_mse(losses_t)

        losses_t = losses_t.mean(dim=-1)
        average_eval_loss = losses_t.mean().item()

        return average_eval_loss
