# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys

# from tensorboardX import SummaryWriter
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from .utils import apply_model, average_metric, center_trim


class Trainer(object):
    """Customized trainer module for Demucs training."""

    def __init__(
        self,
        data_loader,
        sampler,
        augment,
        model,
        quantizer,
        criterion,
        optimizer,
        config,
        device=torch.device("cpu"),
    ):
        """Initialize trainer.

        Args:
            data_loader (dict): Dict of data loaders. It must contrain "train" and "valid" loaders.
            model (dict): Dict of models. It must contrain "generator" and "discriminator" models.
            best_state (dict): Dict of best generator's state.
            criterion (dict): Dict of criterions. It must contrain "stft" and "mse" criterions.
            optimizer (dict): Dict of optimizers. It must contrain "generator" and "discriminator" optimizers.
            metrics (list): List of metrics. It must contrain "train" and "valid", "best", "duration", "model_size", "true_model_size", "compressed_model_size" models.
            device (torch.deive): Pytorch device instance.

        """
        self.metrics = []
        self.data_loader = data_loader
        self.sampler = sampler
        self.augment = augment
        self.model = model
        self.quantizer = quantizer
        self.criterion = criterion
        self.best_state = None
        self.optimizer = optimizer
        self.config = config
        self.device = device
        # self.writer = SummaryWriter(config["outdir"])
        self.total_train_loss = defaultdict(float)

    def save_checkpoint(self, checkpoint_path):
        """Save checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be saved.

        """
        state_dict = {
            "metrics": self.metrics,
            "best_state": self.best_state,
            "optimizer": {
                "generator": self.optimizer["generator"].state_dict(),
            },
        }
        if self.config.device.world_size > 1:
            state_dict["model"] = {
                "generator": self.model["generator"].module.state_dict(),
            }
            if self.config.use_adv:
                state_dict["optimizer"]["discriminator"] = self.optimizer["discriminator"].state_dict()
        else:
            state_dict["model"] = {
                "generator": self.model["generator"].state_dict(),
            }
        if self.config.use_adv:
            state_dict["optimizer"]["discriminator"] = self.optimizer["discriminator"].state_dict()
            state_dict["model"]["discriminator"] = self.model["discriminator"].module.state_dict(
            ) if self.config["distributed"] else self.model["discriminator"].state_dict()

        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))
        torch.save(state_dict, checkpoint_path)

    def load_checkpoint(self, checkpoint_path, load_only_params=False):
        """Load checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be loaded.
            load_only_params (bool): Whether to load only model parameters.

        """
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if self.config.device.world_size > 1:
            self.model["generator"].module.load_state_dict(state_dict["model"]["generator"])
        else:
            self.model["generator"].load_state_dict(state_dict["model"]["generator"])
        if self.config.use_adv:
            if self.config.device.world_size > 1:
                self.model["discriminator"].module.load_state_dict(
                    state_dict["model"]["discriminator"])
            else:
                self.model["discriminator"].load_state_dict(state_dict["model"]["discriminator"])
        if not load_only_params:
            self.metrics = state_dict["metrics"]
            self.best_state = state_dict["best_state"]
            self.optimizer["generator"].load_state_dict(
                state_dict["optimizer"]["generator"]
            )
            if self.config.use_adv:
                self.optimizer["discriminator"].load_state_dict(
                    state_dict["optimizer"]["discriminator"])


    def _train_epoch(self, epoch):
        """Train model one epoch."""
        if self.config.device.world_size > 1:
            sampler_epoch = epoch * self.config.repeat
            if self.config.seed is not None:
                sampler_epoch += self.config.seed * 1000
            self.sampler.set_epoch(sampler_epoch)
        for repetition in range(self.config.repeat):
            tq = tqdm(self.data_loader["train"],
                      ncols=120,
                      desc=f"[{epoch:03d}] train ({repetition + 1}/{self.config.repeat})",
                      leave=False,
                      file=sys.stdout,
                      unit=" batch")
            # reset
            self.total_train_loss = defaultdict(float)
            for idx, sources in enumerate(tq):
                if idx > 0:
                    break
                if len(sources) < self.config.batch_size // self.config.device.world_size:
                    # skip uncomplete batch for augment.Remix to work properly
                    continue
                sources = sources.to(self.device)
                sources = self.augment(sources)
                mix = sources.sum(dim=1)

                for start in range(self.config.batch_divide):
                    sources_divide = sources[start::self.config.batch_divide]
                    mix_divide = mix[start::self.config.batch_divide]
                    estimates = self.model["generator"](mix_divide)
                    sources_divide = center_trim(sources_divide, estimates)

                    # initialize
                    gen_loss = 0.0

                    # l1 loss
                    if self.config.loss.l1["lambda"]:
                        l1_loss = self.criterion["l1"](
                            estimates, sources_divide) / self.config.batch_divide
                        gen_loss += self.config.loss.l1["lambda"] * l1_loss
                        self.total_train_loss["l1_loss"] += l1_loss.item()
                        del l1_loss

                    self.total_train_loss["gen_loss"] += gen_loss.item()
                    gen_loss.backward()

                    # free some space before next round
                    del sources_divide, mix_divide, estimates, gen_loss

                # model size loss
                model_size = 0
                if self.quantizer is not None:
                    model_size = self.quantizer.model_size()
                    (self.config.diffq * model_size).backward()
                    model_size = model_size.item()

                # update generator
                grad_norm = 0
                for p in self.model["generator"].parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.data.norm()**2
                grad_norm = grad_norm**0.5
                self.optimizer["generator"].step()
                self.optimizer["generator"].zero_grad()

                current_loss = self.total_train_loss["gen_loss"] / (1 + idx)
                tq.set_postfix(loss=f"{current_loss:.4f}", ms=f"{model_size:.2f}",
                               grad=f"{grad_norm:.5f}")

            if self.config.device.world_size > 1:
                self.sampler.epoch += 1

        if self.config.device.world_size > 1:
            current_loss = average_metric(current_loss)
        return current_loss, model_size

    def _valid_epoch(self, epoch):
        tq = tqdm(self.data_loader["valid"],
                  ncols=120,
                  desc=f"[{epoch:03d}] valid",
                  leave=False,
                  file=sys.stdout,
                  unit=" track")
        # reset
        self.total_valid_loss = defaultdict(float)
        for idx, streams in enumerate(tq):
            # first five minutes to avoid OOM on --upsample models
            streams = streams[0, ..., :15_000_000]
            streams = streams.to(self.device)
            sources = streams[1:]
            mix = streams[0]
            estimates = apply_model(self.model["generator"], mix, shifts=0,
                                    split=self.config.split_valid, overlap=self.config.dataset.overlap)

            # initialize
            gen_loss = 0.0

            # l1 loss
            if self.config.loss.l1["lambda"]:
                l1_loss = self.criterion["l1"](estimates, sources)
                gen_loss += self.config.loss.l1["lambda"] * l1_loss
                self.total_valid_loss["l1_loss"] += l1_loss.item()

            self.total_valid_loss["gen_loss"] += gen_loss.item()
            del estimates, streams, sources

        current_loss = self.total_valid_loss["gen_loss"] / (1 + idx)
        if self.config.device.world_size > 1:
            current_loss = average_metric(current_loss)
        return current_loss




