# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys

# from tensorboardX import SummaryWriter

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from .utils import apply_model, average_metric, center_trim


class Trainer(object):
    """Customized trainer module for Demucs training."""

    def __init__(
        self,
        model,
        optimizer,
        config,
        device=torch.device("cpu"),
    ):
        """Initialize trainer.

        Args:
            model (dict): Dict of models. It must contrain "generator" and "discriminator" models.
            best_state (dict): Dict of best generator's state.
            optimizer (dict): Dict of optimizers. It must contrain "generator" and "discriminator" optimizers.
            metrics (list): List of metrics. It must contrain "train" and "valid", "best", "duration", "model_size", "true_model_size", "compressed_model_size" models.
            device (torch.deive): Pytorch device instance.

        """
        self.metrics = []
        self.model = model
        self.best_state = None
        self.optimizer = optimizer
        self.config = config
        self.device = device
        # self.writer = SummaryWriter(config["outdir"])

    def save_checkpoint(self, checkpoint_path):
        """Save checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be saved.

        """
        state_dict = {
            "metrics": self.metrics,
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
            self.optimizer["generator"].load_state_dict(
                state_dict["optimizer"]["generator"]
            )
            if self.config.use_adv:
                self.optimizer["discriminator"].load_state_dict(
                    state_dict["optimizer"]["discriminator"])


def train_model(epoch,
                dataset,
                model,
                criterion,
                optimizer,
                augment,
                quantizer=None,
                diffq=0,
                repeat=1,
                device="cpu",
                seed=None,
                workers=4,
                world_size=1,
                batch_size=16,
                batch_divide=1):

    if world_size > 1:
        sampler = DistributedSampler(dataset)
        sampler_epoch = epoch * repeat
        if seed is not None:
            sampler_epoch += seed * 1000
        sampler.set_epoch(sampler_epoch)
        batch_size //= world_size
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=workers)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers, shuffle=True)
    current_loss = 0
    model_size = 0
    for repetition in range(repeat):
        tq = tqdm(loader,
                  ncols=120,
                  desc=f"[{epoch:03d}] train ({repetition + 1}/{repeat})",
                  leave=False,
                  file=sys.stdout,
                  unit=" batch")
        total_loss = 0
        for idx, sources in enumerate(tq):
            if len(sources) < batch_size:
                # skip uncomplete batch for augment.Remix to work properly
                continue
            sources = sources.to(device)
            sources = augment(sources)
            mix = sources.sum(dim=1)

            for start in range(batch_divide):
                sources_divide = sources[start::batch_divide]
                mix_divide = mix[start::batch_divide]
                estimates = model(mix_divide)
                sources_divide = center_trim(sources_divide, estimates)
                loss = criterion(estimates, sources_divide)
                total_loss += loss.item() / batch_divide
                model_size = 0
                if quantizer is not None:
                    model_size = quantizer.model_size()

                train_loss = loss + diffq * model_size
                (train_loss / batch_divide).backward()

                # free some space before next round
                del sources_divide, mix_divide, estimates, loss, train_loss

            grad_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm()**2
            grad_norm = grad_norm**0.5
            optimizer.step()
            optimizer.zero_grad()

            if quantizer is not None:
                model_size = model_size.item()

            current_loss = total_loss / (1 + idx)
            tq.set_postfix(loss=f"{current_loss:.4f}", ms=f"{model_size:.2f}",
                           grad=f"{grad_norm:.5f}")

        if world_size > 1:
            sampler.epoch += 1

    if world_size > 1:
        current_loss = average_metric(current_loss)
    return current_loss, model_size


def validate_model(epoch,
                   dataset,
                   model,
                   criterion,
                   device="cpu",
                   rank=0,
                   world_size=1,
                   shifts=0,
                   overlap=0.25,
                   split=False):
    indexes = range(rank, len(dataset), world_size)
    tq = tqdm.tqdm(indexes,
                   ncols=120,
                   desc=f"[{epoch:03d}] valid",
                   leave=False,
                   file=sys.stdout,
                   unit=" track")
    current_loss = 0
    for index in tq:
        streams = dataset[index]
        # first five minutes to avoid OOM on --upsample models
        streams = streams[..., :15_000_000]
        streams = streams.to(device)
        sources = streams[1:]
        mix = streams[0]
        estimates = apply_model(model, mix, shifts=shifts, split=split, overlap=overlap)
        loss = criterion(estimates, sources)
        current_loss += loss.item() / len(indexes)
        del estimates, streams, sources

    if world_size > 1:
        current_loss = average_metric(current_loss, len(indexes))
    return current_loss
