# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import sys
from pathlib import Path
import time

import musdb
import museval
from scipy.io import wavfile
import numpy as np

from collections import defaultdict

# from tensorboardX import SummaryWriter
import mlflow
from omegaconf import DictConfig, ListConfig

import torch
from torch import distributed
from tqdm import tqdm

# from .test import evaluate
from .audio import convert_audio
from .utils import human_seconds, apply_model, save_model, average_metric, center_trim


ignore_params = ["restart", "split_valid", "show", "save", "save_model", "save_state", "half",
                 "q-min-size", "qat", "diffq", "ms_target", "mlflow", "outdir", "device", "dataset", "name",  "model.generator.params", "model.discriminator.params", "loss.stft.params", "loss.adversarial.params"]


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
        self.run_id = None
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
        self.outdir = Path(config.outdir.out)
        # self.writer = SummaryWriter(self.outdir / "tensorboard" / f"{self.config.name}")
        self.total_train_loss = defaultdict(float)
        self.total_valid_loss = defaultdict(float)
        self.total_eval_loss = defaultdict(float)

    def run(self):
        # mlflow setting
        if self.config.device.rank == 0:
            mlflow.set_tracking_uri(self.config.mlflow.tracking_uri)
            mlflow.set_experiment(self.config.mlflow.experiment_name)
            if self.run_id:
                mlflow.start_run(run_id=self.run_id)
            else:
                mlflow.start_run(run_id=None)
                self.run_id = mlflow.active_run().info.run_id
        self.log_params_from_omegaconf_dict(self.config)

        # checkpoint log
        best_loss = float("inf")
        for epoch, metrics in enumerate(self.metrics):
            print(f"Epoch {epoch:03d}: "
                  f"train={metrics['train']:.8f} "
                  f"valid={metrics['valid']:.8f} "
                  f"best={metrics['best']:.4f} "
                  f"ms={metrics.get('true_model_size', 0):.2f}MB "
                  f"cms={metrics.get('compressed_model_size', 0):.2f}MB "
                  f"duration={human_seconds(metrics['duration'])}")
            best_loss = metrics['best']

        for epoch in range(len(self.metrics), self.config.epochs):
            begin = time.time()
            self.model["generator"].train()
            train_loss, model_size = self._train_epoch(epoch)
            self.model["generator"].eval()
            valid_loss = self._valid_epoch(epoch)

            # compressed model size
            ms = 0
            cms = 0
            if self.quantizer and self.config.device.rank == 0:
                ms = self.quantizer.true_model_size()
                cms = self.quantizer.compressed_model_size(
                    num_workers=min(40, self.config.device.world_size * 10))

            # calculate duration
            duration = time.time() - begin

            # renew best state
            if valid_loss < best_loss and ms <= self.config.ms_target:
                best_loss = valid_loss
                self.best_state = {
                    key: value.to("cpu").clone()
                    for key, value in self.model["generator"].module.state_dict().items()
                } if self.config.device.world_size > 1 else {
                    key: value.to("cpu").clone()
                    for key, value in self.model["generator"].state_dict().items()
                }

            self.metrics.append({
                "train": train_loss,
                "valid": valid_loss,
                "best": best_loss,
                "duration": duration,
                "model_size": model_size,
                "true_model_size": ms,
                "compressed_model_size": cms,
            })
            if self.config.device.rank == 0:
                self._check_log_interval(epoch)
                self._check_save_interval()

            print(f"Epoch {epoch:03d}: "
                  f"train={train_loss:.8f} valid={valid_loss:.8f} best={best_loss:.4f} ms={ms:.2f}MB "
                  f"cms={cms:.2f}MB "
                  f"duration={human_seconds(duration)}")

        # evaluate and save best model
        if self.config.device.world_size > 1:
            distributed.barrier()
            self.model["generator"].module.load_state_dict(self.best_state)
        else:
            self.model["generator"].load_state_dict(self.best_state)
        if self.config.device.eval_cpu:
            device = "cpu"
            self.model["generator"].to(device)
        self.model["generator"].eval()
        stat = self._eval_epoch()
        # eval_folder = self.outdir / "evals" / self.config.name
        # eval_folder.mkdir(exist_ok=True, parents=True)
        # evaluate(self.model["generator"], self.config.dataset.musdb.path, eval_folder,
        #          is_wav=self.config.dataset.musdb.is_wav,
        #          rank=self.config.device.rank,
        #          world_size=self.config.device.world_size,
        #          device=self.device,
        #          save=self.config.save,
        #          split=self.config.split_valid,
        #          shifts=self.config.dataset.shifts,
        #          overlap=self.config.dataset.overlap,
        #          workers=self.config.device.eval_workers)
        self.model["generator"].to("cpu")
        if self.config.device.rank == 0:
            save_model(self.model["generator"], self.quantizer, self.config,
                       self.outdir / "models" / f"{self.config.name}.th")

        mlflow.end_run()
        # self.writer.close()
        return stat

    def save_checkpoint(self, checkpoint_path):
        """Save checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be saved.

        """
        state_dict = {
            "run_id": self.run_id,
            "metrics": self.metrics,
            "best_state": self.best_state,
            "model": {
                "generator": self.model["generator"].module.state_dict() if self.config.device.world_size > 1 else self.model["generator"].state_dict(),
            },
            "optimizer": {
                "generator": self.optimizer["generator"].state_dict(),
            },
        }
        if self.config.loss.adversarial["lambda"]:
            state_dict["model"]["discriminator"] = self.model["discriminator"].module.state_dict(
            ) if self.config.device.world_size > 1 else self.model["discriminator"].state_dict()
            state_dict["optimizer"]["discriminator"] = self.optimizer["discriminator"].state_dict()

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
            self.model["generator"].module.load_state_dict(
                state_dict["model"]["generator"])
        else:
            self.model["generator"].load_state_dict(
                state_dict["model"]["generator"])
        if self.config.loss.adversarial["lambda"]:
            if self.config.device.world_size > 1:
                self.model["discriminator"].module.load_state_dict(
                    state_dict["model"]["discriminator"])
            else:
                self.model["discriminator"].load_state_dict(
                    state_dict["model"]["discriminator"])
        if not load_only_params:
            self.run_id = state_dict["run_id"]
            self.metrics = state_dict["metrics"]
            self.best_state = state_dict["best_state"]
            self.optimizer["generator"].load_state_dict(
                state_dict["optimizer"]["generator"])
            if self.config.loss.adversarial["lambda"]:
                self.optimizer["discriminator"].load_state_dict(
                    state_dict["optimizer"]["discriminator"])

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def log_params_from_omegaconf_dict(self, params):
        for param_name, element in params.items():
            self._explore_recursive(param_name, element)

    def _explore_recursive(self, parent_name, element):
        if parent_name in ignore_params:
            return
        if isinstance(element, DictConfig):
            for k, v in element.items():
                self._explore_recursive(f'{parent_name}.{k}', v)
        elif isinstance(element, ListConfig):
            for i, v in enumerate(element):
                self._explore_recursive(f'{parent_name}.{i}', v)
        else:
            # print(parent_name, "=", element)
            mlflow.log_param(parent_name, element)

    def _train_epoch(self, epoch):
        """Train model one epoch."""
        if self.config.device.world_size > 1:
            sampler_epoch = epoch * self.config.repeat
            if self.config.seed is not None:
                sampler_epoch += self.config.seed * 1000
            self.sampler["train"].set_epoch(sampler_epoch)
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

                #######################
                #      Generator      #
                #######################
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
                            estimates, sources[start::self.config.batch_divide])
                        l1_loss /= self.config.batch_divide
                        self.total_train_loss["train/l1_loss"] += l1_loss.item()
                        gen_loss += self.config.loss.l1["lambda"] * l1_loss
                        del l1_loss

                    # multi-resolution sfft loss
                    if self.config.loss.stft["lambda"]:
                        sc_loss, mag_loss = self.criterion["stft"](
                            estimates, sources[start::self.config.batch_divide])
                        sc_loss /= self.config.batch_divide
                        mag_loss /= self.config.batch_divide
                        self.total_train_loss["train/spectral_convergence_loss"] += sc_loss.item()
                        self.total_train_loss["train/log_stft_magnitude_loss"] += mag_loss.item()
                        gen_loss += self.config.loss.stft["lambda"] * (
                            sc_loss + mag_loss)
                        del sc_loss, mag_loss

                    # adversarial loss
                    if self.config.loss.adversarial["lambda"] and epoch > self.config.loss.adversarial.train_start_epoch:
                        self.set_requires_grad(
                            self.model["discriminator"], False)
                        p_ = self.model["discriminator"](estimates)
                        adv_loss = self.criterion["gen_adv"](p_)
                        self.total_train_loss["train/adversarial_loss"] += adv_loss.item()
                        gen_loss += self.config.loss.adversarial["lambda"] * adv_loss
                        del p_, adv_loss

                    self.total_train_loss["train/gen_loss"] += gen_loss.item()
                    gen_loss.backward()
                    del gen_loss, estimates

                # model size loss
                model_size = 0
                if self.quantizer is not None:
                    model_size = self.quantizer.model_size()
                    (self.config.diffq * model_size).backward()
                    model_size = model_size.item()

                # update generator
                g_grad_norm = 0
                for p in self.model["generator"].parameters():
                    if p.grad is not None:
                        g_grad_norm += p.grad.data.norm()**2
                g_grad_norm = g_grad_norm**0.5
                self.optimizer["generator"].step()
                self.optimizer["generator"].zero_grad()

                #######################
                #    Discriminator    #
                #######################
                if self.config.loss.adversarial["lambda"] and epoch > self.config.loss.adversarial.train_start_epoch:
                    for start in range(self.config.batch_divide):
                        with torch.no_grad():
                            estimates = self.model["generator"](
                                mix[start::self.config.batch_divide])

                        # initialize
                        dis_loss = 0.0

                        # discriminator loss
                        self.set_requires_grad(
                            self.model["discriminator"], True)
                        p = self.model["discriminator"](
                            sources[start::self.config.batch_divide])
                        p_ = self.model["discriminator"](estimates)
                        real_loss, fake_loss = self.criterion["dis_adv"](p_, p)
                        real_loss /= self.config.batch_divide
                        fake_loss /= self.config.batch_divide
                        self.total_train_loss["train/real_loss"] += real_loss.item()
                        self.total_train_loss["train/fake_loss"] += fake_loss.item()
                        dis_loss += real_loss + fake_loss
                        del real_loss, fake_loss

                        self.total_train_loss["train/discriminator_loss"] += dis_loss.item()
                        dis_loss.backward()
                        del dis_loss, estimates

                    # update discriminator
                    d_grad_norm = 0
                    for p in self.model["generator"].parameters():
                        if p.grad is not None:
                            d_grad_norm += p.grad.data.norm()**2
                    d_grad_norm = d_grad_norm**0.5
                    self.optimizer["discriminator"].step()
                    self.optimizer["discriminator"].zero_grad()

                # free some space before next round
                del sources, mix

                for k, v in self.total_train_loss.items():
                    self.total_train_loss[k] = v / (1 + idx)
                current_loss = self.total_train_loss["train/gen_loss"]
                tq.set_postfix(loss=f"{current_loss:.4f}", ms=f"{model_size:.2f}",
                               grad=f"{g_grad_norm:.5f}")

            if self.config.device.world_size > 1:
                self.sampler["train"].epoch += 1

        if self.config.device.world_size > 1:
            current_loss = average_metric(current_loss)
        return current_loss, model_size

    @torch.no_grad()
    def _valid_epoch(self, epoch):
        tq = tqdm(self.data_loader["valid"],
                  ncols=120,
                  desc=f"[{epoch:03d}] valid",
                  leave=False,
                  file=sys.stdout,
                  unit=" track")
        # reset
        self.total_valid_loss = defaultdict(float)
        model = self.model["generator"].module if self.config.device.world_size > 1 else self.model["generator"]
        for idx, streams in enumerate(tq):
            # first five minutes to avoid OOM on --upsample models
            streams = streams[0, ..., :15_000_000]
            streams = streams.to(self.device)
            sources = streams[1:]
            mix = streams[0]
            estimates = apply_model(model, mix, shifts=0,
                                    split=self.config.split_valid, overlap=self.config.dataset.overlap)

            # initialize
            gen_loss = 0.0

            # l1 loss
            if self.config.loss.l1["lambda"]:
                l1_loss = self.criterion["l1"](estimates, sources).item()
                gen_loss += self.config.loss.l1["lambda"] * l1_loss
                self.total_valid_loss["valid/l1_loss"] += l1_loss

            # multi-resolution sfft loss
            if self.config.loss.stft["lambda"]:
                toral_sc_loss, total_mag_loss = 0, 0
                for index in range(sources.size(0)):
                    sc_loss, mag_loss = self.criterion["stft"](
                        estimates[index], sources[index])
                    toral_sc_loss += sc_loss.item()
                    total_mag_loss += mag_loss.item()
                    del sc_loss, mag_loss
                toral_sc_loss /= sources.size(0)
                total_mag_loss /= sources.size(0)
                gen_loss += self.config.loss.stft["lambda"] * \
                    (toral_sc_loss + total_mag_loss)
                self.total_valid_loss["valid/spectral_convergence_loss"] += toral_sc_loss
                self.total_valid_loss["valid/log_stft_magnitude_loss"] += total_mag_loss

            self.total_valid_loss["valid/gen_loss"] += gen_loss
            del estimates, streams, sources

        for k, v in self.total_valid_loss.items():
            self.total_valid_loss[k] = v / (1 + idx)
        current_loss = self.total_valid_loss["valid/gen_loss"]
        if self.config.device.world_size > 1:
            current_loss = average_metric(current_loss)
        return current_loss


    def _check_save_interval(self):
        # save to file
        log_folder = self.outdir / self.config.outdir.logs
        metrics_path = log_folder / f"{self.config.name}.json"
        json.dump(self.metrics, open(metrics_path, "w"))
        checkpoint_folder = self.outdir / self.config.outdir.checkpoints
        checkpoint_folder.mkdir(exist_ok=True, parents=True)
        checkpoint_path = checkpoint_folder / f"{self.config.name}.th"
        checkpoint_tmp_path = checkpoint_folder / f"{self.config.name}.th.tmp"
        self.save_checkpoint(checkpoint_tmp_path)
        checkpoint_tmp_path.rename(checkpoint_path)


    def _check_log_interval(self, epoch):
        # write logs
        mlflow.log_metrics(self.total_train_loss, epoch)
        mlflow.log_metrics(self.total_valid_loss, epoch)
        # self.writer.add_scalars('train_loss', self.total_train_loss, epoch)
        # self.writer.add_scalars('valid_loss', self.total_valid_loss, epoch)

        # reset
        self.total_train_loss = defaultdict(float)
        self.total_valid_loss = defaultdict(float)
