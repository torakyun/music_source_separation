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
import librosa
import matplotlib.pyplot as plt

import numpy as np
from collections import defaultdict

# from tensorboardX import SummaryWriter
import mlflow
from omegaconf import DictConfig, ListConfig

from .audio import convert_audio
from .apply import apply_model
from .utils import gpulife, human_seconds, save_model, average_metric, center_trim

import torch
from torch import distributed
from tqdm import tqdm
from distutils.version import LooseVersion
is_pytorch_17plus = LooseVersion(torch.__version__) >= LooseVersion("1.7")


show_names = {
    "pretrained": "pretrained",
    "epochs": "epochs",
    "dataset.samplerate": "sr",
    "dataset.audio_channels": "ch",
    "loss.l1.lambda": "l1",
    "loss.mag.lambda": "mag",
    "loss.stft.lambda": "stft",
    "loss.mel.lambda": "mel",
    "loss.adversarial.lambda": "adv",
    "loss.feat_match.lambda": "fm",
    "model/generator": "gen",
    "model/discriminator": "dis",
    "model.discriminator.separate": "sep",
}


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
        self.outdir = Path(config.out)
        # self.writer = SummaryWriter(self.outdir / "tensorboard" / f"{self.config.name}")
        self.train_loss = defaultdict(float)
        self.valid_loss = defaultdict(float)
        self.eval_loss = defaultdict(float)
        self.fig = plt.figure(constrained_layout=True, figsize=(20, 15))
        self.axes = self.fig.subplots(
            nrows=len(self.config.dataset.sources), ncols=3, sharex=False)

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

            # train and valid
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

            # save metrics
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
                self._check_eval_interval(epoch)
                self._check_save_interval(epoch)
                self._check_log_interval(epoch)

            # logging
            print(f"Epoch {epoch:03d}: "
                  f"train={train_loss:.8f} valid={valid_loss:.8f} best={best_loss:.4f} ms={ms:.2f}MB "
                  f"cms={cms:.2f}MB "
                  f"duration={human_seconds(duration)}")

        # evaluate and save best model
        if self.config.device.world_size > 1:
            distributed.barrier()
        model, stat = self.evaluate()
        model.to("cpu")
        if self.config.device.rank == 0:
            save_model(model, self.quantizer, self.config,
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
        if self.config.loss.adversarial["lambda"] and "discriminator" in state_dict["model"].keys():
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
            if self.config.loss.adversarial["lambda"] and "discriminator" in state_dict["optimizer"].keys():
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
        if isinstance(element, DictConfig):
            for k, v in element.items():
                self._explore_recursive(f'{parent_name}.{k}', v)
        elif isinstance(element, ListConfig):
            for i, v in enumerate(element):
                self._explore_recursive(f'{parent_name}.{i}', v)
        else:
            if parent_name in show_names.keys():
                mlflow.log_param(show_names[parent_name], element)

    def write_figure(self, title, r, e, d, dir, xmax=None):
        sources = ["drums", "bass", "other", "vocals"]
        targets = ["reference", "estimate", "difference"]
        # fig.suptitle(title, fontsize='xx-large')
        for i, source in enumerate(sources):
            # set label
            for j, target in enumerate(targets):
                # if i == 0:
                #     axes[i, j].text(0.5, 1, target)
                # axes[i, j].text(-0.1, 0.5, source)
                self.axes[i, j].set_xlabel("frame")
                self.axes[i, j].set_ylabel("freq_bin")
                if xmax:
                    self.axes[i, j].set_xlim((0, xmax))

            # draw image
            vmin = torch.min(torch.stack([r[i], e[i], d[i]]))
            vmax = torch.max(torch.stack([r[i], e[i], d[i]]))
            im = self.axes[i, 0].imshow(
                r[i].cpu(), origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
            im = self.axes[i, 1].imshow(
                e[i].cpu(), origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
            im = self.axes[i, 2].imshow(
                d[i].cpu(), origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
            self.fig.colorbar(im, ax=self.axes[i, 2])

        # save figure
        title = title.replace(" ", "_")
        path = Path(dir) / f"{title}.png"
        self.fig.savefig(path)
        mlflow.log_figure(self.fig, f"figure/{title}.png")

        # remove colorbar and clear axis
        for i, _ in enumerate(sources):
            self.axes[i, 2].images[-1].colorbar.remove()
        plt.cla()

    def log_and_save_spectrogram(self, references, estimates, dir, stft_params=None, mel_params=None, window="hann"):
        # Spectrogram params
        stft_params = {
            "n_fft": 2048,
            "hop_length": 240,
            "win_length": 1200,
            "center": True,
            "normalized": False,
            "onesided": True,
        } if not stft_params else stft_params
        if is_pytorch_17plus:
            stft_params["return_complex"] = False
        mel_params = {
            "sr": self.config.dataset.samplerate,
            "n_fft": stft_params["n_fft"],
            "n_mels": 80,
            "fmin": 0,
            "fmax": self.config.dataset.samplerate // 2,
        } if not mel_params else mel_params
        if window is not None:
            window_func = getattr(torch, f"{window}_window")
            stft_params["window"] = window_func(
                stft_params["win_length"], dtype=references.dtype, device=references.device)
        else:
            stft_params["window"] = None

        # calculate STFT
        references_stft = torch.stft(references, **stft_params)
        estimates_stft = torch.stft(estimates, **stft_params)
        del references, estimates
        differences_stft = references_stft - estimates_stft
        differences_stft = torch.sqrt(torch.clamp(
            differences_stft[..., 0]**2 + differences_stft[..., 1]**2, min=1e-7))
        references_mag = torch.sqrt(torch.clamp(
            references_stft[..., 0]**2 + references_stft[..., 1]**2, min=1e-7))
        estimates_mag = torch.sqrt(torch.clamp(
            estimates_stft[..., 0]**2 + estimates_stft[..., 1]**2, min=1e-7))
        del references_stft, estimates_stft

        # STFT spectrogram
        self.write_figure(
            "STFT", references_mag, estimates_mag, differences_stft, dir)
        del differences_stft

        # Magnitude spectrogram
        differences_mag = (references_mag - estimates_mag).abs()
        self.write_figure(
            "Magnitude Spectrogram", references_mag, estimates_mag, differences_mag, dir)
        del differences_mag

        # Log-scale Magnitude spectrogram
        references_log_mag = torch.log(references_mag)
        estimates_log_mag = torch.log(estimates_mag)
        differences_log_mag = (references_log_mag - estimates_log_mag).abs()
        self.write_figure(
            "Log-Scale Magnitude Spectrogram", references_log_mag, estimates_log_mag, differences_log_mag, dir)
        del references_log_mag, estimates_log_mag, differences_log_mag

        # Mel spectrogram
        melmat = librosa.filters.mel(**mel_params)
        melmat = torch.from_numpy(melmat).to(references_mag.device).double()
        references_mel = torch.clamp(torch.matmul(
            melmat, references_mag), min=1e-7)
        estimates_mel = torch.clamp(torch.matmul(
            melmat, estimates_mag), min=1e-7)
        differences_mel = (references_mel - estimates_mel).abs()
        self.write_figure(
            "Mel Spectrogram", references_mel, estimates_mel, differences_mel, dir)
        del melmat, differences_mel

        # Log-scale Mel spectrogram
        references_log_mel = torch.log(references_mel)
        estimates_log_mel = torch.log(estimates_mel)
        differences_log_mel = (references_log_mel - estimates_log_mel).abs()
        self.write_figure(
            "Log-Scale Mel Spectrogram", references_log_mel, estimates_log_mel, differences_log_mel, dir)

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
            self.train_loss = defaultdict(float)
            for idx, sources in enumerate(tq):
                # if idx > 0:
                #     break
                if len(sources) < self.config.batch_size // self.config.device.world_size:
                    # skip uncomplete batch for augment.Remix to work properly
                    continue
                sources = sources.to(self.device)
                sources = self.augment(sources)
                mix = sources.sum(dim=1)

                for start in range(self.config.batch_divide):
                    #######################
                    #      Generator      #
                    #######################
                    # gpulife("start")
                    # start_t = time.time()
                    estimates = self.model["generator"](
                        mix[start::self.config.batch_divide])
                    # gpulife("gen_forward")
                    # print("gen_forward: ", time.time() - start_t)
                    # start_t = time.time()

                    # initialize
                    gen_loss = 0.0

                    # l1 loss
                    if self.config.loss.l1["lambda"]:
                        l1_loss = self.criterion["l1"](
                            estimates, sources[start::self.config.batch_divide])
                        l1_loss /= self.config.batch_divide
                        self.train_loss["train/l1_loss"] += l1_loss.item()
                        gen_loss += self.config.loss.l1["lambda"] * l1_loss
                        del l1_loss
                    # print("l1_loss: ", time.time() - start_t)
                    # gpulife("l1_loss")
                    # start_t = time.time()

                    # multi-resolution magnitude loss
                    if self.config.loss.mag["lambda"]:
                        mag_loss, log_mag_loss = self.criterion["mag"](
                            estimates, sources[start::self.config.batch_divide])
                        mag_loss /= self.config.batch_divide
                        log_mag_loss /= self.config.batch_divide
                        self.train_loss["train/magnitude_spectrogram_loss"] += mag_loss.item()
                        self.train_loss["train/log_magnitude_spectrogram_loss"] += log_mag_loss.item()
                        gen_loss += self.config.loss.mag["lambda"] * (
                            mag_loss + log_mag_loss)
                        del mag_loss, log_mag_loss
                    # print("mag_loss: ", time.time() - start_t)
                    # gpulife("mag_loss")
                    # start_t = time.time()

                    # multi-resolution sfft loss
                    if self.config.loss.stft["lambda"]:
                        stft_loss, log_stft_loss = self.criterion["stft"](
                            estimates, sources[start::self.config.batch_divide])
                        stft_loss /= self.config.batch_divide
                        log_stft_loss /= self.config.batch_divide
                        self.train_loss["train/stft_loss"] += stft_loss.item()
                        self.train_loss["train/log_stft_loss"] += log_stft_loss.item()
                        gen_loss += self.config.loss.stft["lambda"] * (
                            stft_loss + log_stft_loss)
                        del stft_loss, log_stft_loss
                    # print("stft_loss: ", time.time() - start_t)
                    # gpulife("stft_loss")
                    # start_t = time.time()

                    # mel spectrogram loss
                    if self.config.loss.mel["lambda"]:
                        mel_loss = self.criterion["mel"](
                            estimates, sources[start::self.config.batch_divide])
                        mel_loss /= self.config.batch_divide
                        self.train_loss["train/mel_spectrogram_loss"] += mel_loss.item()
                        gen_loss += self.config.loss.mel["lambda"] * mel_loss
                        del mel_loss
                    # print("mel_loss: ", time.time() - start_t)
                    # gpulife("mel_loss")
                    # start_t = time.time()

                    self.train_loss["train/gen_loss"] += gen_loss.item()

                    # adversarial loss
                    if self.config.loss.adversarial["lambda"] and epoch > self.config.loss.adversarial.train_start_epoch:
                        # no need to track gradients
                        self.set_requires_grad(
                            self.model["discriminator"], False)
                        p_ = self.model["discriminator"](estimates)
                        adv_loss = self.criterion["gen_adv"](p_)
                        adv_loss /= self.config.batch_divide
                        self.train_loss["train/adversarial_loss"] += adv_loss.item()
                        gen_loss += self.config.loss.adversarial["lambda"] * adv_loss
                        del adv_loss
                        # print("adv_loss: ", time.time() - start_t)
                        # gpulife("adv_loss")
                        # start_t = time.time()

                        # feature matching loss
                        if self.config.loss.feat_match["lambda"]:
                            p = self.model["discriminator"](
                                sources[start::self.config.batch_divide])
                            fm_loss = self.criterion["feat_match"](p_, p)
                            fm_loss /= self.config.batch_divide
                            self.train_loss["train/feature_matching_loss"] += fm_loss.item()
                            gen_loss += self.config.loss.feat_match["lambda"] * fm_loss
                            del p, fm_loss
                            # print("fm_loss: ", time.time() - start_t)
                            # gpulife("fm_loss")
                            # start_t = time.time()

                        del p_

                    gen_loss.backward()
                    del gen_loss
                    # print("gen_loss.backward(): ", time.time() - start_t)
                    # gpulife("gen_loss.backward()")
                    # start_t = time.time()

                    #######################
                    #    Discriminator    #
                    #######################
                    if self.config.loss.adversarial["lambda"] and epoch > self.config.loss.adversarial.train_start_epoch:
                        # discriminator loss
                        self.set_requires_grad(
                            self.model["discriminator"], True)
                        real_loss, fake_loss = self.criterion["dis_adv"](
                            self.model["discriminator"](estimates.detach()),
                            self.model["discriminator"](sources[start::self.config.batch_divide]))
                        real_loss /= self.config.batch_divide
                        fake_loss /= self.config.batch_divide
                        self.train_loss["train/real_loss"] += real_loss.item()
                        self.train_loss["train/fake_loss"] += fake_loss.item()
                        self.train_loss["train/discriminator_loss"] = self.train_loss["train/real_loss"] + \
                            self.train_loss["train/fake_loss"]
                        (real_loss + fake_loss).backward()
                        del real_loss, fake_loss
                        # print("dis_backward()", time.time() - start_t)
                        # gpulife("dis_backward()")
                        # start_t = time.time()

                    del estimates

                # free some space before next round
                del sources, mix

                # model size loss
                model_size = 0
                if self.quantizer is not None:
                    model_size = self.quantizer.model_size()
                    (self.config.diffq * model_size).backward()
                    model_size = model_size.item()

                # update generator
                g_grad_norm = 0
                if self.config.model.generator.grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model["generator"].parameters(),
                        self.config.model.generator.grad_norm,
                    )
                for p in self.model["generator"].parameters():
                    if p.grad is not None:
                        g_grad_norm += p.grad.data.norm()**2
                g_grad_norm = g_grad_norm**0.5
                self.optimizer["generator"].step()
                self.optimizer["generator"].zero_grad()
                # print("gen_opt: ", time.time() - start_t)
                # gpulife("gen_optimized")
                # start_t = time.time()

                # update discriminator
                if self.config.loss.adversarial["lambda"] and epoch > self.config.loss.adversarial.train_start_epoch:
                    d_grad_norm = 0
                    if self.config.model.discriminator.grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model["discriminator"].parameters(),
                            self.config.model.discriminator.grad_norm,
                        )
                    for p in self.model["discriminator"].parameters():
                        if p.grad is not None:
                            d_grad_norm += p.grad.data.norm()**2
                    d_grad_norm = d_grad_norm**0.5
                    self.optimizer["discriminator"].step()
                    self.optimizer["discriminator"].zero_grad()
                    # print("dis_opt", time.time() - start_t)
                    # gpulife("dis_opt")
                    # start_t = time.time()

                current_loss = self.train_loss["train/gen_loss"] / (
                    1 + idx)
                tq.set_postfix(loss=f"{current_loss:.4f}", ms=f"{model_size:.2f}",
                               grad=f"{g_grad_norm:.5f}")

            for k, v in self.train_loss.items():
                self.train_loss[k] = v / (1 + idx)

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
        self.valid_loss = defaultdict(float)
        model = self.model["generator"].module if self.config.device.world_size > 1 else self.model["generator"]
        for idx, streams in enumerate(tq):
            # first five minutes to avoid OOM on --upsample models
            streams = streams[0, ..., :15_000_000]
            streams = streams.to(self.device)
            sources = streams[1:]
            mix = streams[0]
            estimates = apply_model(
                model, mix[None], split=self.config.split_valid, overlap=0)[0]

            # initialize
            gen_loss = 0.0

            # l1 loss
            if self.config.loss.l1["lambda"]:
                l1_loss = self.criterion["l1"](estimates, sources).item()
                self.valid_loss["valid/l1_loss"] += l1_loss
                gen_loss += self.config.loss.l1["lambda"] * l1_loss

            # multi-resolution magnitude loss
            if self.config.loss.mag["lambda"]:
                total_mag_loss, total_log_mag_loss = 0, 0
                for index in range(sources.size(0)):
                    mag_loss, log_mag_loss = self.criterion["mag"](
                        estimates[index], sources[index])
                    total_mag_loss += mag_loss.item()
                    total_log_mag_loss += log_mag_loss.item()
                mag_loss = total_mag_loss / (index + 1)
                log_mag_loss = total_log_mag_loss / (index + 1)
                self.valid_loss["valid/magnitude_spectrogram_loss"] += mag_loss
                self.valid_loss["valid/log_magnitude_spectrogram_loss"] += log_mag_loss
                gen_loss += self.config.loss.mag["lambda"] * \
                    ((mag_loss + log_mag_loss))

            # multi-resolution sfft loss
            if self.config.loss.stft["lambda"]:
                total_stft_loss, total_log_stft_loss = 0, 0
                for index in range(sources.size(0)):
                    stft_loss, log_stft_loss = self.criterion["stft"](
                        estimates[index], sources[index])
                    total_stft_loss += stft_loss.item()
                    total_log_stft_loss += log_stft_loss.item()
                stft_loss = total_stft_loss / (index + 1)
                log_stft_loss = total_log_stft_loss / (index + 1)
                self.valid_loss["valid/stft_loss"] += stft_loss
                self.valid_loss["valid/log_stft_loss"] += log_stft_loss
                gen_loss += self.config.loss.stft["lambda"] * \
                    ((stft_loss + log_stft_loss))

            # mel spectrogram loss
            if self.config.loss.mel["lambda"]:
                total_mel_loss = 0
                for index in range(sources.size(0)):
                    total_mel_loss += self.criterion["mel"](
                        estimates[index], sources[index]).item()
                mel_loss = total_mel_loss / (index + 1)
                self.valid_loss["valid/mel_spectrogram_loss"] += mel_loss
                gen_loss += self.config.loss.mel["lambda"] * mel_loss

            self.valid_loss["valid/gen_loss"] += gen_loss

            # adversarial loss
            if self.config.loss.adversarial["lambda"] and epoch > self.config.loss.adversarial.train_start_epoch:
                total_adversarial_loss, total_real_loss, total_fake_loss, total_fm_loss = 0, 0, 0, 0
                length = self.config.adversarial_valid_length
                for index, start in enumerate(range(0, sources.size(-1)-1, length)):
                    p_ = self.model["discriminator"](
                        estimates[..., start:start+length])
                    p = self.model["discriminator"](
                        sources[..., start:start+length])
                    total_adversarial_loss += self.criterion["gen_adv"](
                        p_).item()
                    real_loss, fake_loss = self.criterion["dis_adv"](p_, p)
                    total_real_loss += real_loss.item()
                    total_fake_loss += fake_loss.item()
                    del real_loss, fake_loss

                    # feature matching loss
                    if self.config.loss.feat_match["lambda"]:
                        total_fm_loss += self.criterion["feat_match"](
                            p_, p).item()

                    del p_, p
                adversarial_loss = total_adversarial_loss / (index + 1)
                real_loss = total_real_loss / (index + 1)
                fake_loss = total_fake_loss / (index + 1)
                if self.config.loss.feat_match["lambda"]:
                    fm_loss = total_fm_loss / (index + 1)

                self.valid_loss["valid/adversarial_loss"] += adversarial_loss
                self.valid_loss["valid/real_loss"] += real_loss
                self.valid_loss["valid/fake_loss"] += fake_loss
                self.valid_loss["valid/discriminator_loss"] += (
                    real_loss + fake_loss)
                if self.config.loss.feat_match["lambda"]:
                    self.valid_loss["valid/feature_matching_loss"] += fm_loss

            del estimates, streams, sources

        for k, v in self.valid_loss.items():
            self.valid_loss[k] = v / (1 + idx)
        current_loss = self.valid_loss["valid/gen_loss"]
        if self.config.device.world_size > 1:
            current_loss = average_metric(current_loss)
        return current_loss

    @torch.no_grad()
    def _eval_epoch(self, epoch=0):
        # make eval track1
        data_dir = self.outdir / "eval_data"
        data_dir.mkdir(exist_ok=True, parents=True)
        data_file = f"{self.config.dataset.samplerate}_{self.config.dataset.audio_channels}"
        for name in self.config.dataset.sources:
            data_file += f"_{name}"
        data_file += ".th"
        if (data_dir / data_file).is_file():
            track = torch.load(data_dir / data_file, map_location="cpu")
        else:
            test_set = musdb.DB(
                self.config.dataset.musdb.path, subsets=["test"])
            from_samplerate = 44100
            track = test_set.tracks[1]
            mix = torch.from_numpy(track.audio).t().float()
            ref = mix.mean(dim=0)  # mono mixture
            mix = (mix - ref.mean()) / ref.std()
            mix = convert_audio(mix, from_samplerate,
                                self.config.dataset.samplerate, self.config.dataset.audio_channels)
            references = torch.stack(
                [torch.from_numpy(track.targets[name].audio).t() for name in self.config.dataset.sources])
            references = convert_audio(
                references, from_samplerate, self.config.dataset.samplerate, self.config.dataset.audio_channels)
            track = {
                "name": track.name,
                "mix": mix, "mean": ref.mean(),
                "std": ref.std(),
                "targets": references
            }
            torch.save(track, data_dir / data_file)
        else:
            track = torch.load(data_dir / data_file, map_location="cpu")

        """Evaluate model one epoch."""
        eval_folder = self.outdir / "evals" / f"{self.config.name}_{epoch}"
        eval_folder.mkdir(exist_ok=True, parents=True)

        estimates = apply_model(
            self.model["generator"].module if self.config.device.world_size > 1 else self.model["generator"],
            track["mix"][None].to(self.device),
            shifts=self.config.shifts,
            split=self.config.split_valid,
            overlap=self.config.overlap
        )[0]
        estimates = estimates * track["std"] + track["mean"]

        # save spectrogram
        references = track["targets"].to(self.device)
        second = self.config.dataset.samplerate * self.config.eval_second
        self.log_and_save_spectrogram(
            references.mean(dim=1)[..., :second],
            estimates.mean(dim=1)[..., :second],
            eval_folder
        )

        estimates = estimates.transpose(1, 2).cpu().numpy()
        references = references.transpose(1, 2).cpu().numpy()

        # save wav
        track_folder = eval_folder / track["name"]
        track_folder.mkdir(exist_ok=True, parents=True)
        for name, estimate in zip(self.config.dataset.sources, estimates):
            # wavfile.write(
            #     str(track_folder / (f"{name}_epoch{epoch}.wav")), self.config.dataset.samplerate, estimate)
            wavfile.write(
                str(track_folder / (name + ".wav")), self.config.dataset.samplerate, estimate)
            # mlflow.log_artifact(
            #     str(track_folder / (name + ".wav")), "wav")
            # self.writer.add_audio(name, torch.from_numpy(estimate), epoch)

        # cal SDR
        win = int(1. * self.config.dataset.samplerate)
        hop = int(1. * self.config.dataset.samplerate)
        sdr, isr, sir, sar = museval.evaluate(
            references[:, :second, :], estimates[:, :second, :], win=win, hop=hop)
        for idx, source in enumerate(self.config.dataset.sources):
            self.eval_loss[f"eval/sdr_{source}"] = np.nanmedian(
                sdr[idx].tolist())
        self.eval_loss["eval/sdr_all"] = np.array(
            list(self.eval_loss.values())).mean()
        json.dump(self.eval_loss, open(eval_folder / "sdr.json", "w"))

    @torch.no_grad()
    def evaluate(self):
        """Evaluate model one epoch."""
        eval_folder = self.outdir / "evals" / self.config.name
        eval_folder.mkdir(exist_ok=True, parents=True)

        # we load tracks from the original musdb set
        test_set = musdb.DB(self.config.dataset.musdb.path, subsets=[
                            "test"], is_wav=self.config.dataset.musdb.is_wav)
        track_indexes = list(range(len(test_set)))
        src_rate = 44100  # hardcoded for now...
        all_metrics = defaultdict(list)
        model = self.model["generator"].module if self.config.device.world_size > 1 else self.model["generator"]
        del self.model
        model.load_state_dict(self.best_state)
        if self.config.device.eval_cpu:
            device = "cpu"
            model.to(device)
        model.eval()
        for index in tqdm(range(self.config.device.rank, len(track_indexes), self.config.device.world_size), file=sys.stdout):
            track = test_set.tracks[track_indexes[index]]

            mix = torch.from_numpy(track.audio).t().float()
            mix = mix.to(self.device)
            ref = mix.mean(dim=0)  # mono mixture
            mix = (mix - ref.mean()) / ref.std()
            mix = convert_audio(
                mix, src_rate, model.samplerate, model.audio_channels)
            estimates = apply_model(
                model, mix[None],
                shifts=self.config.shifts,
                split=self.config.split_valid,
                overlap=self.config.overlap)[0]
            estimates = estimates * ref.std() + ref.mean()
            references = torch.stack(
                [torch.from_numpy(track.targets[name].audio).t() for name in model.sources])
            references = convert_audio(
                references, src_rate, model.samplerate, model.audio_channels)

            # save spectrogram
            if track_indexes[index] == 1:
                references = references.to(self.device)
                second = self.config.dataset.samplerate * self.config.eval_second
                self.log_and_save_spectrogram(
                    references.mean(dim=1)[..., :second],
                    estimates.mean(dim=1)[..., :second],
                    eval_folder
                )

            references = references.transpose(1, 2).cpu().numpy()
            estimates = estimates.transpose(1, 2).cpu().numpy()

            # save wav
            if track_indexes[index] == 1:
                track_folder = eval_folder / track.name
                track_folder.mkdir(exist_ok=True, parents=True)
                for name, estimate in zip(model.sources, estimates):
                    wavfile.write(
                        str(track_folder / (name + ".wav")), self.config.dataset.samplerate, estimate)
                    # mlflow.log_artifact(
                    #     str(track_folder / (name + ".wav")), "wav")
                    # self.writer.add_audio(name, torch.from_numpy(estimate), epoch)
            # cal SDR
            win = int(1. * model.samplerate)
            hop = int(1. * model.samplerate)
            sdr, isr, sir, sar = museval.evaluate(
                references, estimates, win=win, hop=hop)
            for idx, source in enumerate(model.sources):
                all_metrics[source].append(np.nanmedian(sdr[idx].tolist()))
        json.dump(all_metrics, open(eval_folder /
                                    f"{self.config.device.rank}.json", "w"))
        if self.config.device.world_size > 1:
            distributed.barrier()

        stat = defaultdict(list)
        if self.config.device.rank == 0:
            for rank in range(self.config.device.world_size):
                eval_file = eval_folder / f"{rank}.json"
                parts = json.load(open(eval_file))
                for source, sdr in parts.items():
                    stat[source] += sdr
                eval_file.unlink()
            stat = {source: np.nanmedian(sdr) for source, sdr in stat.items()}
            stat["all"] = np.array(list(stat.values())).mean()
            json.dump(stat, open(eval_folder / "sdr.json", "w"))
            mlflow.log_artifact(str(eval_folder / "sdr.json"))
        return model, stat

    def _check_save_interval(self, epoch):
        # save to file
        log_folder = self.outdir / "logs"
        metrics_path = log_folder / f"{self.config.name}.json"
        json.dump(self.metrics, open(metrics_path, "w"))
        checkpoint_folder = self.outdir / "checkpoints"
        checkpoint_folder.mkdir(exist_ok=True, parents=True)
        checkpoint_path = checkpoint_folder / f"{self.config.name}.th"
        checkpoint_tmp_path = checkpoint_folder / f"{self.config.name}.th.tmp"
        self.save_checkpoint(checkpoint_tmp_path)
        checkpoint_tmp_path.rename(checkpoint_path)
        # interval checkpoint
        if epoch and self.config.save_interval and epoch % self.config.save_interval == 0:
            checkpoint_path = checkpoint_folder / \
                f"{self.config.name}_{epoch}.th"
            self.save_checkpoint(checkpoint_path)

    def _check_eval_interval(self, epoch):
        if epoch and self.config.eval_interval and epoch % self.config.eval_interval == 0:
            # if epoch > 0 and self.metrics[-2]["best"] > self.metrics[-1]["best"]:
            self._eval_epoch(epoch)

    def _check_log_interval(self, epoch):
        # write logs
        mlflow.log_metrics(self.train_loss, epoch)
        mlflow.log_metrics(self.valid_loss, epoch)
        mlflow.log_metrics(self.eval_loss, epoch)
        # self.writer.add_scalars('train_loss', self.train_loss, epoch)
        # self.writer.add_scalars('valid_loss', self.valid_loss, epoch)
        # self.writer.add_scalars('eval_loss', self.eval_loss, epoch)
