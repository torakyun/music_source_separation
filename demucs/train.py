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
from .utils import human_seconds, apply_model, save_model, average_metric, center_trim

import torch
from torch import distributed
from tqdm import tqdm
from distutils.version import LooseVersion
is_pytorch_17plus = LooseVersion(torch.__version__) >= LooseVersion("1.7")


ignore_params = ["restart", "split_valid", "show", "save", "save_model", "save_state", "half", "eval_interval", "eval_second", "eval_epoch_path", "out",
                 "q-min-size", "qat", "diffq", "ms_target", "mlflow", "outdir", "device", "dataset", "name",  "model.generator.params", "model.discriminator.params", "loss.stft.params", "loss.adversarial.generator_params", "loss.adversarial.discriminator_params"]


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
        self.train_loss = defaultdict(float)
        self.valid_loss = defaultdict(float)
        self.eval_loss = defaultdict(float)

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
                self._check_save_interval()
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

    def log_and_save_spectrogram(self, references, estimates, dir, stft_params=None, mel_params=None, window="hann"):
        def log_and_save_spectrogram(title, r, e, d, dir, xmax=None, sources=["drums", "bass", "other", "vocals"], targets=["reference", "estimate", "difference"]):
            fig = plt.figure(constrained_layout=True, figsize=(20, 15))
            # fig.suptitle(title, fontsize='xx-large')
            axes = fig.subplots(
                nrows=len(sources), ncols=len(targets), sharex=False)
            for i, source in enumerate(sources):
                for j, target in enumerate(targets):
                    # if i == 0:
                    #     axes[i, j].text(0.5, 1, target)
                    axes[i, j].set_xlabel("frame")
                    axes[i, j].set_ylabel("freq_bin")
                    if j == 0:
                        # axes[i, j].text(-0.1, 0.5, source)
                        im = axes[i, j].imshow(
                            r[i].cpu(), origin="lower", aspect="auto")
                    elif j == 1:
                        im = axes[i, j].imshow(
                            e[i].cpu(), origin="lower", aspect="auto")
                    elif j == 2:
                        im = axes[i, j].imshow(
                            d[i].cpu(), origin="lower", aspect="auto")
                if xmax:
                    axes[i, j].set_xlim((0, xmax))
                fig.colorbar(im, ax=axes[i, j])
            title = title.replace(" ", "_")
            path = Path(dir) / f"{title}.png"
            fig.savefig(path)
            mlflow.log_figure(fig, f"figure/{title}.png")

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
            "sr": 44100,
            "n_fft": stft_params["n_fft"],
            "n_mels": 80,
            "fmin": 0,
            "fmax": 22050,
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
        log_and_save_spectrogram(
            "STFT", references_mag, estimates_mag, differences_stft, dir)
        del differences_stft

        # Magnitude spectrogram
        differences_mag = (references_mag - estimates_mag).abs()
        log_and_save_spectrogram(
            "Magnitude Spectrogram", references_mag, estimates_mag, differences_mag, dir)
        del differences_mag

        # Log-scale Magnitude spectrogram
        references_log_mag = torch.log(references_mag)
        estimates_log_mag = torch.log(estimates_mag)
        differences_log_mag = (references_log_mag - estimates_log_mag).abs()
        log_and_save_spectrogram(
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
        log_and_save_spectrogram(
            "Mel Spectrogram", references_mel, estimates_mel, differences_mel, dir)
        del melmat, differences_mel

        # Log-scale Mel spectrogram
        references_log_mel = torch.log(references_mel)
        estimates_log_mel = torch.log(estimates_mel)
        differences_log_mel = (references_log_mel - estimates_log_mel).abs()
        log_and_save_spectrogram(
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

                #######################
                #      Generator      #
                #######################
                for start in range(self.config.batch_divide):
                    estimates = self.model["generator"](
                        mix[start::self.config.batch_divide])
                    if start == 0:
                        sources = center_trim(sources, estimates)

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

                    # multi-resolution sfft loss
                    if self.config.loss.stft["lambda"]:
                        sc_loss, mag_loss = self.criterion["stft"](
                            estimates, sources[start::self.config.batch_divide])
                        sc_loss /= self.config.batch_divide
                        mag_loss /= self.config.batch_divide
                        self.train_loss["train/spectral_convergence_loss"] += sc_loss.item()
                        self.train_loss["train/log_stft_magnitude_loss"] += mag_loss.item()
                        gen_loss += self.config.loss.stft["lambda"] * (
                            sc_loss + mag_loss)
                        del sc_loss, mag_loss

                    # mel spectrogram loss
                    if self.config.loss.mel["lambda"]:
                        mel_loss = self.criterion["mel"](
                            estimates, sources[start::self.config.batch_divide])
                        mel_loss /= self.config.batch_divide
                        self.train_loss["train/mel_spectrogram_loss"] += mel_loss.item()
                        gen_loss += self.config.loss.mel["lambda"] * mel_loss
                        del mel_loss

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

                        # feature matching loss
                        if self.config.loss.feat_match["lambda"]:
                            p = self.model["discriminator"](
                                sources[start::self.config.batch_divide])
                            fm_loss = self.criterion["feat_match"](p_, p)
                            fm_loss /= self.config.batch_divide
                            self.train_loss["train/feature_matching_loss"] += fm_loss.item()
                            gen_loss += self.config.loss.feat_match["lambda"] * fm_loss
                            del p, fm_loss

                        del p_

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

                #######################
                #    Discriminator    #
                #######################
                if self.config.loss.adversarial["lambda"] and epoch > self.config.loss.adversarial.train_start_epoch:
                    for start in range(self.config.batch_divide):
                        with torch.no_grad():
                            estimates = self.model["generator"](
                                mix[start::self.config.batch_divide])

                        # discriminator loss
                        self.set_requires_grad(
                            self.model["discriminator"], True)
                        p = self.model["discriminator"](
                            sources[start::self.config.batch_divide])
                        p_ = self.model["discriminator"](estimates)
                        real_loss, fake_loss = self.criterion["dis_adv"](p_, p)
                        real_loss /= self.config.batch_divide
                        fake_loss /= self.config.batch_divide
                        self.train_loss["train/real_loss"] += real_loss.item()
                        self.train_loss["train/fake_loss"] += fake_loss.item()
                        self.train_loss["train/discriminator_loss"] = self.train_loss["train/real_loss"] + \
                            self.train_loss["train/fake_loss"]
                        (real_loss + fake_loss).backward()
                        del real_loss, fake_loss, estimates

                    # update discriminator
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

                # free some space before next round
                del sources, mix

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
            estimates = apply_model(model, mix, shifts=0,
                                    split=self.config.split_valid, overlap=self.config.dataset.overlap)

            # initialize
            gen_loss = 0.0

            # l1 loss
            if self.config.loss.l1["lambda"]:
                l1_loss = self.criterion["l1"](estimates, sources).item()
                self.valid_loss["valid/l1_loss"] += l1_loss
                gen_loss += self.config.loss.l1["lambda"] * l1_loss

            # multi-resolution sfft loss
            if self.config.loss.stft["lambda"]:
                total_sc_loss, total_mag_loss = 0, 0
                for index in range(sources.size(0)):
                    sc_loss, mag_loss = self.criterion["stft"](
                        estimates[index], sources[index])
                    total_sc_loss += sc_loss.item()
                    total_mag_loss += mag_loss.item()
                sc_loss = total_sc_loss / (index + 1)
                mag_loss = total_mag_loss / (index + 1)
                self.valid_loss["valid/spectral_convergence_loss"] += sc_loss
                self.valid_loss["valid/log_stft_magnitude_loss"] += mag_loss
                gen_loss += self.config.loss.stft["lambda"] * \
                    ((sc_loss + mag_loss))

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
        if epoch == 0:
            test_set = musdb.DB(
                self.config.dataset.musdb.path, subsets=["test"])
            from_samplerate = 44100
            track = test_set.tracks[1]
            mix = torch.from_numpy(track.audio).t().float()
            ref = mix.mean(dim=0)  # mono mixture
            mix = (mix - ref.mean()) / ref.std()
            mix = convert_audio(mix, from_samplerate,
                                self.config.dataset.samplerate, self.config.dataset.channels)
            references = torch.stack(
                [torch.from_numpy(track.targets[name].audio).t() for name in ["drums", "bass", "other", "vocals"]])
            references = convert_audio(
                references, from_samplerate, self.config.dataset.samplerate, self.config.dataset.channels)
            track = {
                "name": track.name,
                "mix": mix, "mean": ref.mean(),
                "std": ref.std(),
                "targets": references
            }
            torch.save(track, self.config.eval_epoch_path)
        else:
            track = torch.load(
                Path(self.config.eval_epoch_path), map_location="cpu")

        """Evaluate model one epoch."""
        eval_folder = self.outdir / "evals" / self.config.name
        eval_folder.mkdir(exist_ok=True, parents=True)

        estimates = apply_model(
            self.model["generator"].module if self.config.device.world_size > 1 else self.model["generator"],
            track["mix"].to(self.device),
            shifts=self.config.dataset.shifts,
            split=self.config.split_valid,
            overlap=self.config.dataset.overlap
        )
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
        print("sdr!")
        sdr, isr, sir, sar = museval.evaluate(
            references[:, :second, :], estimates[:, :second, :], win=win, hop=hop)
        print("sdr!")
        for idx, source in enumerate(self.config.dataset.sources):
            self.eval_loss[f"eval/sdr_{source}"] = np.nanmedian(
                sdr[idx].tolist())
            print(source)
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
            ref = mix.mean(dim=0)  # mono mixture
            mix = (mix - ref.mean()) / ref.std()
            mix = convert_audio(
                mix, src_rate, model.samplerate, model.audio_channels)
            estimates = apply_model(model, mix.to(self.device), shifts=self.config.dataset.shifts,
                                    split=self.config.split_valid, overlap=self.config.dataset.overlap)
            estimates = estimates * ref.std() + ref.mean()

            estimates = estimates.transpose(1, 2)
            references = torch.stack(
                [torch.from_numpy(track.targets[name].audio).t() for name in self.model["generator"].sources])
            references = convert_audio(
                references, src_rate, model.samplerate, model.audio_channels)
            references = references.transpose(1, 2).numpy()
            estimates = estimates.cpu().numpy()
            # save wav
            if track_indexes[index] == 1:
                track_folder = eval_folder / track.name
                track_folder.mkdir(exist_ok=True, parents=True)
                for name, estimate in zip(model.sources, estimates):
                    wavfile.write(
                        str(track_folder / (name + ".wav")), 44100, estimate)
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

    def _check_save_interval(self):
        # save to file
        log_folder = self.outdir / "logs"
        metrics_path = log_folder / f"{self.config.name}.json"
        json.dump(self.metrics, open(metrics_path, "w"))
        checkpoint_folder = self.outdir / self.config.outdir.checkpoints
        checkpoint_folder.mkdir(exist_ok=True, parents=True)
        checkpoint_path = checkpoint_folder / f"{self.config.name}.th"
        checkpoint_tmp_path = checkpoint_folder / f"{self.config.name}.th.tmp"
        self.save_checkpoint(checkpoint_tmp_path)
        checkpoint_tmp_path.rename(checkpoint_path)

    def _check_eval_interval(self, epoch):
        if epoch % self.config.eval_interval == 0:
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
