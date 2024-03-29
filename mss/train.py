# Copyright (c) 2023 torakyun
#  MIT License (https://opensource.org/licenses/MIT)

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

import mlflow
from omegaconf import DictConfig, ListConfig

from .audio import convert_audio
from .apply import apply_model
from .states import save_model
from .utils import gpulife, human_seconds, average_metric

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
    "loss.mae.lambda": "mae",
    "loss.mse.lambda": "mse",
    "loss.stft.lambda": "stft",
    "loss.cac.lambda": "cac",
    "loss.mel.lambda": "mel",
    "loss.mfcc.lambda": "mfcc",
    "loss.adversarial.lambda": "adv",
    "loss.feat_match.lambda": "fm",
    "optimizer/generator": "g_opt",
    "optimizer/discriminator": "d_opt",
    "model/generator": "gen",
    "model/discriminator": "dis",
    "model.discriminator.separate": "sep",
}


class Trainer(object):
    """Customized trainer module for training."""

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
        self.train_loss = defaultdict(float)
        self.valid_loss = defaultdict(float)
        self.eval_loss = defaultdict(float)
        self.pretrained_epoch = 0
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

        # show checkpoint's log
        best_loss = float("inf")
        for epoch, metrics in enumerate(self.metrics):
            logging = [
                f"Epoch {epoch:03d}: ",
                f"train={metrics['train']:.4f} ",
                f"valid={metrics['valid']:.4f} ",
                f"best={metrics['best']:.4f} ",
                # f"ms={metrics.get('true_model_size', 0):.2f}MB ",
                # f"cms={metrics.get('compressed_model_size', 0):.2f}MB ",
                f"duration={human_seconds(metrics['duration'])} "]
            if metrics['real']:
                logging += [
                    f"real={metrics['real']:.4f} ",
                    f"fake={metrics['fake']:.4f} "]
            print("".join(logging))
            best_loss = metrics['best']

        # training
        for epoch in range(len(self.metrics), self.config.epochs):
            begin = time.time()

            # train and valid
            # train_loss, real_loss, fake_loss, valid_loss, model_size = 0, 0, 0, 0, 0
            self.model["generator"].train()
            train_loss, real_loss, fake_loss, model_size = self._train_epoch(
                epoch)
            self.model["generator"].eval()
            save_valid_audio = self.config.device.rank == 0 and epoch and self.config.valid_interval and epoch % self.config.valid_interval == 0
            valid_loss = self._valid_epoch(epoch, save_valid_audio)

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
                "real": real_loss,
                "fake": fake_loss,
            })
            if self.config.device.rank == 0:
                self._check_eval_interval(epoch)
                self._check_save_interval(epoch)
                self._check_log_interval(epoch)

            # logging
            logging = [
                f"Epoch {epoch:03d}: ",
                f"train={train_loss:.4f} ",
                f"valid={valid_loss:.4f} ",
                f"best={best_loss:.4f} ",
                # f"ms={ms:.2f}MB ",
                # f"cms={cms:.2f}MB ",
                f"duration={human_seconds(duration)} "]
            if real_loss:
                logging += [
                    f"real={real_loss:.4f} ",
                    f"fake={fake_loss:.4f} "]
            print("".join(logging))

        # save best model
        if self.config.device.world_size > 1:
            distributed.barrier()
        model = self.model["generator"].module if self.config.device.world_size > 1 else self.model["generator"]
        del self.model
        model.load_state_dict(self.best_state)
        model.to("cpu")
        if self.config.device.rank == 0:
            save_model(model, self.quantizer, self.config,
                       self.outdir / "models" / f"{self.config.name}.th")

        mlflow.end_run()

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

    def _write_figure(self, title, ref, est, dif, dir, xmax=None, vmax=None):
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
            min = torch.min(torch.stack([ref[i], est[i]]))
            if min < 0:
                ref[i] = torch.clamp(ref[i], min=min) - min
                est[i] = torch.clamp(est[i], min=min) - min
            if not vmax:
                vmax = torch.max(torch.stack([ref[i], est[i], dif[i]])).item()
            self.axes[i, 0].imshow(
                ref[i].cpu(), origin="lower", aspect="auto", cmap="gnuplot", vmin=0, vmax=vmax)
            self.axes[i, 1].imshow(
                est[i].cpu(), origin="lower", aspect="auto", cmap="gnuplot", vmin=0, vmax=vmax)
            im = self.axes[i, 2].imshow(
                dif[i].cpu(), origin="lower", aspect="auto", cmap="gnuplot", vmin=0, vmax=vmax)
            self.fig.colorbar(im, ax=self.axes[i, 2])

        # save figure
        title = title.replace(" ", "_")
        path = Path(dir) / f"{title}.png"
        self.fig.savefig(path)

        # remove colorbar and clear axis
        for i, _ in enumerate(sources):
            self.axes[i, 2].images[-1].colorbar.remove()
        plt.cla()

    def save_spectrogram(self, references, estimates, dir, stft_params=None, mel_params=None, window="hann"):
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
        self._write_figure(
            "STFT", references_mag, estimates_mag, differences_stft, dir, vmax=5)
        del differences_stft

        # Magnitude spectrogram
        differences_mag = (references_mag - estimates_mag).abs()
        self._write_figure(
            "Magnitude Spectrogram", references_mag, estimates_mag, differences_mag, dir, vmax=5)
        del differences_mag

        # Log-scale Magnitude spectrogram
        references_log_mag = torch.log(references_mag)
        estimates_log_mag = torch.log(estimates_mag)
        differences_log_mag = (references_log_mag - estimates_log_mag).abs()
        self._write_figure(
            "Log-Scale Magnitude Spectrogram", references_log_mag, estimates_log_mag, differences_log_mag, dir)
        del references_log_mag, estimates_log_mag, differences_log_mag

        # Mel spectrogram
        melmat = librosa.filters.mel(**mel_params)
        melmat = torch.from_numpy(melmat).to(references_mag.device)
        references_mel = torch.clamp(torch.matmul(
            melmat, references_mag), min=1e-7)
        estimates_mel = torch.clamp(torch.matmul(
            melmat, estimates_mag), min=1e-7)
        differences_mel = (references_mel - estimates_mel).abs()
        self._write_figure(
            "Mel Spectrogram", references_mel, estimates_mel, differences_mel, dir, vmax=1)
        del melmat, differences_mel

        # Log-scale Mel spectrogram
        references_log_mel = torch.log(references_mel)
        estimates_log_mel = torch.log(estimates_mel)
        differences_log_mel = (references_log_mel - estimates_log_mel).abs()
        self._write_figure(
            "Log-Scale Mel Spectrogram", references_log_mel, estimates_log_mel, differences_log_mel, dir)

    def _train_epoch(self, epoch):
        """Train model one epoch."""
        if self.config.device.world_size > 1:
            sampler_epoch = self.pretrained_epoch + epoch
            # if self.config.seed is not None:
            #     sampler_epoch += self.config.seed * 1000
            self.sampler["train"].set_epoch(sampler_epoch)
        tq = tqdm(self.data_loader["train"],
                  ncols=120,
                  desc=f"[{epoch:03d}] train",
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

                # mae loss
                if self.config.loss.mae["lambda"]:
                    mae_loss = self.criterion["mae"](
                        estimates, sources[start::self.config.batch_divide])
                    mae_loss /= self.config.batch_divide
                    self.train_loss["train/mae_loss"] += mae_loss.item()
                    gen_loss += self.config.loss.mae["lambda"] * mae_loss
                    del mae_loss
                # print("mae_loss: ", time.time() - start_t)
                # gpulife("mae_loss")
                # start_t = time.time()

                # multi-resolution multi-scale stft loss
                if self.config.loss.stft["lambda"]:
                    linear_scale_stft_loss, log_scale_stft_loss = self.criterion["stft"](
                        estimates, sources[start::self.config.batch_divide])
                    linear_scale_stft_loss /= self.config.batch_divide
                    log_scale_stft_loss /= self.config.batch_divide
                    self.train_loss["train/linear_scale_stft_loss"] += linear_scale_stft_loss.item()
                    self.train_loss["train/log_scale_stft_loss"] += log_scale_stft_loss.item()
                    gen_loss += self.config.loss.stft["lambda"] * (
                        linear_scale_stft_loss + log_scale_stft_loss)
                    del linear_scale_stft_loss, log_scale_stft_loss
                # print("stft_loss: ", time.time() - start_t)
                # gpulife("stft_loss")
                # start_t = time.time()

                # multi-resolution multi-scale cac loss
                if self.config.loss.cac["lambda"]:
                    linear_scale_cac_loss, log_scale_cac_loss = self.criterion["cac"](
                        estimates, sources[start::self.config.batch_divide])
                    linear_scale_cac_loss /= self.config.batch_divide
                    log_scale_cac_loss /= self.config.batch_divide
                    self.train_loss["train/linear_scale_cac_loss"] += linear_scale_cac_loss.item()
                    self.train_loss["train/log_scale_cac_loss"] += log_scale_cac_loss.item()
                    gen_loss += self.config.loss.cac["lambda"] * (
                        linear_scale_cac_loss + log_scale_cac_loss)
                    del linear_scale_cac_loss, log_scale_cac_loss
                # print("cac_loss: ", time.time() - start_t)
                # gpulife("cac_loss")
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

                # mfcc loss
                if self.config.loss.mfcc["lambda"]:
                    mfcc_loss = self.criterion["mfcc"](
                        estimates, sources[start::self.config.batch_divide])
                    mfcc_loss /= self.config.batch_divide
                    self.train_loss["train/mfcc_loss"] += mfcc_loss.item()
                    gen_loss += self.config.loss.mfcc["lambda"] * mfcc_loss
                    del mfcc_loss
                # print("mfcc_loss: ", time.time() - start_t)
                # gpulife("mfcc_loss")
                # start_t = time.time()

                self.train_loss["train/gen_loss"] += gen_loss.item(
                ) if isinstance(gen_loss, torch.Tensor) else gen_loss

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
            if self.config.optimizer.generator.grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model["generator"].parameters(),
                    self.config.optimizer.generator.grad_norm,
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
                if self.config.optimizer.discriminator.grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model["discriminator"].parameters(),
                        self.config.optimizer.discriminator.grad_norm,
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
            postfix = {
                "loss": f"{current_loss:.4f}",
                # "ms": f"{model_size:.2f}",
                "grad": f"{g_grad_norm:.4f}",
            }
            current_real_loss, current_fake_loss = 0, 0
            if self.config.loss.adversarial["lambda"] and epoch > self.config.loss.adversarial.train_start_epoch:
                current_real_loss = self.train_loss["train/real_loss"] / (
                    1 + idx)
                current_fake_loss = self.train_loss["train/fake_loss"] / (
                    1 + idx)
                postfix["real"] = f"{current_real_loss:.4f}"
                postfix["fake"] = f"{current_fake_loss:.4f}"
            tq.set_postfix(**postfix)

        for k, v in self.train_loss.items():
            self.train_loss[k] = v / (1 + idx)

        if self.config.device.world_size > 1:
            current_loss = average_metric(current_loss)
        return current_loss, current_real_loss, current_fake_loss, model_size

    @torch.no_grad()
    def _valid_epoch(self, epoch, save_valid_audio=False):
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
            # if idx > 0:
            #     break
            # first five minutes to avoid OOM on --upsample models
            streams = streams[0, ..., :15_000_000]
            streams = streams.to(self.device)
            sources = streams[1:]
            mix = streams[0]
            estimates = apply_model(
                model, mix[None], split=self.config.split_valid, overlap=0)[0]

            # initialize
            gen_loss = 0.0

            # mae loss
            if self.config.loss.mae["lambda"]:
                mae_loss = self.criterion["mae"](estimates, sources).item()
                self.valid_loss["valid/mae_loss"] += mae_loss
                gen_loss += self.config.loss.mae["lambda"] * mae_loss

            # multi-resolution multi-scale stft loss
            if self.config.loss.stft["lambda"]:
                total_linear_scale_stft_loss, total_log_scale_stft_loss = 0, 0
                for index in range(sources.size(0)):
                    linear_scale_stft_loss, log_scale_stft_loss = self.criterion["stft"](
                        estimates[index], sources[index])
                    total_linear_scale_stft_loss += linear_scale_stft_loss.item()
                    total_log_scale_stft_loss += log_scale_stft_loss.item()
                linear_scale_stft_loss = total_linear_scale_stft_loss / (index + 1)
                log_scale_stft_loss = total_log_scale_stft_loss / (index + 1)
                self.valid_loss["valid/linear_scale_stft_loss"] += linear_scale_stft_loss
                self.valid_loss["valid/log_scale_stft_loss"] += log_scale_stft_loss
                gen_loss += self.config.loss.stft["lambda"] * \
                    ((linear_scale_stft_loss + log_scale_stft_loss))

            # multi-resolution multi-scale cac loss
            if self.config.loss.cac["lambda"]:
                total_linear_scale_cac_loss, total_log_scale_cac_loss = 0, 0
                for index in range(sources.size(0)):
                    linear_scale_cac_loss, log_scale_cac_loss = self.criterion["cac"](
                        estimates[index], sources[index])
                    total_linear_scale_cac_loss += linear_scale_cac_loss.item()
                    total_log_scale_cac_loss += log_scale_cac_loss.item()
                linear_scale_cac_loss = total_linear_scale_cac_loss / (index + 1)
                log_scale_cac_loss = total_log_scale_cac_loss / (index + 1)
                self.valid_loss["valid/linear_scale_cac_loss"] += linear_scale_cac_loss
                self.valid_loss["valid/log_scale_cac_loss"] += log_scale_cac_loss
                gen_loss += self.config.loss.cac["lambda"] * \
                    ((linear_scale_cac_loss + log_scale_cac_loss))

            # mel spectrogram loss
            if self.config.loss.mel["lambda"]:
                total_mel_loss = 0
                for index in range(sources.size(0)):
                    total_mel_loss += self.criterion["mel"](
                        estimates[index], sources[index]).item()
                mel_loss = total_mel_loss / (index + 1)
                self.valid_loss["valid/mel_spectrogram_loss"] += mel_loss
                gen_loss += self.config.loss.mel["lambda"] * mel_loss

            # mfcc loss
            if self.config.loss.mfcc["lambda"]:
                total_mfcc_loss = 0
                for index in range(sources.size(0)):
                    total_mfcc_loss += self.criterion["mfcc"](
                        estimates[index], sources[index]).item()
                mfcc_loss = total_mfcc_loss / (index + 1)
                self.valid_loss["valid/mfcc_loss"] += mfcc_loss
                gen_loss += self.config.loss.mfcc["lambda"] * mfcc_loss

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

            # save valid audio
            if save_valid_audio and idx == 0:
                # valid folder
                valid_folder = self.outdir / "valids" / \
                    self.config.name / "interval" / f"{epoch}"
                valid_folder.mkdir(exist_ok=True, parents=True)

                # save spectrogram(.png)
                second = self.config.dataset.samplerate * self.config.valid_second
                self.save_spectrogram(
                    sources.mean(dim=1)[..., :second],
                    estimates.mean(dim=1)[..., :second],
                    valid_folder
                )

                # save audio(.wav)
                track_folder = valid_folder / "track"
                track_folder.mkdir(exist_ok=True, parents=True)
                for name, estimate in zip(self.config.dataset.sources, estimates.transpose(1, 2).cpu().numpy()):
                    wavfile.write(
                        str(track_folder / (name + ".wav")), self.config.dataset.samplerate, estimate)

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
        if self.config.dataset.musdbhq:
            data_file = "musdbhq"
            path = self.config.dataset.musdbhq
            is_wav = True
        else:
            data_file = "musdb"
            path = self.config.dataset.musdb
            is_wav = False
        data_file += f"_{self.config.dataset.samplerate}_{self.config.dataset.audio_channels}"
        for name in self.config.dataset.sources:
            data_file += f"_{name}"
        data_file += ".th"
        if (data_dir / data_file).is_file():
            track = torch.load(data_dir / data_file, map_location="cpu")
        else:
            test_set = musdb.DB(path, subsets=["test"], is_wav=is_wav)
            from_samplerate = 44100
            track = test_set.tracks[1]
            mix = torch.from_numpy(track.audio).t().float()
            ref = mix.mean(dim=0)  # mono mixture
            mix = (mix - ref.mean()) / ref.std()
            mix = convert_audio(mix, from_samplerate,
                                self.config.dataset.samplerate, self.config.dataset.audio_channels)
            references = torch.stack(
                [torch.from_numpy(track.targets[name].audio).t().float() for name in self.config.dataset.sources])
            references = convert_audio(
                references, from_samplerate, self.config.dataset.samplerate, self.config.dataset.audio_channels)
            track = {
                "name": track.name,
                "mix": mix, "mean": ref.mean(),
                "std": ref.std(),
                "targets": references
            }
            torch.save(track, data_dir / data_file)

        """Evaluate model one epoch."""
        eval_folder = self.outdir / "evals" / \
            self.config.name / "interval" / f"{epoch}"
        eval_folder.mkdir(exist_ok=True, parents=True)

        estimates = apply_model(
            self.model["generator"].module if self.config.device.world_size > 1 else self.model["generator"],
            track["mix"][None].to(self.device),
            shifts=self.config.shifts,
            split=self.config.split_valid,
            overlap=self.config.overlap
        )[0]
        estimates = estimates * track["std"] + track["mean"]

        # save spectrogram(.png)
        references = track["targets"].to(self.device)
        second = self.config.dataset.samplerate * self.config.eval_second
        self.save_spectrogram(
            references.mean(dim=1)[..., :second],
            estimates.mean(dim=1)[..., :second],
            eval_folder
        )

        estimates = estimates.transpose(1, 2).cpu().numpy()
        references = references.transpose(1, 2).cpu().numpy()

        # save audio(.wav)
        track_folder = eval_folder / track["name"]
        track_folder.mkdir(exist_ok=True, parents=True)
        for name, estimate in zip(self.config.dataset.sources, estimates):
            wavfile.write(
                str(track_folder / (name + ".wav")), self.config.dataset.samplerate, estimate)

        # calculate SDR
        win = int(1. * self.config.dataset.samplerate)
        hop = int(1. * self.config.dataset.samplerate)
        sdr, isr, sir, sar = museval.evaluate(
            references[:, :second, :], estimates[:, :second, :], win=win, hop=hop)
        for idx, source in enumerate(self.config.dataset.sources):
            self.eval_loss[f"eval/sdr_{source}"] = np.nanmedian(
                sdr[idx].tolist())
        self.eval_loss["eval/sdr_all"] = np.array(
            list(self.eval_loss.values())).mean()
        json.dump(self.eval_loss, open(track_folder / "sdr.json", "w"))

    def _check_save_interval(self, epoch):
        # save checkpoint(.th)
        checkpoint_folder = self.outdir / "checkpoints"
        checkpoint_folder.mkdir(exist_ok=True, parents=True)
        checkpoint_path = checkpoint_folder / f"{self.config.name}.th"
        checkpoint_tmp_path = checkpoint_folder / f"{self.config.name}.th.tmp"
        self.save_checkpoint(checkpoint_tmp_path)
        checkpoint_tmp_path.rename(checkpoint_path)

        # save interval checkpoint(.th)
        if epoch and self.config.save_interval and epoch % self.config.save_interval == 0:
            checkpoint_path = checkpoint_folder / \
                f"{self.config.name}_{epoch}.th"
            self.save_checkpoint(checkpoint_path)

    def _check_eval_interval(self, epoch):
        if epoch and self.config.eval_interval and epoch % self.config.eval_interval == 0:
            self._eval_epoch(epoch)

    def _check_log_interval(self, epoch):
        # write logs
        mlflow.log_metrics(self.train_loss, epoch)
        mlflow.log_metrics(self.valid_loss, epoch)
        mlflow.log_metrics(self.eval_loss, epoch)
