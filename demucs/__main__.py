# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
import sys
from pathlib import Path

import torch as th
from torch import distributed, nn
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel.distributed import DistributedDataParallel

from .augment import FlipChannels, FlipSign, Remix, Scale, Shift
from .compressed import get_compressed_datasets
from .raw import Rawset
from .repitch import RepitchedWrapper
from .train import Trainer
from .utils import (gpulife, save_model, get_state,
                    save_state, sizeof_fmt, get_quantizer)
from .wav import get_wav_datasets, get_musdb_wav_datasets

from . import models

from .losses import DiscriminatorAdversarialLoss
from .losses import FeatureMatchLoss
from .losses import GeneratorAdversarialLoss
from .losses import MelSpectrogramLoss
from .losses import MultiResolutionSTFTLoss

from omegaconf import OmegaConf
import hydra


def get_name():
    args = sys.argv[1:]
    show_names = {
        "pretrained": "pretrained",
        "epochs": "epochs",
        "dataset.samplerate": "sr",
        "dataset.audio_channels": "ch",
        "loss.l1.lambda": "l1",
        "loss.stft.lambda": "stft",
        "loss.mel.lambda": "mel",
        "loss.adversarial.lambda": "adv",
        "loss.feat_match.lambda": "fm",
        "model/generator": "gen",
        "model/discriminator": "dis",
        "model.discriminator.separate": "sep",
    }
    params = {}
    for arg in args:
        name, value = arg.split("=", 1)
        if name[0] == "+":
            name = name[1:]
        if name in show_names.keys():
            params[show_names[name]] = \
                value if name != "pretrained" else f"({value})"
    return "_".join([f"{name}-{value}" for name, value in params.items()]) if params else "default"


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg):
    OmegaConf.set_struct(cfg, False)
    try:
        name = cfg.name
    except:
        name = get_name()
        cfg.name = name
    print(f"Experiment {name}")

    if cfg.dataset.musdb.path is None and cfg.device.rank == 0:
        print(
            "You must provide the path to the MusDB dataset with the --musdb flag. "
            "To download the MusDB dataset, see https://sigsep.github.io/datasets/musdb.html.",
            file=sys.stderr)
        sys.exit(1)

    out = Path(cfg.out)
    log_folder = out / "logs"
    log_folder.mkdir(exist_ok=True, parents=True)
    checkpoint_folder = out / "checkpoints"
    checkpoint_folder.mkdir(exist_ok=True, parents=True)
    checkpoint = checkpoint_folder / f"{name}.th"
    if cfg.restart and checkpoint.exists() and cfg.device.rank == 0:
        checkpoint.unlink()
    model_folder = out / "models"
    model_folder.mkdir(exist_ok=True, parents=True)

    th.manual_seed(cfg.seed)
    # Prevents too many threads to be started when running `museval` as it can be quite
    # inefficient on NUMA architectures.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    if cfg.device.device is None:
        device = "cpu"
        if th.cuda.is_available():
            device = "cuda"
    else:
        device = cfg.device.device
    if cfg.device.world_size > 1:
        if device != "cuda" and cfg.device.rank == 0:
            print(
                "Error: distributed training is only available with cuda device", file=sys.stderr)
            sys.exit(1)
        th.cuda.set_device(cfg.device.rank % th.cuda.device_count())
        distributed.init_process_group(backend="nccl",
                                       init_method="tcp://" + cfg.device.master,
                                       rank=cfg.device.rank,
                                       world_size=cfg.device.world_size)

    # define models
    generator_class = getattr(
        models,
        # keep compatibility
        cfg.model.generator.get("name", "Demucs"),
    )
    model = {
        "generator": generator_class(
            **cfg.model.generator.params,
        ).to(device),
    }
    if cfg.loss.adversarial["lambda"]:
        model["discriminator"] = models.Discriminators(
            name=cfg.model.discriminator.get(
                "name", "ParallelWaveGANDiscriminator"),
            params=cfg.model.discriminator.get("params", {}),
            sources=len(cfg.dataset.sources),
            channels=cfg.dataset.audio_channels,
            separate=cfg.model.discriminator.get("separate", "full"),
        ).to(device)

    if cfg.show:
        print(model)
        size = sizeof_fmt(4 * sum(p.numel()
                                  for p in model["generator"].parameters()))
        print(f"Model size {size}")
        return

    # get dataset

    # Setting number of samples so that all convolution windows are full.
    # Prevents hard to debug mistake with the prediction being shifted compared
    # to the input mixture.
    samples = model["generator"].valid_length(
        cfg.dataset.sample_seconds * cfg.dataset.samplerate)
    print(f"Number of training samples adjusted to {samples}")
    samples += cfg.dataset.stride_seconds * cfg.dataset.samplerate
    if cfg.dataset.repitch:
        # We need a bit more audio samples, to account for potential
        # tempo change.
        samples = math.ceil(samples / (1 - 0.01 * cfg.dataset.max_tempo))

    metadata_folder = Path(cfg.dataset.musdb.metadata)
    metadata_folder.mkdir(exist_ok=True, parents=True)
    if cfg.dataset.raw.path:
        raw_path = Path(cfg.dataset.raw.path)
        train_set = Rawset(raw_path / "train",
                           samples=samples,
                           channels=cfg.dataset.audio_channels,
                           streams=range(
                               1, len(model["generator"].sources) + 1),
                           stride=cfg.dataset.stride_seconds * cfg.dataset.samplerate)

        valid_set = Rawset(raw_path / "valid",
                           channels=cfg.dataset.audio_channels)
    elif cfg.dataset.wav.path:
        train_set, valid_set = get_wav_datasets(
            cfg, samples, model["generator"].sources)

        if cfg.dataset.wav.concat:
            if cfg.dataset.musdb.is_wav:
                mus_train, mus_valid = get_musdb_wav_datasets(
                    cfg, samples, model.sources)
            else:
                mus_train, mus_valid = get_compressed_datasets(cfg, samples)
            train_set = ConcatDataset([train_set, mus_train])
            valid_set = ConcatDataset([valid_set, mus_valid])
    elif cfg.dataset.musdb.is_wav:
        train_set, valid_set = get_musdb_wav_datasets(
            cfg, samples, model["generator"].sources)
    else:
        train_set, valid_set = get_compressed_datasets(cfg, samples)
    print("Train set and valid set sizes", len(train_set), len(valid_set))

    if cfg.dataset.repitch:
        train_set = RepitchedWrapper(
            train_set,
            cfg.dataset.samplerate,
            proba=cfg.dataset.repitch,
            max_tempo=cfg.dataset.max_tempo)

    # get data loader
    sampler = {"train": None, "valid": None}
    if cfg.device.world_size > 1:
        sampler["train"] = DistributedSampler(
            dataset=train_set,
            num_replicas=cfg.device.world_size,
            rank=cfg.device.rank,
            shuffle=True,
        )
        sampler["valid"] = DistributedSampler(
            dataset=valid_set,
            num_replicas=cfg.device.world_size,
            rank=cfg.device.rank,
            shuffle=False,
        )
    batch_size = cfg.batch_size // cfg.device.world_size
    data_loader = {
        "train": DataLoader(
            dataset=train_set,
            shuffle=False if cfg.device.world_size > 1 else True,
            batch_size=batch_size,
            num_workers=cfg.device.workers,
            sampler=sampler["train"],
        ),
        "valid": DataLoader(
            dataset=valid_set,
            shuffle=False,
            batch_size=1,
            num_workers=cfg.device.workers,
            sampler=sampler["valid"],
        ),
    }

    # define augments
    augment = [Shift(cfg.dataset.stride_seconds * cfg.dataset.samplerate)]
    if cfg.dataset.augment:
        augment += [FlipSign(), FlipChannels(), Scale(),
                    Remix(group_size=cfg.dataset.remix_group_size)]
    augment = nn.Sequential(*augment).to(device)
    print("Agumentation pipeline:", augment)

    # define criterions
    criterion = {}
    if cfg.loss.l1["lambda"]:
        criterion["l1"] = nn.L1Loss()
    if cfg.loss.mse["lambda"]:
        criterion["mse"] = nn.MSELoss()
    if cfg.loss.stft["lambda"]:
        criterion["stft"] = MultiResolutionSTFTLoss(
            **cfg.loss.stft.params).to(device)
    if cfg.loss.mel["lambda"]:
        criterion["mel"] = MelSpectrogramLoss(**cfg.loss.mel.params).to(device)
    if cfg.loss.adversarial["lambda"]:
        criterion["gen_adv"] = GeneratorAdversarialLoss(
            **cfg.loss.adversarial.generator_params).to(device)
        criterion["dis_adv"] = DiscriminatorAdversarialLoss(
            **cfg.loss.adversarial.discriminator_params).to(device)
    if cfg.loss.feat_match["lambda"]:
        criterion["feat_match"] = FeatureMatchLoss(
            **cfg.loss.feat_match.params).to(device)
    assert criterion
    print(criterion)

    # define optimizers
    generator_optimizer_class = th.optim.Adam
    optimizer = {
        "generator": generator_optimizer_class(
            model["generator"].parameters(),
            lr=cfg.lr,
        ),
    }
    if cfg.loss.adversarial["lambda"]:
        discriminator_optimizer_class = th.optim.Adam
        optimizer["discriminator"] = discriminator_optimizer_class(
            model["discriminator"].parameters(),
            lr=cfg.lr,
        )

    # define quantizer
    quantizer = None
    quantizer = get_quantizer(model["generator"], cfg, optimizer)

    if cfg.device.world_size > 1:
        model["generator"] = DistributedDataParallel(model["generator"],
                                                     device_ids=[
                                                         th.cuda.current_device()],
                                                     output_device=th.cuda.current_device())
        if model["discriminator"]:
            model["discriminator"] = DistributedDataParallel(model["discriminator"],
                                                             device_ids=[
                                                                 th.cuda.current_device()],
                                                             output_device=th.cuda.current_device())

    # define Trainer
    trainer = Trainer(
        data_loader=data_loader,
        sampler=sampler,
        augment=augment,
        model=model,
        quantizer=quantizer,
        criterion=criterion,
        optimizer=optimizer,
        config=cfg,
        device=device,
    )
    try:
        trainer.load_checkpoint(checkpoint)
        print("load checkpoint")
    except IOError:
        if cfg.pretrained:
            trainer.load_checkpoint(
                checkpoint_folder / (cfg.pretrained + ".th"), load_only_params=True)
            print("load pretrained")
        else:
            print("model init")

    # only save best state
    model_name = f"{name}.th"
    if cfg.save_model:
        if cfg.device.rank == 0:
            model.to("cpu")
            assert trainer.best_state is not None, "model needs to train for 1 epoch at least."
            model.load_state_dict(trainer.best_state)
            save_model(model, quantizer, cfg, model_folder / model_name)
        return
    elif cfg.save_state:
        model_name = f"{cfg.save_state}.th"
        if cfg.device.rank == 0:
            model.to("cpu")
            model.load_state_dict(trainer.best_state)
            state = get_state(model, quantizer)
            save_state(state, model_folder / model_name)
        return

    # run training loop
    if cfg.device.rank == 0:
        done = log_folder / f"{name}.done"
        if done.exists():
            done.unlink()
    stat = trainer.run()
    print(stat, "\n--------------------------")
    if cfg.device.rank == 0:
        print("done")
        done.write_text("done")
    return stat["all"]


if __name__ == "__main__":
    main()
