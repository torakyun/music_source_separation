# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch as th
from torch import distributed, nn
from torch.utils.data import ConcatDataset
from torch.nn.parallel.distributed import DistributedDataParallel

from .augment import FlipChannels, FlipSign, Remix, Scale, Shift
from .compressed import get_compressed_datasets
from .raw import Rawset
from .repitch import RepitchedWrapper
from .pretrained import load_pretrained
from .test import evaluate
from .train import train_model, validate_model, Trainer
from .utils import (human_seconds, load_model, save_model, get_state,
                    save_state, sizeof_fmt, get_quantizer)
from .wav import get_wav_datasets, get_musdb_wav_datasets

from . import models

@dataclass
class SavedState:
    metrics: list = field(default_factory=list)
    last_state: dict = None
    best_state: dict = None
    optimizer: dict = None
from .losses import DiscriminatorAdversarialLoss
from .losses import GeneratorAdversarialLoss
from .losses import MelSpectrogramLoss
from .losses import MultiResolutionSTFTLoss

from omegaconf import DictConfig, OmegaConf
import hydra


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg):
    OmegaConf.set_struct(cfg, False)
    name = cfg.name
    print(f"Experiment {name}")

    if cfg.dataset.musdb.path is None and cfg.device.rank == 0:
        print(
            "You must provide the path to the MusDB dataset with the --musdb flag. "
            "To download the MusDB dataset, see https://sigsep.github.io/datasets/musdb.html.",
            file=sys.stderr)
        sys.exit(1)

    out = Path(cfg.outdir.out)
    eval_folder = out / cfg.outdir.evals / name
    eval_folder.mkdir(exist_ok=True, parents=True)
    log_folder = out / cfg.outdir.logs
    log_folder.mkdir(exist_ok=True)
    metrics_path = out / cfg.outdir.logs / f"{name}.json"
    eval_folder.mkdir(exist_ok=True, parents=True)
    checkpoint_folder = out / cfg.outdir.checkpoints
    checkpoint_folder.mkdir(exist_ok=True, parents=True)
    checkpoint = checkpoint_folder / f"{name}.th"
    checkpoint_tmp = checkpoint_folder / f"{name}.th.tmp"
    if cfg.restart and checkpoint.exists() and cfg.device.rank == 0:
        checkpoint.unlink()
    model_folder = out / cfg.outdir.models
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
    cfg.device.distributed = cfg.device.world_size > 1
    if cfg.device.distributed:
        if device != "cuda" and cfg.device.rank == 0:
            print("Error: distributed training is only available with cuda device", file=sys.stderr)
            sys.exit(1)
        th.cuda.set_device(cfg.device.rank % th.cuda.device_count())
        distributed.init_process_group(backend="nccl",
                                       init_method="tcp://" + cfg.device.master,
                                       rank=cfg.device.rank,
                                       world_size=cfg.device.world_size)

    # define models
    if cfg.test or cfg.test_pretrained:
        cfg.epochs = 1
        cfg.repeat = 0
        if cfg.test:
            model = load_model(model_folder / cfg.test).to(device)
        else:
            model = load_pretrained(cfg.test_pretrained).to(device)
    else:
        generator_class = getattr(
            models,
            # keep compatibility
            cfg.model.generator.get("name", "Demucs"),
        )
        # model = {
        #     "generator": generator_class(
        #         **cfg.model.generator.params,
        #     ).to(device),
        #     # "discriminator": discriminator_class(
        #     #     **cfg.model.discriminator.params,
        #     # ).to(device),
        # }
        model = generator_class(**cfg.model.generator.params).to(device)
    if cfg.init:  # initialize by pretrained
        model.load_state_dict(load_pretrained(cfg.init).state_dict())

    if cfg.show:
        print(model)
        size = sizeof_fmt(4 * sum(p.numel() for p in model.parameters()))
        print(f"Model size {size}")
        return

    try:
        saved = th.load(checkpoint, map_location='cpu')
    except IOError:
        saved = SavedState()
    # define optimizers
    optimizer = th.optim.Adam(model.parameters(), lr=cfg.lr)

    quantizer = None
    quantizer = get_quantizer(model, cfg, optimizer)

    if cfg.device.distributed:
        dmodel = DistributedDataParallel(model,
                                         device_ids=[th.cuda.current_device()],
                                         output_device=th.cuda.current_device())
    else:
        dmodel = model

    cfg.use_adv = False
    trainer = Trainer(
        model={"generator": dmodel},
        optimizer={"generator": optimizer},
        config=cfg
    )
    try:
        trainer.load_checkpoint(checkpoint)
    except IOError:
        pass

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

    if cfg.device.rank == 0:
        done = log_folder / f"{name}.done"
        if done.exists():
            done.unlink()

    augment = [Shift(cfg.dataset.data_stride)]
    if cfg.dataset.augment:
        augment += [FlipSign(), FlipChannels(), Scale(),
                    Remix(group_size=cfg.dataset.remix_group_size)]
    augment = nn.Sequential(*augment).to(device)
    print("Agumentation pipeline:", augment)

    # define criterions
    if cfg.auxiliary_loss == "MSELoss":
        criterion = nn.MSELoss()
    elif cfg.auxiliary_loss == "MultiResolutionSTFTLoss":
        criterion = MultiResolutionSTFTLoss().to(device)
    else:
        criterion = nn.L1Loss()
    valid_criterion = nn.L1Loss()

    # Setting number of samples so that all convolution windows are full.
    # Prevents hard to debug mistake with the prediction being shifted compared
    # to the input mixture.
    samples = model.valid_length(cfg.dataset.samples)
    print(f"Number of training samples adjusted to {samples}")
    samples = samples + cfg.dataset.data_stride
    if cfg.dataset.repitch:
        # We need a bit more audio samples, to account for potential
        # tempo change.
        samples = math.ceil(samples / (1 - 0.01 * cfg.dataset.max_tempo))

    metadata_folder = out / cfg.dataset.musdb.metadata
    metadata_folder.mkdir(exist_ok=True, parents=True)
    if cfg.dataset.raw.path:
        train_set = Rawset(cfg.dataset.raw.path / "train",
                           samples=samples,
                           channels=cfg.dataset.audio_channels,
                           streams=range(1, len(model.sources) + 1),
                           stride=cfg.dataset.data_stride)

        valid_set = Rawset(cfg.dataset.raw.path / "valid", channels=cfg.dataset.audio_channels)
    elif cfg.dataset.wav.path:
        train_set, valid_set = get_wav_datasets(cfg, samples, model.sources)

        if cfg.dataset.wav.concat:
            if cfg.dataset.musdb.is_wav:
                mus_train, mus_valid = get_musdb_wav_datasets(cfg, samples, model.sources)
            else:
                mus_train, mus_valid = get_compressed_datasets(cfg, samples)
            train_set = ConcatDataset([train_set, mus_train])
            valid_set = ConcatDataset([valid_set, mus_valid])
    elif cfg.dataset.musdb.is_wav:
        train_set, valid_set = get_musdb_wav_datasets(cfg, samples, model.sources)
    else:
        train_set, valid_set = get_compressed_datasets(cfg, samples)
    print("Train set and valid set sizes", len(train_set), len(valid_set))

    if cfg.dataset.repitch:
        train_set = RepitchedWrapper(
            train_set,
            proba=cfg.dataset.repitch,
            max_tempo=cfg.dataset.max_tempo)

    best_loss = float("inf")
    for epoch, metrics in enumerate(trainer.metrics):
        print(f"Epoch {epoch:03d}: "
              f"train={metrics['train']:.8f} "
              f"valid={metrics['valid']:.8f} "
              f"best={metrics['best']:.4f} "
              f"ms={metrics.get('true_model_size', 0):.2f}MB "
              f"cms={metrics.get('compressed_model_size', 0):.2f}MB "
              f"duration={human_seconds(metrics['duration'])}")
        best_loss = metrics['best']

    for epoch in range(len(trainer.metrics), cfg.epochs):
        begin = time.time()
        model.train()
        train_loss, model_size = train_model(
            epoch, train_set, dmodel, criterion, optimizer, augment,
            quantizer=quantizer,
            batch_size=cfg.batch_size,
            batch_divide=cfg.batch_divide,
            device=device,
            repeat=cfg.repeat,
            seed=cfg.seed,
            diffq=cfg.diffq,
            workers=cfg.device.workers,
            world_size=cfg.device.world_size)
        model.eval()
        valid_loss = validate_model(
            epoch, valid_set, model, valid_criterion,
            device=device,
            rank=cfg.device.rank,
            split=cfg.split_valid,
            overlap=cfg.dataset.overlap,
            world_size=cfg.device.world_size)

        ms = 0
        cms = 0
        if quantizer and cfg.device.rank == 0:
            ms = quantizer.true_model_size()
            cms = quantizer.compressed_model_size(num_workers=min(40, cfg.device.world_size * 10))

        duration = time.time() - begin
        if valid_loss < best_loss and ms <= cfg.ms_target:
            best_loss = valid_loss
            trainer.best_state = {
                key: value.to("cpu").clone()
                for key, value in model.state_dict().items()
            }

        trainer.metrics.append({
            "train": train_loss,
            "valid": valid_loss,
            "best": best_loss,
            "duration": duration,
            "model_size": model_size,
            "true_model_size": ms,
            "compressed_model_size": cms,
        })
        if cfg.device.rank == 0:
            json.dump(trainer.metrics, open(metrics_path, "w"))

        if cfg.device.rank == 0 and not cfg.test:
            trainer.save_checkpoint(checkpoint_tmp)
            checkpoint_tmp.rename(checkpoint)

        print(f"Epoch {epoch:03d}: "
              f"train={train_loss:.8f} valid={valid_loss:.8f} best={best_loss:.4f} ms={ms:.2f}MB "
              f"cms={cms:.2f}MB "
              f"duration={human_seconds(duration)}")

    if cfg.device.world_size > 1:
        distributed.barrier()

    del dmodel
    model.load_state_dict(trainer.best_state)
    if cfg.device.eval_cpu:
        device = "cpu"
        model.to(device)
    model.eval()
    evaluate(model, cfg.dataset.musdb.path, eval_folder,
             is_wav=cfg.dataset.musdb.is_wav,
             rank=cfg.device.rank,
             world_size=cfg.device.world_size,
             device=device,
             save=cfg.save,
             split=cfg.split_valid,
             shifts=cfg.dataset.shifts,
             overlap=cfg.dataset.overlap,
             workers=cfg.device.eval_workers)
    model.to("cpu")
    if cfg.device.rank == 0:
        if not (cfg.test or cfg.test_pretrained):
            save_model(model, quantizer, cfg, model_folder / model_name)
        print("done")
        done.write_text("done")


if __name__ == "__main__":
    main()
