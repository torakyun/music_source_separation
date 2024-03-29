# Copyright (c) 2023 torakyun
#  MIT License (https://opensource.org/licenses/MIT)

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
from .repitch import RepitchedWrapper
from .states import get_quantizer, get_state, save_state, save_model
from .train import Trainer, show_names
from .utils import gpulife, sizeof_fmt
from .wav import get_wav_datasets, get_musdb_wav_datasets

from . import models
from . import optimizers

from .losses import MultiResolutionMultiScaleSTFTLoss
from .losses import MultiResolutionMultiScaleCACLoss
from .losses import MelSpectrogramLoss
from .losses import MFCCLoss
from .losses import GeneratorAdversarialLoss
from .losses import DiscriminatorAdversarialLoss
from .losses import FeatureMatchLoss

from omegaconf import OmegaConf
import hydra


def get_name():
    args = sys.argv[1:]
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
    if cfg.name:
        name = cfg.name
    else:
        name = get_name()
        cfg.name = name
    print(f"Experiment {name}")

    if cfg.dataset.musdb is None and cfg.dataset.musdbhq is None and cfg.device.rank == 0:
        print(
            "You must provide the path to the MusDB dataset. "
            "To download the MusDB dataset, see https://sigsep.github.io/datasets/musdb.html.",
            file=sys.stderr)
        sys.exit(1)

    out = Path(cfg.out)
    checkpoint_folder = out / "checkpoints"
    checkpoint_folder.mkdir(exist_ok=True, parents=True)
    checkpoint = checkpoint_folder / f"{name}.th"
    if cfg.restart and checkpoint.exists() and cfg.device.rank == 0:
        checkpoint.unlink()
    model_folder = out / "models"
    model_folder.mkdir(exist_ok=True, parents=True)

    # th.manual_seed(cfg.seed)

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
        print(f"Generator size {size}")
        size = sizeof_fmt(4 * sum(p.numel()
                                  for p in model["discriminator"].parameters()))
        print(f"Discriminator size {size}")
        return

    # get dataset

    # Setting number of samples so that all convolution windows are full.
    # Prevents hard to debug mistake with the prediction being shifted compared
    # to the input mixture.
    samples = cfg.dataset.segment * cfg.dataset.samplerate
    print(f"Number of training samples adjusted to {samples}")

    if cfg.dataset.musdbhq:
        train_set, valid_set = get_musdb_wav_datasets(cfg)
    elif cfg.dataset.musdb:
        train_set, valid_set = get_compressed_datasets(cfg)
    if cfg.dataset.wav:
        extra_train_set, extra_valid_set = get_wav_datasets(cfg)
        train_set = ConcatDataset([train_set, extra_train_set])
        valid_set = ConcatDataset([valid_set, extra_valid_set])
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
    augment = [Shift(cfg.dataset.shift * cfg.dataset.samplerate)]
    if cfg.dataset.augment:
        augment += [FlipSign(), FlipChannels(), Scale(),
                    Remix(group_size=cfg.dataset.remix_group_size)]
    augment = nn.Sequential(*augment).to(device)
    print("Agumentation pipeline:", augment)

    # define criterions
    criterion = {}
    if cfg.loss.mae["lambda"]:
        criterion["mae"] = nn.L1Loss()
    if cfg.loss.mse["lambda"]:
        criterion["mse"] = nn.MSELoss()
    if cfg.loss.stft["lambda"]:
        criterion["stft"] = MultiResolutionMultiScaleSTFTLoss(
            **cfg.loss.stft.params).to(device)
    if cfg.loss.cac["lambda"]:
        criterion["cac"] = MultiResolutionMultiScaleCACLoss(
            **cfg.loss.cac.params).to(device)
    if cfg.loss.mel["lambda"]:
        criterion["mel"] = MelSpectrogramLoss(**cfg.loss.mel.params).to(device)
    if cfg.loss.mfcc["lambda"]:
        criterion["mfcc"] = MFCCLoss(**cfg.loss.mfcc.params).to(device)
    if cfg.loss.adversarial["lambda"]:
        criterion["gen_adv"] = GeneratorAdversarialLoss(
            **cfg.loss.adversarial.generator_params).to(device)
        criterion["dis_adv"] = DiscriminatorAdversarialLoss(
            **cfg.loss.adversarial.discriminator_params).to(device)
    if cfg.loss.feat_match["lambda"]:
        criterion["feat_match"] = FeatureMatchLoss(
            **cfg.loss.feat_match.params).to(device)
    assert criterion
    # print(criterion)

    # define optimizers
    generator_optimizer_class = getattr(
        optimizers,
        cfg.optimizer.generator.name,
    )
    optimizer = {
        "generator": generator_optimizer_class(
            model["generator"].parameters(),
            **cfg.optimizer.generator.params,
        ),
    }
    if cfg.loss.adversarial["lambda"]:
        discriminator_optimizer_class = getattr(
            optimizers,
            cfg.optimizer.discriminator.name,
        )
        optimizer["discriminator"] = discriminator_optimizer_class(
            model["discriminator"].parameters(),
            **cfg.optimizer.discriminator.params,
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
            if cfg.device.world_size > 1:
                import re
                epoch = re.findall(r"epochs-(\d+)", cfg.pretrained)
                epoch = sum([int(e) for e in epoch])
                trainer.pretrained_epoch = epoch
            print("load pretrained")
        else:
            print("model init")

    # only save best state
    model_name = f"{name}.th"
    if cfg.save_model:
        if cfg.device.rank == 0:
            model["generator"].to("cpu")
            assert trainer.best_state is not None, "model needs to train for 1 epoch at least."
            # model.load_state_dict(trainer.best_state)
            model["generator"].load_state_dict(
                trainer.model["generator"].module.state_dict() if cfg.device.world_size > 1 
                else trainer.model["generator"].state_dict()
            )
            (model_folder / model_name).parent.mkdir(exist_ok=True, parents=True)
            save_model(model["generator"], quantizer, cfg, model_folder / model_name)
        return
    elif cfg.save_state:
        model_name = f"{cfg.save_state}.th"
        if cfg.device.rank == 0:
            model["generator"].to("cpu")
            # model.load_state_dict(trainer.best_state)
            model["generator"].load_state_dict(
                trainer.model["generator"].module.state_dict() if cfg.device.world_size > 1 
                else trainer.model["generator"].state_dict()
            )
            state = get_state(model["generator"], quantizer)
            (model_folder / model_name).parent.mkdir(exist_ok=True, parents=True)
            save_state(state, model_folder / model_name)
        return

    # run training loop
    trainer.run()
    if cfg.device.rank == 0:
        print("done")


if __name__ == "__main__":
    main()
