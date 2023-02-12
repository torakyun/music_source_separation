# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root folderectory of this source tree.

import argparse
from hashlib import sha256
import sys
from pathlib import Path
import subprocess
import json
import numpy as np
from tqdm import tqdm

import julius
import torch
import torchaudio as ta
import librosa
import musdb
import museval
from scipy.io import wavfile
import matplotlib.pyplot as plt

from demucs.audio import AudioFile, convert_audio_channels, convert_audio
from demucs.apply import apply_model
from demucs.states import load_model

from distutils.version import LooseVersion
is_pytorch_17plus = LooseVersion(torch.__version__) >= LooseVersion("1.7")


def load_track(track, device, audio_channels, samplerate):
    errors = {}
    wav = None

    try:
        wav = AudioFile(track).read(
            streams=0,
            samplerate=samplerate,
            channels=audio_channels).to(device)
    except FileNotFoundError:
        errors['ffmpeg'] = 'Ffmpeg is not installed.'
    except subprocess.CalledProcessError:
        errors['ffmpeg'] = 'FFmpeg could not read the file.'

    if wav is None:
        try:
            wav, sr = ta.load(str(track))
        except RuntimeError as err:
            errors['torchaudio'] = err.args[0]
        else:
            wav = convert_audio_channels(wav, audio_channels)
            wav = wav.to(device)
            wav = julius.resample_frac(wav, sr, samplerate)

    if wav is None:
        print(f"Could not load file {track}. "
              "Maybe it is not a supported file format? ")
        for backend, error in errors.items():
            print(f"When trying to load using {backend}, got the following error: {error}")
        sys.exit(1)
    return wav


def irm(mixture, references, device=None):
    def stft(x):
        x_size = x.size()
        x = x.reshape(-1, x_size[-1])  # (B, C, T) -> (B x C, T)
        if is_pytorch_17plus:
            x_stft = torch.stft(
                x, n_fft=2048, hop_length=2048, win_length=1024, window=torch.hann_window(1024).to(device), return_complex=False
            )
        else:
            x_stft = torch.stft(x, n_fft=2048, hop_length=2048, win_length=1024, window=torch.hann_window(1024).to(device))
        return x_stft.reshape(*x_size[:-1], x_stft.size(-3), x_stft.size(-2), x_stft.size(-1))

    def istft(x_stft):
        x_stft_size = x_stft.size()
        x_stft = x_stft.reshape(-1, x_stft_size[-3], x_stft_size[-2], x_stft_size[-1])  # (B, C, T) -> (B x C, T)
        if is_pytorch_17plus:
            x = torch.istft(
                x_stft, n_fft=2048, hop_length=2048, win_length=1024, window=torch.hann_window(1024).to(device), return_complex=False
            )
        else:
            x = torch.istft(x_stft, n_fft=2048, hop_length=2048, win_length=1024, window=torch.hann_window(1024).to(device))
        return x.reshape(*x_stft_size[:-3], x.size(-1))

    length = references.size(-1)
    mixture = stft(mixture.to(device))
    references = stft(references.to(device))
    references = torch.sqrt(references[..., 0]**2 + references[..., 1]**2)
    references_mag_sum = torch.clamp(references.sum(dim=0), min=1e-7).unsqueeze(0).expand(references.size())
    masks = references / references_mag_sum
    del references, references_mag_sum
    torch.cuda.empty_cache()
    estimates_mag = mixture.unsqueeze(0) * masks.unsqueeze(-1)
    del mixture, masks
    torch.cuda.empty_cache()
    estimates = istft(estimates_mag)
    delta = length - estimates.size(-1)
    return torch.nn.functional.pad(estimates, (0, delta)).cpu()


def _write_figure(title, ref, est, dif, fig, axes, folder, xmax=None, vmax=None):
    sources = ["drums", "bass", "other", "vocals"]
    targets = ["reference", "estimate", "difference"]
    # fig.suptitle(title, fontsize='xx-large')
    for i, source in enumerate(sources):
        # Set label
        for j, target in enumerate(targets):
            # if i == 0:
            #     axes[i, j].text(0.5, 1, target)
            # axes[i, j].text(-0.1, 0.5, source)
            axes[i, j].set_xlabel("frame")
            axes[i, j].set_ylabel("freq_bin")
            if xmax:
                axes[i, j].set_xlim((0, xmax))

        # Draw image
        min = torch.min(torch.stack([ref[i], est[i]]))
        if min < 0:
            ref[i] = torch.clamp(ref[i], min=min) - min
            est[i] = torch.clamp(est[i], min=min) - min
        if not vmax:
            vmax = torch.max(torch.stack([ref[i], est[i], dif[i]])).item()
        axes[i, 0].imshow(
            ref[i].cpu(), origin="lower", aspect="auto", cmap="gnuplot", vmin=0, vmax=vmax)
        axes[i, 1].imshow(
            est[i].cpu(), origin="lower", aspect="auto", cmap="gnuplot", vmin=0, vmax=vmax)
        im = axes[i, 2].imshow(
            dif[i].cpu(), origin="lower", aspect="auto", cmap="gnuplot", vmin=0, vmax=vmax)
        fig.colorbar(im, ax=axes[i, 2])

    # Save figure
    title = title.replace(" ", "_")
    path = Path(folder) / f"{title}.png"
    fig.savefig(path)

    # Remove colorbar and clear axis
    for i, _ in enumerate(sources):
        axes[i, 2].images[-1].colorbar.remove()
    plt.cla()


def save_spectrogram(references, estimates, fig, axes, folder, samplerate=24000, window="hann"):
    # Spectrogram params
    stft_params = {
        "n_fft": 2048,
        "hop_length": 240,
        "win_length": 1200,
        "center": True,
        "normalized": False,
        "onesided": True,
    }
    if window is not None:
        window_func = getattr(torch, f"{window}_window")
        stft_params["window"] = window_func(
            stft_params["win_length"], dtype=references.dtype, device=references.device)
    else:
        stft_params["window"] = None
    if is_pytorch_17plus:
        stft_params["return_complex"] = False
    mel_params = {
        "sr": samplerate,
        "n_fft": stft_params["n_fft"],
        "n_mels": 80,
        "fmin": 0,
        "fmax": samplerate // 2,
    }

    # Calculate STFT
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
    _write_figure(
        "STFT", references_mag, estimates_mag, differences_stft, fig, axes, folder, vmax=5)
    del differences_stft

    # Magnitude spectrogram
    differences_mag = (references_mag - estimates_mag).abs()
    _write_figure(
        "Magnitude Spectrogram", references_mag, estimates_mag, differences_mag, fig, axes, folder, vmax=5)
    del differences_mag

    # Log-scale Magnitude spectrogram
    references_log_mag = torch.log(references_mag)
    estimates_log_mag = torch.log(estimates_mag)
    differences_log_mag = (references_log_mag - estimates_log_mag).abs()
    _write_figure(
        "Log-Scale Magnitude Spectrogram", references_log_mag, estimates_log_mag, differences_log_mag, fig, axes, folder)
    del references_log_mag, estimates_log_mag, differences_log_mag

    # Mel spectrogram
    melmat = librosa.filters.mel(**mel_params)
    melmat = torch.from_numpy(melmat).to(references_mag.device).double()
    references_mel = torch.clamp(torch.matmul(
        melmat, references_mag), min=1e-7)
    estimates_mel = torch.clamp(torch.matmul(
        melmat, estimates_mag), min=1e-7)
    differences_mel = (references_mel - estimates_mel).abs()
    _write_figure(
        "Mel Spectrogram", references_mel, estimates_mel, differences_mel, fig, axes, folder, vmax=1)
    del melmat, differences_mel

    # Log-scale Mel spectrogram
    references_log_mel = torch.log(references_mel)
    estimates_log_mel = torch.log(estimates_mel)
    differences_log_mel = (references_log_mel - estimates_log_mel).abs()
    _write_figure(
        "Log-Scale Mel Spectrogram", references_log_mel, estimates_log_mel, differences_log_mel, fig, axes, folder)


def main():
    parser = argparse.ArgumentParser("demucs.evaluate",
                                     description="Evaluate the sources for the given tracks")

    # Test data params
    parser.add_argument("-m",
                        "--musdb",
                        type=Path,
                        default=None,
                        help="Path to musdb root")
    parser.add_argument("--not_wav",
                        action='store_false',
                        dest="is_wav",
                        default=True)
    parser.add_argument("-t", "--tracks", nargs='+', type=int, default=[], help='Track index')

    # Model params
    parser.add_argument("--irm",
                        action="store_true",
                        help="Calculate IRM's result.")
    parser.add_argument("--samplerate",
                        default=24000,
                        type=int,
                        help="Samplerate(Only IRM).")
    parser.add_argument("--sources",
                        nargs="*",
                        type=str,
                        default=["drums", "bass", "other", "vocals"],
                        help="Sources(Only IRM).")
    parser.add_argument("--channels",
                        default=1,
                        type=int,
                        help="Channels(Only IRM).")
    parser.add_argument("--models",
                        type=Path,
                        default=Path("../out/models"),
                        help="Path to trained models. "
                        "Also used to store downloaded pretrained models")
    parser.add_argument("-n", "--names",
                        nargs="*",
                        type=str,
                        help='Model names')
    parser.add_argument("-d",
                        "--device",
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use, default is cuda if available else cpu")
    parser.add_argument("--shifts",
                        default=0,
                        type=int,
                        help="Number of random shifts for equivariant stabilization."
                        "Increase separation time but improves quality for Demucs. 10 was used "
                        "in the original paper.")
    parser.add_argument("--overlap",
                        default=0.25,
                        type=float,
                        help="Overlap between the splits.")
    parser.add_argument("--no-split",
                        action="store_false",
                        dest="split",
                        default=True,
                        help="Doesn't split audio in chunks. This can use large amounts of memory.")

    # Output params
    parser.add_argument("-o",
                        "--out",
                        type=Path,
                        default=Path("../sdr"),
                        help="Folder where to put sdr results.")
    parser.add_argument("--save_spectrogram",
                        action="store_true",
                        help="Save spectrogram (.png).")
    parser.add_argument("--second",
                        default=600,
                        type=int,
                        help="Track length for STFT.")
    parser.add_argument("--save_audio",
                        action="store_true",
                        help="Save audio (.wav).")
    args = parser.parse_args()

    # Load models
    models = []
    for name in args.names:
        model_path = args.models / (name + ".th")
        model = load_model(model_path).to(args.device)
        model.eval()
        models.append(model)
    samplerate = models[0].samplerate if models else args.samplerate
    sources = list(models[0].sources) if models else args.sources
    channels = models[0].audio_channels if models else args.channels

    model_funcs = {
        name: (lambda mix: 
            apply_model(
                models[i], 
                mix[None].to(args.device),
                shifts=args.shifts,
                split=args.split,
                overlap=args.overlap
            )[0]
        ) for i, name in enumerate(args.names)}

    # Load tracks from the original musdb set
    test_set = musdb.DB(args.musdb, subsets=["test"], is_wav=args.is_wav)
    src_rate = 44100  # hardcoded for now...
    indexes = args.tracks if args.tracks else list(range(len(test_set)))

    print(f"Results will be stored in {args.out.resolve()}")

    # Evaluate
    names = ["IRM"] + args.names if args.irm else args.names
    targets = ["all"] + sources
    fig = plt.figure(constrained_layout=True, figsize=(20, 15))
    axes = fig.subplots(
        nrows=len(sources), ncols=3, sharex=False)
    scores = {name: {target: [] for target in targets} for name in names}
    for index in indexes:
        track = test_set.tracks[index]
        mix = torch.from_numpy(track.audio).t().float()
        ref = mix.mean(dim=0)  # mono mixture
        mix = (mix - ref.mean()) / ref.std()
        mix = convert_audio(mix, src_rate, samplerate, channels)
        references = torch.stack(
            [torch.from_numpy(track.targets[target].audio).t() for target in sources])
        references = convert_audio(references, src_rate, samplerate, channels).to(args.device)

        for name in tqdm(names,
                ncols=120,
                desc=f"[{index+1}/{len(indexes)}]",
                leave=True,
                file=sys.stdout,
                unit=" track"):

            # Separate mixture
            if name == "IRM":
                estimates = irm(mix, references, args.device)
            else:
                estimates = model_funcs[name](mix)
            estimates = estimates * ref.std() + ref.mean()

            # Save spectrogram(.png)
            if args.save_spectrogram:
                if isinstance(references, np.ndarray):
                    references = torch.from_numpy(references).transpose(1, 2).to(args.device)
                track_folder = args.out / name / track.name
                track_folder.mkdir(exist_ok=True, parents=True)
                second = samplerate * args.second
                save_spectrogram(
                    references.mean(dim=1)[..., :second],
                    estimates.mean(dim=1)[..., :second],
                    fig, axes, track_folder, samplerate=samplerate, window="hann"
                )

            if isinstance(references, torch.Tensor):
                references = references.transpose(1, 2).cpu().numpy()
            estimates = estimates.transpose(1, 2).cpu().numpy()

            # Save audio(.wav)
            if args.save_audio:
                track_folder = args.out / name / track.name
                track_folder.mkdir(exist_ok=True, parents=True)
                for target, estimate in zip(sources, estimates):
                    wavfile.write(
                        str(track_folder / (target + ".wav")), samplerate, estimate)

            # Calculate SDR
            win = int(1. * samplerate)
            hop = int(1. * samplerate)
            sdr, isr, sir, sar = museval.evaluate(references, estimates, win=win, hop=hop)
            sdr_all = 0
            for idx, target in enumerate(sources):
                scores[name][target].append(np.nanmedian(sdr[idx].tolist()))
                sdr_all += scores[name][target][-1]
            sdr_all /= (idx + 1)
            scores[name]["all"].append(sdr_all)
            # print()
            # print(scores[name])

    # Save SDR(.json)
    scores = {name: {target: np.nanmedian(scores[name][target]) for target in targets} for name in names}
    texts = ["SDR & "+" & ".join(targets)+" \\\\ \hline\hline"]
    for name in names:
        # Write file
        out = args.out / name
        out.mkdir(parents=True, exist_ok=True)
        json.dump(scores[name], open(out / "sdr.json", "w"))

        # Make latex text
        text = name
        for target in targets:
            score = round(scores[name][target], 3)
            text += f" & {score}"
        text += " \\\\ \hline"
        texts.append(text)
    length = max(map(len, texts))
    [print(text.rjust(length)) for text in texts]


if __name__ == "__main__":
    main()
