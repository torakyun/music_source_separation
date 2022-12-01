import sys
import json
import random
from pathlib import Path
import argparse
from tqdm import tqdm
import musdb
import torch
import torchaudio

from demucs.audio import convert_audio
from demucs.states import load_model
from demucs.utils import human_seconds


def main():
    parser = argparse.ArgumentParser("demucs.mos",
                                     description="Separate test set for MOS")
    parser.add_argument("-n", "--names",
                        required=True,
                        nargs="*",
                        type=str,
                        help='Model names')
    parser.add_argument("-r", "--repeat",
                        type=int,
                        default=5)
    parser.add_argument("-s", "--seed",
                        type=int,
                        default=1)
    parser.add_argument("--extract_seed",
                        type=int,
                        default=0)
    parser.add_argument("--sound_check",
                        action='store_true')
    parser.add_argument("-m", "--musdb",
                        type=Path,
                        default=Path("../musdb18hq"),
                        help="Load musdb folder")
    parser.add_argument("--not_wav",
                        action='store_false',
                        dest="is_wav",
                        default=True)
    parser.add_argument("--models",
                        type=Path,
                        default=Path("../out/models"),
                        help="Load pretrained models folder")
    parser.add_argument("--out",
                        type=Path,
                        default=Path("../mos"),
                        help="Store audio folder")
    parser.add_argument("-d",
                        "--device",
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use, default is cuda if available else cpu")

    # load data and models
    args = parser.parse_args()
    test_set = musdb.DB(args.musdb, subsets=["test"], is_wav=args.is_wav)
    models = []
    for name in args.names:
        model = load_model(args.models / (name + ".th")).to(args.device)
        model.eval()
        models.append(model)
    sources = list(models[0].sources)
    samplerate = models[0].samplerate
    channels = models[0].audio_channels

    # make 8 seconds extracts
    # (offsets: 50 track * (offset))
    src_rate = 44100  # hardcoded for now...
    length = 8 * src_rate
    offsets = [
        2254257,  # extract_seed  1  AM Contra - Heart Peripheral
        1058756,  # extract_seed  1  Al James - Schoolboy Facination
        2139674,  # extract_seed  1  Angels In Amplifiers - I'm Alright
        1978347,  # extract_seed  1  Arise - Run Run Run
        2836752,  # extract_seed  2  BKS - Bulldozer
        7541208,  # extract_seed  1  BKS - Too Much
        7922960,  # extract_seed  1  Ben Carrigan - We'll Talk About It All Tonight
        6368886,  # extract_seed  1  Bobby Nobody - Stitch Up
        3522457,  # extract_seed  1  Buitraker - Revo X
        3560440,  # extract_seed  2  Carlos Gonzalez - A Place For Us
        8184876,  # extract_seed  1  Cristina Vane - So Easy
        1965710,  # extract_seed  3  Detsky Sad - Walkie Talkie
        7493686,  # extract_seed  1  Enda Reilly - Cur An Long Ag Seol
        6539906,  # extract_seed  1  Forkupines - Semantics
        5314447,  # extract_seed  2  Georgia Wonder - Siren
        7225437,  # extract_seed  2  Girls Under Glass - We Feel Alright
        3995940,  # extract_seed  3  Hollow Ground - Ill Fate
        7472357,  # extract_seed  1  James Elder & Mark M Thompson - The English Actor
        4468285,  # extract_seed  1  Juliet's Rescue - Heartbeats
        3837993,  # extract_seed  1  Little Chicago's Finest - My Own
        9129775,  # extract_seed  2  Louis Cressy Band - Good Time
        7463414,  # extract_seed  2  Lyndsey Ollard - Catching Up
        3597042,  # extract_seed  4  M.E.R.C. Music - Knockout
        4500332,  # extract_seed  2  Moosmusic - Big Dummy Shake
        4366340,  # extract_seed  4  Motor Tapes - Shore
        884362,   # extract_seed 27  Mu - Too Bright
        10897575, # extract_seed  1  Nerve 9 - Pray For The Rain
        4541697,  # extract_seed  1  PR - Happy Daze
        1335739,  # extract_seed  2  PR - Oh No
        6395545,  # extract_seed  1  Punkdisco - Oral Hygiene
        3633934,  # extract_seed  1   Raft Monk - Tiring
        7081940,  # extract_seed  1  Sambasevam Shanmugam - Kaathaadi
        10164507, # extract_seed  4  Secretariat - Borderline
        4520313,  # extract_seed  3  Secretariat - Over The Top
        8852152,  # extract_seed  1  Side Effects Project - Sing With Me
        1859684,  # extract_seed  1  Signe Jakobsen - What Have You Done To Me
        13005985, # extract_seed  3  Skelpolu - Resurrection
        8318349,  # extract_seed  1  Speak Softly - Broken Man
        4149378,  # extract_seed  4  Speak Softly - Like Horses
        5454993,  # extract_seed  2  The Doppler Shift - Atrophy
        5799890,  # extract_seed  1  The Easton Ellises (Baumi) - SDRNR
        3873297,  # extract_seed  1  The Easton Ellises - Falcon 69
        8559176,  # extract_seed  2  The Long Wait - Dark Horses
        3670536,  # extract_seed  1  The Mountaineering Club - Mallory
        6034620,  # extract_seed  2  The Sunshine Garcia Band - For I Am The Moon
        7710866,  # extract_seed  1  Timboz - Pony
        4801360,  # extract_seed  4  Tom McKenzie - Directions
        8302765,  # extract_seed  3  Triviul feat. The Fiend - Widow
        3640436,  # extract_seed  3  We Fell From The Sky - Not You
        9335754,  # extract_seed  1  Zeno - Signs
    ]
    random.seed(args.extract_seed)
    for index, offset in enumerate(tqdm(
            offsets,
            ncols=120,
            desc="make extracts",
            leave=True,
            file=sys.stdout,
            unit=" track")):
        folder = args.out / (f"sound_check")
        if args.extract_seed:
            offset = random.randrange(
                test_set.tracks[index].audio.shape[0] - length + 1)
            folder = args.out / (f"sound_check_{args.extract_seed}")
        offsets[index] = offset

        # sound check ( not contain no volume extract ? )
        if args.sound_check:
            data = test_set.tracks[index]
            for target in sources:
                reference = torch.from_numpy(
                    data.targets[target].audio).t().float()[..., offset:offset+length]
                audio = convert_audio(
                    reference, src_rate, samplerate, channels)
                filename = folder / data.name / f"{target}.wav"
                filename.parent.mkdir(exist_ok=True, parents=True)
                torchaudio.save(
                    filepath=str(filename),
                    src=audio,
                    sample_rate=samplerate)
            # print(offset, data.name)

    # make samples
    # (samples: 5 repeat * 20 track * (id, offset, model, target))
    random.seed(args.seed)
    num_track = 20
    samples = [
        [
            (
                index,
                offsets[index],
                random.randrange(1 + len(models)),
                random.randrange(len(sources)),
            ) for index in random.sample(list(range(len(test_set))), num_track)
        ] for _ in range(args.repeat)
    ]

    # separate and write wave
    folder = args.out / f"seed_{args.seed}"
    for repeat, tracks in enumerate(samples, 1):
        repeat_folder = folder / f"MEAN_OPINION_SCORE_{repeat}"
        repeat_folder.mkdir(exist_ok=True, parents=True)
        metadata = []
        for track, sample in enumerate(tqdm(
                tracks,
                ncols=120,
                desc=f"[{repeat}/{args.repeat}] make samples",
                leave=True,
                file=sys.stdout,
                unit=" track"), 1):
            id, offset, model, target = sample
            data = test_set.tracks[id]
            metadata.append(
                {
                    "track": data.name,
                    "offset": human_seconds(offset / samplerate),
                    "target": sources[target],
                    "model": args.names[model-1] if model else "ground truth",
                }
            )
            if model:
                mix = torch.from_numpy(data.audio).t().float()[..., offset:offset+length]
                mix = convert_audio(
                    mix, src_rate, samplerate, channels)
                mix = mix.to(args.device)
                ref = mix.mean(dim=0)  # mono mixture
                mix = (mix - ref.mean()) / ref.std()
                with torch.no_grad():
                    estimate = models[model-1](mix[None])[0, target]
                audio = estimate * ref.std() + ref.mean()
            else:
                reference = torch.from_numpy(
                    data.targets[sources[target]].audio).t().float()[..., offset:offset+length]
                audio = convert_audio(
                    reference, src_rate, samplerate, channels)
            filename = repeat_folder / f"mos_{track}.wav"
            torchaudio.save(
                filepath=str(filename),
                src=audio,
                sample_rate=samplerate)
        json.dump(metadata, open(repeat_folder / "metadata.json", "w"))

    """
    # make 8 seconds extracts
    # (mixtures: 50 song * 1 channel * 8 second)
    # (references: 50 song * 4 source * 1 channel * 8 second)
    src_rate = 44100  # hardcoded for now...
    length = 8 * samplerate
    mixtures = [None] * len(test_set)
    references = [None] * len(test_set)
    for index in tqdm(range(len(test_set)),
                    ncols=120,
                    desc="make 8 seconds extracts",
                    leave=True,
                    file=sys.stdout,
                    unit=" track"):
        track = test_set.tracks[index]
        mix = torch.from_numpy(track.audio).t().float()
        if mix.dim() == 1:
            mix = mix[None]
        mix = mix.to(args.device)
        ref = mix.mean(dim=0)  # mono mixture
        mix = (mix - ref.mean()) / ref.std()
        mix = convert_audio(
            mix, src_rate, samplerate, channels)
        offset = random.randrange(mix.size(-1) - length + 1)
        mixtures[index] = mix[..., offset:offset+length]
        reference = torch.stack(
            [torch.from_numpy(track.targets[name].audio).t() for name in targets])
        if reference.dim() == 2:
            reference = reference[:, None]
        reference = reference.to(args.device)
        reference = convert_audio(
            reference, src_rate, samplerate, channels)
        references[index] = reference[..., offset:offset+length]
        del mix, reference

    # separate 8 seconds extracts
    # (sources: 50 song * (1 + 2) model * 4 source * 1 channel * 8 second)
    sources = [[None for _ in range(1 + len(models))]] * len(test_set)
    for i in tqdm(range(len(test_set)),
                    ncols=120,
                    desc="separate 8 seconds extracts",
                    leave=True,
                    file=sys.stdout,
                    unit=" track"):
        sources[i][0] = references[i]
        for j, model in enumerate(models):
            model.to(args.device)
            model.eval()
            with torch.no_grad():
                estimate = model(mixtures[i])
            estimate = estimate * ref.std() + ref.mean()
            assert length == estimate.size(-1)
            sources[i][j+1] = estimate
    sources = torch.stack(sources)
    print("sources: ", sources.size())

    # make samples
    # (samples: 5 repeat * 4 sources * 20 song * 1 channel * 8 second))
    num_track = 20
    selects = [
        [
            [
                (index, random.randrange(1 + len(models))) for index in random.sample(list(range(len(test_set))), num_track)
            ] for _ in targets
        ] for _ in range(args.repeat)
    ]
    samples = [
        [
            [
                [None] * num_track
            ] for _ in targets
        ] for _ in range(args.repeat)
    ]
    for repeat, targets in enumerate(selects):
        for target, tracks in enumerate(targets):
            for track, sample in enumerate(tracks):
                samples[repeat, target, track] = sources[sample[0], sample[1], target]

    # write wave
    folder = args.out / f"seed_{args.seed}"
    for repeat, targets in enumerate(selects):
        for target, tracks in enumerate(targets):
            for track, sample in enumerate(tracks):
                filename = folder / f"{repeat}" / \
                    targets[target] / f"{track}.wav"
                filename.parent.mkdir(exist_ok=True, parents=True)
                torchaudio.save(
                    filepath=str(filename),
                    src=samples[repeat, target, track],
                    sample_rate=samplerate)
    """


if __name__ == "__main__":
    main()
