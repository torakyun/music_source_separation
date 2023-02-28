# Copyright (c) 2023 torakyun
#  MIT License (https://opensource.org/licenses/MIT)

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import pprint
pp = pprint.PrettyPrinter(indent=4)


def main():
    parser = argparse.ArgumentParser("mss.mos",
                                     description="Make MOS result table")
    parser.add_argument("-f", "--folder",
                        type=Path,
                        default=Path("../mos/csv"),
                        help="Load csv folder")
    args = parser.parse_args()

    # load csv
    df_list = []
    for i, file in enumerate(args.folder.iterdir()):
        df = pd.read_csv(str(file), index_col=0)
        df_list.append(df)
    df = pd.concat(df_list, axis=0).reset_index(drop=True)

    # make choices
    names = df.columns.values[2:]
    models = list(df["model"].unique())
    targets = list(df["target"].unique())
    print(i+1)
    print(names)
    print(models)
    print(targets)
    choices = {
        name: {
            model: {
                target: [] for target in ["all"]+targets
            } for model in models
        } for name in names
    }
    for name in names:
        for model in models:
            for target in targets:
                setting = (df["model"] == model) & (df["target"] == target)
                choices[name][model][target] = df[setting][name].to_numpy().tolist()
                choices[name][model]["all"] += choices[name][model][target]

    # make scores
    scores = {
        name: {
            model: {
                target: None for target in ["all"]+targets
            } for model in models
        } for name in names
    }
    for name in names:
        for model in models:
            for target in ["all"]+targets:
                choice = np.array(choices[name][model][target])
                scores[name][model][target] = (
                    round(choice.mean(), 3), round(choice.std() / choice.shape[0]**0.5, 3))

    # make latex
    length = max(map(len, models))
    print()
    for name in names:
        print(name, end="\n\n")
        texts = []
        for model in models:
            text = model
            for target in ["all"]+targets:
                score = scores[name][model][target]
                text += f" & {score[0]}±{score[1]}"
            text += "\\\\"
            texts.append(text)
        length = max(map(len, texts))
        [print(text.rjust(length)) for text in texts]
        print()


if __name__ == "__main__":
    main()
