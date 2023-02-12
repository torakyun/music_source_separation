# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from hashlib import sha256
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser("demucs.download_v2",
                                     description="Download v2 pretrained model")
    parser.add_argument("-n", "--name", default="demucs_v2")
    parser.add_argument("--models",
                        type=Path,
                        default=Path("../out/models"),
                        help="Store downloaded pretrained models")

    args = parser.parse_args()
    name = args.name + ".th"
    model_path = args.models / name
    url = "https://dl.fbaipublicfiles.com/demucs/v2.0/demucs.th"

    # download to model_path
    import requests
    import tqdm
    try:
        response = requests.get(url, stream=True)
        total_length = int(response.headers.get('content-length', 0))
        with tqdm.tqdm(total=total_length, ncols=120, unit="B", unit_scale=True) as bar:
            with open(model_path, "wb") as output:
                for data in response.iter_content(chunk_size=4096):
                    output.write(data)
                    bar.update(len(data))
    except:  # noqa, re-raising
        if model_path.exists():
            model_path.unlink()
        raise

    # verify model_path
    import hashlib
    sha = 'f6c4148ba0dc92242d82d7b3f2af55c77bd7cb4ff1a0a3028a523986f36a3cfd'
    hasher = hashlib.sha256()
    with open(model_path, "rb") as f:
        while True:
            data = f.read(65536)
            if not data:
                break
            hasher.update(data)
    signature = hasher.hexdigest()
    if signature != sha:
        print(
            f"Invalid sha256 signature for the file {model_path}. Expected {sha256} but got "
            f"{signature}.\nIf you have recently updated the repo, it is possible "
            "the checkpoints have been updated. It is also possible that a previous "
            f"download did not run to completion.\nPlease delete the file '{model_path.absolute()}' "
            "and try again.",
            file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
