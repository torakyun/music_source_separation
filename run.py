#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Run training locally on all visible GPUs. Start only
one task per node as this script will spawn one child for each GPU.
"""
import subprocess as sp
import sys
import time
import re

import torch as th

from demucs.utils import free_port
from demucs.__main__ import get_name


def main():
    args = sys.argv[1:]
    gpus = max(th.cuda.device_count(), 1)

    port = free_port()
    name = get_name()
    args += [f"+name=\"{name}\"", f"device.world_size={gpus}",
             f"+device.master=127.0.0.1:{port}"]
    tasks = []

    for gpu in range(gpus):
        kwargs = {}
        if gpu > 0:
            kwargs['stdin'] = sp.DEVNULL
            kwargs['stdout'] = sp.DEVNULL
            # We keep stderr to see tracebacks from children.
        tasks.append(sp.Popen(["python3", "-m", "demucs"] +
                              args + [f"device.rank={gpu}"], **kwargs))
        tasks[-1].rank = gpu

    failed = False
    try:
        while tasks:
            for task in tasks:
                try:
                    exitcode = task.wait(0.1)
                except sp.TimeoutExpired:
                    continue
                else:
                    tasks.remove(task)
                    if exitcode:
                        print(f"Task {task.rank} died with exit code "
                              f"{exitcode}",
                              file=sys.stderr)
                        failed = True
            if failed:
                break
            time.sleep(1)
    except KeyboardInterrupt:
        for task in tasks:
            task.terminate()
        raise
    if failed:
        for task in tasks:
            task.terminate()
        sys.exit(1)


if __name__ == "__main__":
    main()
