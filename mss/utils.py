# Copyright (c) 2023 torakyun
# Copyright (c) Facebook, Inc. and its affiliates.
#  MIT License (https://opensource.org/licenses/MIT)

import errno
import os
import random
import socket
import tempfile
from contextlib import contextmanager
import subprocess
import shlex
import typing as tp
import math

import torch
from torch import distributed
from torch.nn import functional as F


def gpulife(title):
    """
    Returns GPU usage information in string.

    Parameters
    ----------
    None

    Returns
    -------
    msg: str
    """

    def _gpuinfo():
        command = 'nvidia-smi -q -d MEMORY | sed -n "/FB Memory Usage/,/Free/p" | sed -e "1d" -e "s/ MiB//g" | cut -d ":" -f 2 | cut -c2-'
        commands = [shlex.split(part) for part in command.split(' | ')]
        for i, cmd in enumerate(commands):
            if i == 0:
                res = subprocess.Popen(cmd, stdout=subprocess.PIPE)
            else:
                res = subprocess.Popen(
                    cmd, stdin=res.stdout, stdout=subprocess.PIPE)
        return tuple(map(int, res.communicate()[0].decode('utf-8').strip().split('\n')))

    res = _gpuinfo()
    if len(res) == 3:
        total, used, free = res
    if len(res) == 4:
        total, reserved, used, free = res
    percent = int(used / total * 100)
    msg = 'GPU RAM Usage: {} {}/{} MiB ({:.1f}%)'.format(
        '|' * (percent // 5) + '.' * (20 - percent // 5), used, total, used/total*100)
    print(title, ": ", msg)


def center_trim(tensor: torch.Tensor, reference: tp.Union[torch.Tensor, int]):
    """
    Center trim `tensor` with respect to `reference`, along the last dimension.
    `reference` can also be a number, representing the length to trim to.
    If the size difference != 0 mod 2, the extra sample is removed on the right side.
    """
    ref_size: int
    if isinstance(reference, torch.Tensor):
        ref_size = reference.size(-1)
    else:
        ref_size = reference
    delta = tensor.size(-1) - ref_size
    if delta < 0:
        raise ValueError(
            "tensor must be larger than reference. " f"Delta is {delta}.")
    if delta:
        tensor = tensor[..., delta // 2:-(delta - delta // 2)]
    return tensor


def unfold(a, kernel_size, stride):
    """Given input of size [*OT, T], output Tensor of size [*OT, F, K]
    with K the kernel size, by extracting frames with the given stride.
    This will pad the input so that `F = ceil(T / K)`.
    see https://github.com/pytorch/pytorch/issues/60466
    """
    *shape, length = a.shape
    n_frames = math.ceil(length / stride)
    tgt_length = (n_frames - 1) * stride + kernel_size
    a = F.pad(a, (0, tgt_length - length))
    strides = list(a.stride())
    assert strides[-1] == 1, 'data should be contiguous'
    strides = strides[:-1] + [stride, 1]
    return a.as_strided([*shape, n_frames, kernel_size], strides)


def average_metric(metric, count=1.):
    """
    Average `metric` which should be a float across all hosts. `count` should be
    the weight for this particular host (i.e. number of examples).
    """
    metric = torch.tensor([count, count * metric],
                          dtype=torch.float32, device='cuda')
    distributed.all_reduce(metric, op=distributed.ReduceOp.SUM)
    return metric[1].item() / metric[0].item()


def free_port(host='', low=20000, high=40000):
    """
    Return a port number that is most likely free.
    This could suffer from a race condition although
    it should be quite rare.
    """
    sock = socket.socket()
    while True:
        port = random.randint(low, high)
        try:
            sock.bind((host, port))
        except OSError as error:
            if error.errno == errno.EADDRINUSE:
                continue
            raise
        return port


def sizeof_fmt(num, suffix='B'):
    """
    Given `num` bytes, return human readable size.
    Taken from https://stackoverflow.com/a/1094933
    """
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


def human_seconds(seconds, display='.2f'):
    """
    Given `seconds` seconds, return human readable duration.
    """
    value = seconds * 1e6
    ratios = [1e3, 1e3, 60, 60, 24]
    names = ['us', 'ms', 's', 'min', 'hrs', 'days']
    last = names.pop(0)
    for name, ratio in zip(names, ratios):
        if value / ratio < 0.3:
            break
        value /= ratio
        last = name
    return f"{format(value, display)} {last}"


@contextmanager
def temp_filenames(count, delete=True):
    names = []
    try:
        for _ in range(count):
            names.append(tempfile.NamedTemporaryFile(delete=False).name)
        yield names
    finally:
        if delete:
            for name in names:
                os.unlink(name)
