# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import errno
import functools
import hashlib
import inspect
import io
import os
import random
import socket
import tempfile
import warnings
import zlib
from contextlib import contextmanager
import subprocess
import shlex
import typing as tp
import math

from diffq import UniformQuantizer, DiffQuantizer
import torch
import tqdm
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


class TensorChunk:
    def __init__(self, tensor, offset=0, length=None):
        total_length = tensor.shape[-1]
        assert offset >= 0
        assert offset < total_length

        if length is None:
            length = total_length - offset
        else:
            length = min(total_length - offset, length)

        self.tensor = tensor
        self.offset = offset
        self.length = length
        self.device = tensor.device

    @property
    def shape(self):
        shape = list(self.tensor.shape)
        shape[-1] = self.length
        return shape

    def padded(self, target_length):
        delta = target_length - self.length
        total_length = self.tensor.shape[-1]
        assert delta >= 0

        start = self.offset - delta // 2
        end = start + target_length

        correct_start = max(0, start)
        correct_end = min(total_length, end)

        pad_left = correct_start - start
        pad_right = end - correct_end

        out = F.pad(self.tensor[..., correct_start:correct_end], (pad_left, pad_right))
        assert out.shape[-1] == target_length
        return out


def tensor_chunk(tensor_or_chunk):
    if isinstance(tensor_or_chunk, TensorChunk):
        return tensor_or_chunk
    else:
        assert isinstance(tensor_or_chunk, th.Tensor)
        return TensorChunk(tensor_or_chunk)


def apply_model(model, mix, shifts=None, split=False,
                overlap=0.25, transition_power=1., progress=False):
    """
    Apply model to a given mixture.

    Args:
        shifts (int): if > 0, will shift in time `mix` by a random amount between 0 and 0.5 sec
            and apply the oppositve shift to the output. This is repeated `shifts` time and
            all predictions are averaged. This effectively makes the model time equivariant
            and improves SDR by up to 0.2 points.
        split (bool): if True, the input will be broken down in 8 seconds extracts
            and predictions will be performed individually on each and concatenated.
            Useful for model with large memory footprint like Tasnet.
        progress (bool): if True, show a progress bar (requires split=True)
    """
    assert transition_power >= 1, "transition_power < 1 leads to weird behavior."
    device = mix.device
    channels, length = mix.shape
    if split:
        out = th.zeros(len(model.sources), channels, length, device=device)
        sum_weight = th.zeros(length, device=device)
        segment = model.segment_length
        stride = int((1 - overlap) * segment)
        offsets = range(0, length, stride)
        scale = stride / model.samplerate
        if progress:
            offsets = tqdm.tqdm(offsets, unit_scale=scale, ncols=120, unit='seconds')
        # We start from a triangle shaped weight, with maximal weight in the middle
        # of the segment. Then we normalize and take to the power `transition_power`.
        # Large values of transition power will lead to sharper transitions.
        weight = th.cat([th.arange(1, segment // 2 + 1),
                         th.arange(segment - segment // 2, 0, -1)]).to(device)
        assert len(weight) == segment
        # If the overlap < 50%, this will translate to linear transition when
        # transition_power is 1.
        weight = (weight / weight.max())**transition_power
        for offset in offsets:
            chunk = TensorChunk(mix, offset, segment)
            chunk_out = apply_model(model, chunk, shifts=shifts)
            chunk_length = chunk_out.shape[-1]
            out[..., offset:offset + segment] += weight[:chunk_length] * chunk_out
            sum_weight[offset:offset + segment] += weight[:chunk_length]
            offset += segment
        assert sum_weight.min() > 0
        out /= sum_weight
        return out
    elif shifts:
        max_shift = int(0.5 * model.samplerate)
        mix = tensor_chunk(mix)
        padded_mix = mix.padded(length + 2 * max_shift)
        out = 0
        for _ in range(shifts):
            offset = random.randint(0, max_shift)
            shifted = TensorChunk(padded_mix, offset, length + max_shift - offset)
            shifted_out = apply_model(model, shifted)
            out += shifted_out[..., max_shift - offset:]
        out /= shifts
        return out
    else:
        valid_length = model.valid_length(length)
        mix = tensor_chunk(mix)
        padded_mix = mix.padded(valid_length)
        with th.no_grad():
            out = model(padded_mix.unsqueeze(0))[0]
        return center_trim(out, length)


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


def get_quantizer(model, args, optimizer=None):
    quantizer = None
    if args.diffq:
        quantizer = DiffQuantizer(
            model, min_size=args.q_min_size, group_size=8)
        if optimizer is not None:
            quantizer.setup_optimizer(optimizer)
    elif args.qat:
        quantizer = UniformQuantizer(
            model, bits=args.qat, min_size=args.q_min_size)
    return quantizer


def load_v2_model(path):
    import gzip
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        load_from = path
        if str(path).endswith(".gz"):
            load_from = gzip.open(path, "rb")
        klass, args, kwargs, state = torch.load(load_from, 'cpu')
    model = klass(*args, **kwargs)
    model.load_state_dict(state)
    return model


def load_model(path, strict=False):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        load_from = path
        package = torch.load(load_from, 'cpu')

    klass = package["klass"]
    args = package["args"]
    kwargs = package["kwargs"]

    if strict:
        model = klass(*args, **kwargs)
    else:
        sig = inspect.signature(klass)
        for key in list(kwargs):
            if key not in sig.parameters:
                warnings.warn("Dropping inexistant parameter " + key)
                del kwargs[key]
        model = klass(*args, **kwargs)

    state = package["state"]
    training_args = package["training_args"]
    quantizer = get_quantizer(model, training_args)

    set_state(model, quantizer, state)
    return model


def get_state(model, quantizer, half=False):
    if quantizer is None:
        state = {k: p.data.to(device='cpu', dtype=dtype) for k, p in model.state_dict().items()}
        dtype = torch.half if half else None
    else:
        state = quantizer.get_quantized_state()
        buf = io.BytesIO()
        torch.save(state, buf)
        state = {'compressed': zlib.compress(buf.getvalue())}
    return state


def set_state(model, quantizer, state):
    if quantizer is None:
        model.load_state_dict(state)
    else:
        buf = io.BytesIO(zlib.decompress(state["compressed"]))
        state = torch.load(buf, "cpu")
        quantizer.restore_quantized_state(state)

    return state


def save_state(state, path):
    buf = io.BytesIO()
    torch.save(state, buf)
    sig = hashlib.sha256(buf.getvalue()).hexdigest()[:8]

    path = path.parent / (path.stem + "-" + sig + path.suffix)
    path.write_bytes(buf.getvalue())


def save_model(model, quantizer, training_args, path):
    args, kwargs = model._init_args_kwargs
    klass = model.__class__

    state = get_state(model, quantizer, half=training_args.half)

    save_to = path
    package = {
        'klass': klass,
        'args': args,
        'kwargs': kwargs,
        'state': state,
        'training_args': training_args,
    }
    torch.save(package, save_to)


def capture_init(init):
    @functools.wraps(init)
    def __init__(self, *args, **kwargs):
        self._init_args_kwargs = (args, kwargs)
        init(self, *args, **kwargs)

    return __init__
