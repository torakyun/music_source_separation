# Copyright (c) 2023 torakyun
#  MIT License (https://opensource.org/licenses/MIT)

from .generator.demucs import *  # NOQA
from .generator.hdemucs import *  # NOQA
from .generator.demucs_v2 import *  # NOQA
from .generator.tasnet import *  # NOQA

from .discriminator.parallel_wavegan import *  # NOQA
from .discriminator.hifigan import *  # NOQA
from .discriminator.melgan import *  # NOQA
from .discriminator.style_melgan import *  # NOQA

import torch


class Discriminators(torch.nn.Module):
    """Discriminator modules."""

    def __init__(
        self,
        name,
        params,
        sources=4,
        channels=2,
        separate=None,
    ):
        """Initialize Parallel Discriminator modules.

        Args:
            name (str): Discriminator name.
            params (dict): Discriminator params.
            sources (int): Number of sources.
            channels (int): Number of channels.
            separate (str): How to input separate ( "full" or "sources" or "None").

        """
        super().__init__()
        self.sources = sources
        self.channels = channels
        self.separate = separate
        self.discriminators = torch.nn.ModuleList()

        # set in_channels
        if not separate:
            in_channels = sources * channels
        elif separate == "sources":
            in_channels = channels
        elif separate == "full":
            in_channels = 1
        if name == "HiFiGANMultiScaleMultiPeriodDiscriminator":
            params["scale_params"]["in_channels"] = in_channels
            params["period_params"]["in_channels"] = in_channels
        else:
            params["in_channels"] = in_channels

        # get discriminators
        if not separate:
            self.discriminators.append(globals()[name](**params))
        elif separate == "sources":
            for _ in range(sources):
                self.discriminators.append(globals()[name](**params))
        elif separate == "full":
            for _ in range(sources):
                self.discriminators.append(globals()[name](**params))

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, S, C, T).

        Returns:
            output (list): Output tensors (B, 1, T).

        """
        output = []
        sources, channels, time = x.size(-3), x.size(-2), x.size(-1)
        x = x.view(-1, sources, channels, time)
        for i, f in enumerate(self.discriminators):
            if not self.separate:
                output.append(f(x.view(-1, sources * channels, time)))
            elif self.separate == "sources":
                output.append(f(x[:, i, ...]))
            elif self.separate == "full":
                for j in range(self.channels):
                    output.append(f(x[:, i, j:j+1, :]))
        return output
