from .demucs import *  # NOQA
from .tasnet import *  # NOQA

from .parallel_wavegan_discriminator import *  # NOQA
from .hifigan_discriminator import *  # NOQA
from .melgan_discriminator import *  # NOQA
from .style_melgan_discriminator import *  # NOQA

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
        if not separate:
            params["in_channels"] = sources * channels
            self.discriminators.append(globals()[name](**params))
        elif separate == "sources":
            params["in_channels"] = channels
            for _ in range(sources):
                self.discriminators.append(globals()[name](**params))
        elif separate == "full":
            params["in_channels"] = 1
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
        for i, f in enumerate(self.discriminators):
            if not self.separate:
                batch, sources, channels, time = x.size()
                output.append(f(x.view(batch, sources * channels, time)))
            elif self.separate == "sources":
                output.append(f(x[:, i, ...]))
            elif self.separate == "full":
                for j in range(self.channels):
                    output.append(f(x[:, i, j:j+1, :]))
        return output
