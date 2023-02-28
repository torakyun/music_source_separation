# Copyright (c) 2023 torakyun
# Copyright (c) 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""STFT-based Loss modules."""

import torch
import torch.nn.functional as F

from distutils.version import LooseVersion

is_pytorch_17plus = LooseVersion(torch.__version__) >= LooseVersion("1.7")


def magnitude(x, fft_size, hop_size, win_length, window):
    """Perform STFT and convert to magnitude spectrogram.

    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.

    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).

    """
    if is_pytorch_17plus:
        x_stft = torch.stft(
            x, fft_size, hop_size, win_length, window, return_complex=False
        )
    else:
        x_stft = torch.stft(x, fft_size, hop_size, win_length, window)
    real = x_stft[..., 0]
    imag = x_stft[..., 1]

    return torch.sqrt(torch.clamp(real**2 + imag**2, min=1e-7)).transpose(2, 1)


class MagnitudeSpectrogramLoss(torch.nn.Module):
    """Magnitude loss module."""

    def __init__(self):
        """Initilize magnitude loss module."""
        super(MagnitudeSpectrogramLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.

        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).

        Returns:
            Tensor: Magnitude loss value.

        """
        return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")


class LogMagnitudeSpectrogramLoss(torch.nn.Module):
    """Log STFT magnitude loss module."""

    def __init__(self):
        """Initilize los STFT magnitude loss module."""
        super(LogMagnitudeSpectrogramLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.

        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).

        Returns:
            Tensor: Log magnitude loss value.

        """
        return F.l1_loss(torch.log(y_mag), torch.log(x_mag))


class MagnitudeLoss(torch.nn.Module):
    """STFT loss module."""

    def __init__(
        self, fft_size=1024, shift_size=120, win_length=600, window="hann_window"
    ):
        """Initialize magnitude loss module."""
        super(MagnitudeLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.magnitude_loss = MagnitudeSpectrogramLoss()
        self.log_magnitude_loss = LogMagnitudeSpectrogramLoss()
        self.register_buffer("window", getattr(torch, window)(win_length))

    def forward(self, x, y):
        """Calculate forward propagation.

        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).

        Returns:
            Tensor: Magnitude loss value.
            Tensor: Log magnitude loss value.

        """
        x_mag = magnitude(x, self.fft_size, self.shift_size,
                          self.win_length, self.window)
        y_mag = magnitude(y, self.fft_size, self.shift_size,
                          self.win_length, self.window)
        mag_loss = self.magnitude_loss(x_mag, y_mag)
        log_mag_loss = self.log_magnitude_loss(x_mag, y_mag)

        return mag_loss, log_mag_loss


class MultiResolutionMagnitudeLoss(torch.nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(
        self,
        fft_sizes=[1024, 2048, 512],
        hop_sizes=[120, 240, 50],
        win_lengths=[600, 1200, 240],
        window="hann_window",
    ):
        """Initialize Multi resolution magnitude loss module.

        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.

        """
        super(MultiResolutionMagnitudeLoss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [MagnitudeLoss(fs, ss, wl, window)]

    def forward(self, x, y):
        """Calculate forward propagation.

        Args:
            x (Tensor): Predicted signal (B, T) or (B, #subband, T).
            y (Tensor): Groundtruth signal (B, T) or (B, #subband, T).

        Returns:
            Tensor: Multi resolution magnitude loss value.
            Tensor: Multi resolution log magnitude loss value.

        """
        if len(x.shape) > 2:
            x = x.reshape(-1, x.size(-1))  # (B, C, T) -> (B x C, T)
            y = y.reshape(-1, y.size(-1))  # (B, C, T) -> (B x C, T)
        mag_loss = 0.0
        log_mag_loss = 0.0
        for f in self.stft_losses:
            sc_l, mag_l = f(x, y)
            mag_loss += sc_l
            log_mag_loss += mag_l
        mag_loss /= len(self.stft_losses)
        log_mag_loss /= len(self.stft_losses)

        return mag_loss, log_mag_loss
