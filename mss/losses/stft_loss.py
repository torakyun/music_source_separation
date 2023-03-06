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


class _LinearScaleSTFTLoss(torch.nn.Module):
    """Linear-scale STFT loss module."""

    def __init__(self):
        """Initilize linear-scale STFT loss module."""
        super(_LinearScaleSTFTLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.

        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).

        Returns:
            Tensor: Linear-scale STFT loss value.

        """
        return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")


class _LogScaleSTFTLoss(torch.nn.Module):
    """Log-scale STFT loss module."""

    def __init__(self):
        """Initilize log-scale STFT loss module."""
        super(_LogScaleSTFTLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.

        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).

        Returns:
            Tensor: Log-scale STFT loss value.

        """
        return F.l1_loss(torch.log(y_mag), torch.log(x_mag))


class _MultiScaleSTFTLoss(torch.nn.Module):
    """Multi-scale STFT loss module."""

    def __init__(
        self, fft_size=1024, shift_size=120, win_length=600, window="hann_window"
    ):
        """Initialize Multi-scale STFT loss module."""
        super(_MultiScaleSTFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.linear_scale_stft_loss = _LinearScaleSTFTLoss()
        self.log_scale_stft_loss = _LogScaleSTFTLoss()
        self.register_buffer("window", getattr(torch, window)(win_length))

    def forward(self, x, y):
        """Calculate forward propagation.

        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).

        Returns:
            Tensor: Linear-scale STFT loss value.
            Tensor: Log-scale STFT loss value.

        """
        x_mag = magnitude(x, self.fft_size, self.shift_size,
                          self.win_length, self.window)
        y_mag = magnitude(y, self.fft_size, self.shift_size,
                          self.win_length, self.window)
        linear_scale_stft_loss = self.linear_scale_stft_loss(x_mag, y_mag)
        log_scale_stft_loss = self.log_scale_stft_loss(x_mag, y_mag)

        return linear_scale_stft_loss, log_scale_stft_loss


class MultiResolutionMultiScaleSTFTLoss(torch.nn.Module):
    """Multi-resolution multi-scale STFT loss module."""

    def __init__(
        self,
        fft_sizes=[1024, 2048, 512],
        hop_sizes=[120, 240, 50],
        win_lengths=[600, 1200, 240],
        window="hann_window",
    ):
        """Initialize Multi-resolution multi-scale STFT loss module.

        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.

        """
        super(MultiResolutionMultiScaleSTFTLoss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [_MultiScaleSTFTLoss(fs, ss, wl, window)]

    def forward(self, x, y):
        """Calculate forward propagation.

        Args:
            x (Tensor): Predicted signal (B, T) or (B, #subband, T).
            y (Tensor): Groundtruth signal (B, T) or (B, #subband, T).

        Returns:
            Tensor: Multi-resolution linear-scale STFT loss value.
            Tensor: Multi-resolution log-scale STFT loss value.

        """
        if len(x.shape) > 2:
            x = x.reshape(-1, x.size(-1))  # (B, C, T) -> (B x C, T)
            y = y.reshape(-1, y.size(-1))  # (B, C, T) -> (B x C, T)
        linear_scale_stft_loss = 0.0
        log_scale_stft_loss = 0.0
        for f in self.stft_losses:
            losses = f(x, y)
            linear_scale_stft_loss += losses[0]
            log_scale_stft_loss += losses[1]
        linear_scale_stft_loss /= len(self.stft_losses)
        log_scale_stft_loss /= len(self.stft_losses)

        return linear_scale_stft_loss, log_scale_stft_loss
