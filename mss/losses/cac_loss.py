# Copyright (c) 2023 torakyun
# Copyright (c) 2021 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Complex-As-Channels-based Loss modules."""

import torch

from distutils.version import LooseVersion

is_pytorch_17plus = LooseVersion(torch.__version__) >= LooseVersion("1.7")


def stft(x, fft_size, hop_size, win_length, window):
    """Perform STFT and convert to magnitude spectrogram.

    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.

    Returns:
        Tensor: STFT (B, #frames, fft_size // 2 + 1, 2).
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1)

    """
    if is_pytorch_17plus:
        x_stft = torch.stft(
            x, fft_size, hop_size, win_length, window, return_complex=False
        )
    else:
        x_stft = torch.stft(x, fft_size, hop_size, win_length, window)
    real = x_stft[..., 0]
    imag = x_stft[..., 1]
    x_mag = torch.sqrt(torch.clamp(real**2 + imag**2, min=1e-7))

    return x_stft, x_mag


class _LinearScaleCACLoss(torch.nn.Module):
    """Linear-scale CAC loss module."""

    def __init__(self):
        """Initilize linear-scale CAC loss module."""
        super(_LinearScaleCACLoss, self).__init__()

    def forward(self, x_stft, y_stft, y_mag):
        """Calculate forward propagation.

        Args:
            x_stft (Tensor): STFT of predicted signal (B, #freq_bins, #frames).
            x_stft (Tensor): STFT of groundtruth signal (B, #freq_bins, #frames).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #freq_bins, #frames).

        Returns:
            Tensor: Linear-scale CAC loss value.

        """
        dif_stft = x_stft - y_stft
        dif_mag = torch.sqrt(torch.clamp(
            dif_stft[..., 0]**2 + dif_stft[..., 1]**2, min=1e-7))
        return torch.norm(dif_mag, p="fro") / torch.norm(y_mag, p="fro")


class _LogScaleCACLoss(torch.nn.Module):
    """Log-scale CAC loss module."""

    def __init__(self):
        """Initilize log-scale CAC loss module."""
        super(_LogScaleCACLoss, self).__init__()

    def forward(self, x_stft, y_stft, x_mag, y_mag):
        """Calculate forward propagation.

        Args:
            x_stft (Tensor): STFT of predicted signal (B, #freq_bins, #frames).
            y_stft (Tensor): STFT of groundtruth signal (B, #freq_bins, #frames).
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #freq_bins, #frames).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #freq_bins, #frames).

        Returns:
            Tensor: Log-scale CAC loss value.

        """
        x_log_stft = x_stft * (torch.log(x_mag) / x_mag).unsqueeze(-1)
        y_log_stft = y_stft * (torch.log(y_mag) / y_mag).unsqueeze(-1)
        dif_log_stft = x_log_stft - y_log_stft
        dif_mag = torch.sqrt(torch.clamp(
            dif_log_stft[..., 0]**2 + dif_log_stft[..., 1]**2, min=1e-7))
        return torch.mean(dif_mag)


class _MultiScaleCACLoss(torch.nn.Module):
    """Multi-scale CAC loss module."""

    def __init__(
        self, fft_size=1024, shift_size=120, win_length=600, window="hann_window"
    ):
        """Initialize Multi-scale CAC loss module."""
        super(_MultiScaleCACLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.linear_scale_cac_loss = _LinearScaleCACLoss()
        self.log_scale_cac_loss = _LogScaleCACLoss()
        self.register_buffer("window", getattr(torch, window)(win_length))

    def forward(self, x, y):
        """Calculate forward propagation.

        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).

        Returns:
            Tensor: Linear-scale CAC loss value.
            Tensor: Log-scale CAC loss value.

        """
        x_stft, x_mag = stft(x, self.fft_size, self.shift_size,
                             self.win_length, self.window)
        y_stft, y_mag = stft(y, self.fft_size, self.shift_size,
                             self.win_length, self.window)
        linear_scale_cac_loss = self.linear_scale_cac_loss(x_stft, y_stft, y_mag)
        log_scale_cac_loss = self.log_scale_cac_loss(x_stft, y_stft, x_mag, y_mag)

        return linear_scale_cac_loss, log_scale_cac_loss


class MultiResolutionMultiScaleCACLoss(torch.nn.Module):
    """Multi-resolution Multi-scale CAC loss module."""

    def __init__(
        self,
        fft_sizes=[1024, 2048, 512],
        hop_sizes=[120, 240, 50],
        win_lengths=[600, 1200, 240],
        window="hann_window",
    ):
        """Initialize Multi-resolution CAC loss module.

        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.

        """
        super(MultiResolutionMultiScaleCACLoss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.cac_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.cac_losses += [_MultiScaleCACLoss(fs, ss, wl, window)]

    def forward(self, x, y):
        """Calculate forward propagation.

        Args:
            x (Tensor): Predicted signal (B, T) or (B, #subband, T).
            y (Tensor): Groundtruth signal (B, T) or (B, #subband, T).

        Returns:
            Tensor: Multi-resolution linear-scale CAC loss value.
            Tensor: Multi-resolution log-scale CAC loss value.

        """
        if len(x.shape) > 2:
            x = x.reshape(-1, x.size(-1))  # (B, C, T) -> (B x C, T)
            y = y.reshape(-1, y.size(-1))  # (B, C, T) -> (B x C, T)
        linear_scale_cac_loss = 0.0
        log_scale_cac_loss = 0.0
        for f in self.cac_losses:
            losses = f(x, y)
            linear_scale_cac_loss += losses[0]
            log_scale_cac_loss += losses[1]
        linear_scale_cac_loss /= len(self.cac_losses)
        log_scale_cac_loss /= len(self.cac_losses)

        return linear_scale_cac_loss, log_scale_cac_loss
