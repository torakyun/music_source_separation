import torch
import torchaudio
import torch.nn.functional as F


class MFCCLoss(torch.nn.Module):
    """Mel-spectrogram loss."""

    def __init__(
        self,
        sample_rate=16000,
        n_mfcc=40,
        dct_type=2,
        norm='ortho',
        log_mels=False,
        melkwargs=None,
    ):
        """Initialize Mel-spectrogram loss."""
        super().__init__()
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            dct_type=dct_type,
            norm=norm,
            log_mels=log_mels,
            melkwargs=melkwargs,
        )

    def forward(self, y_hat, y):
        """Calculate Mel-spectrogram loss.

        Args:
            y_hat (Tensor): Generated single tensor (B, 1, T).
            y (Tensor): Groundtruth single tensor (B, 1, T).

        Returns:
            Tensor: Mel-spectrogram loss value.

        """
        mfcc_hat = self.mfcc(y_hat)
        mfcc = self.mfcc(y)
        mfcc_loss = F.l1_loss(mfcc_hat, mfcc)

        return mfcc_loss
