# -*- coding: utf-8 -*-

# Copyright 2021 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Feature matching loss modules."""

import torch
import torch.nn.functional as F


class FeatureMatchLoss(torch.nn.Module):
    """Feature matching loss module."""

    def __init__(
        self,
        average_by_layers=True,
        average_by_scales=True,
        average_by_discriminators=True,
        include_final_outputs=False,
    ):
        """Initialize FeatureMatchLoss module."""
        super().__init__()
        self.average_by_layers = average_by_layers
        self.average_by_scales = average_by_scales
        self.average_by_discriminators = average_by_discriminators
        self.include_final_outputs = include_final_outputs

    def forward(self, feats_hat, feats):
        """Calcualate feature matching loss.

        Args:
            feats_hat (list): List of list of list of discriminator outputs
                calcuated from generater outputs.
            feats (list): List of list of list of discriminator outputs
                calcuated from groundtruth.

        Returns:
            Tensor: Feature matching loss value.

        """
        feat_match_loss = 0.0
        for i, (feats_hat_, feats_) in enumerate(zip(feats_hat, feats)):
            for j, (feat_hat, feat) in enumerate(zip(feats_hat_, feats_)):
                if not self.include_final_outputs:
                    feat_hat = feat_hat[:-1]
                    feat = feat[:-1]
                for k, (feat_hat_, feat_) in enumerate(zip(feat_hat, feat)):
                    feat_match_loss += F.l1_loss(feat_hat_, feat_.detach())
        divide = 1
        if self.average_by_discriminators:
            divide *= i + 1
        if self.average_by_scales:
            divide *= j + 1
        if self.average_by_layers:
            divide *= k + 1
        if divide != 1:
            feat_match_loss /= divide

        return feat_match_loss
