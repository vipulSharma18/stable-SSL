# -*- coding: utf-8 -*-
"""BYOL model."""
#
# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#         Randall Balestriero <randallbalestriero@gmail.com>
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass, field
import torch

from stable_ssl.utils import mlp
from .base import SelfDistillationModel, SelfDistillationConfig


class BYOL(SelfDistillationModel):
    """BYOL model from [GSA+20].

    Reference
    ---------
    .. [GSA+20] Grill, J. B., Strub, F., Altché, ... & Valko, M. (2020).
            Bootstrap Your Own Latent-A New Approach To Self-Supervised Learning.
            Advances in neural information processing systems, 33, 21271-21284.
    """

    def initialize_modules(self):
        super().initialize_modules()

        sizes = [self.config.model.projector[-1]] + self.config.model.predictor
        self.predictor = mlp(sizes)

    def compute_ssl_loss(self, projections, projections_target):
        """Compute the loss of the BYOL model.

        Parameters
        ----------
        projections : list of torch.Tensor
            Projections of the different augmented views from the online network.
        projections_target : list of torch.Tensor
            Projections of the corresponding augmented views from the target network.

        Returns
        -------
        float
            The computed loss.
        """
        if len(projections) > 2 or len(projections_target) > 2:
            logging.warning(
                "BYOL only supports two views. Only the first two views will be used."
            )

        predictions = [self.predictor(proj) for proj in projections]

        sim = torch.nn.CosineSimilarity(dim=1)
        return -0.5 * (
            sim(predictions[0], projections_target[1]).mean()
            + sim(predictions[1], projections_target[0]).mean()
        )


@dataclass
class BYOLConfig(SelfDistillationConfig):
    """Configuration for the BYOL model parameters.

    Parameters
    ----------
    predictor : str
        Architecture of the predictor head. Default is "2048-256".
    """

    predictor: list[int] = field(default_factory=lambda: [2048, 256])

    def __post_init__(self):
        """Convert predictor string to a list of integers if necessary."""
        super().__post_init__()
        if isinstance(self.predictor, str):
            self.predictor = [int(i) for i in self.predictor.split("-")]

    def trainer(self):
        return BYOL
