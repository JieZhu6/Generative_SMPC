"""Sigmoid-output ablation of the non-causal CSNG model."""

from collections.abc import Sequence

import torch
from torch import nn

from Neural_network.noncausal_tcn import (
    PROJECTION_METHOD,
    ConditionalStochasticTCN,
)


SIGMOID_PROJECTION_METHOD = "sigmoid_v1"


class SigmoidConditionalStochasticTCN(ConditionalStochasticTCN):
    """Generate admissible trajectories using sigmoid output coordinates.

    Parameters
    ----------
    condition_dim : int
        Number of pooled uncertainty features per time period.
    free_dim : int
        Number of generated free variables per time period.
    n_free_pg : int
        Number of leading active-power variables.
    projection_method : str, default="sigmoid_v1"
        Checkpoint identifier for the S-CSNG output parameterization.
    hidden_channels : int, default=128
        Width of the temporal feature blocks.
    latent_dim : int, default=16
        Dimension of one trajectory-level Gaussian latent vector.
    latent_embedding_dim : int, default=32
        Width of the latent embedding repeated across the horizon.
    kernel_size : int, default=3
        Positive odd temporal convolution width.
    dilations : sequence of int, default=(1,2,4,8)
        Positive dilation factors for the non-causal TCN blocks.
    """

    def __init__(
        self,
        condition_dim: int,
        free_dim: int,
        n_free_pg: int,
        projection_method: str = SIGMOID_PROJECTION_METHOD,
        hidden_channels: int = 128,
        latent_dim: int = 16,
        latent_embedding_dim: int = 32,
        kernel_size: int = 3,
        dilations: Sequence[int] = (1, 2, 4, 8),
    ):
        """Build the S-CSNG model with a midpoint-centered sigmoid output."""
        if projection_method != SIGMOID_PROJECTION_METHOD:
            raise ValueError(
                f"projection_method must be {SIGMOID_PROJECTION_METHOD!r}, "
                f"got {projection_method!r}"
            )
        super().__init__(
            condition_dim=condition_dim,
            free_dim=free_dim,
            n_free_pg=n_free_pg,
            projection_method=PROJECTION_METHOD,
            hidden_channels=hidden_channels,
            latent_dim=latent_dim,
            latent_embedding_dim=latent_embedding_dim,
            kernel_size=kernel_size,
            dilations=dilations,
        )
        self.projection_method = SIGMOID_PROJECTION_METHOD
        # A zero logit maps to 0.5, matching the physical midpoint used by CSNG.
        nn.init.constant_(self.output_mapping.bias, 0.0)

    def project_unit_interval(self, raw: torch.Tensor) -> torch.Tensor:
        """Map unconstrained logits smoothly into the open unit interval.

        Parameters
        ----------
        raw : torch.Tensor
            Unconstrained generator logits of arbitrary shape.

        Returns
        -------
        torch.Tensor
            Componentwise sigmoid values in ``(0,1)``.
        """
        return torch.sigmoid(raw)
