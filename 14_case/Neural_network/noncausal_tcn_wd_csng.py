"""No-diversity-loss ablation model with the proposed CSNG projection."""

from collections.abc import Sequence

from Neural_network.noncausal_tcn import (
    PROJECTION_METHOD,
    ConditionalStochasticTCN,
)


class WDConditionalStochasticTCN(ConditionalStochasticTCN):
    """Name the WD-CSNG model while preserving the complete CSNG architecture.

    Parameters
    ----------
    condition_dim : int
        Number of pooled uncertainty features per time period.
    free_dim : int
        Number of generated free variables per time period.
    n_free_pg : int
        Number of leading active-power variables.
    projection_method : str, default="inward_clip_v1"
        Proposed constraint-aware projection identifier.
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
        projection_method: str = PROJECTION_METHOD,
        hidden_channels: int = 128,
        latent_dim: int = 16,
        latent_embedding_dim: int = 32,
        kernel_size: int = 3,
        dilations: Sequence[int] = (1, 2, 4, 8),
    ):
        """Build the WD-CSNG model without changing any model-side operation."""
        super().__init__(
            condition_dim=condition_dim,
            free_dim=free_dim,
            n_free_pg=n_free_pg,
            projection_method=projection_method,
            hidden_channels=hidden_channels,
            latent_dim=latent_dim,
            latent_embedding_dim=latent_embedding_dim,
            kernel_size=kernel_size,
            dilations=dilations,
        )
