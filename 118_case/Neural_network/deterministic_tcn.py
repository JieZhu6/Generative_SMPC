"""Deterministic non-causal TCN benchmark for SMPC dispatch trajectories.

The model is the single-solution counterpart of ``ConditionalStochasticTCN``:
it consumes the same temporal min/mean/max uncertainty condition, but contains
no latent variable and returns exactly one shared dispatch trajectory per SMPC
instance. The admissible output map enforces free-variable bounds and the
nonreference-generator ramp limits by construction. It uses the same exact
clip projection and inward-only boundary gradient as the stochastic model.
"""

from collections.abc import Sequence

import torch
from torch import nn

from Neural_network.noncausal_tcn import (
    PROJECTION_METHOD,
    NonCausalTemporalBlock,
    constraint_aware_clip,
)


class DeterministicTCN(nn.Module):
    """Map one pooled uncertainty sequence to one bounded dispatch trajectory.

    Parameters
    ----------
    condition_dim : int
        Number of min/mean/max uncertainty features per time period.
    free_dim : int
        Number of free variables at each time period.
    n_free_pg : int
        Number of leading free-variable components representing active power.
    projection_method : str, default="inward_clip_v1"
        Versioned admissible-output projection stored in model checkpoints.
    hidden_channels : int, default=256
        Width of the input embedding and all temporal blocks.
    kernel_size : int, default=3
        Positive odd temporal convolution width in time steps.
    dilations : sequence of int, default=(1, 2, 4, 8)
        Positive dilation factors. With kernel three, the default receptive
        field covers 31 time periods.
    """

    def __init__(
        self,
        condition_dim: int,
        free_dim: int,
        n_free_pg: int,
        projection_method: str = PROJECTION_METHOD,
        hidden_channels: int = 256,
        kernel_size: int = 3,
        dilations: Sequence[int] = (1, 2, 4, 8),
    ):
        """Build the deterministic input embedding, TCN blocks, and output map.

        Parameters
        ----------
        condition_dim, free_dim, n_free_pg : int
            Per-period input width, output width, and leading active-power count.
        projection_method : str, default="inward_clip_v1"
            Must equal the paper-aligned inward-gradient clip version.
        hidden_channels : int, default=256
            Feature width shared by all temporal blocks.
        kernel_size : int, default=3
            Positive odd temporal kernel width.
        dilations : sequence of int, default=(1, 2, 4, 8)
            Positive dilation factors from shallow to deep blocks.
        """
        super().__init__()
        if min(condition_dim, free_dim, hidden_channels) < 1:
            raise ValueError("all network dimensions must be positive")
        if not 0 < n_free_pg <= free_dim:
            raise ValueError("n_free_pg must lie in [1, free_dim]")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if not dilations or min(dilations) < 1:
            raise ValueError("dilations must contain positive integers")
        if projection_method != PROJECTION_METHOD:
            raise ValueError(
                f"projection_method must be {PROJECTION_METHOD!r}, got {projection_method!r}"
            )

        self.condition_dim = int(condition_dim)
        self.free_dim = int(free_dim)
        self.n_free_pg = int(n_free_pg)
        self.projection_method = projection_method
        self.hidden_channels = int(hidden_channels)
        self.kernel_size = int(kernel_size)
        self.dilations = tuple(int(value) for value in dilations)

        self.input_embedding = nn.Sequential(
            nn.Linear(self.condition_dim, self.hidden_channels),
            nn.ReLU(),
        )
        self.temporal_blocks = nn.ModuleList([
            NonCausalTemporalBlock(
                self.hidden_channels,
                self.hidden_channels,
                self.kernel_size,
                dilation,
            )
            for dilation in self.dilations
        ])
        self.output_mapping = nn.Linear(self.hidden_channels, self.free_dim)
        # The clip projection consumes unit-interval coordinates directly.
        # Start near the interval midpoint and avoid initial saturation.
        nn.init.normal_(self.output_mapping.weight, mean=0.0, std=1e-2)
        nn.init.constant_(self.output_mapping.bias, 0.5)

    @property
    def receptive_field(self) -> int:
        """Return the temporal receptive field in time periods."""
        return 1 + (self.kernel_size - 1) * sum(self.dilations)

    def configuration(self) -> dict[str, int | str | list[int]]:
        """Return constructor arguments suitable for a model checkpoint."""
        return {
            "condition_dim": self.condition_dim,
            "free_dim": self.free_dim,
            "n_free_pg": self.n_free_pg,
            "projection_method": self.projection_method,
            "hidden_channels": self.hidden_channels,
            "kernel_size": self.kernel_size,
            "dilations": list(self.dilations),
        }

    def raw_trajectory(self, condition: torch.Tensor) -> torch.Tensor:
        """Map conditions to unconstrained deterministic trajectories.

        Parameters
        ----------
        condition : torch.Tensor, shape (batch, horizon, condition_dim)
            Normalized min/mean/max uncertainty sequence.

        Returns
        -------
        torch.Tensor, shape (batch, horizon, free_dim)
            Unconstrained active-power and voltage outputs.
        """
        if condition.ndim != 3 or condition.shape[2] != self.condition_dim:
            raise ValueError(
                f"condition must have shape (batch,horizon,{self.condition_dim})"
            )
        features = self.input_embedding(condition).transpose(1, 2)
        for block in self.temporal_blocks:
            features = block(features)
        return self.output_mapping(features.transpose(1, 2))

    def enforce_admissible_output(
        self,
        raw: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        ramp_up: torch.Tensor,
        ramp_down: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Map raw outputs to bounds and recursively feasible active powers.

        Parameters
        ----------
        raw : torch.Tensor, shape (batch, horizon, free_dim)
            Unconstrained deterministic TCN outputs.
        lower, upper : torch.Tensor, shape (free_dim,)
            Static active-power bounds in MW followed by voltage bounds in p.u.
        ramp_up, ramp_down : torch.Tensor, shape (n_free_pg,)
            Positive active-power movement limits in MW per period. The first
            decision uses only its static physical bounds.

        Returns
        -------
        trajectory : torch.Tensor, shape (batch, horizon, free_dim)
            Physically scaled free-variable trajectories.
        admissible_lower, admissible_upper : torch.Tensor
            Dynamic pointwise bounds with the same shape as ``trajectory``.
        """
        if raw.ndim != 3 or raw.shape[-1] != self.free_dim:
            raise ValueError(
                f"raw must have shape (batch,horizon,{self.free_dim})"
            )
        expected_free = (self.free_dim,)
        expected_pg = (self.n_free_pg,)
        if tuple(lower.shape) != expected_free or tuple(upper.shape) != expected_free:
            raise ValueError(f"lower and upper must have shape {expected_free}")
        if tuple(ramp_up.shape) != expected_pg or tuple(ramp_down.shape) != expected_pg:
            raise ValueError(f"ramp tensors must have shape {expected_pg}")
        if torch.any(upper <= lower):
            raise ValueError("every upper bound must exceed its lower bound")
        if torch.any(ramp_up <= 0) or torch.any(ramp_down <= 0):
            raise ValueError("ramp limits must be positive")

        dtype, device = raw.dtype, raw.device
        lower = lower.to(dtype=dtype, device=device)
        upper = upper.to(dtype=dtype, device=device)
        ramp_up = ramp_up.to(dtype=dtype, device=device)
        ramp_down = ramp_down.to(dtype=dtype, device=device)
        batch, horizon, _ = raw.shape

        pg_min, pg_max = lower[:self.n_free_pg], upper[:self.n_free_pg]
        voltage_min = lower[self.n_free_pg:]
        voltage_max = upper[self.n_free_pg:]
        voltage = voltage_min + (voltage_max - voltage_min) * constraint_aware_clip(
            raw[..., self.n_free_pg:]
        )

        active_power = []
        active_lower = []
        active_upper = []
        previous = pg_min + (pg_max - pg_min) * constraint_aware_clip(
            raw[:, 0, :self.n_free_pg]
        )
        active_power.append(previous)
        active_lower.append(pg_min.expand(batch, -1))
        active_upper.append(pg_max.expand(batch, -1))
        for time in range(1, horizon):
            time_lower = torch.maximum(pg_min, previous - ramp_down)
            time_upper = torch.minimum(pg_max, previous + ramp_up)
            previous = time_lower + (time_upper - time_lower) * constraint_aware_clip(
                raw[:, time, :self.n_free_pg]
            )
            active_power.append(previous)
            active_lower.append(time_lower)
            active_upper.append(time_upper)

        active_power = torch.stack(active_power, dim=1)
        active_lower = torch.stack(active_lower, dim=1)
        active_upper = torch.stack(active_upper, dim=1)
        voltage_lower = voltage_min.expand(batch, horizon, -1)
        voltage_upper = voltage_max.expand(batch, horizon, -1)
        trajectory = torch.cat([active_power, voltage], dim=2)
        admissible_lower = torch.cat([active_lower, voltage_lower], dim=2)
        admissible_upper = torch.cat([active_upper, voltage_upper], dim=2)
        return trajectory, admissible_lower, admissible_upper

    def forward(
        self,
        condition: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        ramp_up: torch.Tensor,
        ramp_down: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate one constrained free-variable trajectory per condition.

        Parameters
        ----------
        condition : torch.Tensor, shape (batch, horizon, condition_dim)
            Normalized pooled temporal features.
        lower, upper : torch.Tensor, shape (free_dim,)
            Static free-variable limits in MW and p.u.
        ramp_up, ramp_down : torch.Tensor, shape (n_free_pg,)
            Positive free-generator ramp limits in MW per period.

        Returns
        -------
        trajectory, admissible_lower, admissible_upper : tuple of torch.Tensor
            Deterministic trajectories and their pointwise output intervals.
        """
        raw = self.raw_trajectory(condition)
        return self.enforce_admissible_output(
            raw, lower, upper, ramp_up, ramp_down,
        )
