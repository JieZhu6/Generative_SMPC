"""Non-causal temporal convolutional generator for SMPC dispatch trajectories.

The implementation follows equations (9)-(24) of ``E2E_Transmission_SMPC.pdf``:
one trajectory-level Gaussian latent vector is shared across the horizon, the
condition sequence is processed by symmetrically padded dilated convolutions,
and a constraint-aware clip projection enforces bounds and ramp limits while
retaining only boundary gradients that point back into the admissible interval.
"""

from collections.abc import Sequence

import torch
from torch import nn


PROJECTION_METHOD = "inward_clip_v1"


class _InwardGradientClip(torch.autograd.Function):
    """Clip to ``[0,1]`` and retain only inward boundary gradients."""

    @staticmethod
    def forward(ctx, raw: torch.Tensor) -> torch.Tensor:
        """Return the exact componentwise projection of ``raw`` onto ``[0,1]``.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Autograd context used to save the projected tensor for backward.
        raw : torch.Tensor
            Unconstrained dimensionless generator output.

        Returns
        -------
        torch.Tensor
            ``raw`` clipped componentwise to the closed unit interval.
        """
        projected = raw.clamp(0.0, 1.0)
        ctx.save_for_backward(projected)
        return projected

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor]:
        """Apply equation (24)'s inward-only gradient at saturated bounds.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Autograd context containing the forward projected tensor.
        grad_output : torch.Tensor
            Upstream derivative of the scalar training loss with respect to
            the projected unit-interval output.

        Returns
        -------
        tuple[torch.Tensor]
            Gradient with respect to the unconstrained raw output. Interior
            gradients pass unchanged; a boundary gradient passes only when a
            gradient-descent step moves the variable toward the interval.
        """
        (projected,) = ctx.saved_tensors
        interior = (projected > 0.0) & (projected < 1.0)
        leave_lower = (projected == 0.0) & (grad_output < 0.0)
        leave_upper = (projected == 1.0) & (grad_output > 0.0)
        retain = interior | leave_lower | leave_upper
        return (torch.where(retain, grad_output, torch.zeros_like(grad_output)),)


def constraint_aware_clip(raw: torch.Tensor) -> torch.Tensor:
    """Project a dimensionless raw output with the paper's custom backward.

    Parameters
    ----------
    raw : torch.Tensor
        Unconstrained dimensionless output of arbitrary shape.

    Returns
    -------
    torch.Tensor
        Exact componentwise clip to ``[0,1]`` with inward-only saturated
        gradients as defined by equations (17) and (24).
    """
    return _InwardGradientClip.apply(raw)


class NonCausalTemporalBlock(nn.Module):
    """Apply one symmetric dilated convolution with a residual connection.

    Parameters
    ----------
    in_channels : int
        Number of input temporal features.
    out_channels : int
        Number of output temporal features.
    kernel_size : int
        Positive odd convolution width in time steps.
    dilation : int
        Positive spacing between kernel taps in time steps.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
    ):
        """Build one residual non-causal convolutional block.

        Parameters
        ----------
        in_channels, out_channels : int
            Positive input and output feature counts.
        kernel_size : int
            Positive odd temporal kernel width.
        dilation : int
            Positive spacing between temporal kernel taps.
        """
        super().__init__()
        if min(in_channels, out_channels, dilation) < 1:
            raise ValueError("channels and dilation must be positive")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        padding = dilation * (kernel_size - 1) // 2
        self.convolution = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )
        self.activation = nn.ReLU()

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """Return residual temporal features with unchanged sequence length.

        Parameters
        ----------
        sequence : torch.Tensor, shape (batch, in_channels, horizon)
            Input temporal feature sequence.

        Returns
        -------
        torch.Tensor, shape (batch, out_channels, horizon)
            Non-causal convolutional features.
        """
        return self.activation(self.convolution(sequence) + self.residual(sequence))


class ConditionalStochasticTCN(nn.Module):
    """Generate bounded multi-period free-variable trajectories with a NCTCN.

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
    latent_dim : int, default=32
        Dimension of the trajectory-level standard Gaussian latent vector.
    latent_embedding_dim : int, default=64
        Width of the learned latent embedding concatenated at every time step.
    kernel_size : int, default=3
        Positive odd temporal convolution width in time steps.
    dilations : sequence of int, default=(1, 2, 4, 8)
        Positive block dilation factors. With kernel three, the default
        receptive field is 31 time periods.
    """

    def __init__(
        self,
        condition_dim: int,
        free_dim: int,
        n_free_pg: int,
        projection_method: str = PROJECTION_METHOD,
        hidden_channels: int = 256,
        latent_dim: int = 32,
        latent_embedding_dim: int = 64,
        kernel_size: int = 3,
        dilations: Sequence[int] = (1, 2, 4, 8),
    ):
        """Build the latent/input embeddings, temporal blocks, and output map.

        Parameters
        ----------
        condition_dim, free_dim, n_free_pg : int
            Per-period condition width, output width, and leading active-power
            component count.
        projection_method : str, default="inward_clip_v1"
            Must equal the paper-aligned inward-gradient clip version.
        hidden_channels : int, default=256
            Feature width of all temporal blocks.
        latent_dim : int, default=32
            Trajectory-level Gaussian latent dimension.
        latent_embedding_dim : int, default=64
            Learned latent feature width repeated over time.
        kernel_size : int, default=3
            Positive odd temporal convolution width.
        dilations : sequence of int, default=(1, 2, 4, 8)
            Positive dilation factors from shallow to deep blocks.
        """
        super().__init__()
        dimensions = (
            condition_dim,
            free_dim,
            hidden_channels,
            latent_dim,
            latent_embedding_dim,
        )
        if min(dimensions) < 1:
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
        self.latent_dim = int(latent_dim)
        self.latent_embedding_dim = int(latent_embedding_dim)
        self.kernel_size = int(kernel_size)
        self.dilations = tuple(int(value) for value in dilations)

        self.latent_embedding = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_embedding_dim),
            nn.ReLU(),
        )
        self.input_embedding = nn.Sequential(
            nn.Linear(
                self.condition_dim + self.latent_embedding_dim,
                self.hidden_channels,
            ),
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
        # Clip expects unit-interval raw values. Center the initial projection at
        # 0.5 and keep its random spread small enough to avoid early saturation.
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
            "latent_dim": self.latent_dim,
            "latent_embedding_dim": self.latent_embedding_dim,
            "kernel_size": self.kernel_size,
            "dilations": list(self.dilations),
        }

    def project_unit_interval(self, raw: torch.Tensor) -> torch.Tensor:
        """Map raw outputs to unit coordinates for physical rescaling.

        Parameters
        ----------
        raw : torch.Tensor
            Unconstrained generator output of arbitrary shape.

        Returns
        -------
        torch.Tensor
            Unit-interval coordinates using the proposed inward-gradient clip.
        """
        return constraint_aware_clip(raw)

    def raw_trajectory(
        self,
        condition: torch.Tensor,
        candidates: int,
        latent: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Map conditions and shared latent vectors to unconstrained trajectories.

        Parameters
        ----------
        condition : torch.Tensor, shape (batch, horizon, condition_dim)
            Normalized min/mean/max uncertainty sequence.
        candidates : int
            Number of independent trajectory-level latent samples per instance.
        latent : torch.Tensor or None, shape (batch, candidates, latent_dim)
            Optional fixed latent samples. ``None`` draws standard Gaussian samples.

        Returns
        -------
        torch.Tensor, shape (batch, candidates, horizon, free_dim)
            Unconstrained active-power and voltage outputs.
        """
        if condition.ndim != 3 or condition.shape[2] != self.condition_dim:
            raise ValueError(
                f"condition must have shape (batch,horizon,{self.condition_dim})"
            )
        if candidates < 1:
            raise ValueError("candidates must be positive")
        batch, horizon, _ = condition.shape
        if latent is None:
            latent = torch.randn(
                batch,
                candidates,
                self.latent_dim,
                dtype=condition.dtype,
                device=condition.device,
            )
        expected = (batch, candidates, self.latent_dim)
        if tuple(latent.shape) != expected:
            raise ValueError(f"latent must have shape {expected}")
        if latent.dtype != condition.dtype or latent.device != condition.device:
            latent = latent.to(dtype=condition.dtype, device=condition.device)

        latent_features = self.latent_embedding(latent)
        condition_features = condition[:, None].expand(-1, candidates, -1, -1)
        latent_features = latent_features[:, :, None].expand(-1, -1, horizon, -1)
        features = torch.cat([condition_features, latent_features], dim=3)
        features = self.input_embedding(features).reshape(
            batch * candidates,
            horizon,
            self.hidden_channels,
        ).transpose(1, 2)
        for block in self.temporal_blocks:
            features = block(features)
        features = features.transpose(1, 2)
        return self.output_mapping(features).reshape(
            batch,
            candidates,
            horizon,
            self.free_dim,
        )

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
        raw : torch.Tensor, shape (batch, candidates, horizon, free_dim)
            Unconstrained generator outputs.
        lower, upper : torch.Tensor, shape (free_dim,)
            Static active-power bounds in MW followed by voltage bounds in p.u.
        ramp_up, ramp_down : torch.Tensor, shape (n_free_pg,)
            Positive active-power movement limits in MW per time period. They
            apply only between generated periods; the first decision can take
            any value inside its static physical bounds.

        Returns
        -------
        trajectory : torch.Tensor, shape (batch, candidates, horizon, free_dim)
            Physically scaled free-variable trajectories.
        admissible_lower, admissible_upper : torch.Tensor
            Dynamic bounds used at each generated point, with the same shape as
            ``trajectory``.
        """
        if raw.ndim != 4 or raw.shape[-1] != self.free_dim:
            raise ValueError(
                f"raw must have shape (batch,candidates,horizon,{self.free_dim})"
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
        batch, candidates, horizon, _ = raw.shape

        pg_min, pg_max = lower[:self.n_free_pg], upper[:self.n_free_pg]
        voltage_min = lower[self.n_free_pg:]
        voltage_max = upper[self.n_free_pg:]
        voltage = voltage_min + (voltage_max - voltage_min) * self.project_unit_interval(
            raw[..., self.n_free_pg:]
        )

        active_power = []
        active_lower = []
        active_upper = []
        # The first decision has no predecessor and therefore uses only its
        # static operating range, as assumed in the present SMPC experiment.
        previous = pg_min + (pg_max - pg_min) * self.project_unit_interval(
            raw[:, :, 0, :self.n_free_pg]
        )
        active_power.append(previous)
        active_lower.append(pg_min.expand(batch, candidates, -1))
        active_upper.append(pg_max.expand(batch, candidates, -1))
        for time in range(1, horizon):
            time_lower = torch.maximum(pg_min, previous - ramp_down)
            time_upper = torch.minimum(pg_max, previous + ramp_up)
            previous = time_lower + (time_upper - time_lower) * self.project_unit_interval(
                raw[:, :, time, :self.n_free_pg]
            )
            active_power.append(previous)
            active_lower.append(time_lower)
            active_upper.append(time_upper)

        active_power = torch.stack(active_power, dim=2)
        active_lower = torch.stack(active_lower, dim=2)
        active_upper = torch.stack(active_upper, dim=2)
        voltage_lower = voltage_min.expand(batch, candidates, horizon, -1)
        voltage_upper = voltage_max.expand(batch, candidates, horizon, -1)
        trajectory = torch.cat([active_power, voltage], dim=3)
        admissible_lower = torch.cat([active_lower, voltage_lower], dim=3)
        admissible_upper = torch.cat([active_upper, voltage_upper], dim=3)
        return trajectory, admissible_lower, admissible_upper

    def forward(
        self,
        condition: torch.Tensor,
        candidates: int,
        lower: torch.Tensor,
        upper: torch.Tensor,
        ramp_up: torch.Tensor,
        ramp_down: torch.Tensor,
        latent: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate constrained free-variable trajectories.

        Parameters
        ----------
        condition, candidates, latent
            Inputs documented by :meth:`raw_trajectory`.
        lower, upper, ramp_up, ramp_down
            Physical output parameters documented by
            :meth:`enforce_admissible_output`.

        Returns
        -------
        trajectory, admissible_lower, admissible_upper : tuple of torch.Tensor
            Generated trajectories and their pointwise output intervals.
        """
        raw = self.raw_trajectory(condition, candidates, latent)
        return self.enforce_admissible_output(
            raw,
            lower,
            upper,
            ramp_up,
            ramp_down,
        )
