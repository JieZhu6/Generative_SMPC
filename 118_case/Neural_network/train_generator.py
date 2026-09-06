"""Train the paper-style non-causal TCN stochastic dispatch generator.

The generator consumes a temporal min/mean/max scenario representation and
produces several horizon-wide free-variable trajectories. A frozen DECS model
reconstructs every candidate under every scenario and time period. Training
uses the paper's two-stage hierarchical mean-CVaR feasibility, diversity, and
economic-shaping losses without optimal dispatch labels. Stage two continues
from the stage-one iterate with a reset optimizer and gradually activates the
economic weight after the paper's stage boundary.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from Neural_network.decs import (  # noqa: E402
    DifferentiableEqualityCompletion,
    load_decs_checkpoint,
)
from Neural_network.generator_benchmarks import (  # noqa: E402
    build_generator_model,
    get_benchmark_spec,
)
from Neural_network.noncausal_tcn import ConditionalStochasticTCN  # noqa: E402


def normalize(value: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Map physical loads to dimensionless ``[-1, 1]`` values.

    Parameters
    ----------
    value : torch.Tensor
        Load values with final dimensions ``(n_load_buses, 2)`` in MW/Mvar.
    lower, upper : torch.Tensor, shape (n_load_buses, 2)
        Training-set extrema in the same physical units.

    Returns
    -------
    torch.Tensor
        Normalized values with the same shape as ``value``.
    """
    return 2.0 * (value - lower) / (upper - lower).clamp_min(1e-8) - 1.0


def format_duration(seconds: float) -> str:
    """Format a wall-clock duration as ``HH:MM:SS``.

    Parameters
    ----------
    seconds : float
        Nonnegative duration in seconds.

    Returns
    -------
    str
        Rounded fixed-width duration.
    """
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def file_sha256(path: Path) -> str:
    """Return a stable fingerprint binding Generator training to one DECS model."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_torch_save(payload: dict, path: Path) -> None:
    """Atomically replace a PyTorch checkpoint after a complete temporary write.

    Parameters
    ----------
    payload : dict
        Serializable checkpoint dictionary passed to :func:`torch.save`.
    path : pathlib.Path
        Final checkpoint path. Its parent directory is created when needed.

    Notes
    -----
    Writing a temporary sibling first preserves the previous best checkpoint
    if training is interrupted before the new serialization completes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def improves_validation_economic_checkpoint(
    metrics: dict[str, float],
    validation_instances: int,
    best_cost: float,
    min_delta: float,
) -> bool:
    """Return whether a full-hit validation cost improves the Stage-2 model.

    Parameters
    ----------
    metrics : dict[str, float]
        Validation metrics from :func:`run_epoch`, including the mean
        ``best_feasible_cost`` and number of instances with a feasible candidate.
    validation_instances : int
        Total number of validation SMPC instances; must be positive.
    best_cost : float
        Lowest previously accepted validation mean best-feasible cost in the
        physical generation-cost units used by the benchmark.
    min_delta : float
        Required nonnegative cost decrease in the same physical units.

    Returns
    -------
    bool
        True only when every validation instance has at least one feasible
        candidate and its mean lowest-feasible cost improves by ``min_delta``.
    """
    if validation_instances < 1 or min_delta < 0:
        raise ValueError("validation_instances must be positive and min_delta nonnegative")
    full_hit = int(metrics["instances_with_feasible_cost"]) == validation_instances
    cost = float(metrics["best_feasible_cost"])
    return full_hit and np.isfinite(cost) and cost < best_cost - min_delta


def make_paper_splits(n_instances: int, seed: int) -> dict[str, np.ndarray]:
    """Create the paper's reproducible 8:1:1 split without changing data files.

    Parameters
    ----------
    n_instances : int
        Number of independent SMPC scenario bundles; at least ten.
    seed : int
        Random-permutation seed.

    Returns
    -------
    dict[str, np.ndarray]
        Integer indices for train, validation, and test sets.
    """
    if n_instances < 10:
        raise ValueError("at least ten instances are required for an 8:1:1 split")
    order = np.random.default_rng(seed).permutation(n_instances)
    n_train = int(0.8 * n_instances)
    n_validation = int(0.1 * n_instances)
    return {
        "train": order[:n_train],
        "validation": order[n_train:n_train + n_validation],
        "test": order[n_train + n_validation:],
    }


def compute_training_feature_bounds(
    data_dir: Path,
    train_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute temporal load extrema using only the new training split.

    Parameters
    ----------
    data_dir : pathlib.Path
        SMPC dataset containing ``current_load.npy`` and ``future_pool.npy``.
    train_indices : np.ndarray
        Indices assigned to the 80% training subset.

    Returns
    -------
    feature_min, feature_max : tuple[np.ndarray, np.ndarray]
        Per-load-bus P/Q extrema with shape ``(n_load_buses, 2)``. The same
        bounds apply to every time period and all three pooling channels.
    """
    current = np.load(data_dir / "current_load.npy", mmap_mode="r")
    pool = np.load(data_dir / "future_pool.npy", mmap_mode="r")
    current_train = np.asarray(current[train_indices])
    pool_train = np.asarray(pool[train_indices])
    feature_min = np.minimum(
        current_train.min(axis=0),
        pool_train.min(axis=(0, 1, 2)),
    )
    feature_max = np.maximum(
        current_train.max(axis=0),
        pool_train.max(axis=(0, 1, 2)),
    )
    return feature_min, feature_max


class GeneratorDataset(Dataset):
    """Expose NCTCN conditions and full scenario loads for selected instances.

    Parameters
    ----------
    data_dir : pathlib.Path
        Directory written by ``generate_smpc_dataset.py``.
    indices : np.ndarray
        External SMPC-instance indices in this split.
    feature_min, feature_max : np.ndarray, shape (n_load_buses, 2)
        Load extrema computed from the training split only, in MW/Mvar.
    """

    def __init__(
        self,
        data_dir: Path,
        indices: np.ndarray,
        feature_min: np.ndarray,
        feature_max: np.ndarray,
    ):
        """Memory-map arrays and store training-only temporal scaling.

        Parameters
        ----------
        data_dir : pathlib.Path
            Directory containing current, pooled, and full future load arrays.
        indices : np.ndarray
            Instance indices exposed by this dataset object.
        feature_min, feature_max : np.ndarray, shape (n_load_buses, 2)
            P/Q extrema computed from the training subset only.
        """
        self.current = np.load(data_dir / "current_load.npy", mmap_mode="r")
        self.pool = np.load(data_dir / "future_pool.npy", mmap_mode="r")
        self.future = np.load(data_dir / "future_load.npy", mmap_mode="r")
        self.indices = np.asarray(indices, dtype=np.int64)
        # Temporal convolution shares channel weights, so each physical channel
        # uses one scale across t=1,...,T.
        self.feature_min = torch.tensor(feature_min, dtype=torch.float32)
        self.feature_max = torch.tensor(feature_max, dtype=torch.float32)

        if self.current.ndim != 3 or self.current.shape[-1] != 2:
            raise ValueError("current_load.npy must have shape (N,n_load_buses,2)")
        if self.pool.ndim != 5 or self.pool.shape[1] != 3:
            raise ValueError("future_pool.npy must have shape (N,3,T-1,n_load_buses,2)")
        if self.future.ndim != 5 or self.future.shape[-1] != 2:
            raise ValueError("future_load.npy must have shape (N,S,T-1,n_load_buses,2)")
        if not (
            len(self.current) == len(self.pool) == len(self.future)
            and self.pool.shape[2:] == self.future.shape[2:]
            and self.current.shape[1:] == self.future.shape[3:]
        ):
            raise ValueError("current, pooled, and scenario load dimensions do not match")

        self.n_scenarios = int(self.future.shape[1])
        self.horizon = 1 + int(self.future.shape[2])
        self.condition_dim = 3 * int(np.prod(self.current.shape[1:]))

    def __len__(self) -> int:
        """Return the number of SMPC instances in this split."""
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one temporal condition and all physical load scenarios.

        Parameters
        ----------
        index : int
            Position within this split.

        Returns
        -------
        condition : torch.Tensor, shape (T, 3*n_load_buses*2)
            Time-major min/mean/max load features.
        scenario_load : torch.Tensor, shape (S, T, n_load_buses, 2)
            Current load repeated across scenarios followed by future loads.
        """
        instance = int(self.indices[index])
        current = torch.tensor(np.array(self.current[instance]), dtype=torch.float32)
        pool = torch.tensor(np.array(self.pool[instance]), dtype=torch.float32)
        future = torch.tensor(np.array(self.future[instance]), dtype=torch.float32)

        current_normalized = normalize(current, self.feature_min, self.feature_max)
        pool_normalized = normalize(pool, self.feature_min, self.feature_max)
        current_pool = current_normalized[None, None].expand(3, 1, -1, -1)
        pooled_horizon = torch.cat([current_pool, pool_normalized], dim=1)
        condition = pooled_horizon.permute(1, 0, 2, 3).reshape(
            self.horizon, self.condition_dim,
        )

        current_scenarios = current[None, None].expand(self.n_scenarios, 1, -1, -1)
        scenario_load = torch.cat([current_scenarios, future], dim=1)
        return condition, scenario_load


def mean_cvar_violation(
    residual: torch.Tensor,
    alpha: float,
    tail_fraction: float,
    check_nonnegative: bool = False,
) -> torch.Tensor:
    """Evaluate equation (27) for one normalized constraint family.

    Parameters
    ----------
    residual : torch.Tensor, shape (..., m_c)
        Nonnegative normalized residual vector for one constraint family. The
        leading dimensions normally represent batch, candidate, scenario, and
        time, while ``m_c`` is the number of scalar inequalities in the family.
    alpha : float
        Mean-violation weight in ``(0,1)``; ``1-alpha`` weights upper-tail CVaR.
    tail_fraction : float
        Fraction in ``(0,1]`` defining the empirical upper tail. The largest
        ``ceil(tail_fraction*m_c)`` entries are averaged, with at least one.
    check_nonnegative : bool, default=False
        Debug-only full-tensor check that ``residual`` has no negative entry.
        Keep disabled in normal training to avoid a device synchronization.

    Returns
    -------
    torch.Tensor, shape (...,)
        Hierarchical mean-CVaR violation ``V^(c)`` with ``m_c`` removed.
    """
    if residual.ndim < 1 or residual.shape[-1] < 1:
        raise ValueError("residual must have a nonempty final constraint dimension")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0,1)")
    if not 0.0 < tail_fraction <= 1.0:
        raise ValueError("tail_fraction must lie in (0,1]")
    if check_nonnegative and torch.any(residual < 0):
        raise ValueError("residual must contain positive-part values")

    tail_count = max(1, int(np.ceil(tail_fraction * residual.shape[-1])))
    dense_mean = residual.mean(dim=-1)
    if tail_count == 1:
        upper_tail = residual.max(dim=-1).values
    else:
        # CVaR uses only the selected values' mean, so ordering the tail is wasted work.
        upper_tail = residual.topk(
            tail_count, dim=-1, sorted=False,
        ).values.mean(dim=-1)
    return alpha * dense_mean + (1.0 - alpha) * upper_tail


def hierarchical_feasibility_loss(
    violation_families: dict[str, torch.Tensor],
    alpha: float,
    tail_fraction: float,
    check_nonnegative: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Evaluate equations (27)-(28) over constraint families.

    Parameters
    ----------
    violation_families : dict[str, torch.Tensor]
        Nonempty mapping from family name to tensors shaped
        ``(batch,candidates,scenarios,time,m_c)``. Families may use different
        time lengths; reference-generator ramping uses ``T-1``.
    alpha : float
        Shared mean weight ``alpha_c`` for the present experiment.
    tail_fraction : float
        Shared empirical CVaR fraction ``rho_c`` for the present experiment.
    check_nonnegative : bool, default=False
        Enable expensive checks of all positive-part residual tensors.

    Returns
    -------
    candidate_violation : torch.Tensor, shape (batch, candidates)
        Candidate-wise sum over scenarios, time periods, and families used in
        the soft feasibility score of equation (30).
    loss_feasibility : torch.Tensor
        Batch mean of the candidate sum in equation (28); the candidate,
        scenario, time, and family axes retain the paper's summation.
    shaped_families : dict[str, torch.Tensor]
        Per-family ``V^(c)`` tensors before scenario/time summation.
    """
    if not violation_families:
        raise ValueError("violation_families must be nonempty")

    candidate_violation = None
    shaped_families: dict[str, torch.Tensor] = {}
    leading_shape = None
    for name, residual in violation_families.items():
        if residual.ndim != 5:
            raise ValueError(
                f"constraint family {name!r} must have shape (batch,candidates,scenarios,time,m_c)"
            )
        if leading_shape is None:
            leading_shape = residual.shape[:2]
        elif residual.shape[:2] != leading_shape:
            raise ValueError("constraint families must share batch and candidate dimensions")
        shaped = mean_cvar_violation(
            residual, alpha, tail_fraction, check_nonnegative=check_nonnegative,
        )
        shaped_families[name] = shaped
        contribution = shaped.flatten(start_dim=2).sum(dim=2)
        candidate_violation = (
            contribution if candidate_violation is None
            else candidate_violation + contribution
        )

    loss_feasibility = candidate_violation.sum(dim=1).mean()
    return candidate_violation, loss_feasibility, shaped_families


def soft_feasibility_score(
    candidate_violation: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """Evaluate equation (30)'s differentiable candidate feasibility score.

    Parameters
    ----------
    candidate_violation : torch.Tensor, shape (batch, candidates)
        Sum of hierarchical family violations over scenarios and time.
    tau : float
        Positive temperature on the same dimensionless summed-violation scale.

    Returns
    -------
    torch.Tensor, shape (batch, candidates)
        Scores in ``(0,1]``; exact zeros can occur only through floating-point
        underflow for strongly infeasible candidates.
    """
    if candidate_violation.ndim != 2:
        raise ValueError("candidate_violation must have shape (batch,candidates)")
    if tau <= 0:
        raise ValueError("tau must be positive")
    return torch.exp(-candidate_violation / tau)


def diversity_loss(
    trajectory: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    score: torch.Tensor,
    sigma: float,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """Evaluate equation (31)'s feasibility-weighted Gaussian repulsion.

    Parameters
    ----------
    trajectory : torch.Tensor, shape (batch, candidates, horizon, free_dim)
        Generated horizon-wide free-variable trajectories in MW and p.u. Each
        candidate's last two dimensions are flattened for the dimension-mean
        squared distance in equation (31). The mean keeps ``sigma`` comparable
        when the IEEE-118 control dimension changes.
    lower, upper : torch.Tensor, shape (free_dim,)
        Fixed physical operating limits used by equation (29).
    score : torch.Tensor, shape (batch, candidates)
        Soft feasibility scores from equation (30).
    sigma : float
        Positive Gaussian width in normalized coordinates.
    epsilon : float, default=1e-12
        Positive denominator constant appearing in equation (31).

    Returns
    -------
    torch.Tensor
        Scalar similarity loss whose minimization separates candidates.
    """
    if trajectory.ndim != 4:
        raise ValueError("trajectory must have shape (batch,candidates,horizon,free_dim)")
    if lower.ndim != 1 or upper.shape != lower.shape:
        raise ValueError("lower and upper must share shape (free_dim,)")
    if trajectory.shape[-1] != len(lower) or score.shape != trajectory.shape[:2]:
        raise ValueError("diversity inputs have inconsistent dimensions")
    if sigma <= 0 or epsilon <= 0:
        raise ValueError("sigma and epsilon must be positive")
    center = 0.5 * (upper + lower)
    half_range = 0.5 * (upper - lower).clamp_min(1e-8)
    normalized = ((trajectory - center) / half_range).flatten(start_dim=2)
    mean_distance2 = (
        normalized[:, :, None, :] - normalized[:, None, :, :]
    ).square().mean(dim=-1)
    kernel = torch.exp(-mean_distance2 / (2.0 * sigma**2))
    weights = (score[:, :, None] * score[:, None, :]).detach()
    pair_mask = torch.triu(torch.ones_like(kernel, dtype=torch.bool), diagonal=1)
    numerator = (weights * kernel * pair_mask).sum(dim=(1, 2))
    denominator = (weights * pair_mask).sum(dim=(1, 2)) + epsilon
    return (numerator / denominator).mean()


def economic_shaping_loss(
    objective: torch.Tensor,
    score: torch.Tensor,
    gamma: float,
    cost_scale: float,
    alpha_eco: float,
    best_k: int,
) -> torch.Tensor:
    """Evaluate equation (32)'s mean-best-K economic shaping loss.

    Parameters
    ----------
    objective : torch.Tensor, shape (batch, candidates)
        Horizon cost for every generated trajectory, including the mean
        scenario-dependent slack-generator cost.
    score : torch.Tensor, shape (batch, candidates)
        Differentiable soft feasibility scores from equation (30).
    gamma : float
        Nonnegative amplification ``gamma_eco`` of infeasible candidate costs.
    cost_scale : float
        Positive frozen horizon-cost scale in the same monetary units as
        ``objective``. Dividing by this detached scalar preserves candidate
        ordering while preventing raw costs around 1e4--1e5 from dominating
        feasibility and diversity gradients.
    alpha_eco : float
        Weight in ``(0,1)`` assigned to the mean over all generated candidates;
        ``1-alpha_eco`` weights the best-``best_k`` subset.
    best_k : int
        Number ``K_b`` of candidates with the smallest feasibility-adjusted
        normalized costs selected independently for each batch instance.

    Returns
    -------
    torch.Tensor
        Scalar batch mean of the weighted full-candidate and best-K terms.
    """
    if objective.ndim != 2 or score.shape != objective.shape:
        raise ValueError("objective and score must share shape (batch,candidates)")
    if gamma < 0 or cost_scale <= 0:
        raise ValueError("gamma must be nonnegative and cost_scale must be positive")
    if not 0.0 < alpha_eco < 1.0:
        raise ValueError("alpha_eco must lie in (0,1)")
    if not 1 <= best_k <= objective.shape[1]:
        raise ValueError("best_k must lie in [1,candidates]")

    adjusted = objective / cost_scale * (1.0 + gamma * (1.0 - score))
    full_mean = adjusted.mean(dim=1)
    best_mean = adjusted.topk(best_k, dim=1, largest=False).values.mean(dim=1)
    return (alpha_eco * full_mean + (1.0 - alpha_eco) * best_mean).mean()


def resolve_economic_cost_scale(
    configured_scale: float,
    stage1_history: list[dict],
) -> float:
    """Return a positive fixed cost divisor for all Stage-2 updates.

    Parameters
    ----------
    configured_scale : float
        User-specified cost scale in monetary units. A positive value is used
        directly; zero selects the last finite Stage-1 validation mean cost.
    stage1_history : list[dict]
        Completed Stage-1 epoch records containing ``validation_cost_mean``.

    Returns
    -------
    float
        Frozen positive horizon-cost scale. It is detached from optimization
        and therefore changes only units, not candidate cost ordering.
    """
    if configured_scale < 0:
        raise ValueError("configured economic cost scale must be nonnegative")
    if configured_scale > 0:
        return float(configured_scale)
    for record in reversed(stage1_history):
        value = float(record.get("validation_cost_mean", float("nan")))
        if np.isfinite(value) and value > 0:
            return value
    raise ValueError("cannot infer economic cost scale from Stage-1 history")


def economic_warmup_weight(
    target_weight: float,
    epoch: int,
    stage1_epochs: int,
    warmup_epochs: int,
) -> float:
    """Linearly increase the normalized economic weight during early Stage 2.

    Parameters
    ----------
    target_weight : float
        Final nonnegative normalized economic-loss weight.
    epoch : int
        Current one-based global training epoch.
    stage1_epochs : int
        Last feasibility/diversity-only epoch.
    warmup_epochs : int
        Positive number of Stage-2 epochs used for the linear ramp.

    Returns
    -------
    float
        Zero in Stage 1 and ``target_weight`` after the warmup interval.
    """
    if target_weight < 0 or warmup_epochs < 1:
        raise ValueError("target weight must be nonnegative and warmup positive")
    if epoch <= stage1_epochs:
        return 0.0
    progress = min(1.0, (epoch - stage1_epochs) / warmup_epochs)
    return float(target_weight * progress)


def build_plateau_scheduler(
    optimizer: torch.optim.Optimizer,
    factor: float,
    patience: int,
    min_delta: float,
    min_learning_rate: float,
) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    """Build the validation-loss scheduler shared by both training stages.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Stage-specific Adam optimizer whose learning rate is adjusted.
    factor : float
        Multiplicative learning-rate reduction in ``(0, 1)``.
    patience : int
        Validation epochs without sufficient improvement before one reduction.
    min_delta : float
        Absolute validation-loss improvement required to reset the plateau count.
    min_learning_rate : float
        Positive lower bound for the stage learning rate.

    Returns
    -------
    torch.optim.lr_scheduler.ReduceLROnPlateau
        Scheduler configured to minimize validation total loss.
    """
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=factor,
        patience=patience,
        threshold=min_delta,
        threshold_mode="abs",
        min_lr=min_learning_rate,
    )


def expected_generation_objective(
    scenario_pg: torch.Tensor,
    completion: DifferentiableEqualityCompletion,
) -> torch.Tensor:
    """Compute horizon generation cost including expected slack cost.

    Parameters
    ----------
    scenario_pg : torch.Tensor, shape (batch, candidates, scenarios, horizon, n_gen)
        Reconstructed active generation in MW. Nonreference generation is
        shared across scenarios, while slack generation is scenario-dependent.
    completion : DifferentiableEqualityCompletion
        Completion module whose physics object supplies generator indices and
        quadratic cost coefficients.

    Returns
    -------
    torch.Tensor, shape (batch, candidates)
        For every candidate, total generation cost is summed over the horizon
        and averaged over scenarios. Since nonreference Pg is shared, this
        average only changes the scenario-dependent slack-generator term.
    """
    physics = completion.physics
    if scenario_pg.ndim != 5 or scenario_pg.shape[-1] != physics.n_gen:
        raise ValueError(
            "scenario_pg must have shape (batch,candidates,scenarios,horizon,n_gen)"
        )
    period_cost = physics.generation_cost(scenario_pg)
    return period_cost.sum(dim=3).mean(dim=2)


def combine_generator_losses(
    loss_feasibility: torch.Tensor,
    loss_diversity: torch.Tensor,
    loss_economic: torch.Tensor,
    feasibility_weight: float,
    diversity_weight: float,
    economic_weight: float,
    use_diversity: bool,
) -> torch.Tensor:
    """Combine the active benchmark loss terms.

    Parameters
    ----------
    loss_feasibility, loss_diversity, loss_economic : torch.Tensor
        Scalar losses from equations (28), (31), and (32), respectively.
    feasibility_weight, diversity_weight, economic_weight : float
        Nonnegative coefficients multiplying the three scalar terms.
    use_diversity : bool
        Whether equation (31) belongs to the benchmark objective. WD-CSNG sets
        this to ``False`` so the diversity term is absent from the graph.

    Returns
    -------
    torch.Tensor
        Scalar training objective for the selected benchmark and stage.
    """
    loss = feasibility_weight * loss_feasibility
    if use_diversity:
        loss = loss + diversity_weight * loss_diversity
    if economic_weight > 0.0:
        loss = loss + economic_weight * loss_economic
    return loss


def compute_batch(
    model: ConditionalStochasticTCN,
    completion: DifferentiableEqualityCompletion,
    condition: torch.Tensor,
    scenario_load: torch.Tensor,
    free_lower: torch.Tensor,
    free_upper: torch.Tensor,
    free_ramp: torch.Tensor,
    reference_ramp: torch.Tensor,
    args: argparse.Namespace,
    economic_weight: float,
    latent: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Generate candidates and evaluate the complete self-supervised objective.

    Parameters
    ----------
    model : ConditionalStochasticTCN
        Trainable non-causal temporal generator.
    completion : DifferentiableEqualityCompletion
        Frozen DECS and exact AC reconstruction.
    condition : torch.Tensor, shape (B, T, condition_dim)
        Normalized pooled temporal features.
    scenario_load : torch.Tensor, shape (B, S, T, n_load_buses, 2)
        Physical scenario loads in MW/Mvar.
    free_lower, free_upper : torch.Tensor, shape (free_dim,)
        Static free-variable bounds in MW and p.u.
    free_ramp : torch.Tensor, shape (n_free_pg,)
        Output-layer ramp limits in MW per period.
    reference_ramp : scalar torch.Tensor
        Symmetric reference-generator ramp limit in MW per period. The first
        decision is unconstrained; only consecutive generated periods are
        checked.
    args : argparse.Namespace
        Validated loss and sampling hyperparameters declared in :func:`main`.
    economic_weight : float
        Nonnegative effective weight multiplying the economic loss. Stage one
        uses zero; stage two increases it toward ``args.lambda_eco``.
    latent : torch.Tensor or None, shape (B, K, latent_dim)
        Optional fixed latents for deterministic evaluation.

    Returns
    -------
    loss : torch.Tensor
        Scalar differentiable objective.
    metrics : dict[str, torch.Tensor]
        Detached loss, feasibility, cost, and PF diagnostics.
    """
    batch, scenarios, horizon = scenario_load.shape[:3]
    candidates = args.candidates
    trajectory, _, _ = model(
        condition, candidates, free_lower, free_upper,
        free_ramp, free_ramp, latent,
    )
    free_dim = completion.physics.free_dim
    if tuple(trajectory.shape) != (batch, candidates, horizon, free_dim):
        raise RuntimeError("generator trajectory dimensions do not match the batch")

    candidate_scenarios = trajectory[:, :, None].expand(
        -1, -1, scenarios, -1, -1,
    )
    repeated_load = scenario_load[:, None].expand(
        -1, candidates, -1, -1, -1, -1,
    )
    state = completion(
        candidate_scenarios.reshape(batch * candidates * scenarios * horizon, free_dim),
        repeated_load.reshape(
            batch * candidates * scenarios * horizon, *scenario_load.shape[3:],
        ),
    )

    physical_violation = completion.physics.constraint_violation(state).reshape(
        batch, candidates, scenarios, horizon, -1,
    )
    physics = completion.physics
    category_sizes = physics.remaining_constraint_sizes()
    if sum(category_sizes.values()) != physical_violation.shape[-1]:
        raise RuntimeError("physical constraint categories do not match violation width")
    physical_categories = dict(zip(
        category_sizes,
        torch.split(physical_violation, tuple(category_sizes.values()), dim=-1),
    ))
    reference_pg = state["pg"].reshape(
        batch, candidates, scenarios, horizon, completion.physics.n_gen,
    )[..., completion.physics.reference_generator]
    delta_pg = reference_pg[:, :, :, 1:] - reference_pg[:, :, :, :-1]
    ramp_scale = reference_ramp.clamp_min(1e-8)
    # The dependent reference-generator ramp is the sixth constraint family.
    # Its first valid difference is t=2 minus t=1, hence the T-1 time axis.
    ramp_violation = torch.stack([
        torch.relu((delta_pg - ramp_scale) / ramp_scale),
        torch.relu((-delta_pg - ramp_scale) / ramp_scale),
    ], dim=-1)
    violation_families = {**physical_categories, "ramp": ramp_violation}
    candidate_violation, loss_feasibility, shaped_families = (
        hierarchical_feasibility_loss(
            violation_families,
            args.mean_cvar_alpha,
            args.cvar_tail_fraction,
            check_nonnegative=args.detect_anomaly,
        )
    )
    maximum_violation = torch.stack([
        values.flatten(start_dim=2).amax(dim=2)
        for values in violation_families.values()
    ], dim=2).amax(dim=2)
    score = soft_feasibility_score(candidate_violation, args.tau_feas)
    loss_diversity = torch.zeros(
        (), dtype=condition.dtype, device=condition.device,
    )
    if args.use_diversity:
        loss_diversity = diversity_loss(
            trajectory, free_lower, free_upper,
            score, args.sigma_div, args.loss_epsilon,
        )

    feasible = (maximum_violation <= args.feasibility_tolerance).detach()
    # Sum all-generator cost over time in each scenario, then average scenarios.
    # Only reconstructed slack Pg varies across scenarios; free Pg is shared.
    scenario_pg = state["pg"].reshape(
        batch, candidates, scenarios, horizon, completion.physics.n_gen,
    )
    objective = expected_generation_objective(scenario_pg, completion)
    loss_economic = torch.zeros((), dtype=condition.dtype, device=condition.device)
    if economic_weight > 0.0:
        loss_economic = economic_shaping_loss(
            objective, score, args.gamma_eco, args.economic_cost_scale,
            args.alpha_eco, args.economic_top_k,
        )

    loss = combine_generator_losses(
        loss_feasibility, loss_diversity, loss_economic,
        args.lambda_fea, args.lambda_div, economic_weight,
        args.use_diversity,
    )

    best_feasible_cost = torch.where(
        feasible, objective, torch.full_like(objective, torch.inf),
    ).amin(dim=1)
    has_feasible = torch.isfinite(best_feasible_cost)
    metrics = {
        "loss": loss.detach(),
        "fea": loss_feasibility.detach(),
        "div": loss_diversity.detach(),
        "eco": loss_economic.detach(),
        "candidate_violation": candidate_violation.mean().detach(),
        "soft_score": score.mean().detach(),
        "feasible": feasible.float().mean().detach(),
        "hit": feasible.any(dim=1).float().mean().detach(),
        "max_violation": maximum_violation.mean().detach(),
        "worst_violation": maximum_violation.amax().detach(),
        "violation_pg": shaped_families["pg"].mean().detach(),
        "violation_qg": shaped_families["qg"].mean().detach(),
        "violation_voltage": shaped_families["voltage"].mean().detach(),
        "violation_angle": shaped_families["angle"].mean().detach(),
        "violation_thermal": shaped_families["thermal"].mean().detach(),
        "violation_ramp": shaped_families["ramp"].mean().detach(),
        "cost_mean": objective.mean().detach(),
        "cost_best": objective.amin(dim=1).mean().detach(),
        # Diagnostic only: mean over operating points of the worst normalized
        # nodal P/Q mismatch. It is not currently part of the training loss.
        "pf_residual": state["pf_residual"].mean().detach(),
        "best_cost_sum": torch.where(
            has_feasible, best_feasible_cost, torch.zeros_like(best_feasible_cost),
        ).sum().detach(),
        "best_cost_count": has_feasible.sum().detach(),
    }
    return loss, metrics


def run_epoch(
    model: ConditionalStochasticTCN,
    completion: DifferentiableEqualityCompletion,
    loader: DataLoader,
    free_lower: torch.Tensor,
    free_upper: torch.Tensor,
    free_ramp: torch.Tensor,
    reference_ramp: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    economic_weight: float,
    latent_seed: int,
) -> dict[str, float]:
    """Run one train/evaluation epoch and aggregate instance-level metrics.

    Parameters
    ----------
    model, completion, loader
        Generator, frozen DECS, and split-specific mini-batches.
    free_lower, free_upper, free_ramp, reference_ramp
        Physical tensors documented by :func:`compute_batch`.
    args : argparse.Namespace
        Validated command-line hyperparameters.
    device : torch.device
        CPU or CUDA computation device.
    optimizer : torch.optim.Optimizer or None
        Adam for training; ``None`` selects evaluation.
    economic_weight : float
        Effective nonnegative economic-loss weight for this epoch.
    latent_seed : int
        Fixed evaluation-latent seed; ignored during training.

    Returns
    -------
    dict[str, float]
        Sample-weighted losses, feasibility rates, cost, and PF residual.
    """
    training = optimizer is not None
    model.train(training)
    completion.eval()
    names = (
        "loss", "fea", "div", "eco", "candidate_violation", "soft_score",
        "feasible", "hit",
        "max_violation", "violation_pg", "violation_qg",
        "violation_voltage", "violation_angle", "violation_thermal",
        "violation_ramp", "cost_mean", "cost_best", "pf_residual",
    )
    totals = {name: 0.0 for name in names}
    worst_violation = 0.0
    best_cost_sum = 0.0
    best_cost_count = 0
    seen = 0
    latent_generator = None
    if not training:
        latent_generator = torch.Generator(device=device).manual_seed(latent_seed)

    context = torch.enable_grad() if training else torch.no_grad()
    if training:
        optimizer.zero_grad(set_to_none=True)
    n_batches = len(loader)
    with context:
        for batch_number, (condition, scenario_load) in enumerate(loader, start=1):
            condition = condition.to(device)
            scenario_load = scenario_load.to(device)
            latent = None
            if latent_generator is not None:
                latent = torch.randn(
                    len(condition), args.candidates, model.latent_dim,
                    dtype=condition.dtype, device=device, generator=latent_generator,
                )
            loss, metrics = compute_batch(
                model, completion, condition, scenario_load,
                free_lower, free_upper, free_ramp, reference_ramp,
                args, economic_weight, latent,
            )
            if training:
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite loss before backward in training batch {batch_number}"
                    )
                accumulation_group_start = (
                    (batch_number - 1) // args.gradient_accumulation_steps
                ) * args.gradient_accumulation_steps
                accumulation_group_size = min(
                    args.gradient_accumulation_steps,
                    n_batches - accumulation_group_start,
                )
                (loss / accumulation_group_size).backward()
                update_due = (
                    batch_number % args.gradient_accumulation_steps == 0
                    or batch_number == n_batches
                )
                if update_due:
                    if args.max_grad_norm > 0.0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.max_grad_norm,
                            error_if_nonfinite=True,
                        )
                    else:
                        for name, parameter in model.named_parameters():
                            if (
                                parameter.grad is not None
                                and not torch.isfinite(parameter.grad).all()
                            ):
                                raise FloatingPointError(
                                    "non-finite generator gradient in "
                                    f"{name} at batch {batch_number}"
                                )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            count = len(condition)
            seen += count
            for name in names:
                totals[name] += float(metrics[name]) * count
            worst_violation = max(worst_violation, float(metrics["worst_violation"]))
            best_cost_sum += float(metrics["best_cost_sum"])
            best_cost_count += int(metrics["best_cost_count"])

    result = {name: value / seen for name, value in totals.items()}
    result["worst_violation"] = worst_violation
    result["best_feasible_cost"] = (
        best_cost_sum / best_cost_count if best_cost_count else float("nan")
    )
    result["instances_with_feasible_cost"] = best_cost_count
    return result


def weighted_loss_breakdown(
    metrics: dict[str, float],
    feasibility_weight: float,
    diversity_weight: float,
    economic_weight: float,
) -> dict[str, float]:
    """Return weighted loss components and their percentage contributions.

    Parameters
    ----------
    metrics : dict[str, float]
        Split metrics containing unweighted ``fea``, ``div``, and ``eco``.
    feasibility_weight, diversity_weight, economic_weight : float
        Nonnegative coefficients multiplying the three loss components.

    Returns
    -------
    dict[str, float]
        Weighted components, their sum, and percentages summing to 100 when
        the weighted sum is positive.
    """
    weighted = {
        "fea": feasibility_weight * float(metrics["fea"]),
        "div": diversity_weight * float(metrics["div"]),
        "eco": economic_weight * float(metrics["eco"]),
    }
    total = sum(weighted.values())
    shares = {
        name: (100.0 * value / total if total > 0.0 else 0.0)
        for name, value in weighted.items()
    }
    return {
        **{f"weighted_{name}": value for name, value in weighted.items()},
        "weighted_total": total,
        **{f"share_{name}": value for name, value in shares.items()},
    }


def print_epoch_metrics(
    epoch: int,
    total_epochs: int,
    stage: int,
    elapsed: float,
    eta: float,
    status: str,
    learning_rate: float,
    feasibility_weight: float,
    diversity_weight: float,
    train_economic_weight: float,
    validation_economic_weight: float,
    train_metrics: dict[str, float],
    validation_metrics: dict[str, float],
) -> None:
    """Print optimization, cost, violation, and feasibility data every epoch.

    Parameters
    ----------
    epoch, total_epochs : int
        Current and maximum epoch counts.
    stage : int
        Training stage: one excludes and two includes economic shaping.
    elapsed, eta : float
        Elapsed and estimated remaining wall time in seconds.
    status : str
        Validation-checkpoint status for the current epoch.
    learning_rate : float
        Current Adam step size.
    feasibility_weight, diversity_weight : float
        Fixed feasibility and diversity coefficients.
    train_economic_weight, validation_economic_weight : float
        Effective economic coefficients used for training and checkpoint
        validation in the current epoch.
    train_metrics, validation_metrics : dict[str, float]
        Split metrics returned by :func:`run_epoch`.

    Notes
    -----
    Constraint values are dimensionless positive normalized violations. Costs
    sum all generators over the horizon and average the scenario-dependent
    slack-generator contribution over uncertainty scenarios.
    """
    print(
        f"[epoch {epoch:4d}/{total_epochs} | stage {stage} | lr {learning_rate:.3e} | "
        f"lambda_eco(train) {train_economic_weight:.3e} | "
        f"elapsed {format_duration(elapsed)} | ETA {format_duration(eta)} | {status}]",
        flush=True,
    )
    for split, values, economic_weight in (
        ("train", train_metrics, train_economic_weight),
        ("validation", validation_metrics, validation_economic_weight),
    ):
        feasible_cost = values["best_feasible_cost"]
        feasible_cost_text = (
            f"{feasible_cost:.3f}" if np.isfinite(feasible_cost) else "n/a"
        )
        breakdown = weighted_loss_breakdown(
            values, feasibility_weight, diversity_weight, economic_weight,
        )
        print(
            f"  {split:10s} objective: total={values['loss']:.6e} | "
            f"Lfea={values['fea']:.3e} Ldiv={values['div']:.3e} "
            f"Leco(norm)={values['eco']:.3e}",
            flush=True,
        )
        print(
            f"  {split:10s} weighted: "
            f"fea={breakdown['weighted_fea']:.3e} "
            f"div={breakdown['weighted_div']:.3e} "
            f"eco={breakdown['weighted_eco']:.3e} | "
            f"share: fea={breakdown['share_fea']:.1f}% "
            f"div={breakdown['share_div']:.1f}% "
            f"eco={breakdown['share_eco']:.1f}% | "
            f"lambda_eco={economic_weight:.3e}",
            flush=True,
        )
        print(
            f"  {split:10s} cost: mean={values['cost_mean']:.3f} | "
            f"best_candidate={values['cost_best']:.3f} | "
            f"best_feasible={feasible_cost_text}",
            flush=True,
        )
        print(
            f"  {split:10s} violation(norm): candidate_sum_mean="
            f"{values['candidate_violation']:.3e} | soft_score={values['soft_score']:.3e} | "
            f"candidate_max_mean={values['max_violation']:.3e} | "
            f"worst={values['worst_violation']:.3e}",
            flush=True,
        )
        print(
            f"  {split:10s} mean-CVaR by type: "
            f"Pg={values['violation_pg']:.2e} Qg={values['violation_qg']:.2e} "
            f"V={values['violation_voltage']:.2e} angle={values['violation_angle']:.2e} "
            f"thermal={values['violation_thermal']:.2e} ramp={values['violation_ramp']:.2e}",
            flush=True,
        )
        print(
            f"  {split:10s} recovery: candidate_feasible={values['feasible']:.3f} | "
            f"hit_rate={values['hit']:.3f} | "
            f"DECS_PF={values['pf_residual']:.3e} p.u.",
            flush=True,
        )


def build_parser(benchmark: str = "csng") -> argparse.ArgumentParser:
    """Declare documented command-line parameters for one benchmark.

    Parameters
    ----------
    benchmark : str, default="csng"
        Experiment key: ``csng``, ``s_csng``, or ``wd_csng``.

    Returns
    -------
    argparse.ArgumentParser
        Parser with method-specific checkpoint and diversity defaults.
    """
    spec = get_benchmark_spec(benchmark)
    diversity_enabled = bool(spec["diversity_loss_enabled"])
    parser = argparse.ArgumentParser(
        description=f"Train the {spec['method']} non-causal TCN generator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    N_train = 5000
    S_num = 20
    T_horizon = 16
    parser.add_argument(
        "--data", type=Path,
        default=ROOT / "Data_generation" / "data" / f"e2e118_N{N_train}_S{S_num}_T{T_horizon}",
        help="SMPC dataset containing current, pooled, and full future loads",
    )
    parser.add_argument(
        "--decs", type=Path,
        default=Path(__file__).resolve().parent / "decs_pgm_fixedpv.pt",
        help="trained DECS checkpoint kept frozen during generator training",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent / str(spec["checkpoint"]),
        help="output checkpoint for the best stage-two validation model",
    )
    parser.add_argument(
        "--stage1-output", type=Path,
        default=Path(__file__).resolve().parent / str(spec["stage1_checkpoint"]),
        help="separate checkpoint for the best stage-one model",
    )
    parser.add_argument("--epochs", type=int, default=700, help="maximum epochs across both training stages")
    parser.add_argument(
        "--stage1-epochs", type=int, default=80,
        help=(
            "feasibility/diversity epochs before economic shaping; the IEEE-118 "
            "formal default is longer than the 20-epoch pilot"
            if diversity_enabled else
            "feasibility-only epochs before economic shaping; the IEEE-118 "
            "formal default is longer than the 20-epoch pilot"
        ),
    )
    parser.add_argument(
        "--batch-size", type=int, default=12,
        help=(
            "IEEE-118 SMPC instances per gradient update; one instance already "
            "expands to K*S*T=16000 AC operating points at paper defaults"
        ),
    )
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=1, metavar="STEPS",
        help=(
            "micro-batches averaged before one Adam update; raises the effective "
            "IEEE-118 batch size without storing several 16000-point graphs"
        ),
    )
    parser.add_argument(
        "--candidates", type=int, default=50,
        help="trajectory-level latent candidates per instance; at least two",
    )
    parser.add_argument(
        "--hidden-channels", type=int, default=64,
        help="input-embedding and temporal-block feature width",
    )
    parser.add_argument(
        "--latent-dim", type=int, default=16,
        help="dimension of one Gaussian latent vector per trajectory",
    )
    parser.add_argument(
        "--latent-embedding-dim", type=int, default=32,
        help="latent feature width repeated over the horizon",
    )
    parser.add_argument(
        "--kernel-size", type=int, default=3,
        help="positive odd non-causal convolution width in time steps",
    )
    parser.add_argument(
        "--dilations", type=int, nargs="+", default=(1, 2, 4, 8), metavar="D",
        help="positive dilations whose receptive field must cover the horizon",
    )
    parser.add_argument(
        "--ramp-fraction", type=float, default=0.25,
        help="symmetric one-period ramp limit as a fraction of generator Pmax",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=2.5e-4,
        help="Adam learning rate for the 80-epoch Stage 1",
    )
    parser.add_argument(
        "--stage2-learning-rate", type=float, default=5e-5,
        help="conservative Adam learning rate for Stage-2 economic shaping",
    )
    parser.add_argument(
        "--lr-scheduler", choices=("none", "plateau"), default="plateau",
        help=(
            "adaptive learning-rate schedule; plateau monitors validation total "
            "loss and is paused during the Stage-2 economic warmup"
        ),
    )
    parser.add_argument(
        "--lr-decay-factor", type=float, default=0.5,
        help="multiplicative learning-rate reduction used in both stages",
    )
    parser.add_argument(
        "--stage1-lr-decay-patience", type=int, default=4,
        help="Stage-1 validation plateau epochs before reducing its learning rate",
    )
    parser.add_argument(
        "--stage2-lr-decay-patience", type=int, default=25,
        help="post-warmup Stage-2 plateau epochs before reducing its learning rate",
    )
    parser.add_argument(
        "--lr-min-delta", type=float, default=1e-5,
        help="absolute validation-loss improvement required by both LR schedulers",
    )
    parser.add_argument(
        "--stage1-min-learning-rate", type=float, default=2.5e-5,
        help="positive lower learning-rate bound for Stage 1",
    )
    parser.add_argument(
        "--stage2-min-learning-rate", type=float, default=1e-5,
        help="positive lower learning-rate bound for Stage 2",
    )
    parser.add_argument(
        "--lambda-fea", type=float, default=2e-2,
        help=(
            "weight of equation (28)'s summed hierarchical feasibility loss; "
            "with K=50, the default gives unit weight to the mean candidate "
            "violation before diversity and economic shaping"
        ),
    )
    parser.add_argument(
        "--lambda-eco", type=float, default=3e-3,
        help=(
            "target weight of the normalized stage-two economic loss; the "
            "automatically frozen IEEE-118 cost scale keeps this dimensionless; "
            "the conservative default limits feasibility regression"
        ),
    )
    if diversity_enabled:
        parser.add_argument(
            "--lambda-div", type=float, default=1e-1,
            help="weight of feasibility-aware whole-trajectory diversity",
        )
    else:
        parser.set_defaults(lambda_div=0.0)

    if diversity_enabled:
        parser.add_argument(
            "--sigma-div", type=float, default=0.1,
            help=(
                "Gaussian width applied to the dimension-mean squared distance "
                "in normalized free-variable space"
            ),
        )
    else:
        parser.set_defaults(sigma_div=0.5)
    parser.add_argument(
        "--economic-warmup-epochs", type=int, default=60,
        help="Stage-2 epochs used to increase lambda_eco linearly from zero",
    )
    parser.add_argument(
        "--mean-cvar-alpha", type=float, default=0.05,
        help="alpha_c in (0,1): mean weight; smaller values emphasize tail violations",
    )
    parser.add_argument(
        "--cvar-tail-fraction", type=float, default=0.05,
        help="rho_c in (0,1]: largest residual fraction averaged by empirical CVaR",
    )

    parser.add_argument(
        "--economic-cost-scale", type=float, default=0.0, metavar="COST",
        help=(
            "positive horizon-cost divisor in monetary units; zero freezes the "
            "final Stage-1 validation mean cost as the scale"
        ),
    )

    parser.add_argument(
        "--tau-feas", type=float, default=2.0,
        help=(
            "soft-feasibility temperature calibrated to avoid saturation near the "
            "reactive-power boundary"
        ),
    )

    parser.add_argument(
        "--gamma-eco", type=float, default=5.0,
        help="nonnegative infeasible-cost amplification gamma_eco in equation (32)",
    )
    parser.add_argument(
        "--alpha-eco", type=float, default=0.05,
        help=(
            "alpha_eco in (0,1): weight of the all-candidate economic mean; "
            "1-alpha_eco weights the best-K_b candidate mean in equation (32)"
        ),
    )
    parser.add_argument(
        "--economic-top-k", type=int, default=1, metavar="K_B",
        help=(
            "K_b in equation (32): candidates with the lowest feasibility-adjusted "
            "cost retained in the best-subset economic term; must not exceed candidates"
        ),
    )
    if diversity_enabled:
        parser.add_argument(
            "--loss-epsilon", type=float, default=1e-8,
            help="positive denominator epsilon in equation (31)'s diversity loss",
        )
    else:
        parser.set_defaults(loss_epsilon=1e-8)
    parser.add_argument(
        "--feasibility-tolerance", type=float, default=1e-4,
        help="reporting-only threshold for candidate feasibility and hit-rate metrics",
    )
    parser.add_argument(
        "--patience", type=int, default=50,
        help=(
            "post-warmup Stage-2 epochs without a lower full-hit validation "
            "best-feasible cost before early stopping"
        ),
    )
    parser.add_argument(
        "--min-delta", type=float, default=0.0,
        help="minimum Stage-1 validation-loss decrease counted as improvement",
    )
    parser.add_argument(
        "--economic-min-delta", type=float, default=100.0, metavar="COST",
        help=(
            "minimum Stage-2 decrease in validation mean best-feasible horizon "
            "cost; applied only when every validation instance has a feasible candidate"
        ),
    )
    parser.add_argument(
        "--max-grad-norm", type=float, default=1.0,
        help="global generator-gradient clipping norm; zero disables clipping",
    )
    parser.add_argument(
        "--detect-anomaly", action="store_true",
        help="enable autograd anomaly tracing and expensive residual checks for debugging",
    )
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="DataLoader workers; zero is reliable on Windows",
    )
    parser.add_argument(
        "--seed", type=int, default=2026,
        help="seed for initialization, shuffling, and latent sampling; split is loaded from data",
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto",
        help="training device; auto selects CUDA when available",
    )
    parser.add_argument(
        "--resume-stage1", type=Path, default=None, metavar="CHECKPOINT",
        help=(
            "optional Stage-1 checkpoint to load and continue directly from "
            "epoch stage1_epochs+1 without rerunning Stage 1"
        ),
    )
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    """Reject invalid command-line hyperparameters before loading large arrays.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed values declared by :func:`build_parser`.
    """
    positive_ints = (
        args.epochs, args.batch_size, args.hidden_channels, args.latent_dim,
        args.latent_embedding_dim, args.patience, args.economic_warmup_epochs,
        args.gradient_accumulation_steps, args.stage1_lr_decay_patience,
        args.stage2_lr_decay_patience,
    )
    if min(positive_ints) < 1:
        raise ValueError("epochs, dimensions, batch size, and patience must be positive")
    if args.candidates < 2:
        raise ValueError("candidates must be at least two")
    if not 1 <= args.economic_top_k <= args.candidates:
        raise ValueError("economic_top_k must lie in [1,candidates]")
    if not 1 <= args.stage1_epochs < args.epochs:
        raise ValueError("stage1_epochs must lie in [1, epochs-1] for two-stage training")
    if args.kernel_size < 1 or args.kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    if not args.dilations or min(args.dilations) < 1:
        raise ValueError("dilations must be positive")
    if not 0 < args.ramp_fraction <= 1:
        raise ValueError("ramp_fraction must lie in (0, 1]")
    positive_floats = (
        args.learning_rate, args.stage2_learning_rate,
        args.stage1_min_learning_rate, args.stage2_min_learning_rate,
        args.tau_feas, args.sigma_div,
    )
    if min(positive_floats) <= 0:
        raise ValueError("learning rate, feasibility temperature, and diversity width must be positive")
    if (
        args.stage1_min_learning_rate > args.learning_rate
        or args.stage2_min_learning_rate > args.stage2_learning_rate
    ):
        raise ValueError("minimum learning rate must not exceed its stage initial rate")
    if not 0 < args.lr_decay_factor < 1:
        raise ValueError("lr_decay_factor must lie in (0,1)")
    if not 0 < args.mean_cvar_alpha < 1:
        raise ValueError("mean_cvar_alpha must lie in (0,1)")
    if not 0 < args.alpha_eco < 1:
        raise ValueError("alpha_eco must lie in (0,1)")
    if not 0 < args.cvar_tail_fraction <= 1:
        raise ValueError("cvar_tail_fraction must lie in (0,1]")
    if not 0 < args.loss_epsilon < 1:
        raise ValueError("loss_epsilon must lie in (0,1)")
    if min(
        args.lambda_fea, args.lambda_div, args.lambda_eco,
        args.gamma_eco, args.min_delta, args.economic_min_delta,
        args.lr_min_delta,
    ) < 0:
        raise ValueError(
            "loss weights, gamma_eco, min_delta, and economic_min_delta "
            "must be nonnegative"
        )
    if not args.use_diversity and args.lambda_div != 0.0:
        raise ValueError("WD-CSNG requires lambda_div=0 because diversity loss is disabled")
    if min(
        args.feasibility_tolerance, args.max_grad_norm, args.num_workers,
        args.economic_cost_scale,
    ) < 0:
        raise ValueError(
            "tolerance, gradient norm, cost scale, and num_workers must be nonnegative"
        )
    if not args.data.is_dir() or not args.decs.is_file():
        raise FileNotFoundError("SMPC dataset or DECS checkpoint does not exist")
    if args.output.resolve() == args.stage1_output.resolve():
        raise ValueError("output and stage1-output must be different files")
    if args.resume_stage1 is not None and not args.resume_stage1.is_file():
        raise FileNotFoundError(f"Stage-1 checkpoint does not exist: {args.resume_stage1}")
    if (
        args.resume_stage1 is not None
        and args.output.resolve() == args.resume_stage1.resolve()
    ):
        raise ValueError("output must not overwrite the Stage-1 resume checkpoint")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")


def main(benchmark: str = "csng") -> None:
    """Train one benchmark and select its checkpoint on validation.

    Parameters
    ----------
    benchmark : str, default="csng"
        Experiment key selecting CSNG, sigmoid S-CSNG, or no-diversity WD-CSNG.
    """
    spec = get_benchmark_spec(benchmark)
    args = build_parser(benchmark).parse_args()
    args.benchmark = benchmark
    args.use_diversity = bool(spec["diversity_loss_enabled"])
    validate_arguments(args)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )

    metadata = json.loads((args.data / "metadata.json").read_text(encoding="utf-8"))
    n_instances = int(np.load(args.data / "current_load.npy", mmap_mode="r").shape[0])
    split_dir = args.data / "split"
    split_indices = {
        name: np.load(split_dir / f"{name}_indices.npy")
        for name in ("train", "validation", "test")
    }
    combined_indices = np.concatenate(list(split_indices.values()))
    if (
        len(combined_indices) != n_instances
        or not np.array_equal(np.sort(combined_indices), np.arange(n_instances))
    ):
        raise ValueError("saved train/validation/test indices are not a partition")
    metadata_counts = metadata.get("split", {}).get("counts", {})
    if any(
        int(metadata_counts.get(name, -1)) != len(values)
        for name, values in split_indices.items()
    ):
        raise ValueError("saved split counts differ from metadata.json")
    feature_min, feature_max = compute_training_feature_bounds(
        args.data, split_indices["train"],
    )
    datasets = {
        split: GeneratorDataset(args.data, indices, feature_min, feature_max)
        for split, indices in split_indices.items()
        if split in ("train", "validation")
    }
    loaders = {
        split: DataLoader(
            dataset, batch_size=args.batch_size, shuffle=(split == "train"),
            num_workers=args.num_workers,
            generator=torch.Generator().manual_seed(args.seed),
        )
        for split, dataset in datasets.items()
    }

    completion = load_decs_checkpoint(args.decs, device)
    decs_metadata = torch.load(
        args.decs, map_location="cpu", weights_only=False,
    ).get("data_metadata", {})
    if decs_metadata.get("case_name") != metadata.get("case_name"):
        raise ValueError("DECS and generator data use different network cases")
    if not (
        decs_metadata.get("bus_type_model") == "fixed_PV_PQ"
        and decs_metadata.get("enforce_q_limits") is False
        and decs_metadata.get("pv_to_pq_switching") is False
        and int(decs_metadata.get("chi_dim", -1)) == completion.physics.chi_dim
    ):
        raise ValueError(
            "Generator training requires the IEEE-118 fixed-PV DECS without "
            "PV-to-PQ switching"
        )
    if decs_metadata.get("source_base_usage") != "all_time_load_range_only":
        raise ValueError(
            "DECS must use only the base dataset's configured all-time load "
            "range, not empirical current/future samples"
        )
    source_metadata = decs_metadata.get("source_base_metadata", {})
    # DECS uses the source dataset only to define the absolute load-scale domain.
    # Its sample count, scenario count, and horizon need not equal the downstream
    # generator dataset, especially for small reproducible tuning experiments.
    for field in ("dataset_type", "case_name"):
        if source_metadata.get(field) != metadata.get(field):
            raise ValueError(f"DECS and Generator data differ in metadata field '{field}'")
    for field in ("load_scale_min", "load_scale_max"):
        if not np.isclose(
            float(source_metadata.get(field, np.nan)),
            float(metadata.get(field, np.nan)),
        ):
            raise ValueError(f"DECS and Generator data differ in metadata field '{field}'")
    decs_sha256 = file_sha256(args.decs)
    for parameter in completion.parameters():
        parameter.requires_grad_(False)

    physics = completion.physics
    free_lower, free_upper = physics.free_variable_bounds(1)
    free_lower, free_upper = free_lower[0].to(device), free_upper[0].to(device)
    free_ramp = args.ramp_fraction * physics.pg_max[physics.free_pg]
    reference_ramp = (
        args.ramp_fraction * physics.pg_max[physics.reference_generator]
    )
    active_ramp = args.ramp_fraction * physics.pg_max[physics.active]
    model = build_generator_model(
        benchmark,
        condition_dim=datasets["train"].condition_dim,
        free_dim=physics.free_dim,
        n_free_pg=physics.n_free_pg,
        projection_method=str(spec["projection_method"]),
        hidden_channels=args.hidden_channels,
        latent_dim=args.latent_dim,
        latent_embedding_dim=args.latent_embedding_dim,
        kernel_size=args.kernel_size,
        dilations=tuple(args.dilations),
    ).to(device)
    if model.receptive_field < datasets["train"].horizon:
        raise ValueError(
            f"TCN receptive field {model.receptive_field} is shorter than "
            f"horizon {datasets['train'].horizon}"
        )

    resume_checkpoint = None
    resumed_stage1_history: list[dict] = []
    if args.resume_stage1 is not None:
        resume_checkpoint = torch.load(
            args.resume_stage1, map_location="cpu", weights_only=False,
        )
        if resume_checkpoint.get("training_phase") != spec["stage1_training_phase"]:
            raise ValueError("resume-stage1 must contain a completed Stage-1 checkpoint")
        checkpoint_benchmark = resume_checkpoint.get("benchmark", "csng")
        if checkpoint_benchmark != benchmark:
            raise ValueError(
                f"resume-stage1 benchmark is {checkpoint_benchmark!r}, expected {benchmark!r}"
            )
        if (
            resume_checkpoint.get("model_config", {}).get("projection_method")
            != spec["projection_method"]
        ):
            raise ValueError(
                "resume-stage1 projection does not match the selected benchmark: "
                f"expected {spec['projection_method']!r}"
            )
        if resume_checkpoint.get("model_config") != model.configuration():
            raise ValueError("Stage-1 checkpoint model configuration does not match CLI")
        if resume_checkpoint.get("decs_checkpoint_sha256") != decs_sha256:
            raise ValueError("Stage-1 checkpoint was trained with a different DECS model")
        checkpoint_norm = resume_checkpoint.get("normalization", {})
        if not (
            np.allclose(checkpoint_norm.get("feature_min"), feature_min)
            and np.allclose(checkpoint_norm.get("feature_max"), feature_max)
        ):
            raise ValueError("Stage-1 checkpoint was trained with different data bounds")
        resumed_stage1_history = list(resume_checkpoint.get("training_history", []))
        if (
            not resumed_stage1_history
            or int(resumed_stage1_history[-1]["epoch"]) != args.stage1_epochs
        ):
            raise ValueError("Stage-1 checkpoint history does not end at stage1_epochs")
        model.load_state_dict(resume_checkpoint["model_state"])
        args.economic_cost_scale = resolve_economic_cost_scale(
            args.economic_cost_scale, resumed_stage1_history,
        )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.stage2_learning_rate,
        )
        scheduler = (
            build_plateau_scheduler(
                optimizer,
                args.lr_decay_factor,
                args.stage2_lr_decay_patience,
                args.lr_min_delta,
                args.stage2_min_learning_rate,
            )
            if args.lr_scheduler == "plateau" else None
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = (
            build_plateau_scheduler(
                optimizer,
                args.lr_decay_factor,
                args.stage1_lr_decay_patience,
                args.lr_min_delta,
                args.stage1_min_learning_rate,
            )
            if args.lr_scheduler == "plateau" else None
        )

    print(f"{spec['method']} non-causal TCN training configuration")
    print(f"  device: {device}")
    print(
        f"  projection: {model.projection_method}, "
        f"output initialization: weight~N(0,1e-2), bias={spec['output_bias']}"
    )
    print(
        f"  instances: train={len(datasets['train'])}, "
        f"validation={len(datasets['validation'])}, "
        f"test={len(split_indices['test'])} (reserved; not loaded or evaluated)"
    )
    print(
        f"  condition=(B,{datasets['train'].horizon},{datasets['train'].condition_dim}), "
        f"loads=(B,{datasets['train'].n_scenarios},{datasets['train'].horizon},"
        f"{datasets['train'].current.shape[1]},2)"
    )
    print(
        f"  NCTCN: channels={args.hidden_channels}, latent={args.latent_dim}, "
        f"kernel={args.kernel_size}, dilations={list(args.dilations)}, "
        f"receptive_field={model.receptive_field}"
    )
    print(
        f"  candidates={args.candidates}, ramp={100 * args.ramp_fraction:.1f}% Pmax/period, "
        f"free ramp={free_ramp.detach().cpu().tolist()} MW, "
        f"reference ramp={float(reference_ramp):.3f} MW"
    )
    print(
        f"  optimization batch: micro={args.batch_size}, "
        f"accumulation={args.gradient_accumulation_steps}, "
        f"effective={args.batch_size * args.gradient_accumulation_steps} instances/update"
    )
    print(
        f"  feasibility shaping: alpha_c={args.mean_cvar_alpha:.2f}, "
        f"rho_c={args.cvar_tail_fraction:.2f}, tau_f={args.tau_feas:.3g}"
    )
    print(
        "  diversity shaping: "
        + (
            f"enabled, lambda_div={args.lambda_div:.3e}, sigma_d={args.sigma_div:.3g}"
            if args.use_diversity else
            "disabled (WD-CSNG ablation)"
        )
    )
    print(
        f"  economic shaping: alpha_eco={args.alpha_eco:.2f}, "
        f"K_b={args.economic_top_k}, gamma_eco={args.gamma_eco:.3g}"
    )
    print(
        f"  stage 1: epochs=1..{args.stage1_epochs}, lr={args.learning_rate:.3e}, "
        f"checkpoint={args.stage1_output}"
    )
    print(
        f"  stage 2: epochs={args.stage1_epochs + 1}..{args.epochs}, "
        f"lr={args.stage2_learning_rate:.3e}, lambda_eco(target)={args.lambda_eco:.3e}, "
        f"warmup={args.economic_warmup_epochs}, "
        f"cost_scale={'auto' if args.economic_cost_scale == 0 else f'{args.economic_cost_scale:.3f}'}, "
        f"grad_clip={'off' if args.max_grad_norm == 0 else args.max_grad_norm}"
    )
    print(
        f"  LR schedule: {args.lr_scheduler}, factor={args.lr_decay_factor:g}, "
        f"patience(stage1/stage2)={args.stage1_lr_decay_patience}/"
        f"{args.stage2_lr_decay_patience}, min_lr(stage1/stage2)="
        f"{args.stage1_min_learning_rate:.1e}/{args.stage2_min_learning_rate:.1e}"
    )
    print(
        "  stage 2 selection: validation best_feasible cost with hit_rate=1.000, "
        f"economic_min_delta={args.economic_min_delta:g}, realtime={args.output}"
    )
    if args.resume_stage1 is not None:
        print(f"  resume: loaded completed Stage 1 from {args.resume_stage1}")

    loss_hyperparameters = {
        name: getattr(args, name)
        for name in (
            "lambda_fea", "lambda_eco", "mean_cvar_alpha",
            "cvar_tail_fraction", "tau_feas", "gamma_eco",
            "alpha_eco", "economic_top_k", "feasibility_tolerance",
            "economic_cost_scale",
        )
    }
    if args.use_diversity:
        loss_hyperparameters.update({
            "lambda_div": args.lambda_div,
            "sigma_div": args.sigma_div,
            "loss_epsilon": args.loss_epsilon,
        })

    checkpoint_common = {
        "benchmark": benchmark,
        "method": spec["method"],
        "model_class": spec["model_class"],
        "model_config": model.configuration(),
        "projection_method": model.projection_method,
        "diversity_loss_enabled": args.use_diversity,
        "output_initialization": {
            "weight": "normal(mean=0,std=1e-2)",
            "bias": spec["output_bias"],
        },
        "normalization": {
            "feature_min": datasets["train"].feature_min.numpy(),
            "feature_max": datasets["train"].feature_max.numpy(),
        },
        "free_variable_lower": free_lower.detach().cpu(),
        "free_variable_upper": free_upper.detach().cpu(),
        "ramp_fraction_of_pmax": args.ramp_fraction,
        "free_ramp_mw_per_period": free_ramp.detach().cpu(),
        "reference_ramp_mw_per_period": reference_ramp.detach().cpu(),
        "active_ramp_mw_per_period": active_ramp.detach().cpu(),
        "constraint_scaling": {
            "two_sided_limits": "inverse physical operating range",
            "thermal": "inverse branch MVA rating",
            "thermal_directions_per_branch": 2,
            "per_scenario_time_sizes": physics.remaining_constraint_sizes(),
            "dg_diagonal_per_scenario_time": {
                name: value.detach().cpu()
                for name, value in physics.remaining_constraint_scales().items()
            },
            "reference_ramp_inverse_mw": reference_ramp.reciprocal().detach().cpu(),
            "m_g_per_candidate": (
                datasets["train"].n_scenarios
                * datasets["train"].horizon
                * sum(physics.remaining_constraint_sizes().values())
                + datasets["train"].n_scenarios
                * 2
                * (datasets["train"].horizon - 1)
            ),
            "hierarchical_aggregation": {
                "families": [*physics.remaining_constraint_sizes(), "ramp"],
                "alpha_c": {
                    name: args.mean_cvar_alpha
                    for name in [*physics.remaining_constraint_sizes(), "ramp"]
                },
                "rho_c": {
                    name: args.cvar_tail_fraction
                    for name in [*physics.remaining_constraint_sizes(), "ramp"]
                },
                "cvar_tail_count": {
                    **{
                        name: max(1, int(np.ceil(args.cvar_tail_fraction * size)))
                        for name, size in physics.remaining_constraint_sizes().items()
                    },
                    "ramp": max(1, int(np.ceil(2 * args.cvar_tail_fraction))),
                },
                "reduction": "sum candidates, scenarios, time, and families; mean batch",
            },
        },
        "objective_definition": (
            "sum all-generator cost over time per scenario, including "
            "reconstructed slack Pg, then average over scenarios"
        ),
        "economic_objective_normalization": (
            "divide each raw horizon cost by one frozen Stage-1 validation "
            "mean cost before equation (32)"
        ),
        "decs_checkpoint": str(args.decs.resolve()),
        "decs_checkpoint_sha256": decs_sha256,
        "loss_definition": spec["loss_definition"],
        "loss_hyperparameters": loss_hyperparameters,
        "training_hyperparameters": {
            "stage1_epochs": args.stage1_epochs,
            "stage1_learning_rate": args.learning_rate,
            "stage2_learning_rate": args.stage2_learning_rate,
            "economic_warmup_epochs": args.economic_warmup_epochs,
            "optimizer_reset_at_stage2": True,
            "stage2_checkpoint_selection": (
                "minimum validation mean best-feasible cost with 100% instance hit-rate"
            ),
            "stage1_min_delta": args.min_delta,
            "stage2_economic_min_delta": args.economic_min_delta,
            "max_grad_norm": args.max_grad_norm,
            "detect_anomaly": args.detect_anomaly,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
            "candidates": args.candidates,
            "lr_scheduler": args.lr_scheduler,
            "lr_decay_factor": args.lr_decay_factor,
            "stage1_lr_decay_patience": args.stage1_lr_decay_patience,
            "stage2_lr_decay_patience": args.stage2_lr_decay_patience,
            "lr_min_delta": args.lr_min_delta,
            "stage1_min_learning_rate": args.stage1_min_learning_rate,
            "stage2_min_learning_rate": args.stage2_min_learning_rate,
        },
        "split": {
            "ratio": metadata["split"]["ratio"],
            "seed": metadata["split"]["seed"],
            "counts": {name: len(values) for name, values in split_indices.items()},
            "indices": {
                name: values.astype(np.int64).tolist()
                for name, values in split_indices.items()
            },
        },
        "data_metadata": metadata,
        "seed": args.seed,
    }

    best_loss = np.inf
    best_economic_cost = np.inf
    best_epoch = 0
    best_state = None
    best_validation_hit_rate = 0.0
    best_validation_candidate_feasible = 0.0
    stale_epochs = 0
    stage1_best_loss = np.inf
    stage1_best_epoch = 0
    stage1_best_state = None
    history: list[dict[str, float | int | bool]] = list(resumed_stage1_history)
    stage1_checkpoint_path = args.stage1_output
    start_epoch = 1
    if resume_checkpoint is not None:
        stage1_best_loss = float(resume_checkpoint["best_validation_loss"])
        stage1_best_epoch = int(resume_checkpoint["early_stopping"]["best_epoch"])
        stage1_best_state = {
            name: value.detach().cpu().clone()
            for name, value in resume_checkpoint["model_state"].items()
        }
        stage1_checkpoint_path = args.resume_stage1
        start_epoch = args.stage1_epochs + 1

    def save_best_stage2_checkpoint(
        training_complete: bool,
        stopped_epoch: int,
        stopped_early_value: bool,
    ) -> None:
        """Persist the current economic-best Stage-2 state atomically.

        Parameters
        ----------
        training_complete : bool
            Whether the two-stage loop has ended normally or by early stopping.
        stopped_epoch : int
            Latest fully completed epoch represented in ``history``.
        stopped_early_value : bool
            Whether patience, rather than the maximum epoch, ended training.
        """
        if best_state is None or not np.isfinite(best_economic_cost):
            raise RuntimeError("no full-hit Stage-2 economic checkpoint is available")
        atomic_torch_save({
            **checkpoint_common,
            "model_state": best_state,
            "training_phase": "stage2_economic",
            "training_complete": training_complete,
            "stage1_checkpoint": str(stage1_checkpoint_path.resolve()),
            "stage1_best_epoch": stage1_best_epoch,
            "stage1_best_validation_loss": stage1_best_loss,
            "checkpoint_selection": {
                "metric": "validation_mean_best_feasible_cost",
                "requires_full_instance_hit_rate": True,
                "economic_min_delta": args.economic_min_delta,
            },
            "early_stopping": {
                "patience": args.patience,
                "economic_min_delta": args.economic_min_delta,
                "best_epoch": best_epoch,
                "stopped_epoch": stopped_epoch,
                "stopped_early": stopped_early_value,
            },
            "best_validation_feasible_cost": best_economic_cost,
            "best_validation_loss": best_loss,
            "best_validation_hit_rate": best_validation_hit_rate,
            "best_validation_candidate_feasible": best_validation_candidate_feasible,
            "test_evaluated_during_training": False,
            "training_history": list(history),
            "lr_scheduler_state": None if scheduler is None else scheduler.state_dict(),
        }, args.output)

    start_time = perf_counter()
    stopped_early = False
    for epoch in range(start_epoch, args.epochs + 1):
        use_economic = epoch > args.stage1_epochs
        if epoch == args.stage1_epochs + 1 and resume_checkpoint is None:
            if best_state is None:
                raise RuntimeError("stage one did not produce a finite validation checkpoint")
            stage1_best_loss = best_loss
            stage1_best_epoch = best_epoch
            stage1_best_state = best_state
            args.economic_cost_scale = resolve_economic_cost_scale(
                args.economic_cost_scale, history,
            )
            checkpoint_common["loss_hyperparameters"]["economic_cost_scale"] = (
                args.economic_cost_scale
            )
            atomic_torch_save({
                **checkpoint_common,
                "model_state": stage1_best_state,
                "training_phase": spec["stage1_training_phase"],
                "best_validation_loss": stage1_best_loss,
                "training_history": list(history),
                "lr_scheduler_state": (
                    None if scheduler is None else scheduler.state_dict()
                ),
                "early_stopping": {
                    "patience": None, "min_delta": args.min_delta,
                    "best_epoch": stage1_best_epoch,
                    "stopped_epoch": args.stage1_epochs,
                    "stopped_early": False,
                },
            }, args.stage1_output)
            print(
                f"saved stage-one best checkpoint from epoch {stage1_best_epoch} "
                f"to {args.stage1_output}",
                flush=True,
            )

            # The objective changes scale at the stage boundary, so Stage 2
            # restarts from the best Stage-1 weights with fresh Adam moments
            # and its lower dedicated step size.
            model.load_state_dict(stage1_best_state)
            optimizer = torch.optim.Adam(
                model.parameters(), lr=args.stage2_learning_rate,
            )
            scheduler = (
                build_plateau_scheduler(
                    optimizer,
                    args.lr_decay_factor,
                    args.stage2_lr_decay_patience,
                    args.lr_min_delta,
                    args.stage2_min_learning_rate,
                )
                if args.lr_scheduler == "plateau" else None
            )
            best_loss, best_economic_cost = np.inf, np.inf
            best_state, best_epoch, stale_epochs = None, 0, 0
            best_validation_hit_rate = 0.0
            best_validation_candidate_feasible = 0.0

        train_economic_weight = economic_warmup_weight(
            args.lambda_eco, epoch, args.stage1_epochs,
            args.economic_warmup_epochs,
        )
        # Use the final target weight for validation throughout Stage 2 so
        # checkpoint losses remain comparable while the training weight warms up.
        validation_economic_weight = args.lambda_eco if use_economic else 0.0

        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_metrics = run_epoch(
            model, completion, loaders["train"], free_lower, free_upper,
            free_ramp, reference_ramp, args, device, optimizer,
            train_economic_weight,
            latent_seed=args.seed + 11,
        )
        validation_metrics = run_epoch(
            model, completion, loaders["validation"], free_lower, free_upper,
            free_ramp, reference_ramp, args, device, optimizer=None,
            economic_weight=validation_economic_weight,
            latent_seed=args.seed + 11,
        )
        if not np.isfinite(validation_metrics["loss"]):
            raise FloatingPointError(f"non-finite validation loss at epoch {epoch}")
        validation_instances = len(datasets["validation"])
        full_validation_hit = (
            int(validation_metrics["instances_with_feasible_cost"])
            == validation_instances
        )
        if use_economic:
            improved = improves_validation_economic_checkpoint(
                validation_metrics,
                validation_instances,
                best_economic_cost,
                args.economic_min_delta,
            )
        else:
            improved = validation_metrics["loss"] < best_loss - args.min_delta
        warmup_complete = (
            not use_economic
            or epoch >= args.stage1_epochs + args.economic_warmup_epochs
        )
        if improved:
            best_loss = validation_metrics["loss"]
            if use_economic:
                best_economic_cost = validation_metrics["best_feasible_cost"]
                best_validation_hit_rate = validation_metrics["hit"]
                best_validation_candidate_feasible = validation_metrics["feasible"]
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale_epochs = 0
        elif warmup_complete:
            stale_epochs += 1
        else:
            stale_epochs = 0

        scheduler_active = scheduler is not None and warmup_complete
        if scheduler_active:
            scheduler.step(validation_metrics["loss"])
        next_learning_rate = float(optimizer.param_groups[0]["lr"])

        history.append({
            "epoch": epoch, "economic_stage": use_economic,
            "learning_rate": learning_rate,
            "next_learning_rate": next_learning_rate,
            "lr_scheduler_active": scheduler_active,
            "train_lambda_eco_effective": train_economic_weight,
            "validation_lambda_eco": validation_economic_weight,
            "economic_cost_scale": args.economic_cost_scale,
            **{f"train_{name}": value for name, value in train_metrics.items()},
            **{f"validation_{name}": value for name, value in validation_metrics.items()},
        })
        elapsed = perf_counter() - start_time
        completed_this_run = epoch - start_epoch + 1
        eta = elapsed / completed_this_run * (args.epochs - epoch)
        stage = 2 if use_economic else 1
        if improved:
            status = "new economic best" if use_economic else "new Stage-1 best"
        elif not warmup_complete:
            status = "economic warmup"
        elif use_economic and not full_validation_hit:
            status = f"no full-hit validation | stale {stale_epochs}/{args.patience}"
        else:
            status = f"stale {stale_epochs}/{args.patience}"
        if next_learning_rate < learning_rate:
            status += f" | lr -> {next_learning_rate:.2e}"
        print_epoch_metrics(
            epoch=epoch,
            total_epochs=args.epochs,
            stage=stage,
            elapsed=elapsed,
            eta=eta,
            status=status,
            learning_rate=learning_rate,
            feasibility_weight=args.lambda_fea,
            diversity_weight=args.lambda_div,
            train_economic_weight=train_economic_weight,
            validation_economic_weight=validation_economic_weight,
            train_metrics=train_metrics,
            validation_metrics=validation_metrics,
        )

        if use_economic and improved:
            save_best_stage2_checkpoint(
                training_complete=False,
                stopped_epoch=epoch,
                stopped_early_value=False,
            )
            print(
                f"  realtime checkpoint: validation best_feasible="
                f"{best_economic_cost:.3f} at epoch {best_epoch} -> {args.output}",
                flush=True,
            )

        if use_economic and warmup_complete and stale_epochs >= args.patience:
            stopped_early = True
            print(
                f"early stopping at epoch {epoch}; best epoch {best_epoch}, "
                f"validation best_feasible {best_economic_cost:.3f}"
            )
            break

    if best_state is None:
        raise RuntimeError(
            "Stage 2 did not produce a checkpoint with a feasible candidate for "
            "every validation instance"
        )
    model.load_state_dict(best_state)
    print(
        f"best checkpoint epoch {best_epoch} | validation best_feasible "
        f"{best_economic_cost:.3f} | validation loss {best_loss:.6e} | "
        "test set not evaluated"
    )

    save_best_stage2_checkpoint(
        training_complete=True,
        stopped_epoch=int(history[-1]["epoch"]),
        stopped_early_value=stopped_early,
    )
    print(f"saved non-causal TCN generator checkpoint to {args.output}")


if __name__ == "__main__":
    main()
