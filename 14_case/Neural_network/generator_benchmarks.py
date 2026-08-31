"""Shared identifiers and model construction for CSNG benchmark variants."""

from torch import nn

from Neural_network.noncausal_tcn import (
    PROJECTION_METHOD,
    ConditionalStochasticTCN,
)
from Neural_network.noncausal_tcn_s_csng import (
    SIGMOID_PROJECTION_METHOD,
    SigmoidConditionalStochasticTCN,
)
from Neural_network.noncausal_tcn_wd_csng import WDConditionalStochasticTCN


BENCHMARK_SPECS = {
    "csng": {
        "method": "CSNG",
        "model_class": "ConditionalStochasticTCN",
        "projection_method": PROJECTION_METHOD,
        "diversity_loss_enabled": True,
        "stage1_training_phase": "stage1_feasibility_diversity",
        "checkpoint": "generator_tcn_clip.pt",
        "stage1_checkpoint": "generator_tcn_clip_stage1.pt",
        "pgm_output_prefix": "smpc_pgm",
        "output_bias": 0.5,
        "loss_definition": "hierarchical_mean_cvar_mean_best_k_v3",
    },
    "s_csng": {
        "method": "S-CSNG",
        "model_class": "SigmoidConditionalStochasticTCN",
        "projection_method": SIGMOID_PROJECTION_METHOD,
        "diversity_loss_enabled": True,
        "stage1_training_phase": "stage1_feasibility_diversity",
        "checkpoint": "generator_tcn_s_csng.pt",
        "stage1_checkpoint": "generator_tcn_s_csng_stage1.pt",
        "pgm_output_prefix": "s_csng_pgm",
        "output_bias": 0.0,
        "loss_definition": "hierarchical_mean_cvar_mean_best_k_sigmoid_v1",
    },
    "wd_csng": {
        "method": "WD-CSNG",
        "model_class": "WDConditionalStochasticTCN",
        "projection_method": PROJECTION_METHOD,
        "diversity_loss_enabled": False,
        "stage1_training_phase": "stage1_feasibility",
        "checkpoint": "generator_tcn_wd_csng.pt",
        "stage1_checkpoint": "generator_tcn_wd_csng_stage1.pt",
        "pgm_output_prefix": "wd_csng_pgm",
        "output_bias": 0.5,
        "loss_definition": "hierarchical_mean_cvar_mean_best_k_without_diversity_v1",
    },
}


def get_benchmark_spec(benchmark: str) -> dict[str, object]:
    """Return a copy of one benchmark's immutable experiment metadata.

    Parameters
    ----------
    benchmark : str
        One of ``csng``, ``s_csng``, or ``wd_csng``.

    Returns
    -------
    dict[str, object]
        Method identifiers, projection/loss choices, and default filenames.
    """
    if benchmark not in BENCHMARK_SPECS:
        raise ValueError(
            f"unknown benchmark {benchmark!r}; choose from {tuple(BENCHMARK_SPECS)}"
        )
    return dict(BENCHMARK_SPECS[benchmark])


def build_generator_model(benchmark: str, **model_config: object) -> nn.Module:
    """Construct the model class assigned to a benchmark.

    Parameters
    ----------
    benchmark : str
        Benchmark key accepted by :func:`get_benchmark_spec`.
    **model_config : object
        Constructor arguments stored in the generator checkpoint.

    Returns
    -------
    torch.nn.Module
        CSNG, sigmoid-output S-CSNG, or no-diversity WD-CSNG model.
    """
    model_classes = {
        "csng": ConditionalStochasticTCN,
        "s_csng": SigmoidConditionalStochasticTCN,
        "wd_csng": WDConditionalStochasticTCN,
    }
    get_benchmark_spec(benchmark)
    return model_classes[benchmark](**model_config)
