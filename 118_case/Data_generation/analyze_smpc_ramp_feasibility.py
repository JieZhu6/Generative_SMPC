"""Screen SMPC scenario parameters against the shared-trajectory ramp condition.

For each generated scenario tree, the script measures the cross-scenario span
of total active-load increments at every time transition. A shared dispatch
increment can place all scenario-dependent slack increments inside ``[-R,R]``
in the lossless balance approximation only when every span is at most ``2R``.
This is a fast necessary-condition diagnostic, not a complete AC feasibility
certificate.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Data_generation.case118_pglib import load_case118  # noqa: E402
from Data_generation.generate_smpc_scenarios import generate_bundle  # noqa: E402


def maximum_increment_span(bundle) -> float:
    """Return the largest cross-scenario total-P increment span in MW.

    Parameters
    ----------
    bundle : ScenarioBundle
        One current load and ``S`` future scenario trajectories.

    Returns
    -------
    float
        Maximum over time of ``max_s(delta P_s)-min_s(delta P_s)`` in MW.
    """
    current_total = float(bundle.current_pd.sum())
    future_total = bundle.future_pd.sum(axis=2)
    total_trajectory = np.concatenate([
        np.full((len(future_total), 1), current_total), future_total,
    ], axis=1)
    return float(np.ptp(np.diff(total_trajectory, axis=1), axis=0).max())


def evaluate_configuration(
    case,
    samples: int,
    scenarios: int,
    horizon: int,
    deviation: float,
    rho: float,
    load_scale_min: float,
    load_scale_max: float,
    seed: int,
    reference_ramp_mw: float,
) -> dict[str, float | int]:
    """Evaluate one scenario parameter pair over reproducible samples.

    Parameters
    ----------
    case
        IEEE-118 case data used by the production scenario generator.
    samples : int
        Number of independent scenario trees.
    scenarios : int
        Future scenarios per tree.
    horizon : int
        Number of 15-minute periods including the known current period.
    deviation : float
        Maximum multiplicative forecast deviation before absolute clipping.
    rho : float
        AR(1) temporal correlation coefficient.
    load_scale_min, load_scale_max : float
        Absolute load-factor clipping bounds.
    seed : int
        Base seed; sample ``i`` uses ``seed + 1009*i`` as in dataset generation.
    reference_ramp_mw : float
        Symmetric slack-generator ramp limit in MW per period.

    Returns
    -------
    dict
        Failure count/rate and span/required-ramp distribution statistics.
    """
    spans = np.empty(samples)
    for index in range(samples):
        bundle = generate_bundle(
            sample_seed=seed + 1009 * index,
            horizon=horizon,
            n_scenarios=scenarios,
            forecast_deviation=deviation,
            rho=rho,
            case=case,
            load_scale_min=load_scale_min,
            load_scale_max=load_scale_max,
        )
        spans[index] = maximum_increment_span(bundle)
    required_ramp = spans / 2.0
    failures = required_ramp > reference_ramp_mw
    return {
        "forecast_deviation": deviation,
        "rho": rho,
        "samples": samples,
        "failures": int(failures.sum()),
        "failure_rate": float(failures.mean()),
        "pass_rate": float((~failures).mean()),
        "span_mean_mw": float(spans.mean()),
        "span_p95_mw": float(np.quantile(spans, 0.95)),
        "span_p99_mw": float(np.quantile(spans, 0.99)),
        "span_max_mw": float(spans.max()),
        "required_ramp_max_mw": float(required_ramp.max()),
        "reference_ramp_mw": reference_ramp_mw,
    }


def main() -> None:
    """Run the parameter grid and save JSON/CSV summaries."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--samples", type=int, default=500,
                        help="scenario trees evaluated per parameter pair")
    parser.add_argument("--scenarios", type=int, default=20,
                        help="future scenarios per tree")
    parser.add_argument("--horizon", type=int, default=16,
                        help="15-minute periods including the current period")
    parser.add_argument(
        "--deviations", type=float, nargs="+",
        default=(0.03, 0.05, 0.08, 0.10, 0.12, 0.15),
        help="forecast-deviation values in [0,1) to screen",
    )
    parser.add_argument(
        "--rhos", type=float, nargs="+", default=(0.82, 0.90, 0.95, 0.98),
        help="AR(1) temporal correlations in [0,1) to screen",
    )
    parser.add_argument("--load-scale-min", type=float, default=0.75,
                        help="absolute minimum load multiplier")
    parser.add_argument("--load-scale-max", type=float, default=1.05,
                        help="absolute maximum load multiplier")
    parser.add_argument("--ramp-fraction", type=float, default=0.25,
                        help="reference-generator ramp as a fraction of Pmax")
    parser.add_argument("--seed", type=int, default=2026,
                        help="base random seed matching dataset generation")
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "output" / "smpc_scenario_ramp_grid.json",
        help="JSON output path; a CSV table is written beside it",
    )
    args = parser.parse_args()
    if min(args.samples, args.scenarios, args.horizon) < 1:
        raise ValueError("samples, scenarios, and horizon must be positive")
    if any(not 0 <= value < 1 for value in [*args.deviations, *args.rhos]):
        raise ValueError("deviations and rhos must lie in [0,1)")
    if not 0 < args.load_scale_min < args.load_scale_max:
        raise ValueError("load scale limits must be positive and ordered")
    if args.ramp_fraction <= 0:
        raise ValueError("ramp_fraction must be positive")

    case = load_case118()
    reference_ramp_mw = (
        args.ramp_fraction * case.gen[case.reference_generator, 8]
    )
    results = []
    for deviation in args.deviations:
        for rho in args.rhos:
            result = evaluate_configuration(
                case, args.samples, args.scenarios, args.horizon,
                deviation, rho, args.load_scale_min, args.load_scale_max,
                args.seed, reference_ramp_mw,
            )
            results.append(result)
            print(
                f"D={deviation:.3f}, rho={rho:.3f}: "
                f"fail={result['failures']}/{args.samples}, "
                f"max span={result['span_max_mw']:.1f} MW",
                flush=True,
            )

    report = {
        "diagnostic": "lossless shared-trajectory reference-ramp necessary condition",
        "not_full_ac_certificate": True,
        "scenarios": args.scenarios,
        "horizon": args.horizon,
        "load_scale_min": args.load_scale_min,
        "load_scale_max": args.load_scale_max,
        "ramp_fraction": args.ramp_fraction,
        "reference_ramp_mw": float(reference_ramp_mw),
        "allowed_cross_scenario_increment_span_mw": float(2 * reference_ramp_mw),
        "seed": args.seed,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"saved {args.output} and {csv_path}")


if __name__ == "__main__":
    main()
