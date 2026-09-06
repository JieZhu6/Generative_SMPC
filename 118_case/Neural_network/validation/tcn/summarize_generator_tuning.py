"""Summarize short Generator tuning runs from Stage-1 checkpoints."""

import argparse
from pathlib import Path

import torch


METRICS = (
    "validation_loss",
    "validation_candidate_violation",
    "validation_max_violation",
    "validation_worst_violation",
    "validation_violation_qg",
    "validation_violation_thermal",
    "validation_soft_score",
    "validation_pf_residual",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()

    columns = ("file", "hidden", "latent", "candidates", "lr", "best_epoch", *METRICS)
    print("|".join(columns))
    for path in sorted(args.directory.glob("*stage1.pt")):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        history = checkpoint["training_history"]
        best = min(history, key=lambda record: record["validation_loss"])
        model = checkpoint["model_config"]
        training = checkpoint["training_hyperparameters"]
        values = (
            path.name,
            str(model["hidden_channels"]),
            str(model["latent_dim"]),
            str(training["candidates"]),
            f"{training['stage1_learning_rate']:.1e}",
            str(best["epoch"]),
            *(f"{best[name]:.7g}" for name in METRICS),
        )
        print("|".join(values))


if __name__ == "__main__":
    main()
