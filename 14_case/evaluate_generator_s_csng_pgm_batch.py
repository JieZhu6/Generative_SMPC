"""Evaluate the sigmoid-output S-CSNG benchmark with native PGM batches."""

from evaluate_generator_tcn_pgm_batch import main as run_pgm_benchmark


def main() -> None:
    """Run held-out PGM screening for an S-CSNG checkpoint."""
    run_pgm_benchmark("s_csng")


if __name__ == "__main__":
    main()
