"""Evaluate the no-diversity WD-CSNG benchmark with native PGM batches."""

from evaluate_generator_tcn_pgm_batch import main as run_pgm_benchmark


def main() -> None:
    """Run held-out PGM screening for a WD-CSNG checkpoint."""
    run_pgm_benchmark("wd_csng")


if __name__ == "__main__":
    main()
