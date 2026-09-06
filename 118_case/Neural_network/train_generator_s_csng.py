"""Train S-CSNG with the shared CSNG optimization defaults."""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from Neural_network.train_generator import main as run_generator_training


def main() -> None:
    """Run sigmoid-output S-CSNG through the shared training entry point."""
    run_generator_training("s_csng")


if __name__ == "__main__":
    main()
