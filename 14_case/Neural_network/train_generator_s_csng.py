"""Train the sigmoid-output S-CSNG benchmark."""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from Neural_network.train_generator import main as run_generator_training


def main() -> None:
    """Run two-stage S-CSNG training with sigmoid output parameterization."""
    run_generator_training("s_csng")


if __name__ == "__main__":
    main()
