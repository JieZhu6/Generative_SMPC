"""Train the WD-CSNG benchmark without the diversity-preservation loss."""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from Neural_network.train_generator import main as run_generator_training


def main() -> None:
    """Run feasibility/economic WD-CSNG training with no diversity term."""
    run_generator_training("wd_csng")


if __name__ == "__main__":
    main()
