"""Train WD-CSNG with shared defaults and no diversity-preservation loss."""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from Neural_network.train_generator import main as run_generator_training


def main() -> None:
    """Run WD-CSNG through the shared training entry point without diversity."""
    run_generator_training("wd_csng")


if __name__ == "__main__":
    main()
