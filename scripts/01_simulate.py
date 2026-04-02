"""Generate simulated multi-channel time-series for the FC benchmark.

Produces data for all subjects, models, and epochs, then saves the result as a
compressed NumPy archive at the path given by ``--output``.

Called by the ``simulate`` Snakemake rule.

Usage (standalone)
------------------
    python scripts/01_simulate.py --output data/simulated_connectivity_benchmark.npz
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from simulation.simulation_models import N_CHANNELS, simulate  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/simulated_connectivity_benchmark.npz")
    parser.add_argument("--n-subjects", type=int, default=25)
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument(
        "--t", type=int, default=1000, help="Time points per epoch."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "random", "henon", "lorenz", "sweep",
            "cascadear", "freqarlin", "freqarnonlin",
        ],
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the simulation script."""
    args = parse_args()
    output_path = Path(args.output)
    n_subjects = args.n_subjects
    n_epochs = args.n_epochs
    time_points = args.t
    model_names = args.models

    logger.info(
        "Simulating %d subjects × %d models × %d epochs × %d channels × %d samples",
        n_subjects, len(model_names), n_epochs, N_CHANNELS, time_points,
    )

    data = np.zeros(
        (n_subjects, len(model_names), n_epochs, N_CHANNELS, time_points),
        dtype=np.float32,
    )

    invalid_entries: list[tuple] = []
    for subj in range(n_subjects):
        for model_id, model_name in enumerate(model_names):
            for epoch in range(n_epochs):
                # Deterministic seed: unique per (subject, model, epoch) combination
                seed = subj * 100 + model_id * 10 + epoch
                signal = simulate(model=model_name, T=time_points, seed=seed)
                data[subj, model_id, epoch] = signal

                if np.isnan(signal).any() or np.isinf(signal).any():
                    invalid_entries.append((subj, model_name, epoch))

        logger.info("  sub-%02d done", subj + 1)

    if invalid_entries:
        logger.warning("%d invalid entries detected (NaN or Inf):", len(invalid_entries))
        for entry in invalid_entries[:10]:
            logger.warning("  subj=%d, model=%s, epoch=%d", *entry)
    else:
        logger.info("All entries valid — no NaN or Inf detected.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(output_path), data=data, model_names=model_names)
    logger.info("Saved: %s  shape=%s", output_path, data.shape)


if __name__ == "__main__":
    main()
