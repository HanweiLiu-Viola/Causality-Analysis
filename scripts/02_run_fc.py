"""Run FC methods on benchmark data and evaluate against ground-truth adjacency.

Loads the benchmark NPZ produced by ``01_simulate.py``, runs all configured
FC methods, computes MCC per subject / model / method, and saves the results
as a pickle file.

Called by the ``run_fc`` Snakemake rule.

Usage (standalone)
------------------
    python scripts/02_run_fc.py \\
        --data   data/simulated_connectivity_benchmark.npz \\
        --output data/mcc_benchmark.pkl
"""

import argparse
import logging
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import matthews_corrcoef

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from methods.fc_pipeline import FCMethods          # noqa: E402
from simulation.ground_truth import GROUND_TRUTH   # noqa: E402

sys.path.insert(0, str(ROOT / "utils"))
from metrics import binarize_matrix                # noqa: E402

# Default FC method parameters (matching fc_benchmark.ipynb)
DEFAULT_METHODS_PARAMS: dict[str, dict] = {
    "ADTF": {"fmin": 8, "fmax": 12, "n_freqs": 100, "maxlags": 10, "integrate": True},
    "PDC":  {"model_order": 5, "n_fft": 128, "ica_method": "infomax_extended", "integrate": True},
    "DTF":  {"model_order": 5, "n_fft": 128, "ica_method": "infomax_extended", "integrate": True},
    "cGC":  {},
    "PLI":  {"fmin": 8, "fmax": 12, "integrate": True},
    "PSI":  {"fmin": 8, "fmax": 12, "integrate": True},
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", default="data/simulated_connectivity_benchmark.npz"
    )
    parser.add_argument("--output", default="data/mcc_benchmark.pkl")
    parser.add_argument("--fs", type=float, default=256.0)
    parser.add_argument(
        "--n-subjects",
        type=int,
        default=None,
        help="Limit to first N subjects (default: all).",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=75.0,
        help="Binarisation percentile threshold (default: 75).",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Subset of methods to run (default: all).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the FC benchmark script."""
    args = parse_args()

    if not (0.0 < args.percentile < 100.0):
        raise ValueError(
            f"--percentile must be in (0, 100), got {args.percentile}."
        )

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Benchmark data not found: {data_path}. "
            "Run 01_simulate.py first."
        )

    loaded = np.load(data_path, allow_pickle=True)
    all_data = loaded["data"]           # (n_subjects, n_models, n_epochs, n_nodes, n_times)
    model_names = list(loaded["model_names"])
    n_subj_total = all_data.shape[0]
    n_subjects = min(args.n_subjects or n_subj_total, n_subj_total)

    methods_params = {
        name: params
        for name, params in DEFAULT_METHODS_PARAMS.items()
        if args.methods is None or name in args.methods
    }
    method_names = list(methods_params.keys())

    logger.info("Data shape   : %s", all_data.shape)
    logger.info("Subjects     : %d / %d", n_subjects, n_subj_total)
    logger.info("Models       : %s", model_names)
    logger.info("Methods      : %s", method_names)
    logger.info("Percentile   : %.1f", args.percentile)

    mcc_dict: dict[str, dict[str, list[float]]] = {
        model: {method: [] for method in method_names}
        for model in model_names
    }

    for subj in range(n_subjects):
        for model_id, model_name in enumerate(model_names):
            ground_truth = GROUND_TRUTH.get(model_name)
            if ground_truth is None:
                logger.warning("No ground truth for model %r; skipping.", model_name)
                continue

            epoch_data = all_data[subj, model_id]   # (n_epochs, n_nodes, n_times)
            fc = FCMethods(epoch_data, fs=args.fs)
            results = fc.compute_all(methods_params)

            for method in method_names:
                result = results.get(method, {})
                binary = binarize_matrix(result, method, percentile=args.percentile)
                if binary is not None:
                    mcc = matthews_corrcoef(ground_truth.flatten(), binary.flatten())
                    mcc_dict[model_name][method].append(float(mcc))

        logger.info("  sub-%02d done", subj + 1)

    # Summary
    logger.info("Mean MCC per method:")
    for method in method_names:
        all_vals = [v for model in model_names for v in mcc_dict[model][method]]
        logger.info("  %-8s: %.3f", method, float(np.nanmean(all_vals)))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as fh:
        pickle.dump(mcc_dict, fh)
    logger.info("Saved: %s", output_path)


if __name__ == "__main__":
    main()
