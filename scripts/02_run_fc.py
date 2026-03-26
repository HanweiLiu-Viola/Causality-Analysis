"""
scripts/02_run_fc.py
--------------------
Load the benchmark NPZ, run all FC methods against the known ground-truth
adjacency matrices, compute MCC per subject / model / method, and save the
results dictionary as a pickle file.

Called by the ``run_fc`` Snakemake rule.

Usage (standalone)
------------------
    python scripts/02_run_fc.py \\
        --data   data/simulated_connectivity_benchmark.npz \\
        --output data/mcc_benchmark.pkl
"""

import argparse
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import matthews_corrcoef

warnings.filterwarnings("ignore")

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from methods.fc_pipeline import FCMethods  # noqa: E402

# ── Ground truth adjacency (source → target) ──────────────────────────────────
GROUND_TRUTH = {
    "random": np.array([
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0],
    ]),
    "henon": np.array([
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
    ]),
    "lorenz": np.array([
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0],
    ]),
    "sweep": np.array([
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]),
    "cascadear": np.array([
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
    ]),
    "freqarlin": np.array([
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
    ]),
    "freqarnonlin": np.array([
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
    ]),
}

# Default FC method parameters (matching fc_benchmark.ipynb)
DEFAULT_METHODS_PARAMS = {
    "ADTF":  {"fmin": 8, "fmax": 12, "n_freqs": 100, "maxlags": 10, "integrate": True},
    "PDCoh": {"model_order": 5, "n_fft": 128, "ica_method": "infomax_extended", "integrate": True},
    "DTF":   {"model_order": 5, "n_fft": 128, "ica_method": "infomax_extended", "integrate": True},
    "cGC":   {},
    "PLI":   {"fmin": 8, "fmax": 12, "integrate": True},
    "PSI":   {"fmin": 8, "fmax": 12, "integrate": True},
}


def binarize_fc(result: dict, method: str, percentile: float = 75) -> np.ndarray:
    """Return binary adjacency matrix in [source, target] convention."""
    if "matrix" not in result:
        return None
    mat = result["matrix"].copy()
    np.fill_diagonal(mat, 0)
    if method == "Corr":
        mat = np.abs(mat)
    # Non-PSI methods store [target, source] internally → transpose
    if method != "PSI":
        mat = mat.T
    mask  = ~np.eye(mat.shape[0], dtype=bool)
    thr   = np.percentile(np.abs(mat[mask]), percentile)
    return (np.abs(mat) >= thr).astype(int)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",       default="data/simulated_connectivity_benchmark.npz")
    p.add_argument("--output",     default="data/mcc_benchmark.pkl")
    p.add_argument("--fs",         type=float, default=256.0)
    p.add_argument("--n-subjects", type=int,   default=None,
                   help="Limit to first N subjects (default: all)")
    p.add_argument("--percentile", type=float, default=75.0)
    p.add_argument("--methods",    nargs="+",  default=None,
                   help="Subset of methods to run (default: all)")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Load data ──────────────────────────────────────────────────────────────
    loaded      = np.load(args.data, allow_pickle=True)
    all_data    = loaded["data"]          # (subjects, models, epochs, nodes, time)
    model_names = list(loaded["model_names"])
    n_subj_total = all_data.shape[0]
    n_subjects   = min(args.n_subjects or n_subj_total, n_subj_total)

    methods_params = {
        k: v for k, v in DEFAULT_METHODS_PARAMS.items()
        if args.methods is None or k in args.methods
    }
    method_names = list(methods_params.keys())

    print(f"Data shape   : {all_data.shape}")
    print(f"Subjects     : {n_subjects}/{n_subj_total}")
    print(f"Models       : {model_names}")
    print(f"Methods      : {method_names}")
    print(f"Percentile   : {args.percentile}")

    mcc_dict = {
        model: {method: [] for method in method_names}
        for model in model_names
    }

    for subj in range(n_subjects):
        for model_id, model_name in enumerate(model_names):
            gt  = GROUND_TRUTH.get(model_name)
            if gt is None:
                continue

            data = all_data[subj, model_id]          # (epochs, nodes, time)
            fc   = FCMethods(data, fs=args.fs)
            results = fc.compute_all(methods_params)

            for method in method_names:
                res    = results.get(method, {})
                binary = binarize_fc(res, method, percentile=args.percentile)
                if binary is not None:
                    mcc = matthews_corrcoef(gt.flatten(), binary.flatten())
                    mcc_dict[model_name][method].append(float(mcc))

        print(f"  sub-{subj+1:02d}  done", flush=True)

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\nMean MCC per method:")
    for method in method_names:
        vals = [v for model in model_names for v in mcc_dict[model][method]]
        print(f"  {method:<8}: {np.nanmean(vals):.3f}")

    # ── Save ───────────────────────────────────────────────────────────────────
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fh:
        pickle.dump(mcc_dict, fh)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
