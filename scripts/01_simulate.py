"""
scripts/01_simulate.py
----------------------
Generate simulated multi-channel time-series for all subjects, models, and
epochs, then save them as a compressed NumPy archive.

Called by the ``simulate`` Snakemake rule.

Usage (standalone)
------------------
    python scripts/01_simulate.py --output data/simulated_connectivity_benchmark.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from simulation.simulation_models import simulate  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",     default="data/simulated_connectivity_benchmark.npz")
    p.add_argument("--n-subjects", type=int, default=25)
    p.add_argument("--n-epochs",   type=int, default=5)
    p.add_argument("--t",          type=int, default=1000, help="Time points per epoch")
    p.add_argument("--models",     nargs="+", default=[
        "random", "henon", "lorenz", "sweep",
        "cascadear", "freqarlin", "freqarnonlin",
    ])
    return p.parse_args()


def main():
    args    = parse_args()
    out     = Path(args.output)
    models  = args.models
    n_subj  = args.n_subjects
    n_ep    = args.n_epochs
    T       = args.t
    n_nodes = 5

    print(f"Simulating {n_subj} subjects x {len(models)} models x {n_ep} epochs "
          f"x {n_nodes} nodes x {T} samples …")

    data = np.zeros(
        (n_subj, len(models), n_ep, n_nodes, T), dtype=np.float32
    )

    invalid = []
    for subj in range(n_subj):
        for model_id, model_name in enumerate(models):
            for epoch in range(n_ep):
                seed   = subj * 100 + model_id * 10 + epoch
                signal = simulate(model=model_name, T=T, seed=seed)
                data[subj, model_id, epoch] = signal

                if np.isnan(signal).any() or np.isinf(signal).any():
                    invalid.append((subj, model_name, epoch))

        print(f"  sub-{subj+1:02d} done", flush=True)

    if invalid:
        print(f"WARNING: {len(invalid)} invalid entry/entries detected:")
        for entry in invalid[:10]:
            print(f"  subj={entry[0]}, model={entry[1]}, epoch={entry[2]}")
    else:
        print("All entries valid (no NaN / Inf).")

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out), data=data, model_names=models)
    print(f"Saved: {out}  shape={data.shape}")


if __name__ == "__main__":
    main()
