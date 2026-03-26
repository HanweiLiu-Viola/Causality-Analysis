"""
src/methods/convert_to_bids.py
------------------------------
Write simulated connectivity data into a BIDS-compliant EEG dataset using
MNE-BIDS.

Simulated nodes are represented as EEG channels — BIDS provides the most
complete specification for this modality.  Channel names are kept as x1–x5
so it is immediately clear that the data are simulated.

Public API
----------
init_dataset(bids_root, name, authors)
    Write dataset_description.json, README and participants.json once before
    adding any subjects.

write_subject(signal, subject_id, bids_root, fs, ch_names, task, run)
    Convert one (channels x time) numpy array and write it as a BIDS EEG
    run for the given subject.

finalise_dataset(bids_root)
    Run mne_bids.inspect_dataset and report any issues.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import mne
import mne_bids
from mne_bids import BIDSPath, make_dataset_description, write_raw_bids
import numpy as np

mne.set_log_level("WARNING")

# ── Default channel names (lowercase, making simulation origin obvious) ────────
DEFAULT_CH_NAMES = [f"x{i+1}" for i in range(5)]


# ── Dataset-level helpers ─────────────────────────────────────────────────────

def init_dataset(
    bids_root: str | Path,
    name: str = "Simulated Functional Connectivity Dataset",
    authors: Optional[list[str]] = None,
    readme_extra: str = "",
) -> None:
    """Create the BIDS root directory and write dataset-level sidecar files.

    Call this once before writing any subjects.

    Parameters
    ----------
    bids_root : path
        Root directory for the BIDS dataset (created if absent).
    name : str
        Human-readable dataset name written to dataset_description.json.
    authors : list of str, optional
        Author list for dataset_description.json.
    readme_extra : str
        Additional text appended to the README.
    """
    bids_root = Path(bids_root)
    bids_root.mkdir(parents=True, exist_ok=True)

    make_dataset_description(
        path=str(bids_root),
        name=name,
        dataset_type="raw",
        authors=authors or [],
        data_license="CC0",
        overwrite=True,
    )

    _write_readme(bids_root, name, readme_extra)
    _write_participants_json(bids_root)


def _write_readme(bids_root: Path, name: str, extra: str) -> None:
    txt = (
        f"{name}\n"
        f"{'=' * len(name)}\n\n"
        "Simulated multi-channel EEG time-series with known ground-truth\n"
        "directed connectivity, generated for benchmarking functional\n"
        "connectivity (FC) estimation methods.\n\n"
        "Channel names (x1-x5) are intentionally non-standard to indicate\n"
        "that the signals are simulated and do not correspond to real\n"
        "electrode positions.\n\n"
        f"Converted with mne-bids {mne_bids.__version__}.\n"
    )
    if extra:
        txt += "\n" + extra + "\n"
    (bids_root / "README").write_text(txt, encoding="utf-8")


def _write_participants_json(bids_root: Path) -> None:
    desc = {
        "participant_id": {"Description": "Unique participant identifier"},
        "simulation_seed": {
            "Description": "Random seed used to generate this subject's data",
        },
        "simulation_model": {
            "Description": "Name of the simulation model used",
        },
    }
    (bids_root / "participants.json").write_text(
        json.dumps(desc, indent=2), encoding="utf-8"
    )


# ── Per-subject writer ────────────────────────────────────────────────────────

def write_subject(
    signal: np.ndarray,
    subject_id: str,
    bids_root: str | Path,
    fs: float,
    ch_names: Optional[list[str]] = None,
    task: str = "rest",
    run: Optional[str] = None,
) -> BIDSPath:
    """Write one subject's simulated signal as a BIDS EEG run.

    Parameters
    ----------
    signal : np.ndarray, shape (n_channels, n_times)
        Time-series data (z-scored, arbitrary units).
    subject_id : str
        BIDS subject label without the ``sub-`` prefix, e.g. ``"01"``.
    bids_root : path
        Root of the BIDS dataset (must already be initialised with
        :func:`init_dataset`).
    fs : float
        Sampling frequency in Hz.
    ch_names : list of str, optional
        Channel names.  Defaults to ``["x1", "x2", ..., "x5"]``.
    task : str
        BIDS task label (alphanumeric only).
    run : str, optional
        BIDS run label, e.g. ``"01"``.  Omit when there is only one run.

    Returns
    -------
    BIDSPath
        The path object pointing to the written file.
    """
    ch_names = ch_names or DEFAULT_CH_NAMES[: signal.shape[0]]
    bids_root = Path(bids_root)

    # Build MNE Raw object
    info = mne.create_info(
        ch_names=list(ch_names),
        sfreq=float(fs),
        ch_types=["eeg"] * len(ch_names),
    )
    raw = mne.io.RawArray(signal.astype(np.float64), info, verbose=False)
    # Use a virtual average reference — no physical electrode for simulated data
    raw.set_eeg_reference(ref_channels="average", projection=True, verbose=False)

    bids_path = BIDSPath(
        subject=subject_id,
        task=task,
        run=run,
        datatype="eeg",
        suffix="eeg",
        root=str(bids_root),
    )

    write_raw_bids(
        raw,
        bids_path,
        format="BrainVision",
        overwrite=True,
        allow_preload=True,
        verbose=False,
    )

    # Guarantee all BIDS-required EEG sidecar fields
    _patch_eeg_sidecar(bids_path, task=task, fs=fs, n_ch=len(ch_names))

    return bids_path


def _patch_eeg_sidecar(
    bids_path: BIDSPath, task: str, fs: float, n_ch: int
) -> None:
    """Add any missing required fields to the per-run EEG sidecar JSON."""
    sidecar = Path(str(bids_path.update(extension=".json").fpath))
    if not sidecar.exists():
        return

    with open(sidecar, encoding="utf-8") as fh:
        meta = json.load(fh)

    meta.setdefault("TaskName", task)
    meta.setdefault("SamplingFrequency", float(fs))
    meta.setdefault(
        "EEGReference",
        "average (virtual — simulated data, no physical electrode)",
    )
    meta.setdefault("PowerLineFrequency", 50)
    meta.setdefault("EEGGround", "n/a")
    meta.setdefault("EEGPlacementScheme", "n/a")
    meta.setdefault("Manufacturer", "n/a")
    meta.setdefault("NumberOfEEGChannels", n_ch)
    meta.setdefault("RecordingType", "continuous")

    with open(sidecar, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)


# ── Dataset finalisation ──────────────────────────────────────────────────────

def finalise_dataset(bids_root: str | Path) -> list:
    """Validate the dataset with mne_bids.inspect_dataset.

    Parameters
    ----------
    bids_root : path
        BIDS root directory.

    Returns
    -------
    list
        List of issues reported by the inspector (empty == valid).
    """
    bids_path = mne_bids.BIDSPath(root=str(bids_root))
    issues = mne_bids.inspect_dataset(bids_path, verbose=False)
    if issues:
        print(f"[BIDS] {len(issues)} issue(s) found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("[BIDS] Dataset is valid — no issues found.")
    return issues
