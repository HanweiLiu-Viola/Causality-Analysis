"""Convert simulated connectivity data into a BIDS-compliant EEG dataset.

Simulated nodes are represented as EEG channels — BIDS provides the most
complete specification for this modality. Channel names are kept as x1–x5
so it is immediately clear that the data are simulated.

Public API
----------
init_dataset(bids_root, name, authors)
    Write dataset_description.json, README, and participants.json once before
    adding any subjects.

write_subject(signal, subject_id, bids_root, fs, ch_names, task, run)
    Convert one (channels, times) numpy array and write it as a BIDS EEG
    run for the given subject.

finalise_dataset(bids_root)
    Run mne_bids.inspect_dataset and report any issues.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import mne
import mne_bids
import numpy as np
from mne_bids import BIDSPath, make_dataset_description, write_raw_bids

logger = logging.getLogger(__name__)

# Suppress MNE's verbose output; the caller controls logging level via the
# standard logging framework.
mne.set_log_level("WARNING")

# Default channel names signal that these are simulated (non-anatomical) sources
_DEFAULT_CH_NAMES: list[str] = [f"x{i + 1}" for i in range(5)]


# ---------------------------------------------------------------------------
# Dataset-level helpers
# ---------------------------------------------------------------------------


def init_dataset(
    bids_root: str | Path,
    name: str = "Simulated Effective Connectivity Dataset",
    authors: list[str] | None = None,
    readme_extra: str = "",
) -> None:
    """Create the BIDS root directory and write dataset-level sidecar files.

    Must be called once before writing any subjects.

    Parameters
    ----------
    bids_root : str or Path
        Root directory for the BIDS dataset (created if absent).
    name : str
        Human-readable dataset name for dataset_description.json.
    authors : list of str, optional
        Author list for dataset_description.json.
    readme_extra : str
        Additional text appended to the README file.
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
    logger.info("Initialised BIDS dataset at %s", bids_root)


def _write_readme(bids_root: Path, name: str, extra: str) -> None:
    """Write a plain-text README to the BIDS root.

    Parameters
    ----------
    bids_root : Path
        BIDS dataset root.
    name : str
        Dataset name used as the README title.
    extra : str
        Additional text appended after the standard boilerplate.
    """
    content = (
        f"{name}\n"
        f"{'=' * len(name)}\n\n"
        "Simulated multi-channel EEG time-series with known ground-truth\n"
        "directed connectivity, generated for benchmarking effective\n"
        "connectivity (EC) estimation methods.\n\n"
        "Channel names (x1–x5) are intentionally non-standard to indicate\n"
        "that the signals are simulated and do not correspond to real\n"
        "electrode positions.\n\n"
        f"Converted with mne-bids {mne_bids.__version__}.\n"
    )
    if extra:
        content += "\n" + extra + "\n"
    (bids_root / "README").write_text(content, encoding="utf-8")


def _write_participants_json(bids_root: Path) -> None:
    """Write participants.json schema describing the columns in participants.tsv.

    Parameters
    ----------
    bids_root : Path
        BIDS dataset root.
    """
    schema = {
        "participant_id": {"Description": "Unique participant identifier"},
        "simulation_seed": {
            "Description": "Random seed used to generate this subject's data",
        },
        "simulation_model": {
            "Description": "Name of the simulation model used",
        },
    }
    (bids_root / "participants.json").write_text(
        json.dumps(schema, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Per-subject writer
# ---------------------------------------------------------------------------


def write_subject(
    signal: np.ndarray,
    subject_id: str,
    bids_root: str | Path,
    fs: float,
    ch_names: list[str] | None = None,
    task: str = "rest",
    run: str | None = None,
) -> BIDSPath:
    """Write one subject's simulated signal as a BIDS EEG run.

    Parameters
    ----------
    signal : np.ndarray, shape (n_channels, n_times)
        Time-series data (z-scored, arbitrary units).
    subject_id : str
        BIDS subject label without the ``sub-`` prefix, e.g. ``"01"``.
    bids_root : str or Path
        Root of the BIDS dataset (must already be initialised with
        :func:`init_dataset`).
    fs : float
        Sampling frequency in Hz. Must be positive.
    ch_names : list of str, optional
        Channel names. Defaults to ``["x1", "x2", ..., "x{n_channels}"]``.
    task : str
        BIDS task label (alphanumeric only). Default: ``"rest"``.
    run : str, optional
        BIDS run label, e.g. ``"01"``. Omit when there is only one run.

    Returns
    -------
    BIDSPath
        Path object pointing to the written file.

    Raises
    ------
    ValueError
        If ``signal`` is not 2-D, or if ``fs`` is not positive.
    """
    if signal.ndim != 2:
        raise ValueError(
            f"signal must be 2-D (n_channels, n_times), got shape {signal.shape}."
        )
    if fs <= 0:
        raise ValueError(f"fs must be a positive sampling frequency, got {fs}.")

    n_channels = signal.shape[0]
    ch_names = ch_names or _DEFAULT_CH_NAMES[:n_channels]
    bids_root = Path(bids_root)

    info = mne.create_info(
        ch_names=list(ch_names),
        sfreq=float(fs),
        ch_types=["eeg"] * n_channels,
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

    _patch_eeg_sidecar(bids_path, task=task, fs=fs, n_ch=n_channels)
    logger.info("Written subject %s to %s", subject_id, bids_path.fpath)
    return bids_path


def _patch_eeg_sidecar(
    bids_path: BIDSPath,
    task: str,
    fs: float,
    n_ch: int,
) -> None:
    """Add any missing required fields to the per-run EEG sidecar JSON.

    MNE-BIDS may omit certain fields for simulated data that lacks real
    electrode metadata. This function fills them in with sensible defaults.

    Parameters
    ----------
    bids_path : BIDSPath
        Path object for the EEG run (used to locate the JSON sidecar).
    task : str
        Task label, written as ``TaskName`` if absent.
    fs : float
        Sampling frequency written as ``SamplingFrequency`` if absent.
    n_ch : int
        Channel count written as ``NumberOfEEGChannels`` if absent.
    """
    sidecar_path = Path(str(bids_path.update(extension=".json").fpath))
    if not sidecar_path.exists():
        return

    with sidecar_path.open(encoding="utf-8") as fh:
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

    with sidecar_path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)


# ---------------------------------------------------------------------------
# Dataset finalisation
# ---------------------------------------------------------------------------


def finalise_dataset(bids_root: str | Path) -> list:
    """Validate the dataset with mne_bids.inspect_dataset.

    Parameters
    ----------
    bids_root : str or Path
        BIDS root directory.

    Returns
    -------
    list
        List of issues reported by the inspector (empty list = valid dataset).
    """
    bids_path = mne_bids.BIDSPath(root=str(bids_root))
    issues = mne_bids.inspect_dataset(bids_path, verbose=False)
    if issues:
        logger.warning("BIDS validation found %d issue(s):", len(issues))
        for issue in issues:
            logger.warning("  %s", issue)
    else:
        logger.info("BIDS validation passed — no issues found.")
    return issues
