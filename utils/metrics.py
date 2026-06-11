"""Evaluation utilities for effective-connectivity benchmarks.

Provides binarization and scoring helpers used by both scripts and notebooks.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def binarize_matrix(
    result: dict,
    method: str,
    *,
    percentile: float = 75.0,
) -> np.ndarray | None:
    """Convert a raw connectivity result dict to a binary [source, target] adjacency matrix.

    Thresholds the top ``percentile`` percent of off-diagonal absolute values,
    and handles the internal [target, source] convention used by most EC methods.

    Convention note
    ---------------
    Ground-truth matrices follow [source, target] (GT[i, j] = 1 → i drives j).
    Most EC methods (ADTF, PDC, DTF, cGC, PLI) return [target, source] internally
    and are transposed here. PSI already returns [source, target] and is not
    transposed.

    Parameters
    ----------
    result : dict
        Output of ``FCMethods.compute_all()`` for a single method. Must contain
        a ``"matrix"`` key with a 2-D or 3-D numpy array.
    method : str
        Method name (e.g. ``"ADTF"``, ``"PSI"``). Used to select the
        correct transpose behaviour.
    percentile : float
        Threshold percentile applied to off-diagonal absolute values.
        Default: 75.0.

    Returns
    -------
    np.ndarray or None
        Binary (n_channels, n_channels) matrix in [source, target] convention,
        with diagonal zeroed. Returns ``None`` if result has no ``"matrix"`` key
        or if the matrix is missing.
    """
    if "matrix" not in result or result["matrix"] is None:
        return None

    mat = np.array(result["matrix"], dtype=float).copy()
    np.fill_diagonal(mat, 0.0)

    if method == "Corr":
        mat = np.abs(mat)

    # PSI is already in [source, target]; all others store [target, source]
    if method != "PSI":
        mat = mat.T

    off_diag_mask = ~np.eye(mat.shape[0], dtype=bool)
    threshold = np.percentile(np.abs(mat[off_diag_mask]), percentile)
    return (np.abs(mat) >= threshold).astype(int)


def binarize_matrix_raw(
    matrix: np.ndarray,
    method: str,
    *,
    percentile: float = 75.0,
) -> np.ndarray:
    """Thin wrapper around ``binarize_matrix`` for callers that hold a raw array.

    Use this when you have the connectivity matrix directly (not a result dict).

    Parameters
    ----------
    matrix : np.ndarray
        Raw connectivity matrix, shape (n_channels, n_channels).
    method : str
        Method name (determines transpose behaviour; see ``binarize_matrix``).
    percentile : float
        Threshold percentile. Default: 75.0.

    Returns
    -------
    np.ndarray
        Binary adjacency matrix, shape (n_channels, n_channels).
    """
    return binarize_matrix({"matrix": matrix}, method, percentile=percentile)
