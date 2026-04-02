"""Statistical validation tests for functional connectivity (FC) pipelines.

Three tests are implemented:

1. ``bootstrap_performance``
   Estimates confidence intervals for FC method performance scores
   (AUC-ROC, Average Precision, or MCC) via non-parametric bootstrap
   resampling across subjects.

2. ``surrogate_test``
   Assesses whether detected connectivity is significantly above chance
   by comparing true FC scores against a null distribution built from
   phase-randomized surrogate time series.

3. ``time_reversal_test``
   Validates the directionality of FC estimates by comparing forward
   and time-reversed connectivity matrices. Genuine directed connections
   should change under time reversal; symmetric or noise-driven estimates
   should not.

Dependencies
------------
    numpy, scipy, sklearn
    (all present in the project's Docker image)

Example usage
-------------
    from methods.statistical_tests import (
        bootstrap_performance,
        surrogate_test,
        time_reversal_test,
    )
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import stats  # noqa: F401 — available for callers who import this module
from sklearn.metrics import average_precision_score, roc_auc_score


# =============================================================================
# Private helpers
# =============================================================================


def _compute_matrix_score(
    matrix: np.ndarray,
    y_true: np.ndarray,
    metric: str,
) -> float:
    """Score a connectivity matrix against flattened binary ground-truth labels.

    Parameters
    ----------
    matrix : np.ndarray
        Connectivity matrix, shape (n_channels, n_channels).
    y_true : np.ndarray
        Flattened binary ground-truth labels.
    metric : str
        ``"auc"`` for AUC-ROC, ``"ap"`` for Average Precision.

    Returns
    -------
    float
        Performance score.
    """
    y_score = np.abs(matrix.flatten())
    if metric == "auc":
        return float(roc_auc_score(y_true, y_score))
    return float(average_precision_score(y_true, y_score))


# =============================================================================
# 1. Bootstrap confidence intervals
# =============================================================================


def bootstrap_performance(
    scores: np.ndarray,
    n_iterations: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """Estimate bootstrap confidence intervals for a set of per-subject scores.

    Parameters
    ----------
    scores : np.ndarray, shape (n_subjects,)
        Per-subject performance scores (e.g. AUC-ROC or Average Precision).
    n_iterations : int
        Number of bootstrap resamples.
    ci : float
        Confidence level, e.g. 0.95 for a 95 % interval.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        mean       : float  — mean of the original scores
        ci_lower   : float  — lower bound of the confidence interval
        ci_upper   : float  — upper bound of the confidence interval
        bootstrap_distribution : np.ndarray — all bootstrap means
    """
    rng = np.random.default_rng(seed)
    n = len(scores)

    bootstrap_means = np.array([
        np.mean(rng.choice(scores, size=n, replace=True))
        for _ in range(n_iterations)
    ])

    alpha = (1.0 - ci) / 2.0
    ci_lower = float(np.percentile(bootstrap_means, 100 * alpha))
    ci_upper = float(np.percentile(bootstrap_means, 100 * (1 - alpha)))

    return {
        "mean": float(np.mean(scores)),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "bootstrap_distribution": bootstrap_means,
    }


def bootstrap_all_methods(
    auc_dict: dict[str, np.ndarray],
    n_iterations: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, dict]:
    """Apply ``bootstrap_performance`` to every method in a results dictionary.

    Parameters
    ----------
    auc_dict : dict
        Mapping from method name to array of per-subject AUC scores,
        e.g. ``{"ADTF": array([0.8, 0.7, ...]), "cGC": array([...]), ...}``.
    n_iterations : int
        Number of bootstrap resamples per method.
    ci : float
        Confidence level.
    seed : int
        Base random seed (each method uses seed + method_index).

    Returns
    -------
    dict
        Mapping from method name to the output of ``bootstrap_performance``.
    """
    return {
        method: bootstrap_performance(
            np.asarray(scores),
            n_iterations=n_iterations,
            ci=ci,
            seed=seed + i,
        )
        for i, (method, scores) in enumerate(auc_dict.items())
    }


# =============================================================================
# 2. Surrogate test
# =============================================================================


def phase_randomize(data: np.ndarray, seed: int | None = None) -> np.ndarray:
    """Generate a phase-randomized surrogate of a multichannel time series.

    Phase randomization preserves the power spectrum of each channel while
    destroying temporal structure and cross-channel phase relationships.
    The same random phase shift is applied to all channels to preserve the
    instantaneous amplitude envelope correlations (Prichard & Theiler, 1994).

    Parameters
    ----------
    data : np.ndarray, shape (n_channels, n_times)
    seed : int or None

    Returns
    -------
    surrogate : np.ndarray, shape (n_channels, n_times)
    """
    rng = np.random.default_rng(seed)
    n_times = data.shape[1]
    fft_data = np.fft.rfft(data, axis=1)

    # Shared phase shifts across channels preserve amplitude-envelope correlations
    n_freqs = fft_data.shape[1]
    phase_shift = np.exp(1j * rng.uniform(0, 2 * np.pi, size=n_freqs))

    fft_surrogate = fft_data * phase_shift[np.newaxis, :]
    surrogate = np.fft.irfft(fft_surrogate, n=n_times, axis=1)
    return surrogate.astype(data.dtype)


def surrogate_test(
    data: np.ndarray,
    ground_truth: np.ndarray,
    fc_func: Callable[[np.ndarray], np.ndarray],
    n_surrogates: int = 200,
    seed: int = 42,
    metric: str = "auc",
) -> dict:
    """Test whether FC estimates are significantly above a surrogate null.

    A null distribution is built by computing the chosen performance metric
    on phase-randomized versions of the input data. The true score is then
    compared against this distribution to obtain an empirical p-value.

    Parameters
    ----------
    data : np.ndarray, shape (n_channels, n_times)
        Single-epoch time series.
    ground_truth : np.ndarray, shape (n_channels, n_channels)
        Binary ground-truth adjacency matrix (GT[i, j] = 1 means i -> j).
    fc_func : Callable
        Function that accepts data (n_channels, n_times) and returns a
        connectivity matrix (n_channels, n_channels).
    n_surrogates : int
        Number of surrogate datasets to generate.
    seed : int
        Base random seed.
    metric : str
        Performance metric: ``"auc"`` (AUC-ROC) or ``"ap"`` (Average Precision).

    Returns
    -------
    dict with keys:
        true_score        : float — metric on real data
        surrogate_scores  : np.ndarray — null distribution
        p_value           : float — fraction of surrogates >= true score
        z_score           : float — standardised distance from null mean
    """
    y_true = ground_truth.flatten()
    true_matrix = fc_func(data)
    true_score = _compute_matrix_score(true_matrix, y_true, metric)

    surrogate_scores = np.array([
        _compute_matrix_score(
            fc_func(phase_randomize(data, seed=seed + i)), y_true, metric
        )
        for i in range(n_surrogates)
    ])

    p_value = float(np.mean(surrogate_scores >= true_score))
    null_mean = np.mean(surrogate_scores)
    null_std = np.std(surrogate_scores)
    z_score = float((true_score - null_mean) / null_std) if null_std > 0 else np.nan

    return {
        "true_score": true_score,
        "surrogate_scores": surrogate_scores,
        "p_value": p_value,
        "z_score": z_score,
    }


# =============================================================================
# 3. Time reversal test
# =============================================================================


def time_reversal_test(
    data: np.ndarray,
    fc_func: Callable[[np.ndarray], np.ndarray],
    ground_truth: np.ndarray | None = None,
    metric: str = "auc",
) -> dict:
    """Validate FC directionality using time reversal.

    Genuine directed connections should produce asymmetric connectivity
    matrices that change when the time series is reversed. Symmetric or
    noise-driven connections should be invariant under time reversal.

    The asymmetry index is defined as the mean absolute difference between
    the forward and reversed connectivity matrices (off-diagonal elements
    only). A high asymmetry index suggests the method is sensitive to
    temporal direction.

    Parameters
    ----------
    data : np.ndarray, shape (n_channels, n_times)
        Single-epoch time series.
    fc_func : Callable
        Function that accepts data (n_channels, n_times) and returns a
        connectivity matrix (n_channels, n_channels).
    ground_truth : np.ndarray or None
        Binary ground-truth adjacency matrix. If provided, performance
        scores are computed for both forward and reversed data.
    metric : str
        ``"auc"`` or ``"ap"``. Used only when ground_truth is not None.

    Returns
    -------
    dict with keys:
        forward_matrix    : np.ndarray — FC matrix on original data
        reversed_matrix   : np.ndarray — FC matrix on time-reversed data
        asymmetry_index   : float — mean |forward - reversed| off-diagonal
        forward_score     : float or None — metric score on forward data
        reversed_score    : float or None — metric score on reversed data
        score_difference  : float or None — forward_score - reversed_score
    """
    forward_matrix = fc_func(data)
    reversed_matrix = fc_func(data[:, ::-1])

    n = forward_matrix.shape[0]
    off_diag = ~np.eye(n, dtype=bool)
    asymmetry_index = float(
        np.mean(np.abs(forward_matrix[off_diag] - reversed_matrix[off_diag]))
    )

    forward_score = None
    reversed_score = None
    score_difference = None

    if ground_truth is not None:
        y_true = ground_truth.flatten()
        forward_score = _compute_matrix_score(forward_matrix, y_true, metric)
        reversed_score = _compute_matrix_score(reversed_matrix, y_true, metric)
        score_difference = forward_score - reversed_score

    return {
        "forward_matrix": forward_matrix,
        "reversed_matrix": reversed_matrix,
        "asymmetry_index": asymmetry_index,
        "forward_score": forward_score,
        "reversed_score": reversed_score,
        "score_difference": score_difference,
    }


def time_reversal_all_methods(
    data: np.ndarray,
    fc_funcs: dict[str, Callable[[np.ndarray], np.ndarray]],
    ground_truth: np.ndarray | None = None,
    metric: str = "auc",
) -> dict[str, dict]:
    """Apply ``time_reversal_test`` to every method in a dictionary.

    Parameters
    ----------
    data : np.ndarray, shape (n_channels, n_times)
    fc_funcs : dict
        Mapping from method name to callable FC function.
    ground_truth : np.ndarray or None
    metric : str

    Returns
    -------
    dict
        Mapping from method name to the output of ``time_reversal_test``.
    """
    return {
        method: time_reversal_test(data, func, ground_truth, metric)
        for method, func in fc_funcs.items()
    }
