"""Plotting utilities for effective-connectivity results.

Provides thin wrappers around matplotlib/seaborn for the most common
connectivity visualizations used in the benchmark and demo notebooks.

All functions save figures to disk and close them — they do not display
interactively. Pass ``output_dir`` to control where files are written.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr

logger = logging.getLogger(__name__)


def plot_corr_matrix(
    corr: np.ndarray,
    *,
    output_dir: str | Path = "results",
) -> None:
    """Save a heatmap of a correlation or connectivity matrix.

    Parameters
    ----------
    corr : np.ndarray
        Square matrix, shape (n_channels, n_channels). Complex values are
        reduced to their real part before plotting.
    output_dir : str or Path
        Directory where the figure is saved. Created if absent.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(np.real(corr), cmap="coolwarm", center=0, square=True, ax=ax)
    ax.set_title("Correlation Matrix")
    output_path = output_dir / "corr_matrix.png"
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("Saved correlation matrix plot to %s", output_path)


def plot_cgc_timeseries(
    cgc: xr.DataArray,
    *,
    output_dir: str | Path = "results",
) -> None:
    """Save per-ROI cGC time-series plots.

    Parameters
    ----------
    cgc : xr.DataArray
        Data array with dimensions ``(roi, times, direction)``.
    output_dir : str or Path
        Directory where figures are saved.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    times = cgc.coords["times"].values
    for i, roi in enumerate(cgc.coords["roi"].values):
        fig, ax = plt.subplots(figsize=(10, 6))
        for j, direction in enumerate(cgc.coords["direction"].values):
            ax.plot(times, cgc[i, :, j], label=str(direction))
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("cGC")
        ax.set_title(f"cGC for ROI: {roi}")
        ax.legend()
        output_path = output_dir / f"cgc_{roi}.png"
        fig.savefig(output_path)
        plt.close(fig)
    logger.info("Saved cGC time-series plots to %s", output_dir)


def plot_freq_response(
    data: np.ndarray,
    title: str,
    *,
    output_dir: str | Path = "results",
) -> None:
    """Save frequency-response plots for each source channel.

    Parameters
    ----------
    data : np.ndarray
        Shape (n_channels, n_channels, n_freqs). ``data[i, j, :]`` is the
        frequency profile from channel i to channel j.
    title : str
        Plot title and filename prefix.
    output_dir : str or Path
        Directory where figures are saved.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_channels = data.shape[0]
    for i in range(n_channels):
        fig, ax = plt.subplots(figsize=(10, 5))
        for j in range(n_channels):
            ax.plot(data[i, j, :], label=f"{i}->{j}")
        ax.set_xlabel("Frequency bin")
        ax.set_ylabel("Value")
        ax.set_title(f"{title}: From channel {i}")
        ax.legend()
        output_path = output_dir / f"{title.lower()}_from_{i}.png"
        fig.savefig(output_path)
        plt.close(fig)
    logger.info("Saved frequency-response plots (%s) to %s", title, output_dir)


def visualize_all_results(
    results: dict,
    *,
    output_dir: str | Path = "results",
) -> None:
    """Dispatch to all applicable plot functions based on available result keys.

    Parameters
    ----------
    results : dict
        Mapping from method name to result data. Recognized keys:
        ``"corr"``, ``"cgc"``, ``"adtf"``, ``"dtf"``, ``"PDC"``.
    output_dir : str or Path
        Directory where all figures are saved.
    """
    logger.info("Generating visualizations in %s ...", output_dir)
    if "corr" in results:
        plot_corr_matrix(results["corr"], output_dir=output_dir)
    if "cgc" in results:
        plot_cgc_timeseries(results["cgc"], output_dir=output_dir)
    if "adtf" in results:
        plot_freq_response(results["adtf"], "ADTF", output_dir=output_dir)
    if "dtf" in results:
        plot_freq_response(results["dtf"], "DTF", output_dir=output_dir)
    if "PDC" in results:
        plot_freq_response(results["PDC"], "PDC", output_dir=output_dir)
    logger.info("All plots saved in '%s'.", output_dir)
