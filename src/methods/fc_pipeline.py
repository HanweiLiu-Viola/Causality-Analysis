"""Unified interface for functional connectivity estimation methods.

Provides eight FC methods under a single ``FCMethods`` dispatcher class:

    VAR-based:          ADTF, PDC, DTF
    Phase-based:        PLI, PSI
    Granger-causality:  cGC
    Information theory: TE, MI  (require IDTxl installation)

All methods share the same [target, source] output convention internally.
Use ``utils.metrics.binarize_matrix()`` to convert to [source, target] for
comparison against ground-truth adjacency matrices.

Matrix convention (internal)
-----------------------------
``result[i, j]`` = connectivity from channel ``j`` to channel ``i``.
This matches the PDC/DTF spectral definition: A[i, j, f] is the contribution
of channel j to the prediction of channel i.
"""

import logging
import warnings

import numpy as np
from mne_connectivity import phase_slope_index, spectral_connectivity_epochs
from statsmodels.tsa.vector_ar.var_model import VAR

from core.mvarica import MVAR, connectivity_mvarica

logger = logging.getLogger(__name__)

# IDTxl is an optional dependency (may not be installed in all environments)
try:
    from idtxl.data import Data
    from idtxl.multivariate_mi import MultivariateMI
    from idtxl.multivariate_te import MultivariateTE
    _IDTXL_AVAILABLE = True
except ImportError:
    _IDTXL_AVAILABLE = False
    logger.debug("IDTxl not installed; TE and MI methods will raise ImportError.")


class BasePreprocessor:
    """Standardize input data shape and prepare ICA parameter dictionaries.

    Accepts 2-D (n_channels, n_times) or 3-D (n_epochs, n_channels, n_times)
    arrays and ensures the output is always 3-D.

    Parameters
    ----------
    data : np.ndarray
        Input time-series, 2-D or 3-D.

    Attributes
    ----------
    data : np.ndarray
        Standardized data, shape (n_epochs, n_channels, n_times).
    """

    def __init__(self, data: np.ndarray) -> None:
        self.data = self._standardize(data)

    @staticmethod
    def _standardize(data: np.ndarray) -> np.ndarray:
        """Add a leading epoch dimension if data is 2-D."""
        if data.ndim == 2:
            return data[np.newaxis, ...]
        if data.ndim == 3:
            return data
        raise ValueError(
            f"Data must be 2-D (n_channels, n_times) or 3-D (n_epochs, n_channels, n_times), "
            f"got {data.ndim}-D array."
        )

    def prepare_ica_params(
        self,
        ica_params: dict | None = None,
        **kwargs,
    ) -> dict:
        """Build or validate an ICA parameter dictionary.

        Parameters
        ----------
        ica_params : dict or None
            If provided, returned unchanged. If None, built from ``kwargs``.
        **kwargs
            ``ica_method`` (str, default ``"infomax_extended"``) and
            ``random_state`` (int | None).

        Returns
        -------
        dict
            Dict with keys ``"method"`` and ``"random_state"``.
        """
        if ica_params is not None:
            return ica_params
        ica_method = kwargs.get("ica_method", "infomax_extended")
        valid_methods = ("infomax_extended", "infomax", "fastica")
        if ica_method not in valid_methods:
            raise ValueError(
                f"Unsupported ICA method {ica_method!r}. Valid: {valid_methods}"
            )
        return {
            "method": ica_method,
            "random_state": kwargs.get("random_state", None),
        }


class ADTFModel:
    """Adaptive Directed Transfer Function estimator.

    Fits one VAR model per epoch using AIC order selection and computes the
    normalized squared transfer function (ADTF) at the requested frequencies.

    Parameters
    ----------
    data : np.ndarray
        Shape (n_epochs, n_channels, n_times).
    """

    def __init__(self, data: np.ndarray) -> None:
        self.data = data
        self.models: list = []

    def fit_var_model(self, *, maxlags: int = 20) -> None:
        """Fit one VAR model per epoch with AIC order selection.

        Falls back to maxlags=1 on singular-matrix failure to ensure
        downstream code always receives a fitted model.

        Parameters
        ----------
        maxlags : int
            Maximum lag order for AIC selection. Default: 20.
        """
        self.models = []
        for trial in self.data:
            model = VAR(trial.T)
            try:
                results = model.fit(maxlags=maxlags, ic="aic")
            except np.linalg.LinAlgError:
                logger.warning(
                    "VAR fitting with AIC failed (singular matrix); falling back to maxlags=1."
                )
                results = model.fit(maxlags=1)
            self.models.append(results)

    def _compute_transfer_matrix(self, coef_matrices: np.ndarray, freq: float) -> np.ndarray:
        """Compute the VAR transfer function H(f) at a single normalized frequency.

        Parameters
        ----------
        coef_matrices : np.ndarray
            VAR coefficient matrices, shape (model_order, n_channels, n_channels).
        freq : float
            Normalized frequency in [0, 0.5].

        Returns
        -------
        np.ndarray
            Transfer function matrix H(f), shape (n_channels, n_channels).
        """
        p, n_channels, _ = coef_matrices.shape
        z_powers = np.exp(-1j * 2 * np.pi * freq * np.arange(1, p + 1))
        a_sum = sum(coef_matrices[k] * z_powers[k] for k in range(p))
        return np.linalg.inv(np.eye(n_channels) - a_sum)

    def _compute_adtf(self, transfer: np.ndarray) -> np.ndarray:
        """Compute normalised squared transfer function (ADTF) from H.

        Parameters
        ----------
        transfer : np.ndarray
            Transfer matrix, shape (n_channels, n_channels [, n_freqs]).

        Returns
        -------
        np.ndarray
            ADTF matrix, same shape as input.
        """
        h_sq = np.abs(transfer) ** 2
        row_sums = h_sq.sum(axis=1, keepdims=True)
        # Avoid division by zero for channels with no inflow
        safe_sums = np.where(row_sums == 0, 1.0, row_sums)
        return np.where(row_sums == 0, 0.0, h_sq / safe_sums)

    def run(
        self,
        freqs: np.ndarray,
        *,
        maxlags: int = 20,
        integrate: bool = True,
    ) -> dict:
        """Fit VAR models and compute the ADTF matrix over the requested bands.

        Parameters
        ----------
        freqs : np.ndarray
            Normalized frequencies in [0, 0.5] to evaluate.
        maxlags : int
            Maximum VAR lag order. Default: 20.
        integrate : bool
            If True, return the mean ADTF over all frequencies as a single
            (n, n) matrix. If False, return per-trial, per-frequency results.

        Returns
        -------
        dict
            Keys: ``"matrix"`` (integrated) or ``"matrix_list"`` (non-integrated),
            ``"freqs"``, ``"type"``, ``"integrated"``.
        """
        self.models.clear()
        self.fit_var_model(maxlags=maxlags)

        if integrate:
            adtf_per_trial = []
            for model in self.models:
                h_stack = np.stack(
                    [self._compute_transfer_matrix(model.coefs, f) for f in freqs],
                    axis=-1,
                )
                adtf_per_trial.append(self._compute_adtf(h_stack))
            adtf_mean = np.mean(np.stack(adtf_per_trial, axis=0), axis=0)
            return {
                "matrix": np.mean(adtf_mean, axis=-1),
                "freqs": freqs,
                "type": "ADTF",
                "integrated": True,
            }

        matrix_list = []
        for model in self.models:
            h_list = [self._compute_transfer_matrix(model.coefs, f) for f in freqs]
            matrix_list.append([self._compute_adtf(h) for h in h_list])
        return {
            "matrix_list": matrix_list,
            "freqs": freqs,
            "type": "ADTF",
            "integrated": False,
        }


class FCMethods:
    """Dispatcher for all supported functional connectivity methods.

    Parameters
    ----------
    data : np.ndarray
        Input time-series, 2-D (n_channels, n_times) or 3-D
        (n_epochs, n_channels, n_times).
    fs : float
        Sampling frequency in Hz. Default: 1.0.
    """

    def __init__(self, data: np.ndarray, fs: float = 1.0) -> None:
        self.data = BasePreprocessor(data).data
        self.fs = fs

    def compute_all(self, methods_params: dict) -> dict:
        """Run every method listed in ``methods_params``.

        Parameters
        ----------
        methods_params : dict
            Mapping from method name (e.g. ``"PDC"``) to a dict of keyword
            arguments forwarded to the corresponding ``_{name}_func`` method.

        Returns
        -------
        dict
            Mapping from method name to result dict or error dict.
        """
        results = {}
        for name, params in methods_params.items():
            func = getattr(self, f"_{name.lower()}_func", None)
            if func is None:
                results[name] = {"error": f"Method {name!r} not implemented."}
                continue
            output = func(**params)
            results[name] = (
                output
                if isinstance(output, dict)
                else {"matrix": output, "type": name, "params": params}
            )
        return results

    def _corr_func(
        self,
        *,
        regularize: bool = False,
        normalize: bool = False,
        threshold: float | None = None,
    ) -> np.ndarray:
        """Compute epoch-averaged Pearson correlation matrix.

        Parameters
        ----------
        regularize : bool
            Apply Ledoit-Wolf-style ridge regularization. Default: False.
        normalize : bool
            Shift values to [0, 1] range. Default: False.
        threshold : float or None
            Zero out entries with |value| below this threshold.

        Returns
        -------
        np.ndarray
            Correlation matrix, shape (n_channels, n_channels).
        """
        n_epochs, n_channels, _ = self.data.shape
        if n_epochs == 1:
            corr = np.corrcoef(self.data[0])
        else:
            corr = np.mean(
                [np.corrcoef(self.data[epoch]) for epoch in range(n_epochs)], axis=0
            )

        if regularize:
            eigenvalues = np.sort(np.linalg.eigvals(corr).real)
            positive_eigs = eigenvalues[eigenvalues > 1e-10]
            eig_min = positive_eigs[0] if len(positive_eigs) > 0 else eigenvalues[-1]
            delta = max(0.0, (eigenvalues[-1] - 50 * eig_min) / 49)
            corr = (corr + delta * np.eye(n_channels)) / (1 + delta)

        if normalize:
            corr = (corr + 1) / 2

        if threshold is not None:
            corr[np.abs(corr) < threshold] = 0

        return corr

    def _adtf_func(
        self,
        *,
        fmin: float = 1.0,
        fmax: float = 40.0,
        n_freqs: int = 100,
        maxlags: int = 20,
        threshold: float | None = None,
        integrate: bool = True,
    ) -> dict:
        """Compute Adaptive Directed Transfer Function over a frequency band.

        Parameters
        ----------
        fmin, fmax : float
            Frequency band boundaries in Hz. Default: 1.0–40.0 Hz.
        n_freqs : int
            Number of frequency points to evaluate. Default: 100.
        maxlags : int
            Maximum VAR lag order. Default: 20.
        threshold : float or None
            Zero out matrix entries below this absolute value.
        integrate : bool
            Return a single (n, n) matrix averaged over the band if True.

        Returns
        -------
        dict
            Keys ``"matrix"`` (or ``"matrix_list"``), ``"freqs"``, ``"type"``,
            ``"integrated"``.
        """
        # Convert Hz to normalized frequency [0, 0.5] for the VAR model
        freqs_norm = np.linspace(fmin, fmax, n_freqs) / self.fs
        adtf = ADTFModel(self.data).run(freqs=freqs_norm, maxlags=maxlags, integrate=integrate)

        if threshold is not None:
            if "matrix" in adtf:
                adtf["matrix"][np.abs(adtf["matrix"]) < threshold] = 0
            elif "matrix_list" in adtf:
                adtf["matrix_list"] = [
                    [np.where(np.abs(m) < threshold, 0, m) for m in trial]
                    for trial in adtf["matrix_list"]
                ]
        return adtf

    def _mvarica_func(
        self,
        measure_name: str,
        *,
        model_order: int = 5,
        delta: float = 0,
        ica_params: dict | None = None,
        n_fft: int = 128,
        threshold: float | None = None,
        integrate: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """Shared MVARICA computation for PDC and DTF.

        Parameters
        ----------
        measure_name : str
            Either ``"pdc"`` or ``"dtf"``.
        model_order : int
            MVAR model order. Default: 5.
        delta : float
            Ridge regularization coefficient. Default: 0.
        ica_params : dict or None
            ICA settings. Built from ``kwargs`` if None.
        n_fft : int
            Number of FFT frequency bins. Default: 128.
        threshold : float or None
            Zero out entries with absolute value below this threshold.
        integrate : bool
            If True, average the frequency-resolved matrix over all bins.
        **kwargs
            Passed to ``prepare_ica_params`` (e.g. ``ica_method``).

        Returns
        -------
        np.ndarray
            Connectivity matrix, shape (n_channels, n_channels [, n_fft]).
        """
        preprocessor = BasePreprocessor(self.data)
        ica_params = preprocessor.prepare_ica_params(ica_params, **kwargs)
        var_model = MVAR(model_order=model_order, delta=delta)
        result = connectivity_mvarica(
            preprocessor.data,
            ica_params=ica_params,
            measure_name=measure_name,
            n_fft=n_fft,
            var_model=var_model,
        )
        if integrate:
            result = result.mean(axis=2)
        if threshold is not None:
            result[np.abs(result) < threshold] = 0
        return result

    def _pdc_func(self, **kwargs) -> np.ndarray:
        """Compute Partial Directed Coherence (PDC) via MVARICA.

        Keyword arguments are forwarded to ``_mvarica_func``. See its
        docstring for the full parameter list.

        Returns
        -------
        np.ndarray
            PDC matrix, shape (n_channels, n_channels [, n_fft]).
        """
        return self._mvarica_func("pdc", **kwargs)

    def _dtf_func(self, **kwargs) -> np.ndarray:
        """Compute Directed Transfer Function (DTF) via MVARICA.

        Keyword arguments are forwarded to ``_mvarica_func``. See its
        docstring for the full parameter list.

        Returns
        -------
        np.ndarray
            DTF matrix, shape (n_channels, n_channels [, n_fft]).
        """
        return self._mvarica_func("dtf", **kwargs)

    def _cgc_func(self, *, maxlag: int = 5, mean: bool = True) -> np.ndarray:
        """Compute epoch-averaged conditional Granger causality matrix.

        Uses VAR model residual variance reduction as the GC measure:
        GC(j→i) = log(var_i^{reduced} / var_i^{full}).

        Parameters
        ----------
        maxlag : int
            Maximum VAR lag for order selection. Default: 5.
        mean : bool
            Average across epochs if True, else return epoch-stacked array.

        Returns
        -------
        np.ndarray
            cGC matrix, shape (n_channels, n_channels) or
            (n_epochs, n_channels, n_channels) when ``mean=False``.
        """

        def _compute_cgc_matrix(data: np.ndarray, maxlag: int) -> np.ndarray:
            """Compute cGC for a single (n_times, n_channels) data slice."""
            n_times, n_nodes = data.shape
            cgc_matrix = np.full((n_nodes, n_nodes), np.nan)
            np.fill_diagonal(cgc_matrix, 0.0)

            try:
                res_full = VAR(data).fit(maxlags=maxlag)
                resid_full = res_full.resid
            except Exception:
                logger.warning("Full VAR fitting failed; returning NaN cGC matrix.")
                return cgc_matrix

            for i in range(n_nodes):
                for j in range(n_nodes):
                    if i == j:
                        continue
                    reduced_channels = [k for k in range(n_nodes) if k != j]
                    try:
                        res_reduced = VAR(data[:, reduced_channels]).fit(maxlags=maxlag)
                        new_i = reduced_channels.index(i)
                        var_reduced = np.var(res_reduced.resid[:, new_i])
                        var_full = np.var(resid_full[:, i])
                        if var_full > 0 and var_reduced > 0:
                            cgc_matrix[i, j] = np.log(var_reduced / var_full)
                    except Exception:
                        logger.debug(
                            "Reduced VAR fitting failed for target=%d, source=%d; skipping.",
                            i, j,
                        )

            return cgc_matrix

        n_epochs, n_nodes, n_samples = self.data.shape

        if n_epochs == 1:
            return _compute_cgc_matrix(self.data[0].T, maxlag)

        cgc_all = np.stack(
            [_compute_cgc_matrix(self.data[epoch].T, maxlag) for epoch in range(n_epochs)]
        )
        return np.nanmean(cgc_all, axis=0) if mean else cgc_all

    def _pli_func(
        self,
        *,
        fmin: float = 8.0,
        fmax: float = 13.0,
        mode: str = "multitaper",
        integrate: bool = True,
        **kwargs,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Compute Phase Lag Index (PLI) using MNE-Connectivity.

        Parameters
        ----------
        fmin, fmax : float
            Frequency band in Hz. Default: 8–13 Hz (alpha).
        mode : str
            Spectral estimation method (e.g. ``"multitaper"``). Default: ``"multitaper"``.
        integrate : bool
            If True, return the band-averaged (n, n) PLI matrix.

        Returns
        -------
        np.ndarray or tuple
            Band-averaged PLI matrix, or (PLI array, freqs) when integrate=False.
        """
        n_epochs, n_channels, _ = self.data.shape
        if n_epochs == 1:
            warnings.warn(
                "At least 2 epochs are required for PLI; single-epoch PLI is always 1 "
                "and not statistically meaningful.",
                RuntimeWarning,
                stacklevel=2,
            )

        channel_names = [f"Ch{i}" for i in range(n_channels)]
        con = spectral_connectivity_epochs(
            self.data,
            names=channel_names,
            method="pli",
            sfreq=self.fs,
            mode=mode,
            fmin=fmin,
            fmax=fmax,
            faverage=integrate,
            n_jobs=1,
            verbose=False,
            **kwargs,
        )

        if integrate:
            pli = con.get_data(output="dense")[:, :, 0]
            # MNE fills only the upper triangle; symmetrize for undirected PLI
            return pli + pli.T

        pli = con.get_data(output="dense")
        return pli + np.transpose(pli, (1, 0, 2)), con.freqs

    def _psi_func(
        self,
        *,
        fmin: float = 8.0,
        fmax: float = 13.0,
        mode: str = "multitaper",
        band_width: float = 2.0,
        integrate: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """Compute Phase Slope Index (PSI) for all channel pairs.

        PSI output convention: [source, target] (positive PSI[i, j] means i drives j).
        This is the only method that does NOT need to be transposed before
        comparison with ground-truth matrices.

        Parameters
        ----------
        fmin, fmax : float
            Frequency band in Hz. Default: 8–13 Hz.
        mode : str
            Spectral method. Default: ``"multitaper"``.
        band_width : float
            Multitaper bandwidth in Hz. Default: 2.0.
        integrate : bool
            Return frequency-integrated matrix if True (default).

        Returns
        -------
        np.ndarray
            PSI matrix, shape (n_channels, n_channels).
        """
        n_epochs, n_channels, _ = self.data.shape
        sources, targets = np.meshgrid(np.arange(n_channels), np.arange(n_channels))
        indices = (sources.ravel(), targets.ravel())

        psi_result = phase_slope_index(
            self.data,
            indices=indices,
            sfreq=self.fs,
            mode="multitaper",
            fmin=fmin,
            fmax=fmax,
            mt_bandwidth=band_width,
            mt_adaptive=True,
            mt_low_bias=True,
            verbose=False,
            **kwargs,
        )

        psi_values = psi_result.get_data()[:, 0]
        psi_matrix = np.zeros((n_channels, n_channels))
        for (src, tgt), val in zip(zip(*indices), psi_values):
            psi_matrix[src, tgt] = val

        return psi_matrix if integrate else psi_values

    def _te_func(self, setting: dict | None = None) -> dict:
        """Compute Multivariate Transfer Entropy using IDTxl.

        Requires IDTxl to be installed (``pip install idtxl``).

        Parameters
        ----------
        setting : dict or None
            IDTxl analysis settings. Uses Gaussian CMI estimator with lags
            1–5 by default.

        Returns
        -------
        dict
            Keys ``"type"`` and ``"result"`` (IDTxl Results object).
        """
        if not _IDTXL_AVAILABLE:
            raise ImportError("IDTxl is not installed. Install it to use TE estimation.")

        if setting is None:
            setting = {
                "cmi_estimator": "JidtGaussianCMI",
                "max_lag_sources": 5,
                "min_lag_sources": 1,
            }
        n_replications = self.data.shape[0]
        if n_replications == 1:
            data_obj = Data(self.data[0], dim_order="ps")
        else:
            data_obj = Data(
                np.transpose(self.data, (1, 2, 0)),
                dim_order="psr",
            )
        result = MultivariateTE().analyse_network(settings=setting, data=data_obj)
        return {"type": "TE", "result": result}

    def _mi_func(self, setting: dict | None = None) -> dict:
        """Compute Multivariate Mutual Information using IDTxl.

        Requires IDTxl to be installed (``pip install idtxl``).

        Parameters
        ----------
        setting : dict or None
            IDTxl analysis settings. Uses Gaussian CMI estimator with lags
            1–5 by default.

        Returns
        -------
        dict
            Keys ``"type"`` and ``"result"`` (IDTxl Results object).
        """
        if not _IDTXL_AVAILABLE:
            raise ImportError("IDTxl is not installed. Install it to use MI estimation.")

        if setting is None:
            setting = {
                "cmi_estimator": "JidtGaussianCMI",
                "max_lag_sources": 5,
                "min_lag_sources": 1,
            }
        n_replications = self.data.shape[0]
        if n_replications == 1:
            data_obj = Data(self.data[0], dim_order="ps")
        else:
            data_obj = Data(
                np.transpose(self.data, (1, 2, 0)),
                dim_order="psr",
            )
        result = MultivariateMI().analyse_network(settings=setting, data=data_obj)
        return {"type": "MI", "result": result}
