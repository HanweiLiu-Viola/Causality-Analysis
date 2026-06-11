"""MVARICA pipeline for directed brain connectivity estimation.

Implements Multivariate Vector Autoregressive Independent Component Analysis
(MVARICA): fits an MVAR model, applies ICA to the residuals, transforms
coefficients to source space, and computes PDC or DTF in the frequency domain.

References
----------
Baccalá & Sameshima (2001). Partial directed coherence. Biol. Cybern., 84, 463–474.
Kaminski & Blinowska (1991). A new method of information flow description in
    brain structures. Biol. Cybern., 65, 203–210.
"""

import logging

import numpy as np
import scipy.linalg
from numpy.fft import fft

logger = logging.getLogger(__name__)

# An MVAR model is stable when all eigenvalues of the companion matrix
# have modulus strictly less than 1 (ensures stationarity).
_STABILITY_THRESHOLD: float = 1.0


class MVAR:
    """Multivariate Vector Autoregressive (MVAR) model.

    Fits, predicts with, and checks the stability of an MVAR model.
    MVAR models capture how past values of all channels jointly predict the
    current value of each channel, enabling directed connectivity analysis.

    Parameters
    ----------
    model_order : int
        Number of past time lags included in the model.
    fitting_method : str or object
        ``"default"`` uses least-squares (or ridge regression when delta > 0).
        A custom object must implement ``fit(x, y)`` and expose a ``coef``
        attribute.
    delta : float
        Ridge penalty for regularization. ``0`` means no regularization.

    Attributes
    ----------
    order : int
        Model order (number of lags).
    coeff : np.ndarray
        Estimated coefficients, shape (n_channels, n_channels * order).
    residuals : np.ndarray
        Model residuals, same shape as the input signal.

    Examples
    --------
    >>> data = np.random.randn(2, 3, 1000)   # 2 epochs, 3 channels, 1000 samples
    >>> model = MVAR(model_order=5, delta=0.1)
    >>> model.fit(data)
    >>> model.stability()
    True
    """

    def __init__(
        self,
        model_order: int,
        fitting_method: str | object = "default",
        delta: float = 0,
    ) -> None:
        self.order = model_order
        self.fit_method = fitting_method
        self.fitting = None
        self.coeff: np.ndarray = np.asarray([])
        self.residuals: np.ndarray = np.asarray([])
        self.delta = delta

    def copy(self) -> "MVAR":
        """Return a deep copy of this model instance.

        Returns
        -------
        MVAR
            New instance with the same order, coefficients, and residuals.
        """
        mvar_copy = self.__class__(self.order)
        mvar_copy.coeff = self.coeff.copy()
        mvar_copy.residuals = self.residuals.copy()
        return mvar_copy

    def predict(self, signal: np.ndarray) -> np.ndarray:
        """Predict time series values using the fitted MVAR coefficients.

        Predictions start from the (model_order)-th time point because earlier
        points have insufficient history.

        Parameters
        ----------
        signal : np.ndarray
            Input signal, shape (n_epochs, n_channels, n_samples).

        Returns
        -------
        np.ndarray
            Predicted signal, same shape as input.
        """
        n_epochs, n_channels, n_samples = signal.shape
        p = self.coeff.shape[1] // n_channels
        predicted = np.zeros_like(signal)

        if n_epochs > n_samples - n_channels:
            for lag in range(1, p + 1):
                lag_coeff = self.coeff[:, (lag - 1)::p]
                for t in range(p, n_samples):
                    predicted[:, :, t] += np.dot(signal[:, :, t - lag], lag_coeff.T)
        else:
            for lag in range(1, p + 1):
                lag_coeff = self.coeff[:, (lag - 1)::p]
                for epoch in range(n_epochs):
                    predicted[epoch, :, p:] += np.dot(
                        lag_coeff, signal[epoch, :, (p - lag) : (n_samples - lag)]
                    )
        return predicted

    def stability(self) -> bool:
        """Check that all eigenvalues of the companion matrix have modulus < 1.

        Stability is required for the MVAR to represent a stationary process.
        Unstable models produce physiologically implausible, diverging signals.

        Returns
        -------
        bool
            ``True`` if the model is stable.
        """
        n_channels, n_coeff = self.coeff.shape
        p = n_coeff // n_channels
        assert n_coeff == n_channels * p, "Coefficient matrix shape is inconsistent."

        # Build the companion matrix: top block = AR coefficients, lower = identity shift
        top_block = np.hstack([self.coeff[:, i::p] for i in range(p)])
        identity_block = np.eye(n_channels * (p - 1))
        zero_pad = np.zeros((n_channels * (p - 1), n_channels))
        companion = np.vstack([top_block, np.hstack([identity_block, zero_pad])])

        eigenvalues = np.linalg.eigvals(companion)
        return bool(np.all(np.abs(eigenvalues) < _STABILITY_THRESHOLD))

    def construct_equation(
        self,
        signal: np.ndarray,
        delta_1: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build the design matrix X and target matrix Y for least-squares fitting.

        Rearranges lagged signal values into a regression design. When
        ``delta_1`` is provided, ridge-regularization rows are appended.

        Parameters
        ----------
        signal : np.ndarray
            Input signal, shape (n_epochs, n_channels, n_samples).
        delta_1 : float or None
            Ridge penalty to append as regularization rows.

        Returns
        -------
        x : np.ndarray
            Design matrix, shape (n_obs [+ n_reg], n_channels * order).
        y : np.ndarray
            Target matrix, shape (n_obs [+ n_reg], n_channels).
        """
        p = self.order
        n_epochs, n_channels, n_samples = signal.shape
        n_obs = (n_samples - p) * n_epochs
        n_rows = n_obs if delta_1 is None else n_obs + n_channels * p

        x = np.zeros((n_rows, n_channels * p))
        for ch in range(n_channels):
            for lag in range(1, p + 1):
                x[:n_obs, ch * p + lag - 1] = np.reshape(
                    signal[:, ch, p - lag : -lag].T, n_obs
                )
        if delta_1 is not None:
            np.fill_diagonal(x[n_obs:, :], delta_1)

        y = np.zeros((n_rows, n_channels))
        for ch in range(n_channels):
            y[:n_obs, ch] = np.reshape(signal[:, ch, p:].T, n_obs)

        return x, y

    def fit(self, signal: np.ndarray) -> "MVAR":
        """Estimate MVAR coefficients from the input signal.

        Uses ordinary least squares when ``delta == 0``, ridge regression
        when ``delta > 0``, or a custom sklearn-compatible estimator when
        ``fitting_method`` is an object.

        Parameters
        ----------
        signal : np.ndarray
            Input signal, shape (n_epochs, n_channels, n_samples).

        Returns
        -------
        MVAR
            This instance (for method chaining).
        """
        is_default = isinstance(self.fit_method, str) and self.fit_method.lower() == "default"

        if is_default:
            if self.delta == 0 or self.delta is None:
                x, y = self.construct_equation(signal)
            else:
                x, y = self.construct_equation(signal, self.delta)
            coeff, _residuals, _rank, _sv = scipy.linalg.lstsq(x, y)
            self.coeff = coeff.T
        else:
            x, y = self.construct_equation(signal)
            self.fitting = self.fit_method.fit(x, y)
            self.coeff = self.fitting.coef

        self.residuals = signal - self.predict(signal)
        return self


def ica_wrapper(
    ica_input: np.ndarray,
    ica_method: str = "infomax_extended",
    random_state: int | None = None,
) -> np.ndarray:
    """Apply ICA to the input data and return the unmixing matrix.

    Serves as a unified interface to Extended Infomax (MNE), standard Infomax
    (MNE), and FastICA (scikit-learn).

    Parameters
    ----------
    ica_input : np.ndarray
        Shape (n_samples, n_features). Observations in rows, variables in columns.
    ica_method : str
        ICA algorithm. One of ``"infomax_extended"``, ``"infomax"``,
        ``"fastica"``.
    random_state : int or None
        Seed for the ICA random initialisation.

    Returns
    -------
    np.ndarray
        Unmixing matrix, shape (n_features, n_features).

    Raises
    ------
    ValueError
        If ``ica_method`` is not one of the supported options.
    """
    method_lower = ica_method.lower()
    if method_lower in ("infomax_extended", "infomax"):
        from mne.preprocessing.infomax_ import infomax
        return infomax(
            ica_input,
            extended=(method_lower == "infomax_extended"),
            random_state=random_state,
        )
    if method_lower == "fastica":
        from sklearn.decomposition import FastICA
        estimator = FastICA(random_state=random_state)
        estimator.fit(ica_input)
        return estimator.components_

    raise ValueError(
        f"Unsupported ICA method {ica_method!r}. "
        "Valid options: 'infomax_extended', 'infomax', 'fastica'."
    )


def connectivity_mvarica(
    real_signal: np.ndarray,
    ica_params: dict,
    measure_name: str,
    n_fft: int = 512,
    var_model: MVAR = MVAR,
) -> np.ndarray:
    """Estimate directed connectivity using the MVARICA pipeline.

    Steps
    -----
    1. Fit an MVAR model to the input signals.
    2. Extract residuals (model innovations).
    3. Apply ICA to the residuals to estimate the source unmixing matrix.
    4. Transform MVAR coefficients to the ICA source space.
    5. Compute PDC or DTF from the spectral representation.

    Parameters
    ----------
    real_signal : np.ndarray
        Input data, shape (n_epochs, n_channels, n_samples).
    ica_params : dict
        ICA settings with keys ``"method"`` (str) and ``"random_state"`` (int | None).
    measure_name : str
        Connectivity measure to return. One of ``"mvar_spectral"``,
        ``"mvar_tf"``, ``"pdc"``, ``"dtf"``.
    n_fft : int
        Number of frequency bins. Default: 512.
    var_model : MVAR
        Pre-initialised (but unfitted) MVAR model instance.

    Returns
    -------
    np.ndarray
        Connectivity matrix, shape (n_channels, n_channels, n_fft).
        Entry ``[i, j, f]`` represents connectivity from channel j to channel i
        at frequency bin f.

    Notes
    -----
    The transfer function is computed via:
        A(f) = FFT([I, -A_1, -A_2, ..., -A_p])
        H(f) = A(f)^{-1}
    where A_k are the MVAR coefficient matrices in source space.
    """
    fitted_model = var_model.fit(real_signal)
    residuals = real_signal - var_model.predict(real_signal)

    # Concatenate epochs along time axis for ICA
    residuals_cat = np.concatenate(
        np.split(residuals, residuals.shape[0], axis=0), axis=2
    ).squeeze(0)
    unmixing_matrix = ica_wrapper(
        residuals_cat.T,
        ica_method=ica_params["method"],
        random_state=ica_params["random_state"],
    ).T
    mixing_matrix = scipy.linalg.pinv(unmixing_matrix)

    # Project residuals into source space
    source_residuals = np.concatenate([
        (unmixing_matrix.T @ residuals[epoch])[np.newaxis]
        for epoch in range(residuals.shape[0])
    ])

    # Transform MVAR coefficients to source space
    source_model = fitted_model.copy()
    for lag in range(fitted_model.order):
        source_model.coeff[:, lag :: fitted_model.order] = (
            mixing_matrix
            @ fitted_model.coeff[:, lag :: fitted_model.order].T
            @ unmixing_matrix
        ).T

    noise_cov = np.cov(
        np.concatenate(
            np.split(source_residuals, source_residuals.shape[0], axis=0), axis=2
        ).squeeze(0).T,
        rowvar=False,
    )

    # Reshape coefficients to (n_channels, n_channels, model_order) for FFT
    coeffs = np.asarray(source_model.coeff)
    n_channels = coeffs.shape[0]
    model_order = coeffs.shape[1] // n_channels
    coeffs_reshaped = np.reshape(coeffs, (n_channels, n_channels, model_order), order="C")

    # Spectral representation: FFT of [I, -A_1, ..., -A_p] over (2*n_fft-1) points,
    # then keep only the first n_fft bins (causal / positive-frequency part).
    spectral_a = fft(
        np.dstack([np.eye(n_channels), -coeffs_reshaped]),
        n_fft * 2 - 1,
    )[:, :, :n_fft]

    # Transfer function H = A^{-1} computed per frequency bin
    transfer_h = np.array([
        np.linalg.solve(a_f, np.eye(n_channels)) for a_f in spectral_a.T
    ]).T

    name = measure_name.lower()
    if name == "mvar_spectral":
        return spectral_a
    if name == "mvar_tf":
        return transfer_h
    if name == "pdc":
        # PDC: normalised by total outflow from each source
        return np.abs(
            spectral_a / np.sqrt(np.sum(spectral_a.conj() * spectral_a, axis=0, keepdims=True))
        )
    if name == "dtf":
        # DTF: normalised by total inflow to each target
        return np.abs(
            transfer_h / np.sqrt(np.sum(transfer_h * transfer_h.conj(), axis=1, keepdims=True))
        )

    raise ValueError(
        f"Unknown measure {measure_name!r}. "
        "Valid options: 'mvar_spectral', 'mvar_tf', 'pdc', 'dtf'."
    )


def connectivity_mvar(
    real_signal: np.ndarray,
    measure_name: str,
    n_fft: int = 512,
    var_model: MVAR = MVAR,
) -> np.ndarray:
    """Estimate PDC or DTF directly in the observed channel space.

    This path should be used when outputs are compared against ground-truth
    adjacency matrices defined over the original simulated channels. MVARICA
    rotates the model into ICA source space, so its component indices do not
    necessarily correspond to the original channel labels.
    """
    fitted_model = var_model.fit(real_signal)

    coeffs = np.asarray(fitted_model.coeff)
    n_channels = coeffs.shape[0]
    model_order = coeffs.shape[1] // n_channels
    coeffs_reshaped = np.reshape(coeffs, (n_channels, n_channels, model_order), order="C")

    spectral_a = fft(
        np.dstack([np.eye(n_channels), -coeffs_reshaped]),
        n_fft * 2 - 1,
    )[:, :, :n_fft]
    transfer_h = np.array([
        np.linalg.solve(a_f, np.eye(n_channels)) for a_f in spectral_a.T
    ]).T

    name = measure_name.lower()
    if name == "mvar_spectral":
        return spectral_a
    if name == "mvar_tf":
        return transfer_h
    if name == "pdc":
        denom = np.sqrt(np.sum(spectral_a.conj() * spectral_a, axis=0, keepdims=True))
        return np.abs(spectral_a / denom)
    if name == "dtf":
        denom = np.sqrt(np.sum(transfer_h * transfer_h.conj(), axis=1, keepdims=True))
        return np.abs(transfer_h / denom)

    raise ValueError(
        f"Unknown measure {measure_name!r}. "
        "Valid options: 'mvar_spectral', 'mvar_tf', 'pdc', 'dtf'."
    )
