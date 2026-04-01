"""Synthetic time-series simulation models with known ground-truth connectivity.

Each model generates a ``(5, T)`` array representing five multivariate channels.
All models share the same channel count (``N_CHANNELS = 5``) and expose a single
``simulate()`` entry point for uniform access.

Ground-truth directed edges (source -> target, 1-indexed) per model:
    random      : 1->2 (delay 3), 1->3 (delay 2), 4->5 (delay 5)
    henon       : 1->2, 2->3, 4->5
    lorenz      : 1->2, 2->3, 3->4, 4->5
    sweep       : 1->2 (delay 2 samples), 1->3 (delay 4 samples)
    cascadear   : 1<->2<->3<->4<->5 (bidirectional chain)
    pinkarlin   : 1->2, 1->3, 1->4, 4<->5
    pinkarnonlin: same structure as pinkarlin (quadratic coupling)
    freqarlin   : 1->2 (gamma), 1->3 (gamma), 1->4 (alpha), 4->5 (gamma)
    freqarnonlin: same structure as freqarlin (quadratic coupling)
"""

import logging
from collections.abc import Callable

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import butter, filtfilt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

N_CHANNELS = 5  # number of channels produced by every model

# Hénon map parameters — classic chaotic regime (Hénon 1976)
HENON_A = 1.4
HENON_B = 0.3

# Lorenz system parameters — classic chaotic regime (Lorenz 1963)
LORENZ_SIGMA = 10.0
LORENZ_RHO = 28.0
LORENZ_BETA = 8.0 / 3.0

# AR model spectral-radius coefficient
AR_RHO = 0.95          # pole radius; keeps AR(2) stationary
AR_STABLE = AR_RHO**2  # = 0.9025, second AR(2) coefficient

# Frequency bands used by freq_ar (Hz)
ALPHA_LOW: float = 8.0
ALPHA_HIGH: float = 12.0
GAMMA_LOW: float = 25.0

# Safety margin so the high cutoff never reaches exactly Nyquist
NYQUIST_SAFETY: float = 0.99

# Seizure sweep: instantaneous frequency decreases from START to END (Hz)
SWEEP_F_START: float = 12.0
SWEEP_F_END: float = 8.0

# Ordered model keys used for integer-indexed access in simulate()
_MODEL_KEYS: list[str] = [
    "random",
    "henon",
    "lorenz",
    "sweep",
    "cascadear",
    "pinkarlin",
    "pinkarnonlin",
    "freqarlin",
    "freqarnonlin",
]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _resolve_rng(rng: np.random.Generator | None) -> np.random.Generator:
    """Return the given generator, or create a new unseeded one."""
    return rng if rng is not None else np.random.default_rng()


def _zscore(x: np.ndarray) -> np.ndarray:
    """Z-score normalize each row of a 2-D array in place.

    Parameters
    ----------
    x : np.ndarray
        Shape (n_channels, n_samples).

    Returns
    -------
    np.ndarray
        Row-wise z-scored copy of x.
    """
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    return (x - mean) / std


# ---------------------------------------------------------------------------
# Public helpers (used across models)
# ---------------------------------------------------------------------------


def generate_pink_noise(
    n_samples: int,
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate approximate pink (1/f) noise using power-law spectral scaling.

    Produces a zero-mean, unit-variance signal whose power spectrum decays as
    1/f. The approximation is achieved by scaling white-noise Fourier coefficients
    and inverting the transform.

    Parameters
    ----------
    n_samples : int
        Length of the output signal in samples.
    rng : np.random.Generator, optional
        Random generator for reproducibility. A new generator is created if None.

    Returns
    -------
    np.ndarray
        Shape (n_samples,), zero-mean and unit-variance.
    """
    rng = _resolve_rng(rng)
    n_freq = n_samples // 2 + 1 + (n_samples % 2)
    spectrum = (rng.standard_normal(n_freq) + 1j * rng.standard_normal(n_freq))
    spectrum *= np.sqrt(1.0 / np.arange(1, n_freq + 1))
    signal = np.fft.irfft(spectrum).real[:n_samples]
    signal -= signal.mean()
    return signal / signal.std()


def bandpass(
    data: np.ndarray,
    low: float,
    high: float,
    fs: float,
    *,
    order: int = 4,
) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass filter to a 1-D signal.

    The high cutoff is clamped to ``NYQUIST_SAFETY * fs/2`` to prevent filter
    instability at the Nyquist boundary.

    Parameters
    ----------
    data : np.ndarray
        1-D input signal.
    low : float
        Lower cutoff frequency in Hz.
    high : float
        Upper cutoff frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    order : int
        Butterworth filter order. Default: 4.

    Returns
    -------
    np.ndarray
        Filtered signal, same shape as data.

    Raises
    ------
    ValueError
        If low >= high after clamping the high cutoff.
    """
    nyquist = 0.5 * fs
    high = min(high, nyquist * NYQUIST_SAFETY)
    if low >= high:
        raise ValueError(
            f"Low cutoff {low} Hz must be less than high cutoff {high:.2f} Hz "
            f"(after clamping to {NYQUIST_SAFETY} * Nyquist)."
        )
    b, a = butter(order, [low / nyquist, high / nyquist], btype="band")
    return filtfilt(b, a, data)


# ---------------------------------------------------------------------------
# Simulation models
# ---------------------------------------------------------------------------


def random_system(
    T: int = 1000,
    *,
    c: float = 0.5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a 5-channel linear delay system with two independent sub-networks.

    Sub-network A (channels 1–3): channel 1 drives channels 2 and 3 with delays.
    Sub-network B (channels 4–5): channel 4 drives channel 5 with a delay.
    Ground-truth edges: 1->2 (delay 3), 1->3 (delay 2), 4->5 (delay 5).

    Parameters
    ----------
    T : int
        Number of time samples. Default: 1000.
    c : float
        Coupling strength in [0, 1]; weight given to the lagged source signal.
        Default: 0.5.
    rng : np.random.Generator, optional
        Random generator for reproducibility.

    Returns
    -------
    np.ndarray
        Shape (5, T), z-score normalized per channel.
    """
    rng = _resolve_rng(rng)
    # Propagation delays indexed by channel (0-based); 0 means no coupling
    delays = [0, 3, 2, 0, 5]
    x = np.zeros((N_CHANNELS, T))
    w = rng.standard_normal((N_CHANNELS, T))

    # Channels 1 and 4 (0-indexed: 0 and 3) are pure independent noise sources
    x[0] = w[0]
    x[3] = w[3]

    # Each driven channel is a weighted mixture of its own noise and a lagged source
    for target_idx, source_idx, delay_idx in [(1, 0, 1), (2, 0, 2), (4, 3, 4)]:
        d = delays[delay_idx]
        x[target_idx, :d] = (1 - c) * w[target_idx, :d]
        x[target_idx, d:] = (1 - c) * w[target_idx, d:] + c * x[source_idx, :-d]

    return _zscore(x)


def henon_system(
    T: int = 1000,
    *,
    c: float = 0.5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a 5-channel unidirectional chain of coupled Hénon maps.

    Channels 1 and 4 are autonomous Hénon oscillators. Coupled channels replace
    the self-squared term with a cross-channel product scaled by coupling strength c.
    Ground-truth edges: 1->2, 2->3, 4->5.

    Parameters
    ----------
    T : int
        Number of time samples. Default: 1000.
    c : float
        Coupling strength. At c=0 all channels are autonomous; at c=1 the target
        is fully replaced by cross-channel dynamics. Default: 0.5.
    rng : np.random.Generator, optional
        Random generator for reproducibility.

    Returns
    -------
    np.ndarray
        Shape (5, T), z-score normalized per channel.
    """
    rng = _resolve_rng(rng)
    x = np.zeros((N_CHANNELS, T))
    # Small random initialisation to avoid early-transient divergence
    x[:, :2] = rng.standard_normal((N_CHANNELS, 2)) * 0.1

    for t in range(2, T):
        # Autonomous Hénon map: x[n] = A - x[n-1]^2 + B * x[n-2]
        x[0, t] = HENON_A - x[0, t - 1] ** 2 + HENON_B * x[0, t - 2]
        x[3, t] = HENON_A - x[3, t - 1] ** 2 + HENON_B * x[3, t - 2]

        # Coupled channels: x_i[n-1]^2 term is replaced by c * x_{i-1}[n-1] * x_i[n-1]
        x[1, t] = HENON_A - c * x[0, t - 1] * x[1, t - 1] + HENON_B * x[1, t - 2]
        x[2, t] = HENON_A - c * x[1, t - 1] * x[2, t - 1] + HENON_B * x[2, t - 2]
        x[4, t] = HENON_A - c * x[3, t - 1] * x[4, t - 1] + HENON_B * x[4, t - 2]

    return _zscore(x)


def _lorenz_ode(
    t: float,
    y: np.ndarray,
    n_systems: int,
    c: float,
) -> np.ndarray:
    """ODE right-hand side for N unidirectionally coupled Lorenz oscillators.

    Internal helper passed to ``scipy.integrate.solve_ivp``.
    State layout: [x1, y1, z1, x2, y2, z2, ..., xN, yN, zN].

    Parameters
    ----------
    t : float
        Current time (required by solve_ivp interface; unused internally).
    y : np.ndarray
        State vector of length 3 * n_systems.
    n_systems : int
        Number of Lorenz oscillators in the chain.
    c : float
        Unidirectional coupling strength from x_{i-1} to x_i.

    Returns
    -------
    np.ndarray
        Derivative vector, same shape as y.
    """
    dydt = np.zeros_like(y)
    for i in range(n_systems):
        xi, yi, zi = y[3 * i : 3 * (i + 1)]
        dx = LORENZ_SIGMA * (yi - xi)
        dy = LORENZ_RHO * xi - yi - xi * zi
        dz = xi * yi - LORENZ_BETA * zi

        # Each oscillator (except the first) is driven by the x-component of its predecessor
        if i > 0:
            x_prev = y[3 * (i - 1)]
            dx += c * (x_prev - xi)

        dydt[3 * i : 3 * (i + 1)] = [dx, dy, dz]

    return dydt


def lorenz_system(
    T: int = 1000,
    *,
    sampling_period: float = 0.5,
    dt: float = 0.01,
    c: float = 4.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a 5-channel unidirectional chain of coupled Lorenz oscillators.

    The continuous-time system is integrated with RK45 at step dt and subsampled
    to T points separated by sampling_period.
    Ground-truth edges: 1->2, 2->3, 3->4, 4->5.

    Parameters
    ----------
    T : int
        Number of output time samples. Default: 1000.
    sampling_period : float
        Physical time (seconds) between consecutive output samples. Default: 0.5.
    dt : float
        Integration step size (seconds). Default: 0.01.
    c : float
        Coupling strength between adjacent oscillators. Default: 4.0.
    rng : np.random.Generator, optional
        Random generator for initial conditions.

    Returns
    -------
    np.ndarray
        Shape (5, T), z-score normalized x-components of each oscillator.
    """
    rng = _resolve_rng(rng)
    total_steps = T * int(sampling_period / dt)
    t_span = (0.0, total_steps * dt)
    t_eval = np.arange(0, total_steps * dt, sampling_period)

    y0 = rng.random(3 * N_CHANNELS) * 5.0
    sol = solve_ivp(
        _lorenz_ode,
        t_span,
        y0,
        method="RK45",
        t_eval=t_eval,
        args=(N_CHANNELS, c),
        rtol=1e-6,
        atol=1e-9,
    )

    # Extract x-components (every 3rd state variable) and trim to exactly T samples
    x = sol.y[::3, :T]
    return _zscore(x)


def seizure_sweep(
    T: int = 1000,
    *,
    fs: float = 256.0,
    snr_db: float = -5.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a 5-channel seizure propagation model with a frequency-sweeping source.

    Channel 1 carries a seizure oscillation that sweeps from SWEEP_F_START to
    SWEEP_F_END Hz, embedded in pink noise at the requested SNR. The seizure
    propagates to channels 2 and 3 with fixed sample delays. Channels 4 and 5
    are independent pink noise.
    Ground-truth edges: 1->2 (delay 2 samples), 1->3 (delay 4 samples).

    Parameters
    ----------
    T : int
        Number of time samples. Default: 1000.
    fs : float
        Sampling frequency in Hz. Default: 256.0.
    snr_db : float
        Signal-to-noise ratio in dB (negative = noise-dominated). Default: -5.
    rng : np.random.Generator, optional
        Random generator for reproducibility.

    Returns
    -------
    np.ndarray
        Shape (5, T), z-score normalized per channel.
    """
    rng = _resolve_rng(rng)
    time_vec = np.arange(T) / fs

    # Instantaneous frequency decreases linearly from SWEEP_F_START to SWEEP_F_END
    inst_freq = SWEEP_F_START + (SWEEP_F_END - SWEEP_F_START) * (time_vec / time_vec[-1])
    phase = 2 * np.pi * np.cumsum(inst_freq) / fs
    seizure = np.sin(phase)

    # Scale noise so that signal power / noise power equals 10^(snr_db/10)
    noises = np.array([generate_pink_noise(T, rng=rng) for _ in range(N_CHANNELS)])
    signal_power = np.mean(seizure**2)
    noise_power = np.mean(noises[0] ** 2)
    noise_scale = np.sqrt(signal_power / (10 ** (snr_db / 10)) / noise_power)
    noises *= noise_scale

    x = np.zeros((N_CHANNELS, T))
    x[0] = seizure + noises[0]

    # Propagation: channel 1 drives channels 2 and 3 with 2- and 4-sample delays
    x[1] = noises[1]
    x[2] = noises[2]
    x[1, 2:] += x[0, :-2]
    x[2, 4:] += x[0, :-4]

    # Channels 4 and 5: uncoupled, pure noise
    x[3] = noises[3]
    x[4] = noises[4]

    return _zscore(x)


def cascade_ar(
    T: int = 1000,
    *,
    c: float = 0.8,
    rho: float = 0.9,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a 5-channel bidirectional AR cascade model.

    Interior channels are each driven by their two immediate neighbours plus own
    AR(2) dynamics. The AR(2) coefficients are derived from spectral radius rho
    (poles at rho * e^(±j*pi/4)) to produce oscillatory but stationary dynamics.
    Ground-truth edges: 1<->2<->3<->4<->5.

    Parameters
    ----------
    T : int
        Number of time samples. Default: 1000.
    c : float
        Coupling strength in [0, 1]. At c=0 all channels are independent AR(2).
        Default: 0.8.
    rho : float
        AR spectral radius in (0, 1); controls oscillation frequency. Default: 0.9.
    rng : np.random.Generator, optional
        Random generator for reproducibility.

    Returns
    -------
    np.ndarray
        Shape (5, T), z-score normalized per channel.
    """
    rng = _resolve_rng(rng)
    theta = np.array([generate_pink_noise(T, rng=rng) for _ in range(N_CHANNELS)])
    x = np.zeros((N_CHANNELS, T))

    # AR(2) coefficients: x[t] = ar1*x[t-1] - ar2*x[t-2] gives poles at rho*e^(±j*pi/4)
    ar1_coeff = np.sqrt(2) * rho
    ar2_coeff = rho**2

    for t in range(2, T):
        # Channels 1 and 5: autonomous AR(2) processes (no neighbours)
        x[0, t] = ar1_coeff * x[0, t - 1] - ar2_coeff * x[0, t - 2] + theta[0, t]
        x[4, t] = ar1_coeff * x[4, t - 1] - ar2_coeff * x[4, t - 2] + theta[4, t]

        # Interior channels: equally weighted mixture of both neighbours and own AR(2)
        x[1, t] = (
            0.5 * c * (x[0, t - 1] + x[2, t - 1])
            + (1 - c) * (ar1_coeff * x[1, t - 1] - ar2_coeff * x[1, t - 2])
            + theta[1, t]
        )
        x[2, t] = (
            0.5 * c * (x[1, t - 1] + x[3, t - 1])
            + (1 - c) * (ar1_coeff * x[2, t - 1] - ar2_coeff * x[2, t - 2])
            + theta[2, t]
        )
        x[3, t] = (
            0.5 * c * (x[2, t - 1] + x[4, t - 1])
            + (1 - c) * (ar1_coeff * x[3, t - 1] - ar2_coeff * x[3, t - 2])
            + theta[3, t]
        )

    return _zscore(x)


def pink_ar(
    T: int = 1000,
    *,
    nonlinear: bool = False,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a 5-channel pink-noise-driven AR model.

    Channel 1 is an autonomous AR(2) oscillator. It drives channels 2, 3, and 4
    via lagged linear (or quadratic) couplings. Channels 4 and 5 have a
    bidirectional AR(1) interaction.
    Ground-truth edges: 1->2, 1->3, 1->4, 4<->5.

    Parameters
    ----------
    T : int
        Number of time samples. Default: 1000.
    nonlinear : bool
        If True, coupling from channel 1 to channels 2 and 4 uses the squared
        signal (quadratic interaction). Default: False.
    rng : np.random.Generator, optional
        Random generator for reproducibility.

    Returns
    -------
    np.ndarray
        Shape (5, T), z-score normalized per channel.
    """
    rng = _resolve_rng(rng)
    theta = np.array([generate_pink_noise(T, rng=rng) for _ in range(N_CHANNELS)])
    x = np.zeros((N_CHANNELS, T))

    # AR(2) coefficient for channel 1: spectral radius AR_RHO at lag 2
    ar2_coeff = AR_RHO * np.sqrt(2)
    # AR(1) weight used for the bidirectional channel-4/5 interaction
    ar1_weight = 0.25 * np.sqrt(2)

    for t in range(3, T):
        x[0, t] = ar2_coeff * x[0, t - 2] + theta[0, t]

        # Linear or quadratic driver from channel 1 at lag 2
        driver_1 = x[0, t - 2] ** 2 if nonlinear else x[0, t - 2]
        x[1, t] = 0.5 * driver_1 + theta[1, t]

        # Channel 3: always linear coupling at lag 3
        x[2, t] = -0.4 * x[0, t - 3] + theta[2, t]

        # Channel 4: linear/quadratic driver from channel 1 plus AR(1) interaction with 5
        driver_4 = x[0, t - 2] ** 2 if nonlinear else x[0, t - 2]
        x[3, t] = (
            -0.5 * driver_4
            + ar1_weight * x[3, t - 1]
            + ar1_weight * x[4, t - 1]
            + theta[3, t]
        )

        # Channel 5: AR(1) interaction with channel 4 (bidirectional)
        x[4, t] = (
            -ar1_weight * x[3, t - 1]
            + ar1_weight * x[4, t - 1]
            + theta[4, t]
        )

    return _zscore(x)


def freq_ar(
    T: int = 1000,
    *,
    fs: float = 256.0,
    nonlinear: bool = False,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a 5-channel frequency-band-dependent AR model.

    Connectivity is carried selectively in alpha (8–12 Hz) and gamma (25+ Hz)
    bands: channels 2 and 3 receive gamma-band input from channel 1, channel 4
    receives alpha-band input from channel 1, and channel 5 receives gamma-band
    input from channel 4.
    Ground-truth edges: 1->2 (γ), 1->3 (γ), 1->4 (α), 4->5 (γ).

    Parameters
    ----------
    T : int
        Number of time samples. Default: 1000.
    fs : float
        Sampling frequency in Hz. Default: 256.0.
    nonlinear : bool
        If True, apply quadratic coupling for 1->2 and 1->4. Default: False.
    rng : np.random.Generator, optional
        Random generator for reproducibility.

    Returns
    -------
    np.ndarray
        Shape (5, T), z-score normalized per channel.
    """
    rng = _resolve_rng(rng)
    theta = np.array([generate_pink_noise(T, rng=rng) for _ in range(N_CHANNELS)])

    # Bandpass-filter noise to isolate frequency-specific coupling pathways
    gamma_high = fs / 2 * NYQUIST_SAFETY
    theta_alpha = np.array([bandpass(sig, ALPHA_LOW, ALPHA_HIGH, fs) for sig in theta])
    theta_gamma = np.array([bandpass(sig, GAMMA_LOW, gamma_high, fs) for sig in theta])

    x = np.zeros((N_CHANNELS, T))
    # AR(2) coefficients matching pink_ar for channel 1
    ar1_coeff = AR_RHO * np.sqrt(2)
    ar1_weight = 0.25 * np.sqrt(2)

    for t in range(6, T):
        # Channel 1: AR(2) with spectral radius AR_RHO driven by broadband pink noise
        x[0, t] = ar1_coeff * x[0, t - 1] - AR_STABLE * x[0, t - 2] + theta[0, t]

        # Channel 2: driven by gamma-band of channel 1 at lag 2
        gamma_driver_1 = theta_gamma[0, t - 2] ** 2 if nonlinear else theta_gamma[0, t - 2]
        x[1, t] = 0.5 * gamma_driver_1 + theta[1, t]

        # Channel 3: driven by gamma-band of channels 1 and 2 at lag 3
        x[2, t] = (
            -0.4 * theta_gamma[0, t - 3]
            + ar1_weight * theta_gamma[1, t - 3]
            + theta[2, t]
        )

        # Channel 4: driven by alpha-band of channel 1 at lag 5, plus AR(1) with channel 5
        alpha_driver_1 = theta_alpha[0, t - 5] ** 2 if nonlinear else theta_alpha[0, t - 5]
        x[3, t] = (
            -0.5 * alpha_driver_1
            + ar1_weight * x[3, t - 1]
            + ar1_weight * x[4, t - 1]
            + theta[3, t]
        )

        # Channel 5: driven by gamma-band of channel 4 at lag 1, plus AR(1) self
        x[4, t] = (
            -ar1_weight * theta_gamma[3, t - 1]
            + ar1_weight * x[4, t - 1]
            + theta[4, t]
        )

    return _zscore(x)


# ---------------------------------------------------------------------------
# Model registry and unified entry point
# ---------------------------------------------------------------------------

_MODEL_REGISTRY: dict[str, Callable[..., np.ndarray]] = {
    "random": random_system,
    "henon": henon_system,
    "lorenz": lorenz_system,
    "sweep": seizure_sweep,
    "cascadear": cascade_ar,
    "pinkarlin": lambda T, **kw: pink_ar(T, nonlinear=False, **kw),
    "pinkarnonlin": lambda T, **kw: pink_ar(T, nonlinear=True, **kw),
    "freqarlin": lambda T, **kw: freq_ar(T, nonlinear=False, **kw),
    "freqarnonlin": lambda T, **kw: freq_ar(T, nonlinear=True, **kw),
}

VALID_MODELS: list[str] = list(_MODEL_REGISTRY)


def simulate(
    model: str | int,
    T: int = 1000,
    *,
    seed: int | None = None,
    **kwargs,
) -> np.ndarray:
    """Run a named or index-addressed simulation model.

    All randomness is routed through a single ``numpy.random.Generator`` created
    from ``seed``, so results are fully reproducible without mutating global RNG state.

    Parameters
    ----------
    model : str or int
        Model name (case-insensitive) or integer index 0–8. Valid names:
        ``random``, ``henon``, ``lorenz``, ``sweep``, ``cascadear``,
        ``pinkarlin``, ``pinkarnonlin``, ``freqarlin``, ``freqarnonlin``.
    T : int
        Number of time samples to generate. Must be a positive integer.
        Default: 1000.
    seed : int, optional
        Random seed for reproducibility. Does not affect global ``numpy`` RNG state.
    **kwargs
        Additional keyword arguments forwarded to the chosen model function
        (e.g. ``c``, ``rho``, ``snr_db``, ``fs``).

    Returns
    -------
    np.ndarray
        Simulated data of shape ``(5, T)``, z-score normalized per channel.

    Raises
    ------
    ValueError
        If ``model`` is an unrecognised name, an out-of-range index, or T <= 0.
    TypeError
        If ``model`` is neither a string nor an integer.

    Examples
    --------
    >>> x = simulate("random", T=2000, seed=42)
    >>> x.shape
    (5, 2000)
    >>> x = simulate(2, T=500, seed=0)  # lorenz by index
    >>> x.shape
    (5, 500)
    """
    if not isinstance(T, int) or T <= 0:
        raise ValueError(f"T must be a positive integer, got {T!r}.")

    if isinstance(model, str):
        model_key = model.lower()
        if model_key not in _MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model {model!r}. Valid names: {VALID_MODELS}"
            )
        func = _MODEL_REGISTRY[model_key]
    elif isinstance(model, int):
        if not (0 <= model < len(_MODEL_KEYS)):
            raise ValueError(
                f"Model index {model} out of range [0, {len(_MODEL_KEYS) - 1}]."
            )
        func = _MODEL_REGISTRY[_MODEL_KEYS[model]]
    else:
        raise TypeError(
            f"model must be str or int, got {type(model).__name__!r}."
        )

    # Isolated generator: seeding here does not pollute numpy's global RNG
    rng = np.random.default_rng(seed)
    logger.debug("Simulating model=%r T=%d seed=%s", model, T, seed)
    return func(T, rng=rng, **kwargs)
