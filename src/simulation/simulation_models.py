import numpy as np
from scipy.signal import butter, filtfilt
from scipy.integrate import solve_ivp
# from numba import njit


def generate_pink_noise(N):
    """Approximate pink noise using power-law scaling."""
    uneven = N % 2
    X = np.random.randn(N // 2 + 1 + uneven) + 1j * np.random.randn(N // 2 + 1 + uneven)
    S = np.sqrt(1.0 / np.arange(1, len(X) + 1))
    y = (np.fft.irfft(X * S)).real
    y = y - np.mean(y)
    return y / np.std(y)

def bandpass(data, low, high, fs, order=4):
    nyq = 0.5 * fs
    high = min(high, nyq * 0.99)
    if low >= high:
        raise ValueError("Low cutoff must be smaller than high cutoff!")
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, data)

def zscore_normalize(x):
    return (x - np.mean(x, axis=1, keepdims=True)) / np.std(x, axis=1, keepdims=True)


# ============== 每个模型的生成器 ==================

def random_system(T=1000, c=0.5):

  # Initialize
  N=5
  x = np.zeros((N, T))
  w = np.random.randn(N, T)
  delays = [0, 3, 2, 0, 5]

  # x1 and x4: pure noise
  x[0] = w[0]
  x[3] = w[3]

  # x2: depends on x1 with delay 3
  d = delays[1]
  x[1, :d] = (1 - c) * w[1, :d]
  x[1, d:] = (1 - c) * w[1, d:] + c * x[0, :-d]

  # x3: depends on x1 with delay 2
  d = delays[2]
  x[2, :d] = (1 - c) * w[2, :d]
  x[2, d:] = (1 - c) * w[2, d:] + c * x[0, :-d]

  # x5: depends on x4 with delay 5
  d = delays[4]
  x[4, :d] = (1 - c) * w[4, :d]
  x[4, d:] = (1 - c) * w[4, d:] + c * x[3, :-d]

  x = zscore_normalize(x)
  return x

def henon_system(T=1000, c=0.5):
    N = 5
    x = np.zeros((N, T))
    x[:, :2] = np.random.randn(N, 2) * 0.1
    for t in range(2, T):
        x1_1, x1_2 = x[0, t-1], x[0, t-2]
        x2_1, x2_2 = x[1, t-1], x[1, t-2]
        x3_1, x3_2 = x[2, t-1], x[2, t-2]
        x4_1, x4_2 = x[3, t-1], x[3, t-2]
        x5_1, x5_2 = x[4, t-1], x[4, t-2]

        x[0, t] = 1.4 - x1_1**2 + 0.3 * x1_2
        x[1, t] = 1.4 - c * x1_1 * x2_1 + 0.3 * x2_2
        x[2, t] = 1.4 - c * x2_1 * x3_1 + 0.3 * x3_2
        x[3, t] = 1.4 - x4_1**2 + 0.3 * x4_2
        x[4, t] = 1.4 - c * x4_1 * x5_1 + 0.3 * x5_2

    x = zscore_normalize(x)
    return x

def lorenz_chain(t, y, N, c):
    dydt = np.zeros_like(y)
    for i in range(N):
        xi, yi, zi = y[3*i:3*(i+1)]
        dx = 10 * (yi - xi)
        dy = 28 * xi - yi - xi * zi
        dz = xi * yi - 8/3 * zi

        # Add coupling for all but first system
        if i > 0:
            x_prev = y[3*(i-1)]
            dx += c * (x_prev - xi)

        dydt[3*i:3*(i+1)] = [dx, dy, dz]

    return dydt


def lorenz_system(T=1000, sampling_period=0.5, dt=0.01, c=4.0):
    """
    Simulate 5 coupled Lorenz systems in a chain: x1->x2->x3->x4->x5
    Returns:
        X: array (5, T) -- sampled x components
    """
    N = 5
    steps_per_sample = int(sampling_period / dt)
    total_steps = T * steps_per_sample
    t_span = (0, total_steps * dt)
    t_eval = np.arange(0, total_steps * dt, sampling_period)

    # Initial condition: random for each (x,y,z)
    y0 = np.random.rand(3 * N) * 5.0

    sol = solve_ivp(
        lorenz_chain,
        t_span,
        y0,
        method='RK45',
        t_eval=t_eval,
        args=(N, c),
        rtol=1e-6,
        atol=1e-9
    )

    # Extract only the x components
    x = sol.y[::3, :]
    x = zscore_normalize(x)
    return x

def seizure_sweep(T=1000, fs=256, snr_db=-5):
    """
    Fast implementation for the 'Sweep' seizure model.
    """
    t = np.arange(T) / fs

    # Instantaneous frequency: linear from 12 Hz to 8 Hz
    f0, f1 = 12, 8
    ft = f0 + (f1 - f0) * (t / t[-1])
    phase = 2 * np.pi * np.cumsum(ft) / fs
    seizure = np.sin(phase)

    # Compute signal power
    Ps = np.mean(seizure**2)

    # Generate 5 pink noises
    noises = np.array([generate_pink_noise(T) for _ in range(5)])
    Pn = np.mean(noises[0]**2)
    scale = np.sqrt(Ps / (10**(snr_db/10)) / Pn)
    noises *= scale

    X = np.zeros((5, T))
    X[0] = seizure + noises[0]

    # Propagation with delays
    X[1] = noises[1]
    X[2] = noises[2]
    X[1, 2:] += X[0, :-2]
    X[2, 4:] += X[0, :-4]

    # Pure noise channels
    X[3] = noises[3]
    X[4] = noises[4]

    X = zscore_normalize(X)
    return X


def cascade_ar(T=1000, c=0.8, rho=0.9):
    """
    Simulate one realization of the Cascade AR model.
    """
    K = 5  # number of variables

    # Generate 5 pink noises
    theta = np.array([generate_pink_noise(T) for _ in range(K)])

    x = np.zeros((K, T))

    sqrt2rho = np.sqrt(2) * rho
    rho2 = rho ** 2

    for t in range(2, T):
        # x1 and x5: pure AR(2)
        x[0, t] = sqrt2rho * x[0, t-1] - rho2 * x[0, t-2] + theta[0, t]
        x[4, t] = sqrt2rho * x[4, t-1] - rho2 * x[4, t-2] + theta[4, t]

        # x2: driven by x1 and x3
        x[1, t] = (0.5 * c * (x[0, t-1] + x[2, t-1])
                   + (1 - c) * (sqrt2rho * x[1, t-1] - rho2 * x[1, t-2])
                   + theta[1, t])

        # x3: driven by x2 and x4
        x[2, t] = (0.5 * c * (x[1, t-1] + x[3, t-1])
                   + (1 - c) * (sqrt2rho * x[2, t-1] - rho2 * x[2, t-2])
                   + theta[2, t])

        # x4: driven by x3 and x5
        x[3, t] = (0.5 * c * (x[2, t-1] + x[4, t-1])
                   + (1 - c) * (sqrt2rho * x[3, t-1] - rho2 * x[3, t-2])
                   + theta[3, t])

    x = zscore_normalize(x)
    return x



def pink_ar(T=1000, nonlinear=False):
    """
    Simulate one realization of Pink AR linear or nonlinear model.
    - nonlinear = True: quadratic interactions for X1->X2 and X1->X4.
    """
    K = 5
    theta = np.array([generate_pink_noise(T) for _ in range(K)])
    x = np.zeros((K, T))
    sqrt2 = np.sqrt(2)

    for t in range(3, T):
        # x1: AR(2)
        x[0, t] = 0.95 * sqrt2 * x[0, t-2] + theta[0, t]

        # X1 -> X2, quadratic if nonlinear
        if nonlinear:
            x[1, t] = 0.5 * (x[0, t-2] ** 2) + theta[1, t]
        else:
            x[1, t] = 0.5 * x[0, t-2] + theta[1, t]

        # X1 -> X3: always linear
        x[2, t] = -0.4 * x[0, t-3] + theta[2, t]

        # X1 -> X4, quadratic if nonlinear + AR(1) terms
        if nonlinear:
            x[3, t] = (-0.5 * (x[0, t-2] ** 2) +
                       0.25 * sqrt2 * x[3, t-1] +
                       0.25 * sqrt2 * x[4, t-1] +
                       theta[3, t])
        else:
            x[3, t] = (-0.5 * x[0, t-2] +
                       0.25 * sqrt2 * x[3, t-1] +
                       0.25 * sqrt2 * x[4, t-1] +
                       theta[3, t])

        # X4 <-> X5: AR(1)
        x[4, t] = (-0.25 * sqrt2 * x[3, t-1] +
                    0.25 * sqrt2 * x[4, t-1] +
                    theta[4, t])
        
    x = zscore_normalize(x)
    return x



def freq_ar(T=1000, fs=256, nonlinear=False):
    """
    Simulate one realization of Frequency-Dependent AR model.
    fs: sampling frequency in Hz (default 256 Hz)
    """
    K = 5
    sqrt2 = np.sqrt(2)

    # Generate pink noise θ1,...,θ5
    theta = np.array([generate_pink_noise(T) for _ in range(K)])

    # # Bandpass filtered versions for alpha (8–12 Hz) & gamma (25–100 Hz)
    # theta_alpha = np.array([bandpass(sig, 8, 12, fs) for sig in theta])
    # theta_gamma = np.array([bandpass(sig, 25, 100, fs) for sig in theta])

    # Use safe band edges:
    gamma_low, gamma_high = 25, fs/2 * 0.99
    alpha_low, alpha_high = 8, 12

    theta_alpha = np.array([bandpass(sig, alpha_low, alpha_high, fs) for sig in theta])
    theta_gamma = np.array([bandpass(sig, gamma_low, gamma_high, fs) for sig in theta])


    x = np.zeros((K, T))

    for t in range(6, T):
        # x1: AR(2)
        x[0, t] = (0.95 * sqrt2 * x[0, t-1] - 0.9025 * x[0, t-2] + theta[0, t])

        # X1γ -> X2: quadratic if nonlinear
        if nonlinear:
            x[1, t] = 0.5 * (theta_gamma[0, t-2] ** 2) + theta[1, t]
        else:
            x[1, t] = 0.5 * theta_gamma[0, t-2] + theta[1, t]

        # X1γ, X2γ -> X3
        x[2, t] = (-0.4 * theta_gamma[0, t-3] +
                   0.25 * sqrt2 * theta_gamma[1, t-3] +
                   theta[2, t])

        # X1α -> X4: quadratic if nonlinear, plus AR(1) terms
        if nonlinear:
            x[3, t] = (-0.5 * (theta_alpha[0, t-5] ** 2) +
                       0.25 * sqrt2 * x[3, t-1] +
                       0.25 * sqrt2 * x[4, t-1] +
                       theta[3, t])
        else:
            x[3, t] = (-0.5 * theta_alpha[0, t-5] +
                       0.25 * sqrt2 * x[3, t-1] +
                       0.25 * sqrt2 * x[4, t-1] +
                       theta[3, t])

        # X4γ -> X5 + AR(1)
        x[4, t] = (-0.25 * sqrt2 * theta_gamma[3, t-1] +
                    0.25 * sqrt2 * x[4, t-1] +
                    theta[4, t])

    x = zscore_normalize(x)
    return x


# =============================
# 统一入口
# =============================
_MODEL_MAP = {
    'random': random_system,
    'henon': henon_system,
    'lorenz': lorenz_system,
    'sweep': seizure_sweep,
    'cascadear': cascade_ar,
    'pinkarlin': lambda T, **kw: pink_ar(T=T, nonlinear=False, **kw),
    'pinkarnonlin': lambda T, **kw: pink_ar(T=T, nonlinear=True, **kw),
    'freqarlin': lambda T, **kw: freq_ar(T=T, nonlinear=False, **kw),
    'freqarnonlin': lambda T, **kw: freq_ar(T=T, nonlinear=True, **kw)
}

# 支持数字索引：按顺序生成同样的列表
_MODEL_LIST = [
    _MODEL_MAP['random'],
    _MODEL_MAP['henon'],
    _MODEL_MAP['lorenz'],
    _MODEL_MAP['sweep'],
    _MODEL_MAP['cascadear'],
    _MODEL_MAP['pinkarlin'],
    _MODEL_MAP['pinkarnonlin'],
    _MODEL_MAP['freqarlin'],
    _MODEL_MAP['freqarnonlin'],
]

# ==== 统一接口 ====
def simulate(model, T=1000, seed=None,**kwargs):
    """
    Unified simulator for all models.

    Parameters:
        model : str or int
            Model name (case-insensitive) or index (0~8)
        T : int
            Time series length
        **kwargs : model-specific arguments

    Returns:
        np.ndarray : Simulated data (5, T)
    """

    if seed is not None:
        np.random.seed(seed)

    if isinstance(model, str):
        model_key = model.lower()
        if model_key not in _MODEL_MAP:
            raise ValueError(f"Unknown model name: {model}")
        func = _MODEL_MAP[model_key]

    elif isinstance(model, int):
        if not (0 <= model < len(_MODEL_LIST)):
            raise ValueError(f"Model index out of range: {model}")
        func = _MODEL_LIST[model]

    else:
        raise TypeError(f"Model must be str or int, got {type(model)}")

    return func(T=T, **kwargs)

