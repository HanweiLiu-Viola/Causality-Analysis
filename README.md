# Causality Analysis — Project Description

**Author:** Hanwei Liu
**Institution:** University Hospital Würzburg (UKW)
**Last Updated:** 2026-05-14

---

## Docker

A pre-built image is available on DockerHub:

```bash
docker pull viola1003/causality-analysis:latest
```

Run with Jupyter on port 8888:

```bash
docker run -p 8888:8888 -v "$(pwd):/home/jovyan/work" viola1003/causality-analysis:latest
```

Then open the URL printed in the terminal.

---
## 1. Project Overview

This project implements and benchmarks a suite of **Functional Connectivity (FC)**
methods for directed brain connectivity analysis. Given multichannel time-series data
(real or simulated), each method estimates a connectivity matrix describing which
signals influence which others and in which direction.

The pipeline is validated against synthetic models with known ground-truth causal
structure, using the **Matthews Correlation Coefficient (MCC)** as the evaluation metric.


---

## 2. Directory Structure

```
Causality-Analysis/
├── src/
│   ├── methods/
│   │   └── fc_pipeline.py          # All FC methods (FCMethods class)
│   ├── core/
│   │   └── mvarica.py              # MVAR model + ICA + MVARICA pipeline
│   ├── simulation/
│   │   └── simulation_models.py    # 9 simulation models with known GT
│   └── data/
│       ├── brain_data.py           # MNE-based EEG/MEG data loader
│       └── mne_loader.py           # MNEData class (lazy load, caching)
├── notebooks/
│   ├── fc_benchmark.ipynb          # Multi-subject, multi-model benchmark
│   └── fc_random_demo.ipynb        # Thesis-quality demo on the random model
├── figures/                        # Exported PDF + PNG figures
├── data/                           # Simulated data + MCC results (.npz, .pkl)
├── requirements.txt                # Python dependencies
├── pyproject.toml                  # Package metadata (src/ layout)
└── PROJECT_DESCRIPTION.md          # This file
```

---

## 3. Simulation Models (`simulation_models.py`)

The simulation models implemented in this project are adapted from:
> Heyse, J., Sheybani, L., Vulliémoz, S., & van Mierlo, P. (2021).
> Evaluation of Directed Causality Measures and Lag Estimations in
> Multivariate Time-Series. *Frontiers in Systems Neuroscience*, 15,
> 620338. https://doi.org/10.3389/fnsys.2021.620338

![Simulation Models](figures/simulation_models.webp)

All models generate 5-node time series with known directed connectivity.
Unified entry point: `simulate(model, T=1000, seed=None, **kwargs)`.


| Key | Function | Ground Truth Connectivity | Notes |
|---|---|---|---|
| `random` | `random_system` | x1→x2, x1→x3, x4→x5 | Fixed delays (2–5 samples), linear |
| `henon` | `henon_system` | x1→x2→x3, x4↔x5 | Nonlinear Hénon map |
| `lorenz` | `lorenz_system` | x1→x2→x3→x4→x5 (chain) | Coupled Lorenz oscillators (ODE) |
| `sweep` | `seizure_sweep` | x1→x2, x1→x3 | Frequency-swept seizure model + pink noise |
| `cascadear` | `cascade_ar` | x1→x2→x3→x4, x5→x4 | AR(2) cascade with bidirectional middle |
| `pinkarlin` | `pink_ar(nonlinear=False)` | x1→x2,x3,x4; x5→x4 | Pink-noise driven linear AR |
| `pinkarnonlin` | `pink_ar(nonlinear=True)` | x1→x2,x3,x4; x5→x4 | Same with quadratic coupling |
| `freqarlin` | `freq_ar(nonlinear=False)` | x1→x2,x3,x4; x2→x3; x5→x4 | Frequency-band specific coupling |
| `freqarnonlin` | `freq_ar(nonlinear=True)` | same as freqarlin | Nonlinear version |

All models apply z-score normalization (`zscore_normalize`) before returning.

---

## 4. FC Methods (`fc_pipeline.py`)

### 4.1 Class Structure

```
BasePreprocessor          — standardizes data to (epochs, nodes, time)
ADTFModel                 — VAR fitting + transfer matrix + ADTF computation
FCMethods                 — unified interface; calls _<method>_func per method
```

### 4.2 Available Methods

| Method | Function | Internal Convention | Description |
|---|---|---|---|
| `ADTF` | `_adtf_func` | `[target, source]` | Adaptive DTF via VAR; integrated over frequency band |
| `PDC` | `_pdc_func` | `[target, source]` | Partial Directed Coherence via MVARICA |
| `DTF` | `_dtf_func` | `[target, source]` | Directed Transfer Function via MVARICA |
| `cGC` | `_cgc_func` | `[target, source]` | Conditional Granger Causality via VAR residuals |
| `PLI` | `_pli_func` | `[target, source]` (symmetric) | Phase Lag Index via MNE; symmetrized |
| `PSI` | `_psi_func` | `[source, target]` | Phase Slope Index via MNE; all pairs |
| `TE` | `_te_func` | IDTxl result | Multivariate Transfer Entropy (IDTxl) |
| `MI` | `_mi_func` | IDTxl result | Multivariate Mutual Information (IDTxl) |


## 5. Notebook 1: `fc_benchmark.ipynb`

**Purpose:** Systematic multi-subject, multi-model benchmark to quantify FC method performance.

### Tasks Completed

1. **Data Generation**
   - Parameters: `N_SUBJECTS=25`, `N_EPOCHS=5`, `N_NODES=5`, `T=1000`, `FS=256`
   - 7 models: `random, henon, lorenz, sweep, cascadear, freqarlin, freqarnonlin`
   - Seeds fixed as `subj*100 + model_id*10 + epoch` for full reproducibility
   - Data shape: `(25, 7, 5, 5, 1000)`, saved as `.npz`

2. **FC Computation**
   - Methods: ADTF, PDC, DTF, cGC, PLI, PSI
   - ADTF: alpha band 8–12 Hz, `maxlags=10`, mean over band (not trapz integral)
   - PDC/DTF: MVARICA with Extended Infomax ICA, `model_order=5`, `n_fft=128`
   - Quick sanity check on Subject 0 / Model 0 with visualization

3. **MCC Evaluation**
   - Ground truth adjacency matrices defined for all 9 models
   - Binarization: top-25% percentile threshold on off-diagonal values (in source→target space)
   - All non-PSI matrices transposed before thresholding to match GT convention
   - Results: violin plots per model + mean MCC heatmap (models × methods)

### Key Parameters

```python
METHODS_PARAMS = {
    "ADTF":  {"fmin": 8, "fmax": 12, "n_freqs": 100, "maxlags": 10, "integrate": True},
    "PDC": {"model_order": 5, "n_fft": 128, "ica_method": "infomax_extended", "integrate": True},
    "DTF":   {"model_order": 5, "n_fft": 128, "ica_method": "infomax_extended", "integrate": True},
    "cGC":   {},
    "PLI":   {"fmin": 8, "fmax": 12, "integrate": True},
    "PSI":   {"fmin": 8, "fmax": 12, "integrate": True},
}
```

---

## 6. Notebook 2: `fc_random_demo.ipynb`

**Purpose:** Thesis-quality demonstration of all FC methods on the *random* simulation model

### Tasks Completed

**Data Generation** 
- Random model parameters: `N_EPOCHS=10`, `T=1000`, `FS=256`, `SEED=42`
- True edges: x1→x2 (Δ=3), x1→x3 (Δ=2), x4→x5 (Δ=5)
- Data shape: `(10, 5, 1000)`,

