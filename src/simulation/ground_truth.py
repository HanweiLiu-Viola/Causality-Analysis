"""Known ground-truth directed adjacency matrices for all simulation models.

Each matrix is a 5×5 binary array where ``GT[source, target] = 1`` indicates
a directed connection from channel (source+1) to channel (target+1),
using 0-based indexing for the array.

This module is the single source of truth for ground-truth connectivity.
Both the benchmark script and analysis notebooks should import from here
rather than re-defining these matrices inline.
"""

import numpy as np

# GT[i, j] = 1 means channel (i+1) drives channel (j+1).
# Consistent with the [source, target] convention used throughout the pipeline.
GROUND_TRUTH: dict[str, np.ndarray] = {
    # Two independent sub-networks: (1->2, 1->3) and (4->5)
    "random": np.array([
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0],
    ]),
    # Unidirectional chain via Hénon coupling: 1->2->3, 4->5
    "henon": np.array([
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
    ]),
    # Unidirectional chain via Lorenz coupling: 1->2->3->4->5
    "lorenz": np.array([
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0],
    ]),
    # Seizure propagation with fixed delays: 1->2, 1->3
    "sweep": np.array([
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]),
    # Bidirectional AR cascade; dominant direction shown: 1->2->3->4, 5->4
    "cascadear": np.array([
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
    ]),
    # Frequency-dependent AR (linear): 1->2(γ), 1->3(γ), 1->4(α), 2->3(γ), 5->4(γ)
    "freqarlin": np.array([
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
    ]),
    # Frequency-dependent AR (nonlinear): same structure as freqarlin
    "freqarnonlin": np.array([
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
    ]),
    # Pink-noise AR (linear): 1->2, 1->3, 1->4, 5->4
    "pinkarlin": np.array([
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
    ]),
    # Pink-noise AR (nonlinear): same structure as pinkarlin
    "pinkarnonlin": np.array([
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
    ]),
}

VALID_MODELS: list[str] = list(GROUND_TRUTH)
