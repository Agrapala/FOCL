"""
Data Utilities for FLoBC
========================
Generates synthetic datasets — no internet downloads required.

  generate_mnist_like()        → MNIST-style 784-dim / 10-class data
  generate_bayesian_data()     → Alarm-Network-style tabular data
  train_val_test_split()       → stratified split helper
"""

import numpy as np
from typing import Tuple


def generate_mnist_like(
    n_samples: int = 5000,
    n_classes: int = 10,
    input_dim: int = 784,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Synthetic MNIST-like dataset.
    Each class has a distinct sparse activation pattern shifted across dims.
    Samples are drawn from a Gaussian around each class centre.
    """
    np.random.seed(seed)
    X_parts, y_parts = [], []
    per_class = n_samples // n_classes

    for c in range(n_classes):
        # Class-specific sparse pattern
        centre = np.zeros(input_dim, dtype=np.float32)
        idx = np.random.choice(input_dim,
                               size=input_dim // 4,
                               replace=False)
        centre[idx] = np.random.uniform(0.3, 0.9, len(idx)).astype(np.float32)
        centre = np.roll(centre, int(c * input_dim / n_classes))

        samples = centre + (np.random.randn(per_class, input_dim) * 0.15).astype(np.float32)
        samples = np.clip(samples, 0, 1)
        X_parts.append(samples)
        y_parts.append(np.full(per_class, c, dtype=np.int32))

    X = np.vstack(X_parts)
    y = np.hstack(y_parts)
    perm = np.random.permutation(len(X))
    return X[perm], y[perm]


def generate_bayesian_data(
    n_samples: int = 2000,
    n_features: int = 12,
    n_classes: int = 4,
    seed: int = 7,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Synthetic Alarm-Network-style Bayesian tabular data.
    Each class has distinct Beta-distributed feature marginals.
    """
    np.random.seed(seed)
    priors = np.random.dirichlet(np.ones(n_classes))
    y = np.random.choice(n_classes, size=n_samples, p=priors).astype(np.int32)

    X = np.zeros((n_samples, n_features), dtype=np.float32)
    for f in range(n_features):
        for c in range(n_classes):
            mask = y == c
            X[mask, f] = np.random.beta(c + 1, n_classes - c,
                                         size=mask.sum()).astype(np.float32)

    # Standardise
    mu  = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    return (X - mu) / std, y


def train_val_test_split(
    X: np.ndarray, y: np.ndarray,
    val_ratio:  float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray,
           np.ndarray, np.ndarray,
           np.ndarray, np.ndarray]:
    np.random.seed(seed)
    n     = len(X)
    perm  = np.random.permutation(n)
    n_te  = int(n * test_ratio)
    n_val = int(n * val_ratio)
    te_idx  = perm[:n_te]
    val_idx = perm[n_te: n_te + n_val]
    tr_idx  = perm[n_te + n_val:]
    return (X[tr_idx], y[tr_idx],
            X[val_idx], y[val_idx],
            X[te_idx],  y[te_idx])
