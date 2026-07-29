from __future__ import annotations

import numpy as np
from sklearn.utils.extmath import randomized_svd


def standardize_from_rows(
    values: np.ndarray,
    fit_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Standardize columns using only ``fit_rows`` and preserve the observation mask."""
    matrix = np.asarray(values, dtype=np.float64)
    observed = np.isfinite(matrix)
    fit = matrix[fit_rows]
    counts = np.isfinite(fit).sum(axis=0)
    if np.any(counts < 2):
        bad = np.flatnonzero(counts < 2)[:10].tolist()
        raise ValueError(f"Columns with fewer than two fit observations: {bad}")
    means = np.nanmean(fit, axis=0)
    scales = np.nanstd(fit, axis=0, ddof=1)
    scales = np.where(np.isfinite(scales) & (scales > 1e-12), scales, 1.0)
    standardized = (matrix - means) / scales
    return standardized, observed, means, scales


def iterative_svd_impute(
    standardized: np.ndarray,
    observed: np.ndarray,
    rank: int,
    iterations: int,
    random_state: int,
    n_iter: int,
) -> np.ndarray:
    """Low-rank iterative imputation used only to fit the PCA basis.

    Observed entries never change. Missing entries start at zero (the training-column
    mean after standardization) and are replaced by low-rank reconstructions.
    """
    matrix = np.array(standardized, dtype=np.float64, copy=True, order="C")
    matrix[~observed] = 0.0
    effective_rank = max(1, min(int(rank), min(matrix.shape) - 1))
    for iteration in range(max(int(iterations), 1)):
        u, s, vt = randomized_svd(
            matrix,
            n_components=effective_rank,
            n_iter=int(n_iter),
            random_state=int(random_state) + iteration,
        )
        reconstruction = (u * s) @ vt
        matrix[~observed] = reconstruction[~observed]
    return matrix


def masked_factor_scores(
    standardized: np.ndarray,
    observed: np.ndarray,
    components: np.ndarray,
    ridge: float = 1e-4,
    minimum_observed_fraction: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Project each week with observed-only ridge least squares.

    This avoids interpreting missing standardized returns as genuine zero returns.
    Rows with too little observed data are returned as NaN and accompanied by their
    observed fraction.
    """
    matrix = np.asarray(standardized, dtype=np.float64)
    mask = np.asarray(observed, dtype=bool)
    basis = np.asarray(components, dtype=np.float64)
    rank, stock_count = basis.shape
    scores = np.full((matrix.shape[0], rank), np.nan, dtype=np.float64)
    fractions = mask.mean(axis=1)
    identity = np.eye(rank, dtype=np.float64) * float(ridge)
    for row_index in range(matrix.shape[0]):
        row_mask = mask[row_index]
        if fractions[row_index] < float(minimum_observed_fraction) or row_mask.sum() < rank:
            continue
        design = basis[:, row_mask]
        gram = design @ design.T + identity
        rhs = design @ matrix[row_index, row_mask]
        try:
            scores[row_index] = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            scores[row_index] = np.linalg.lstsq(gram, rhs, rcond=None)[0]
    return scores, fractions


def masked_reconstruction_mse(
    standardized: np.ndarray,
    observed: np.ndarray,
    scores: np.ndarray,
    components: np.ndarray,
) -> float:
    reconstructed = np.asarray(scores) @ np.asarray(components)
    valid = np.asarray(observed, dtype=bool) & np.isfinite(reconstructed)
    if not valid.any():
        return float("nan")
    residual = np.asarray(standardized)[valid] - reconstructed[valid]
    return float(np.mean(residual**2))
