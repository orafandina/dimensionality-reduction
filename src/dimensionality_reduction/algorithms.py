"""
Core approximation algorithms for dimensionality reduction.

This module implements the approximation algorithm with provable guarantees
for dimensionality reduction as described in:

Bartal, Y., Fandina, N., & Neiman, O. (2019).
Dimensionality reduction: theoretical perspective on practical measures.
NeurIPS 2019.
"""

import logging
from typing import Literal

import cvxpy as cp
import numpy as np
from sklearn.random_projection import GaussianRandomProjection

from . import spaces
from .exceptions import OptimizationError

logger = logging.getLogger(__name__)

ObjectiveType = Literal["lq_dist", "stress", "energy", "sigma"]


def johnson_lindenstrauss_transform(
    space: np.ndarray, target_dimension: int
) -> np.ndarray:
    """
    Apply Johnson-Lindenstrauss random projection.

    Uses Gaussian random projection to reduce dimensionality while
    approximately preserving distances.

    Parameters
    ----------
    space : np.ndarray
        High-dimensional vectors (n x d)
    target_dimension : int
        Target dimension k

    Returns
    -------
    np.ndarray
        Low-dimensional vectors (n x k)
    """
    transformer = GaussianRandomProjection(target_dimension)
    result = transformer.fit_transform(space)
    return result


def approx_algo(
    distance_matrix: np.ndarray,
    target_dimension: int,
    q: float,
    objective: ObjectiveType = "lq_dist",
) -> np.ndarray:
    """
    Approximation algorithm for dimensionality reduction with guarantees.

    Computes an embedding F: X → ℓ₂^k with distortion ≈ (1 + O(q/k)) * OPT,
    where OPT is the optimal distortion.

    The algorithm works in two steps:
    1. Convex optimization to find optimal high-dimensional embedding
    2. Johnson-Lindenstrauss projection to reduce to target dimension

    Parameters
    ----------
    distance_matrix : np.ndarray
        Pairwise distance matrix of input metric space (n x n)
    target_dimension : int
        Target dimension k (should be ≥ 3)
    q : float
        Order parameter for distortion measure (q ≥ 1)
    objective : {'lq_dist', 'stress', 'energy', 'sigma'}, optional
        Distortion measure to optimize (default: 'lq_dist')

    Returns
    -------
    np.ndarray
        Embedded vectors in target dimension (n x k)

    Raises
    ------
    OptimizationError
        If convex optimization fails
    ValueError
        If target_dimension < 1 or q < 1

    Notes
    -----
    The convex optimization step can be slow for large inputs.
    Runtime: O(n^3) for optimization, O(ndk) for projection.
    """
    if target_dimension < 1:
        raise ValueError(f"target_dimension must be >= 1, got {target_dimension}")
    if q < 1:
        raise ValueError(f"q must be >= 1, got {q}")

    rows, cols = distance_matrix.shape
    if rows != cols:
        raise ValueError("distance_matrix must be square")

    logger.info(
        f"Starting approximation algorithm: n={rows}, k={target_dimension}, "
        f"q={q}, objective={objective}"
    )

    # Step 1: Normalize by largest distance (doesn't change optimal embedding)
    max_dist = np.amax(distance_matrix)
    if max_dist == 0:
        raise ValueError("distance_matrix contains only zeros")

    normalized_dists = distance_matrix / max_dist

    # Step 1: Solve convex optimization
    logger.info("Step 1/2: Solving convex optimization problem...")
    
    if objective == "lq_dist":
        recovered_vectors = _solve_optimization_lq_dist(normalized_dists, q)
    elif objective == "energy":
        recovered_vectors = _solve_optimization_energy(normalized_dists, q)
    elif objective == "stress":
        recovered_vectors = _solve_optimization_stress(normalized_dists, q)
    elif objective == "sigma":
        recovered_vectors = _solve_optimization_sigma(normalized_dists, q)
    else:
        raise ValueError(f"Unknown objective: {objective}")

    logger.info("Convex optimization complete")

    # Step 2: Johnson-Lindenstrauss projection
    logger.info("Step 2/2: Applying random projection...")
    low_dim_space = johnson_lindenstrauss_transform(recovered_vectors, target_dimension)

    # Restore original scale
    result = low_dim_space * max_dist

    logger.info("Approximation algorithm complete")
    return result


def _solve_optimization_lq_dist(distance_matrix: np.ndarray, q: float) -> np.ndarray:
    """
    Solve convex optimization for lq-distortion objective.

    Parameters
    ----------
    distance_matrix : np.ndarray
        Normalized distance matrix (n x n)
    q : float
        Order parameter

    Returns
    -------
    np.ndarray
        Optimal high-dimensional embedding (n x n)

    Raises
    ------
    OptimizationError
        If optimization fails
    """
    rows, cols = distance_matrix.shape

    # Variables
    G = cp.Variable((rows, cols), PSD=True)  # Gram matrix
    Z = cp.Variable((rows, cols), symmetric=True)  # New squared distances
    E = cp.Variable((rows, cols), symmetric=True)  # Squared expansions
    C = cp.inv_pos(E)  # Squared contractions
    M = cp.maximum(E, C)  # Squared distortions

    # Constraints
    constraints = [
        Z >= 0,
        Z[0] == cp.diag(G),  # z_0j = ||v_j||^2
        cp.diag(E) == 1,  # Technical constraint
        Z == cp.multiply(E, distance_matrix ** 2),  # z_ij = expansion_ij^2 * d_ij^2
    ]

    # Gram matrix relation: z_ij = G_ii + G_jj - 2G_ij (vectorized)
    G_expression = cp.Variable((rows, cols))
    for i in range(rows):
        constraints += [G_expression[i] == cp.diag(G)]

    constraints += [G_expression.T + G_expression - 2 * G == Z]

    # Objective function
    if q / 2 == 1:
        objective = cp.Minimize(cp.norm1(M))
    else:
        objective = cp.Minimize(cp.pnorm(M, p=q / 2))

    problem = cp.Problem(objective, constraints)

    try:
        problem.solve(solver="SCS", verbose=False)
    except Exception as e:
        raise OptimizationError(f"Convex optimization failed: {e}") from e

    if problem.status not in ["optimal", "optimal_inaccurate"]:
        raise OptimizationError(f"Optimization failed with status: {problem.status}")

    if G.value is None:
        raise OptimizationError("Optimization returned no solution")

    # Recover vectors from Gram matrix
    is_psd = spaces.is_positive_semidefinite(G.value)
    recovered_vectors = spaces.space_from_gram(G.value, is_psd)

    return recovered_vectors


def _solve_optimization_energy(distance_matrix: np.ndarray, q: float) -> np.ndarray:
    """
    Solve convex optimization for energy distortion objective.

    Parameters
    ----------
    distance_matrix : np.ndarray
        Normalized distance matrix (n x n)
    q : float
        Order parameter

    Returns
    -------
    np.ndarray
        Optimal high-dimensional embedding (n x n)
    """
    rows, cols = distance_matrix.shape

    G = cp.Variable((rows, cols), PSD=True)
    Z = cp.Variable((rows, cols), symmetric=True)
    E = cp.Variable((rows, cols), symmetric=True)

    constraints = [
        Z >= 0,
        Z[0] == cp.diag(G),
        cp.diag(E) == 1,
        Z == cp.multiply(E, distance_matrix ** 2),
    ]

    G_expression = cp.Variable((rows, cols))
    for i in range(rows):
        constraints += [G_expression[i] == cp.diag(G)]

    constraints += [G_expression.T + G_expression - 2 * G == Z]

    # Objective: |√E - 1| = |expansion - 1|
    obj_expression = cp.abs(E - 1)

    if q == 1:
        objective = cp.Minimize(cp.norm1(obj_expression))
    else:
        objective = cp.Minimize(cp.pnorm(obj_expression, p=q))

    problem = cp.Problem(objective, constraints)

    try:
        problem.solve(solver="SCS", verbose=False)
    except Exception as e:
        raise OptimizationError(f"Convex optimization failed: {e}") from e

    if problem.status not in ["optimal", "optimal_inaccurate"]:
        raise OptimizationError(f"Optimization failed with status: {problem.status}")

    if G.value is None:
        raise OptimizationError("Optimization returned no solution")

    is_psd = spaces.is_positive_semidefinite(G.value)
    return spaces.space_from_gram(G.value, is_psd)


def _solve_optimization_stress(distance_matrix: np.ndarray, q: float) -> np.ndarray:
    """
    Solve convex optimization for stress distortion objective.

    Parameters
    ----------
    distance_matrix : np.ndarray
        Normalized distance matrix (n x n)
    q : float
        Order parameter

    Returns
    -------
    np.ndarray
        Optimal high-dimensional embedding (n x n)
    """
    rows, cols = distance_matrix.shape

    G = cp.Variable((rows, cols), PSD=True)
    Z = cp.Variable((rows, cols), symmetric=True)
    E = cp.Variable((rows, cols), symmetric=True)

    constraints = [
        Z >= 0,
        Z[0] == cp.diag(G),
        cp.diag(E) == 1,
        Z == cp.multiply(E, distance_matrix ** 2),
    ]

    G_expression = cp.Variable((rows, cols))
    for i in range(rows):
        constraints += [G_expression[i] == cp.diag(G)]

    constraints += [G_expression.T + G_expression - 2 * G == Z]

    # Objective: |z_ij - d_ij^2|
    obj_expression = cp.abs(Z - distance_matrix ** 2)

    if q == 1:
        objective = cp.Minimize(cp.norm1(obj_expression))
    else:
        objective = cp.Minimize(cp.pnorm(obj_expression, p=q))

    problem = cp.Problem(objective, constraints)

    try:
        problem.solve(solver="SCS", verbose=False)
    except Exception as e:
        raise OptimizationError(f"Convex optimization failed: {e}") from e

    if problem.status not in ["optimal", "optimal_inaccurate"]:
        raise OptimizationError(f"Optimization failed with status: {problem.status}")

    if G.value is None:
        raise OptimizationError("Optimization returned no solution")

    is_psd = spaces.is_positive_semidefinite(G.value)
    return spaces.space_from_gram(G.value, is_psd)


def _solve_optimization_sigma(distance_matrix: np.ndarray, q: float) -> np.ndarray:
    """
    Solve convex optimization for sigma distortion objective.

    Parameters
    ----------
    distance_matrix : np.ndarray
        Normalized distance matrix (n x n)
    q : float
        Order parameter

    Returns
    -------
    np.ndarray
        Optimal high-dimensional embedding (n x n)
    """
    rows, cols = distance_matrix.shape
    num_pairs = rows * (rows - 1) / 2

    G = cp.Variable((rows, cols), PSD=True)
    Z = cp.Variable((rows, cols), symmetric=True)
    E = cp.Variable((rows, cols), symmetric=True)

    constraints = [
        Z >= 0,
        Z[0] == cp.diag(G),
        cp.diag(E) == 1,
        Z == cp.multiply(E, distance_matrix ** 2),
        cp.sum(E) == num_pairs,  # Normalization constraint for sigma
    ]

    G_expression = cp.Variable((rows, cols))
    for i in range(rows):
        constraints += [G_expression[i] == cp.diag(G)]

    constraints += [G_expression.T + G_expression - 2 * G == Z]

    obj_expression = cp.abs(E - 1)

    if q == 1:
        objective = cp.Minimize(cp.norm1(obj_expression))
    else:
        objective = cp.Minimize(cp.pnorm(obj_expression, p=q))

    problem = cp.Problem(objective, constraints)

    try:
        problem.solve(solver="SCS", verbose=False)
    except Exception as e:
        raise OptimizationError(f"Convex optimization failed: {e}") from e

    if problem.status not in ["optimal", "optimal_inaccurate"]:
        raise OptimizationError(f"Optimization failed with status: {problem.status}")

    if G.value is None:
        raise OptimizationError("Optimization returned no solution")

    is_psd = spaces.is_positive_semidefinite(G.value)
    return spaces.space_from_gram(G.value, is_psd)

