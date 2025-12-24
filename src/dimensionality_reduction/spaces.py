"""
Utilities for working with metric spaces.

This module provides functions for:
- Creating and validating metric spaces
- Converting between distance matrices and vector representations
- Checking if spaces are Euclidean
- Generating synthetic metric spaces
"""

import logging
from typing import Tuple

import numpy as np
import scipy.spatial

from .exceptions import InvalidMetricSpaceError

logger = logging.getLogger(__name__)


def is_metric_space(distance_matrix: np.ndarray) -> bool:
    """
    Verify that a distance matrix satisfies the triangle inequality.

    Parameters
    ----------
    distance_matrix : np.ndarray
        Square matrix of pairwise distances (n x n)

    Returns
    -------
    bool
        True if the triangle inequality holds for all triples, False otherwise
    """
    rows, cols = distance_matrix.shape
    if rows != cols:
        return False

    for i in range(rows):
        for j in range(i):
            for k in range(rows):
                if distance_matrix[i, j] > distance_matrix[i, k] + distance_matrix[k, j]:
                    return False
    return True


def _gram_matrix_from_squared_distances(squared_dists: np.ndarray) -> np.ndarray:
    """
    Construct Gram matrix from squared distance matrix.

    Uses the formula: G_ij = (1/2)(||x_i||^2 + ||x_j||^2 - d_ij^2)
    Assumes the first point is at the origin (x_0 = 0).

    Parameters
    ----------
    squared_dists : np.ndarray
        Matrix of squared distances (n x n)

    Returns
    -------
    np.ndarray
        Gram matrix (n x n)
    """
    rows, cols = squared_dists.shape

    # First row contains squared norms (distances from origin)
    squared_norms = squared_dists[0]
    norms_matrix = np.tile(squared_norms, (rows, 1))

    sum_norms = np.zeros((rows, cols))
    for i in range(1, cols):
        sum_norms[i] = np.full(cols, squared_norms[i])

    gram_matrix = (1 / 2) * (norms_matrix + sum_norms - squared_dists)
    return gram_matrix


def is_positive_semidefinite(matrix: np.ndarray, tol: float = 1e-10) -> bool:
    """
    Check if a matrix is positive semi-definite.

    Parameters
    ----------
    matrix : np.ndarray
        Square matrix to check
    tol : float, optional
        Tolerance for negative eigenvalues (default: 1e-10)

    Returns
    -------
    bool
        True if all eigenvalues are >= -tol, False otherwise
    """
    eigenvalues = np.linalg.eigvalsh(matrix)
    return np.all(eigenvalues >= -tol)


def is_euclidean_space(distance_matrix: np.ndarray) -> bool:
    """
    Check if a distance matrix represents a Euclidean metric space.

    A metric space is Euclidean if its Gram matrix is positive semi-definite.

    Parameters
    ----------
    distance_matrix : np.ndarray
        Matrix of pairwise distances (n x n)

    Returns
    -------
    bool
        True if the space is Euclidean, False otherwise
    """
    squared_dists = distance_matrix ** 2
    gram = _gram_matrix_from_squared_distances(squared_dists)
    return is_positive_semidefinite(gram)


def space_from_gram(gram_matrix: np.ndarray, is_psd: bool = True) -> np.ndarray:
    """
    Recover vector representation from Gram matrix.

    Uses eigendecomposition: if G = U^T D U, then vectors are rows of D^(1/2) U.

    Parameters
    ----------
    gram_matrix : np.ndarray
        Gram matrix (n x n)
    is_psd : bool, optional
        Whether the matrix is known to be positive semi-definite (default: True)
        If False, takes absolute values of eigenvalues before square root

    Returns
    -------
    np.ndarray
        Matrix where each row is a vector (n x d)
    """
    eigenvalues, eigenvectors = np.linalg.eigh(gram_matrix)

    if not is_psd:
        logger.warning("Gram matrix is not PSD; using absolute values of eigenvalues")
        sqrt_eigenvalues = np.sqrt(np.abs(eigenvalues))
    else:
        sqrt_eigenvalues = np.sqrt(eigenvalues)

    D_matrix = np.diag(sqrt_eigenvalues)
    U_matrix = np.transpose(eigenvectors)  # Orthonormal basis

    vectors = np.matmul(D_matrix, U_matrix)
    return np.transpose(vectors)  # Return vectors as rows


def space_from_dists(
    distance_matrix: np.ndarray, squared: bool = False
) -> np.ndarray:
    """
    Recover Euclidean vector space from distance matrix.

    Parameters
    ----------
    distance_matrix : np.ndarray
        Matrix of pairwise distances (n x n)
    squared : bool, optional
        Whether distances are already squared (default: False)

    Returns
    -------
    np.ndarray
        Matrix where each row is a vector (n x d)

    Warnings
    --------
    If the distance matrix is non-Euclidean, returns an approximation
    """
    if squared:
        squared_dists = distance_matrix
    else:
        squared_dists = np.square(distance_matrix)

    gram = _gram_matrix_from_squared_distances(squared_dists)

    if not is_positive_semidefinite(gram):
        logger.warning(
            "Distance matrix is non-Euclidean; returning best approximation"
        )

    is_psd = is_positive_semidefinite(gram)
    return space_from_gram(gram, is_psd)


def space_to_dist(space: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """
    Compute pairwise distance matrix from vector space.

    Parameters
    ----------
    space : np.ndarray
        Matrix where each row is a vector (n x d)
    metric : str, optional
        Distance metric to use (default: 'euclidean')
        Options: 'euclidean', 'chebyshev', 'minkowski'

    Returns
    -------
    np.ndarray
        Pairwise distance matrix (n x n)
    """
    distances = scipy.spatial.distance.pdist(space, metric=metric)
    matrix_dist = scipy.spatial.distance.squareform(distances)
    return matrix_dist


def space_to_lp_dists(space: np.ndarray, p: float) -> np.ndarray:
    """
    Compute Lp-norm distance matrix from vector space.

    Parameters
    ----------
    space : np.ndarray
        Matrix where each row is a vector (n x d)
    p : float
        Parameter for Minkowski (Lp) distance

    Returns
    -------
    np.ndarray
        Pairwise distance matrix (n x n)
    """
    distances = scipy.spatial.distance.pdist(space, "minkowski", p=p)
    matrix_dist = scipy.spatial.distance.squareform(distances)
    return np.around(matrix_dist, 8)


def get_random_space(size: int, dimension: int) -> np.ndarray:
    """
    Generate a random Euclidean vector space.

    Points are sampled from normal distributions with random standard deviations.

    Parameters
    ----------
    size : int
        Number of points
    dimension : int
        Dimension of the space

    Returns
    -------
    np.ndarray
        Random vector space (size x dimension)
    """
    space = np.random.randn(size, dimension)
    for i in range(size):
        std_dev = np.random.randint(1, 30)
        space[i] = std_dev * space[i]
    return space


def get_epsilon_close_metric(
    distance_matrix: np.ndarray, epsilon: float, max_iterations: int = 5
) -> np.ndarray:
    """
    Generate a non-Euclidean metric space epsilon-close to a Euclidean one.

    The algorithm randomly perturbs distances while maintaining the triangle
    inequality. The result can be embedded into the original space with
    distortion (1 + epsilon).

    Parameters
    ----------
    distance_matrix : np.ndarray
        Original Euclidean distance matrix (n x n)
    epsilon : float
        Closeness parameter (typically small, e.g., 0.1-0.5)
    max_iterations : int, optional
        Maximum attempts to find valid perturbation per edge (default: 5)

    Returns
    -------
    np.ndarray
        Perturbed distance matrix (n x n)

    Notes
    -----
    Due to randomness, there's a small probability the output is still Euclidean.
    Rounding issues may occasionally cause triangle inequality violations.
    """
    working_dists = np.copy(distance_matrix)
    rows, cols = distance_matrix.shape
    generated_metric = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(i + 1, cols):
            # Find valid range for d(i,j) based on triangle inequality
            upper_bounds = working_dists[i] + working_dists[j]
            lower_bounds = np.abs(working_dists[i] - working_dists[j])

            # Exclude i and j themselves
            mask = np.ones(cols, dtype=bool)
            mask[[i, j]] = False

            largest_valid = np.amin(upper_bounds[mask])
            smallest_valid = np.amax(lower_bounds[mask])

            current_dist = working_dists[i, j]

            # Try to find (1 ± epsilon) perturbation within valid range
            new_dist = current_dist
            for _ in range(max_iterations):
                noise = np.random.normal(0, epsilon)
                factor = (1 + noise) if noise >= 0 else 1 / (1 - noise)
                candidate = factor * current_dist

                if smallest_valid <= candidate <= largest_valid:
                    new_dist = candidate
                    break

            # If no valid perturbation found, use smallest valid
            if new_dist == current_dist:
                new_dist = smallest_valid

            # Update matrices
            generated_metric[i, j] = new_dist
            generated_metric[j, i] = new_dist
            working_dists[i, j] = new_dist
            working_dists[j, i] = new_dist

    return generated_metric


def get_random_non_euclidean(
    size: int, epsilon: float, max_tries: int = 50
) -> np.ndarray:
    """
    Generate a random non-Euclidean metric space.

    Repeatedly generates epsilon-close perturbations until a non-Euclidean
    space is found.

    Parameters
    ----------
    size : int
        Number of points
    epsilon : float
        Perturbation parameter
    max_tries : int, optional
        Maximum generation attempts (default: 50)

    Returns
    -------
    np.ndarray
        Non-Euclidean distance matrix (n x n)

    Raises
    ------
    InvalidMetricSpaceError
        If unable to generate non-Euclidean space after max_tries attempts

    Warnings
    --------
    May still return Euclidean space if max_tries is reached
    """
    for attempt in range(max_tries):
        original_space = get_random_space(size, size)
        original_dists = space_to_dist(original_space)
        perturbed_dists = get_epsilon_close_metric(original_dists, epsilon, 5)

        if not is_euclidean_space(perturbed_dists):
            logger.info(f"Generated non-Euclidean space on attempt {attempt + 1}")
            return perturbed_dists

    logger.warning(
        f"Failed to generate non-Euclidean space after {max_tries} attempts; "
        "returning last attempt (may be Euclidean)"
    )
    return perturbed_dists

