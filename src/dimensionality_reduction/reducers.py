"""
Class-based interfaces for dimensionality reduction.

This module provides object-oriented interfaces that:
- Cache optimization results for reuse
- Follow scikit-learn API pattern (fit/transform)
- Support future backend implementations (CPU, GPU)
"""

import logging
from abc import ABC, abstractmethod
from typing import Literal, Optional

import numpy as np

from . import spaces
from .algorithms import (
    _solve_optimization_energy,
    _solve_optimization_lq_dist,
    _solve_optimization_sigma,
    _solve_optimization_stress,
    johnson_lindenstrauss_transform,
)
from .exceptions import OptimizationError

logger = logging.getLogger(__name__)

ObjectiveType = Literal["lq_dist", "stress", "energy", "sigma"]
BackendType = Literal["numpy", "pytorch"]


class BaseDimensionalityReducer(ABC):
    """
    Abstract base class for dimensionality reduction algorithms.
    
    All reducers should implement the fit, transform, and fit_transform methods
    following the scikit-learn API pattern.
    """

    @abstractmethod
    def fit(self, distance_matrix: np.ndarray) -> "BaseDimensionalityReducer":
        """
        Fit the reducer to the distance matrix.

        Parameters
        ----------
        distance_matrix : np.ndarray
            Pairwise distance matrix (n x n)

        Returns
        -------
        self
            Fitted reducer instance
        """
        pass

    @abstractmethod
    def transform(self, target_dimension: int) -> np.ndarray:
        """
        Transform to target dimension using fitted model.

        Parameters
        ----------
        target_dimension : int
            Target dimension k

        Returns
        -------
        np.ndarray
            Embedded vectors (n x k)
        """
        pass

    def fit_transform(
        self, distance_matrix: np.ndarray, target_dimension: int
    ) -> np.ndarray:
        """
        Fit and transform in one step.

        Parameters
        ----------
        distance_matrix : np.ndarray
            Pairwise distance matrix (n x n)
        target_dimension : int
            Target dimension k

        Returns
        -------
        np.ndarray
            Embedded vectors (n x k)
        """
        return self.fit(distance_matrix).transform(target_dimension)


class ApproximationAlgorithm(BaseDimensionalityReducer):
    """
    Approximation algorithm with provable guarantees.

    This class provides a stateful interface that caches the optimization
    result, allowing efficient projection to multiple target dimensions
    without re-solving the expensive convex program.

    Parameters
    ----------
    q : float, default=2
        Order parameter for distortion measure (q >= 1)
    objective : {'lq_dist', 'stress', 'energy', 'sigma'}, default='lq_dist'
        Distortion measure to optimize
    backend : {'numpy', 'pytorch'}, default='numpy'
        Computation backend (pytorch support coming soon)
    device : str, optional
        Device for computation (e.g., 'cuda:0' for GPU, 'cpu' for CPU)
        Only used with pytorch backend

    Attributes
    ----------
    high_dim_embedding_ : np.ndarray or None
        Cached high-dimensional embedding from optimization
    distance_matrix_ : np.ndarray or None
        Fitted distance matrix
    normalization_factor_ : float or None
        Scale factor for denormalization

    Examples
    --------
    >>> import dimensionality_reduction as dr
    >>> space = dr.get_random_space(100, 50)
    >>> distances = dr.space_to_dist(space)
    
    >>> # Create reducer and fit once
    >>> reducer = dr.ApproximationAlgorithm(q=2, objective='lq_dist')
    >>> reducer.fit(distances)
    
    >>> # Transform to multiple dimensions (reuses optimization!)
    >>> embedded_10d = reducer.transform(10)  # Fast!
    >>> embedded_5d = reducer.transform(5)    # Fast!
    >>> embedded_3d = reducer.transform(3)    # Fast!
    
    >>> # Or do both at once
    >>> embedded = reducer.fit_transform(distances, 10)

    Notes
    -----
    The expensive convex optimization is performed once during `fit()`.
    Subsequent `transform()` calls only perform fast random projection.
    
    For GPU acceleration (coming soon):
    >>> reducer = dr.ApproximationAlgorithm(backend='pytorch', device='cuda:0')
    """

    def __init__(
        self,
        q: float = 2,
        objective: ObjectiveType = "lq_dist",
        backend: BackendType = "numpy",
        device: Optional[str] = None,
    ):
        if q < 1:
            raise ValueError(f"q must be >= 1, got {q}")

        if backend not in ["numpy", "pytorch"]:
            raise ValueError(f"backend must be 'numpy' or 'pytorch', got {backend}")

        if backend == "pytorch":
            raise NotImplementedError(
                "PyTorch backend not yet implemented. "
                "Use backend='numpy' for now. GPU support coming soon!"
            )

        self.q = q
        self.objective = objective
        self.backend = backend
        self.device = device

        # State (filled during fit)
        self.high_dim_embedding_: Optional[np.ndarray] = None
        self.distance_matrix_: Optional[np.ndarray] = None
        self.normalization_factor_: Optional[float] = None

    def fit(self, distance_matrix: np.ndarray) -> "ApproximationAlgorithm":
        """
        Solve convex optimization and cache the result.

        This performs the expensive optimization step once. The result
        is cached so that subsequent calls to transform() are fast.

        Parameters
        ----------
        distance_matrix : np.ndarray
            Pairwise distance matrix (n x n)

        Returns
        -------
        self
            Fitted instance

        Raises
        ------
        ValueError
            If distance_matrix is invalid
        OptimizationError
            If optimization fails
        """
        if distance_matrix.ndim != 2:
            raise ValueError("distance_matrix must be 2-dimensional")

        rows, cols = distance_matrix.shape
        if rows != cols:
            raise ValueError("distance_matrix must be square")

        if rows == 0:
            raise ValueError("distance_matrix cannot be empty")

        logger.info(
            f"Fitting ApproximationAlgorithm: n={rows}, q={self.q}, "
            f"objective={self.objective}, backend={self.backend}"
        )

        # Store original
        self.distance_matrix_ = distance_matrix

        # Normalize by largest distance
        max_dist = np.amax(distance_matrix)
        if max_dist == 0:
            raise ValueError("distance_matrix contains only zeros")

        self.normalization_factor_ = max_dist
        normalized_dists = distance_matrix / max_dist

        # Solve optimization (expensive step - cached!)
        logger.info("Solving convex optimization (this may take a while)...")

        if self.objective == "lq_dist":
            self.high_dim_embedding_ = _solve_optimization_lq_dist(
                normalized_dists, self.q
            )
        elif self.objective == "energy":
            self.high_dim_embedding_ = _solve_optimization_energy(
                normalized_dists, self.q
            )
        elif self.objective == "stress":
            self.high_dim_embedding_ = _solve_optimization_stress(
                normalized_dists, self.q
            )
        elif self.objective == "sigma":
            self.high_dim_embedding_ = _solve_optimization_sigma(
                normalized_dists, self.q
            )
        else:
            raise ValueError(f"Unknown objective: {self.objective}")

        logger.info("Optimization complete - result cached for reuse")
        return self

    def transform(self, target_dimension: int) -> np.ndarray:
        """
        Project to target dimension using cached optimization result.

        This is FAST because it only performs random projection,
        not the expensive convex optimization.

        Parameters
        ----------
        target_dimension : int
            Target dimension k (>= 1)

        Returns
        -------
        np.ndarray
            Embedded vectors (n x k)

        Raises
        ------
        ValueError
            If not fitted or target_dimension is invalid
        """
        if self.high_dim_embedding_ is None:
            raise ValueError(
                "This ApproximationAlgorithm instance is not fitted yet. "
                "Call fit() before transform()."
            )

        if target_dimension < 1:
            raise ValueError(f"target_dimension must be >= 1, got {target_dimension}")

        n = self.high_dim_embedding_.shape[0]
        if target_dimension > n:
            logger.warning(
                f"target_dimension ({target_dimension}) > n ({n}). "
                f"Result will be padded with zeros."
            )

        logger.info(f"Projecting to {target_dimension} dimensions (fast operation)")

        # Apply Johnson-Lindenstrauss projection
        low_dim_space = johnson_lindenstrauss_transform(
            self.high_dim_embedding_, target_dimension
        )

        # Restore original scale
        result = low_dim_space * self.normalization_factor_

        logger.info("Projection complete")
        return result

    def __repr__(self) -> str:
        """String representation."""
        fitted_status = "fitted" if self.high_dim_embedding_ is not None else "not fitted"
        return (
            f"ApproximationAlgorithm(q={self.q}, objective='{self.objective}', "
            f"backend='{self.backend}', {fitted_status})"
        )

    def __str__(self) -> str:
        """User-friendly string representation."""
        return self.__repr__()


# TODO: Future PyTorch implementation
class ApproximationAlgorithmPyTorch(BaseDimensionalityReducer):
    """
    PyTorch-based approximation algorithm for GPU acceleration.

    This will be implemented in a future version to support:
    - GPU acceleration with CUDA
    - Faster optimization with GPU solvers
    - Batch processing of multiple datasets

    Parameters
    ----------
    q : float
        Order parameter
    objective : str
        Distortion measure
    device : str, default='cuda:0'
        PyTorch device (e.g., 'cuda:0', 'cuda:1', 'cpu')

    Notes
    -----
    Coming soon! This will use:
    - cvxpylayers for differentiable optimization
    - PyTorch tensors for GPU computation
    - Mixed precision for speed

    Examples
    --------
    >>> # Future usage:
    >>> reducer = dr.ApproximationAlgorithmPyTorch(
    ...     q=2, device='cuda:0'
    ... )
    >>> reducer.fit(distances)  # Runs on GPU!
    >>> embedded = reducer.transform(10)
    """

    def __init__(self, q: float = 2, objective: str = "lq_dist", device: str = "cuda:0"):
        raise NotImplementedError(
            "PyTorch backend not yet implemented. "
            "This will be added in a future version for GPU acceleration. "
            "For now, use ApproximationAlgorithm with backend='numpy'."
        )

    def fit(self, distance_matrix: np.ndarray) -> "ApproximationAlgorithmPyTorch":
        raise NotImplementedError

    def transform(self, target_dimension: int) -> np.ndarray:
        raise NotImplementedError

