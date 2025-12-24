"""
Dimensionality Reduction with Provable Guarantees

This package implements approximation algorithms for dimensionality reduction
with theoretical guarantees as described in:

Bartal, Y., Fandina, N., & Neiman, O. (2019). 
Dimensionality reduction: theoretical perspective on practical measures. 
NeurIPS 2019.

Main modules:
    - algorithms: Core approximation algorithms
    - metrics: Distortion measures for embeddings
    - spaces: Utilities for metric space operations
"""

__version__ = "0.1.0"
__author__ = "Ora Fandina"
__license__ = "MIT"

from .algorithms import approx_algo
from .metrics import (
    lq_dist,
    wc_distortion,
    rem_q,
    sigma_q,
    energy,
    stress,
    expansion,
    contraction,
    distortion,
)
from .spaces import (
    is_metric_space,
    is_euclidean_space,
    space_from_dists,
    space_to_dist,
    get_random_space,
    get_random_non_euclidean,
    space_to_lp_dists,
)
from .exceptions import (
    DimensionalityReductionError,
    DistortionCalculationError,
    InvalidMetricSpaceError,
    OptimizationError,
    DivisionByZeroError,
)
from .reducers import (
    ApproximationAlgorithm,
    BaseDimensionalityReducer,
)

__all__ = [
    # Main algorithm (functional API)
    "approx_algo",
    # Class-based API (object-oriented)
    "ApproximationAlgorithm",
    "BaseDimensionalityReducer",
    # Distortion metrics
    "lq_dist",
    "wc_distortion",
    "rem_q",
    "sigma_q",
    "energy",
    "stress",
    "expansion",
    "contraction",
    "distortion",
    # Space utilities
    "is_metric_space",
    "is_euclidean_space",
    "space_from_dists",
    "space_to_dist",
    "get_random_space",
    "get_random_non_euclidean",
    "space_to_lp_dists",
    # Exceptions
    "DimensionalityReductionError",
    "DistortionCalculationError",
    "InvalidMetricSpaceError",
    "OptimizationError",
    "DivisionByZeroError",
]

