"""
Distortion measures for evaluating embedding quality.

This module provides various distortion measures including:
- Worst-case distortion
- lq-distortion
- REM (Relative Error Measure)
- Sigma distortion
- Energy distortion
- Stress distortion
"""

import logging
from typing import Tuple

import numpy as np
from numpy import linalg as LA

from .exceptions import DivisionByZeroError, DistortionCalculationError

logger = logging.getLogger(__name__)


def expansion(old_dist: float, new_dist: float) -> float:
    """
    Calculate expansion factor for a pair of points.

    Parameters
    ----------
    old_dist : float
        Original distance between points
    new_dist : float
        New distance after embedding

    Returns
    -------
    float
        Expansion factor (new_dist / old_dist)

    Raises
    ------
    DivisionByZeroError
        If old_dist is zero
    """
    if old_dist == 0:
        raise DivisionByZeroError("Cannot compute expansion: original distance is zero")
    return new_dist / old_dist


def contraction(old_dist: float, new_dist: float) -> float:
    """
    Calculate contraction factor for a pair of points.

    Parameters
    ----------
    old_dist : float
        Original distance between points
    new_dist : float
        New distance after embedding

    Returns
    -------
    float
        Contraction factor (old_dist / new_dist)

    Raises
    ------
    DivisionByZeroError
        If new_dist is zero
    """
    if new_dist == 0:
        raise DivisionByZeroError("Cannot compute contraction: new distance is zero")
    return old_dist / new_dist


def distortion(old_dist: float, new_dist: float) -> float:
    """
    Calculate distortion for a single pair of points.

    Distortion is defined as max(expansion, contraction).

    Parameters
    ----------
    old_dist : float
        Original distance between points
    new_dist : float
        New distance after embedding

    Returns
    -------
    float
        Distortion factor
    """
    exp = expansion(old_dist, new_dist)
    con = contraction(old_dist, new_dist)
    return max(exp, con)


def _distortion_vectors(
    input_dists: np.ndarray, embedded_dists: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute vectors of contractions and expansions for all point pairs.

    Parameters
    ----------
    input_dists : np.ndarray
        Original distance matrix (n x n)
    embedded_dists : np.ndarray
        Embedded distance matrix (n x n)

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (contractions, expansions) vectors containing values for lower triangle

    Raises
    ------
    DistortionCalculationError
        If any pair has been contracted to zero distance
    """
    # Extract lower triangle (excluding diagonal)
    mask = np.tri(input_dists.shape[0], input_dists.shape[0], -1, bool)
    old_dists = input_dists[mask]
    new_dists = embedded_dists[mask]

    if np.any(new_dists == 0):
        raise DistortionCalculationError(
            "Cannot compute distortion: some pairs contracted to zero distance"
        )

    if np.any(old_dists == 0):
        logger.warning("Original distances contain zeros; results may be unreliable")

    contractions = old_dists / new_dists
    expansions = new_dists / old_dists

    return contractions, expansions


def wc_distortion(input_dists: np.ndarray, embedded_dists: np.ndarray) -> float:
    """
    Calculate worst-case distortion.

    Worst-case distortion is the product of maximum contraction and
    maximum expansion across all pairs.

    Parameters
    ----------
    input_dists : np.ndarray
        Original distance matrix (n x n)
    embedded_dists : np.ndarray
        Embedded distance matrix (n x n)

    Returns
    -------
    float
        Worst-case distortion value
    """
    contractions, expansions = _distortion_vectors(input_dists, embedded_dists)
    max_contraction = np.amax(contractions)
    max_expansion = np.amax(expansions)
    return max_contraction * max_expansion


def lq_dist(input_dists: np.ndarray, embedded_dists: np.ndarray, q: float) -> float:
    """
    Calculate lq-distortion measure.

    The lq-distortion is defined as:
        (1/C(n,2))^(1/q) * ||distortions||_q

    where distortions = max(expansions, contractions) for each pair.

    Parameters
    ----------
    input_dists : np.ndarray
        Original distance matrix (n x n)
    embedded_dists : np.ndarray
        Embedded distance matrix (n x n)
    q : float
        Order parameter (q >= 1)

    Returns
    -------
    float
        lq-distortion value
    """
    contractions, expansions = _distortion_vectors(input_dists, embedded_dists)
    distortions = np.maximum(contractions, expansions)
    num_pairs = len(distortions)
    return LA.norm(distortions, ord=q) / (num_pairs ** (1 / q))


def rem_q(input_dists: np.ndarray, embedded_dists: np.ndarray, q: float) -> float:
    """
    Calculate REM (Relative Error Measure) distortion.

    REM is defined as the lq-norm of (distortion - 1) values.

    Parameters
    ----------
    input_dists : np.ndarray
        Original distance matrix (n x n)
    embedded_dists : np.ndarray
        Embedded distance matrix (n x n)
    q : float
        Order parameter (q >= 1)

    Returns
    -------
    float
        REM distortion value
    """
    contractions, expansions = _distortion_vectors(input_dists, embedded_dists)
    distortions = np.maximum(contractions, expansions)
    num_pairs = len(distortions)
    rem_values = distortions - np.ones((num_pairs,))
    return LA.norm(rem_values, ord=q) / (num_pairs ** (1 / q))


def sigma_q(input_dists: np.ndarray, embedded_dists: np.ndarray, q: float) -> float:
    """
    Calculate sigma distortion measure (with r=1).

    Sigma distortion normalizes by average expansion before computing error.

    Parameters
    ----------
    input_dists : np.ndarray
        Original distance matrix (n x n)
    embedded_dists : np.ndarray
        Embedded distance matrix (n x n)
    q : float
        Order parameter (q >= 1)

    Returns
    -------
    float
        Sigma distortion value
    """
    contractions, expansions = _distortion_vectors(input_dists, embedded_dists)
    num_pairs = len(contractions)

    avg_expansion = LA.norm(expansions, 1) / num_pairs
    normalized_values = (expansions / avg_expansion) - np.ones((num_pairs,))

    return LA.norm(normalized_values, ord=q) / (num_pairs ** (1 / q))


def energy(input_dists: np.ndarray, embedded_dists: np.ndarray, q: float) -> float:
    """
    Calculate energy distortion measure.

    Energy is the lq-norm of (expansion - 1) values.

    Parameters
    ----------
    input_dists : np.ndarray
        Original distance matrix (n x n)
    embedded_dists : np.ndarray
        Embedded distance matrix (n x n)
    q : float
        Order parameter (q >= 1)

    Returns
    -------
    float
        Energy distortion value
    """
    contractions, expansions = _distortion_vectors(input_dists, embedded_dists)
    num_pairs = len(contractions)
    energy_values = expansions - np.ones((num_pairs,))
    return LA.norm(energy_values, ord=q) / (num_pairs ** (1 / q))


def stress(input_dists: np.ndarray, embedded_dists: np.ndarray, q: float) -> float:
    """
    Calculate stress distortion measure.

    Stress is the relative lq-norm of additive distance errors.

    Parameters
    ----------
    input_dists : np.ndarray
        Original distance matrix (n x n)
    embedded_dists : np.ndarray
        Embedded distance matrix (n x n)
    q : float
        Order parameter (q >= 1)

    Returns
    -------
    float
        Stress distortion value
    """
    additive_error = LA.norm(input_dists - embedded_dists, ord=q)
    return additive_error / LA.norm(input_dists, ord=q)

