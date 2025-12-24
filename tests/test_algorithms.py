"""Tests for algorithms module."""

import numpy as np
import pytest

from dimensionality_reduction import algorithms, metrics, spaces
from dimensionality_reduction.exceptions import OptimizationError


class TestJohnsonLindenstraussTransform:
    """Tests for JL random projection."""

    def test_jl_transform_shape(self) -> None:
        """Test that JL transform produces correct output shape."""
        space = np.random.randn(20, 50)
        target_dim = 10

        result = algorithms.johnson_lindenstrauss_transform(space, target_dim)
        assert result.shape == (20, target_dim)

    def test_jl_transform_approximately_preserves_distances(self) -> None:
        """Test that JL transform approximately preserves distances."""
        # Use enough points and dimensions for JL lemma to apply
        space = np.random.randn(100, 100)
        original_dists = spaces.space_to_dist(space)

        target_dim = 20
        projected = algorithms.johnson_lindenstrauss_transform(space, target_dim)
        projected_dists = spaces.space_to_dist(projected)

        # Check that distances are roughly preserved (not exact, due to randomness)
        # This is a weak test; JL lemma gives probabilistic guarantees
        distortion = metrics.lq_dist(original_dists, projected_dists, q=2)
        # Expect reasonably small distortion (though not 1.0)
        assert distortion < 5.0  # Very loose bound


class TestApproxAlgo:
    """Tests for the main approximation algorithm."""

    def test_approx_algo_small_euclidean(self) -> None:
        """Test approx_algo on small Euclidean space."""
        # Small test case
        original_space = spaces.get_random_space(8, 8)
        original_dists = spaces.space_to_dist(original_space)

        target_dim = 3
        q = 2

        embedded = algorithms.approx_algo(
            original_dists, target_dim, q, objective="lq_dist"
        )

        # Check output shape
        assert embedded.shape == (8, target_dim)

        # Check that result is a valid embedding
        embedded_dists = spaces.space_to_dist(embedded)
        assert embedded_dists.shape == original_dists.shape

        # Distortion should be finite and reasonable
        distortion = metrics.lq_dist(original_dists, embedded_dists, q)
        assert distortion > 0
        assert distortion < 100  # Sanity check

    def test_approx_algo_invalid_dimension(self) -> None:
        """Test that invalid target dimension raises error."""
        dists = np.array([[0, 1], [1, 0]])

        with pytest.raises(ValueError, match="target_dimension must be >= 1"):
            algorithms.approx_algo(dists, target_dimension=0, q=2)

    def test_approx_algo_invalid_q(self) -> None:
        """Test that invalid q raises error."""
        dists = np.array([[0, 1], [1, 0]])

        with pytest.raises(ValueError, match="q must be >= 1"):
            algorithms.approx_algo(dists, target_dimension=2, q=0.5)

    def test_approx_algo_non_square_matrix(self) -> None:
        """Test that non-square distance matrix raises error."""
        dists = np.array([[0, 1, 2]])

        with pytest.raises(ValueError, match="must be square"):
            algorithms.approx_algo(dists, target_dimension=2, q=2)

    def test_approx_algo_zero_distances(self) -> None:
        """Test that all-zero distance matrix raises error."""
        dists = np.zeros((5, 5))

        with pytest.raises(ValueError, match="contains only zeros"):
            algorithms.approx_algo(dists, target_dimension=2, q=2)

    def test_approx_algo_different_objectives(self) -> None:
        """Test that different objectives run without error."""
        original_space = spaces.get_random_space(6, 6)
        original_dists = spaces.space_to_dist(original_space)

        target_dim = 3
        q = 2

        objectives = ["lq_dist", "energy", "stress", "sigma"]

        for objective in objectives:
            embedded = algorithms.approx_algo(
                original_dists, target_dim, q, objective=objective
            )
            assert embedded.shape == (6, target_dim)

    def test_approx_algo_unknown_objective(self) -> None:
        """Test that unknown objective raises error."""
        dists = spaces.space_to_dist(spaces.get_random_space(5, 5))

        with pytest.raises(ValueError, match="Unknown objective"):
            algorithms.approx_algo(dists, target_dimension=2, q=2, objective="invalid")


class TestOptimizationSolvers:
    """Tests for individual optimization solvers."""

    def test_solve_optimization_lq_dist(self) -> None:
        """Test lq-dist optimization solver."""
        dists = spaces.space_to_dist(spaces.get_random_space(5, 5))
        normalized_dists = dists / np.amax(dists)

        result = algorithms._solve_optimization_lq_dist(normalized_dists, q=2)

        assert result.shape[0] == 5  # Should have 5 vectors
        assert result.ndim == 2

    def test_solve_optimization_energy(self) -> None:
        """Test energy optimization solver."""
        dists = spaces.space_to_dist(spaces.get_random_space(5, 5))
        normalized_dists = dists / np.amax(dists)

        result = algorithms._solve_optimization_energy(normalized_dists, q=2)

        assert result.shape[0] == 5
        assert result.ndim == 2

    def test_solve_optimization_stress(self) -> None:
        """Test stress optimization solver."""
        dists = spaces.space_to_dist(spaces.get_random_space(5, 5))
        normalized_dists = dists / np.amax(dists)

        result = algorithms._solve_optimization_stress(normalized_dists, q=2)

        assert result.shape[0] == 5
        assert result.ndim == 2

    def test_solve_optimization_sigma(self) -> None:
        """Test sigma optimization solver."""
        dists = spaces.space_to_dist(spaces.get_random_space(5, 5))
        normalized_dists = dists / np.amax(dists)

        result = algorithms._solve_optimization_sigma(normalized_dists, q=2)

        assert result.shape[0] == 5
        assert result.ndim == 2

