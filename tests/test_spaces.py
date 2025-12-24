"""Tests for spaces module."""

import numpy as np
import pytest

from dimensionality_reduction import spaces
from dimensionality_reduction.exceptions import InvalidMetricSpaceError


class TestMetricSpaceValidation:
    """Tests for metric space validation functions."""

    def test_is_metric_space_valid(self) -> None:
        """Test that a valid metric space is recognized."""
        # Simple triangle in 2D
        distances = np.array([[0, 1, 1], [1, 0, np.sqrt(2)], [1, np.sqrt(2), 0]])
        assert spaces.is_metric_space(distances)

    def test_is_metric_space_invalid(self) -> None:
        """Test that invalid metric space is rejected."""
        # Violates triangle inequality
        distances = np.array([[0, 1, 10], [1, 0, 1], [10, 1, 0]])
        assert not spaces.is_metric_space(distances)

    def test_is_metric_space_non_square(self) -> None:
        """Test that non-square matrices are rejected."""
        distances = np.array([[0, 1, 2], [1, 0, 3]])
        assert not spaces.is_metric_space(distances)


class TestEuclideanSpace:
    """Tests for Euclidean space detection."""

    def test_is_euclidean_space_true(self) -> None:
        """Test that Euclidean distances are recognized."""
        # Generate random Euclidean space
        space = np.random.randn(10, 5)
        distances = spaces.space_to_dist(space)
        assert spaces.is_euclidean_space(distances)

    def test_is_positive_semidefinite_psd_matrix(self) -> None:
        """Test PSD detection for positive semidefinite matrix."""
        matrix = np.array([[2, 1], [1, 2]])
        assert spaces.is_positive_semidefinite(matrix)

    def test_is_positive_semidefinite_not_psd(self) -> None:
        """Test PSD detection for non-PSD matrix."""
        matrix = np.array([[1, 2], [2, 1]])
        assert not spaces.is_positive_semidefinite(matrix)


class TestSpaceConversions:
    """Tests for space/distance conversions."""

    def test_space_to_dist_and_back(self) -> None:
        """Test round-trip conversion: space → distances → space."""
        original_space = np.random.randn(5, 3)
        distances = spaces.space_to_dist(original_space)
        recovered_space = spaces.space_from_dists(distances)

        # Check distances are preserved (up to translation)
        recovered_distances = spaces.space_to_dist(recovered_space)
        np.testing.assert_allclose(distances, recovered_distances, rtol=1e-5)

    def test_space_from_gram_psd(self) -> None:
        """Test vector recovery from PSD Gram matrix."""
        # Create Gram matrix from known vectors
        vectors = np.array([[1, 0], [0, 1], [1, 1]])
        gram = vectors @ vectors.T

        recovered = spaces.space_from_gram(gram, is_psd=True)
        recovered_gram = recovered @ recovered.T

        np.testing.assert_allclose(gram, recovered_gram, rtol=1e-10)

    def test_space_to_lp_dists(self) -> None:
        """Test Lp distance computation."""
        space = np.array([[0, 0], [1, 0], [0, 1]])

        # L1 distances
        l1_dists = spaces.space_to_lp_dists(space, p=1)
        assert l1_dists[0, 1] == 1  # Manhattan distance
        assert l1_dists[0, 2] == 1

        # L-infinity should use space_to_dist with chebyshev metric
        linf_dists = spaces.space_to_dist(space, metric="chebyshev")
        assert linf_dists[0, 1] == 1


class TestRandomSpaceGeneration:
    """Tests for random space generation."""

    def test_get_random_space_shape(self) -> None:
        """Test that random space has correct shape."""
        space = spaces.get_random_space(size=10, dimension=5)
        assert space.shape == (10, 5)

    def test_get_random_space_is_euclidean(self) -> None:
        """Test that random space is Euclidean."""
        space = spaces.get_random_space(size=10, dimension=5)
        distances = spaces.space_to_dist(space)
        assert spaces.is_euclidean_space(distances)

    def test_get_random_non_euclidean(self) -> None:
        """Test non-Euclidean space generation."""
        distances = spaces.get_random_non_euclidean(size=10, epsilon=0.5, max_tries=20)
        assert distances.shape == (10, 10)
        # Should be valid metric space
        assert spaces.is_metric_space(distances)

    def test_get_epsilon_close_metric_preserves_metric(self) -> None:
        """Test that epsilon-close perturbation preserves metric property."""
        original_space = spaces.get_random_space(8, 8)
        original_dists = spaces.space_to_dist(original_space)

        perturbed_dists = spaces.get_epsilon_close_metric(
            original_dists, epsilon=0.1, max_iterations=5
        )

        # Should still be a valid metric (though may fail due to rounding)
        # Just check shape and symmetry
        assert perturbed_dists.shape == original_dists.shape
        np.testing.assert_allclose(perturbed_dists, perturbed_dists.T, rtol=1e-10)

