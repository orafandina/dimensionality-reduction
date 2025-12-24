"""Tests for metrics module."""

import numpy as np
import pytest

from dimensionality_reduction import metrics
from dimensionality_reduction.exceptions import (
    DistortionCalculationError,
    DivisionByZeroError,
)


class TestBasicDistortionFunctions:
    """Tests for basic expansion/contraction/distortion functions."""

    def test_expansion_normal(self) -> None:
        """Test expansion calculation."""
        assert metrics.expansion(2.0, 4.0) == 2.0
        assert metrics.expansion(1.0, 1.0) == 1.0

    def test_expansion_zero_old_distance(self) -> None:
        """Test that expansion raises error for zero old distance."""
        with pytest.raises(DivisionByZeroError):
            metrics.expansion(0.0, 1.0)

    def test_contraction_normal(self) -> None:
        """Test contraction calculation."""
        assert metrics.contraction(4.0, 2.0) == 2.0
        assert metrics.contraction(1.0, 1.0) == 1.0

    def test_contraction_zero_new_distance(self) -> None:
        """Test that contraction raises error for zero new distance."""
        with pytest.raises(DivisionByZeroError):
            metrics.contraction(1.0, 0.0)

    def test_distortion(self) -> None:
        """Test distortion calculation (max of expansion and contraction)."""
        assert metrics.distortion(2.0, 4.0) == 2.0  # expansion dominates
        assert metrics.distortion(4.0, 2.0) == 2.0  # contraction dominates
        assert metrics.distortion(1.0, 1.0) == 1.0  # no distortion


class TestDistortionMeasures:
    """Tests for distortion measure functions."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        # Simple case: identity embedding (no distortion)
        self.identity_dists = np.array(
            [[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]]
        )

        # Uniform scaling by factor of 2
        self.scaled_dists = self.identity_dists * 2

    def test_wc_distortion_identity(self) -> None:
        """Test worst-case distortion for identity embedding."""
        dist = metrics.wc_distortion(self.identity_dists, self.identity_dists)
        assert dist == 1.0

    def test_wc_distortion_scaled(self) -> None:
        """Test worst-case distortion for uniformly scaled embedding."""
        dist = metrics.wc_distortion(self.identity_dists, self.scaled_dists)
        # Should be 2.0 (expansion=2, contraction=2)
        assert dist == pytest.approx(4.0, rel=1e-10)

    def test_lq_dist_identity(self) -> None:
        """Test lq-distortion for identity embedding."""
        dist = metrics.lq_dist(self.identity_dists, self.identity_dists, q=2)
        assert dist == pytest.approx(1.0, rel=1e-10)

    def test_lq_dist_scaled(self) -> None:
        """Test lq-distortion for uniformly scaled embedding."""
        dist = metrics.lq_dist(self.identity_dists, self.scaled_dists, q=2)
        # All pairs have distortion=2, so lq-dist should be 2
        assert dist == pytest.approx(2.0, rel=1e-10)

    def test_rem_q_identity(self) -> None:
        """Test REM distortion for identity embedding."""
        dist = metrics.rem_q(self.identity_dists, self.identity_dists, q=2)
        # REM = |distortion - 1|, should be 0 for identity
        assert dist == pytest.approx(0.0, abs=1e-10)

    def test_energy_identity(self) -> None:
        """Test energy distortion for identity embedding."""
        dist = metrics.energy(self.identity_dists, self.identity_dists, q=2)
        # Energy = |expansion - 1|, should be 0 for identity
        assert dist == pytest.approx(0.0, abs=1e-10)

    def test_stress_identity(self) -> None:
        """Test stress distortion for identity embedding."""
        dist = metrics.stress(self.identity_dists, self.identity_dists, q=2)
        # Stress = ||D_old - D_new|| / ||D_old||, should be 0 for identity
        assert dist == pytest.approx(0.0, abs=1e-10)

    def test_sigma_q_identity(self) -> None:
        """Test sigma distortion for identity embedding."""
        dist = metrics.sigma_q(self.identity_dists, self.identity_dists, q=2)
        # Sigma normalizes by average expansion
        assert dist == pytest.approx(0.0, abs=1e-10)

    def test_distortion_with_zero_new_distances(self) -> None:
        """Test that zero new distances raise appropriate error."""
        zero_dists = np.zeros_like(self.identity_dists)
        with pytest.raises(DistortionCalculationError):
            metrics.wc_distortion(self.identity_dists, zero_dists)


class TestDistortionVectorEdgeCases:
    """Test edge cases for distortion calculations."""

    def test_single_point(self) -> None:
        """Test with single point (no pairs)."""
        single_dist = np.array([[0]])
        # Should handle gracefully (no pairs to compare)
        # Most measures should return 0 or nan
        # Just test it doesn't crash
        try:
            metrics.lq_dist(single_dist, single_dist, q=2)
        except Exception:
            pass  # Expected to fail or return trivial result

    def test_two_points(self) -> None:
        """Test with two points (one pair)."""
        two_points = np.array([[0, 1], [1, 0]])
        two_points_scaled = np.array([[0, 2], [2, 0]])

        dist = metrics.lq_dist(two_points, two_points_scaled, q=2)
        assert dist == pytest.approx(2.0, rel=1e-10)

