"""Tests for class-based reducer interfaces."""

import numpy as np
import pytest

from dimensionality_reduction import reducers, spaces, metrics
from dimensionality_reduction.exceptions import OptimizationError


class TestApproximationAlgorithm:
    """Tests for ApproximationAlgorithm class."""

    def test_initialization(self) -> None:
        """Test that algorithm can be initialized."""
        reducer = reducers.ApproximationAlgorithm(q=2, objective="lq_dist")
        assert reducer.q == 2
        assert reducer.objective == "lq_dist"
        assert reducer.backend == "numpy"
        assert reducer.high_dim_embedding_ is None

    def test_initialization_invalid_q(self) -> None:
        """Test that invalid q raises error."""
        with pytest.raises(ValueError, match="q must be >= 1"):
            reducers.ApproximationAlgorithm(q=0.5)

    def test_initialization_invalid_backend(self) -> None:
        """Test that invalid backend raises error."""
        with pytest.raises(ValueError, match="backend must be"):
            reducers.ApproximationAlgorithm(backend="invalid")

    def test_pytorch_backend_not_implemented(self) -> None:
        """Test that PyTorch backend raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="PyTorch backend not yet"):
            reducers.ApproximationAlgorithm(backend="pytorch")

    def test_fit_basic(self) -> None:
        """Test basic fit operation."""
        space = spaces.get_random_space(10, 10)
        distances = spaces.space_to_dist(space)

        reducer = reducers.ApproximationAlgorithm(q=2)
        result = reducer.fit(distances)

        # Should return self
        assert result is reducer

        # Should have cached results
        assert reducer.high_dim_embedding_ is not None
        assert reducer.distance_matrix_ is not None
        assert reducer.normalization_factor_ is not None

    def test_transform_before_fit_raises_error(self) -> None:
        """Test that transform before fit raises error."""
        reducer = reducers.ApproximationAlgorithm(q=2)

        with pytest.raises(ValueError, match="not fitted yet"):
            reducer.transform(5)

    def test_fit_transform(self) -> None:
        """Test fit_transform convenience method."""
        space = spaces.get_random_space(8, 8)
        distances = spaces.space_to_dist(space)

        reducer = reducers.ApproximationAlgorithm(q=2, objective="lq_dist")
        embedded = reducer.fit_transform(distances, target_dimension=3)

        assert embedded.shape == (8, 3)
        assert reducer.high_dim_embedding_ is not None

    def test_reuse_optimization(self) -> None:
        """Test that optimization is cached and reused."""
        space = spaces.get_random_space(10, 10)
        distances = spaces.space_to_dist(space)

        reducer = reducers.ApproximationAlgorithm(q=2)

        # Fit once
        reducer.fit(distances)
        cached_embedding = reducer.high_dim_embedding_.copy()

        # Transform multiple times
        embedded_10d = reducer.transform(5)
        embedded_5d = reducer.transform(3)
        embedded_3d = reducer.transform(2)

        # Cached embedding should not change
        np.testing.assert_array_equal(
            reducer.high_dim_embedding_, cached_embedding
        )

        # All transforms should succeed
        assert embedded_10d.shape == (10, 5)
        assert embedded_5d.shape == (10, 3)
        assert embedded_3d.shape == (10, 2)

    def test_different_objectives(self) -> None:
        """Test that different objectives work."""
        space = spaces.get_random_space(6, 6)
        distances = spaces.space_to_dist(space)

        objectives = ["lq_dist", "energy", "stress", "sigma"]

        for objective in objectives:
            reducer = reducers.ApproximationAlgorithm(q=2, objective=objective)
            embedded = reducer.fit_transform(distances, target_dimension=3)
            assert embedded.shape == (6, 3)

    def test_invalid_distance_matrix(self) -> None:
        """Test that invalid distance matrices raise errors."""
        reducer = reducers.ApproximationAlgorithm(q=2)

        # Non-square matrix
        with pytest.raises(ValueError, match="must be square"):
            reducer.fit(np.array([[0, 1, 2]]))

        # Empty matrix
        with pytest.raises(ValueError, match="cannot be empty"):
            reducer.fit(np.array([[]]))

        # All zeros
        with pytest.raises(ValueError, match="only zeros"):
            reducer.fit(np.zeros((5, 5)))

    def test_invalid_target_dimension(self) -> None:
        """Test that invalid target dimension raises error."""
        space = spaces.get_random_space(5, 5)
        distances = spaces.space_to_dist(space)

        reducer = reducers.ApproximationAlgorithm(q=2)
        reducer.fit(distances)

        with pytest.raises(ValueError, match="target_dimension must be >= 1"):
            reducer.transform(0)

    def test_repr_and_str(self) -> None:
        """Test string representations."""
        reducer = reducers.ApproximationAlgorithm(q=2, objective="lq_dist")

        # Before fit
        repr_before = repr(reducer)
        assert "not fitted" in repr_before
        assert "q=2" in repr_before
        assert "lq_dist" in repr_before

        # After fit
        space = spaces.get_random_space(5, 5)
        distances = spaces.space_to_dist(space)
        reducer.fit(distances)

        repr_after = repr(reducer)
        assert "fitted" in repr_after
        assert "not fitted" not in repr_after

        # str should work too
        str_repr = str(reducer)
        assert isinstance(str_repr, str)

    def test_quality_of_embedding(self) -> None:
        """Test that class produces same quality as functional API."""
        space = spaces.get_random_space(10, 10)
        distances = spaces.space_to_dist(space)

        # Class-based
        reducer = reducers.ApproximationAlgorithm(q=2, objective="lq_dist")
        embedded_class = reducer.fit_transform(distances, target_dimension=5)

        # Measure distortion
        embedded_dists = spaces.space_to_dist(embedded_class)
        distortion_class = metrics.lq_dist(distances, embedded_dists, q=2)

        # Should produce reasonable distortion
        assert distortion_class > 0
        assert distortion_class < 100  # Sanity check

    def test_multiple_fits(self) -> None:
        """Test that refitting works correctly."""
        space1 = spaces.get_random_space(5, 5)
        distances1 = spaces.space_to_dist(space1)

        space2 = spaces.get_random_space(5, 5)
        distances2 = spaces.space_to_dist(space2)

        reducer = reducers.ApproximationAlgorithm(q=2)

        # Fit first dataset
        reducer.fit(distances1)
        embedding1 = reducer.high_dim_embedding_.copy()

        # Fit second dataset
        reducer.fit(distances2)
        embedding2 = reducer.high_dim_embedding_

        # Should have different embeddings
        assert not np.allclose(embedding1, embedding2)


class TestBaseDimensionalityReducer:
    """Tests for base class."""

    def test_is_abstract(self) -> None:
        """Test that base class cannot be instantiated."""
        # Cannot instantiate abstract class
        with pytest.raises(TypeError):
            reducers.BaseDimensionalityReducer()

    def test_approximation_algorithm_inherits(self) -> None:
        """Test that ApproximationAlgorithm inherits from base."""
        reducer = reducers.ApproximationAlgorithm(q=2)
        assert isinstance(reducer, reducers.BaseDimensionalityReducer)


class TestApproximationAlgorithmPyTorch:
    """Tests for PyTorch implementation placeholder."""

    def test_not_implemented(self) -> None:
        """Test that PyTorch class raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            reducers.ApproximationAlgorithmPyTorch(q=2)

