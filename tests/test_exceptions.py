"""Tests for custom exceptions."""

import pytest

from dimensionality_reduction.exceptions import (
    DimensionalityReductionError,
    DistortionCalculationError,
    DivisionByZeroError,
    InvalidMetricSpaceError,
    OptimizationError,
)


def test_base_exception() -> None:
    """Test base exception can be raised and caught."""
    with pytest.raises(DimensionalityReductionError):
        raise DimensionalityReductionError("Test error")


def test_distortion_calculation_error() -> None:
    """Test distortion calculation error."""
    with pytest.raises(DistortionCalculationError):
        raise DistortionCalculationError("Test distortion error")


def test_invalid_metric_space_error() -> None:
    """Test invalid metric space error."""
    with pytest.raises(InvalidMetricSpaceError):
        raise InvalidMetricSpaceError("Test metric space error")


def test_optimization_error() -> None:
    """Test optimization error."""
    with pytest.raises(OptimizationError):
        raise OptimizationError("Test optimization error")


def test_division_by_zero_error() -> None:
    """Test division by zero error."""
    with pytest.raises(DivisionByZeroError):
        raise DivisionByZeroError("Test division error")


def test_exception_hierarchy() -> None:
    """Test that specific exceptions inherit from base."""
    assert issubclass(DistortionCalculationError, DimensionalityReductionError)
    assert issubclass(InvalidMetricSpaceError, DimensionalityReductionError)
    assert issubclass(OptimizationError, DimensionalityReductionError)
    assert issubclass(DivisionByZeroError, DistortionCalculationError)

