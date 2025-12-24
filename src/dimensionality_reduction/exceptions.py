"""Custom exceptions for the dimensionality_reduction package."""


class DimensionalityReductionError(Exception):
    """Base exception for all package-specific errors."""
    pass


class DistortionCalculationError(DimensionalityReductionError):
    """Raised when distortion calculation fails."""
    pass


class InvalidMetricSpaceError(DimensionalityReductionError):
    """Raised when a metric space is invalid or malformed."""
    pass


class OptimizationError(DimensionalityReductionError):
    """Raised when convex optimization fails."""
    pass


class DivisionByZeroError(DistortionCalculationError):
    """Raised when attempting to calculate distortion with zero distances."""
    pass

