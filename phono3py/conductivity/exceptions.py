"""Exceptions for conductivity modules."""


class LBTECollisionReadError(RuntimeError):
    """Raised when collision data requested for LBTE cannot be loaded."""


class LBTEInputError(ValueError):
    """Raised when LBTE cannot run with given grid points or band indices."""
