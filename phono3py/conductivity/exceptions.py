"""Exceptions for conductivity modules."""


class LBTECollisionReadError(RuntimeError):
    """Raised when collision data requested for LBTE cannot be loaded."""
