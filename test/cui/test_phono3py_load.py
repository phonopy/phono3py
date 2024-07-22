"""Tests of Phono3py load."""

from __future__ import annotations

from phono3py import Phono3py


def test_phono3py_load(si_pbesol_without_forcesets: Phono3py):
    """Test phono3py.load.

    Check phono3py.load can read displacements from phono3py_disp.yaml like file
    that doesn't contain forces.

    """
    ph3 = si_pbesol_without_forcesets
    assert ph3.dataset is not None
    assert ph3.displacements.shape == (111, 64, 3)
    assert ph3.forces is None
