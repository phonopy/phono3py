"""Compare the Rust real-self-energy backend against the C one.

Uses ``RealSelfEnergy`` with ``lang="C"`` vs ``lang="Rust"`` at the same
grid point, checking both the band-index and the frequency-point paths
for finite and zero temperature.

"""

from __future__ import annotations

import numpy as np
import pytest

from phono3py import Phono3py
from phono3py.phonon3.real_self_energy import RealSelfEnergy

pytest.importorskip("phonors")
pytest.importorskip("phono3py._phono3py")


def _build_interaction(ph3: Phono3py) -> None:
    ph3.mesh_numbers = [9, 9, 9]
    ph3.init_phph_interaction()


def _run_rse(
    ph3: Phono3py,
    *,
    lang: str,
    grid_point: int,
    temperature: float,
    frequency_points: np.ndarray | None = None,
) -> np.ndarray:
    assert ph3.phph_interaction is not None
    rse = RealSelfEnergy(
        ph3.phph_interaction,
        grid_point=grid_point,
        temperature=temperature,
        lang=lang,
    )
    if frequency_points is not None:
        rse.frequency_points = frequency_points
    rse.run()
    return np.array(rse.real_self_energy)


@pytest.mark.parametrize("temperature", [300.0, 0.0])
def test_real_self_energy_rust_vs_c_bands(si_pbesol: Phono3py, temperature: float):
    """Band-index path: Rust matches C at T > 0 and T = 0."""
    _build_interaction(si_pbesol)
    gp = int(si_pbesol.grid.grg2bzg[103])
    out_c = _run_rse(si_pbesol, lang="C", grid_point=gp, temperature=temperature)
    out_rust = _run_rse(si_pbesol, lang="Rust", grid_point=gp, temperature=temperature)
    np.testing.assert_allclose(out_rust, out_c, rtol=1e-10, atol=1e-14)


@pytest.mark.parametrize("temperature", [300.0, 0.0])
def test_real_self_energy_rust_vs_c_frequency_points(
    si_pbesol: Phono3py, temperature: float
):
    """Frequency-point path: Rust matches C at T > 0 and T = 0."""
    _build_interaction(si_pbesol)
    gp = int(si_pbesol.grid.grg2bzg[103])
    fpoints = np.linspace(0.0, 16.0, 9, dtype="double")
    out_c = _run_rse(
        si_pbesol,
        lang="C",
        grid_point=gp,
        temperature=temperature,
        frequency_points=fpoints,
    )
    out_rust = _run_rse(
        si_pbesol,
        lang="Rust",
        grid_point=gp,
        temperature=temperature,
        frequency_points=fpoints,
    )
    np.testing.assert_allclose(out_rust, out_c, rtol=1e-10, atol=1e-14)
