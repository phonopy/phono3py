"""Compare the Rust imag_self_energy_with_g backend against the C one."""

from __future__ import annotations

import numpy as np
import pytest

from phono3py import Phono3py
from phono3py.phonon3.imag_self_energy import (
    ImagSelfEnergy,
    run_imag_self_energy_with_g_rust,
)

pytest.importorskip("phonors")


def _random_inputs(
    num_triplets: int,
    num_band0: int,
    num_band: int,
    g_dim: int,
    *,
    seed: int = 0,
    g_zero_density: float = 0.3,
):
    """Build a consistent set of synthetic inputs for both backends."""
    rng = np.random.default_rng(seed)
    pp_strength = rng.standard_normal(
        (num_triplets, num_band0, num_band, num_band)
    ).astype("double")
    # Make some frequencies small to exercise the cutoff path.
    num_grid = max(8, num_triplets + 3)
    frequencies = (rng.random((num_grid, num_band)) * 5.0 + 0.01).astype("double")
    triplets = np.stack(
        [
            rng.integers(0, num_grid, size=num_triplets, dtype="int64"),
            rng.integers(0, num_grid, size=num_triplets, dtype="int64"),
            rng.integers(0, num_grid, size=num_triplets, dtype="int64"),
        ],
        axis=1,
    )
    weights = rng.integers(1, 5, size=num_triplets, dtype="int64")
    g = rng.standard_normal((2, num_triplets, g_dim, num_band, num_band)).astype(
        "double"
    )
    g_zero = (
        rng.random((num_triplets, g_dim, num_band, num_band)) < g_zero_density
    ).astype("byte")
    return pp_strength, frequencies, triplets, weights, g, g_zero


def _run_c(
    num_band0: int,
    pp_strength,
    triplets,
    weights,
    frequencies,
    temperature_thz: float,
    g,
    g_zero,
    cutoff_frequency: float,
    frequency_point_index: int,
):
    import phono3py._phono3py as phono3c  # type: ignore[import-untyped]

    out = np.zeros(num_band0, dtype="double")
    phono3c.imag_self_energy_with_g(
        out,
        pp_strength,
        triplets,
        weights,
        frequencies,
        temperature_thz,
        g,
        g_zero,
        cutoff_frequency,
        frequency_point_index,
    )
    return out


@pytest.mark.parametrize("temperature_thz", [0.0, 5.0])
def test_imag_self_energy_rust_vs_c_band_mode(temperature_thz: float):
    """Band-index mode: frequency_point_index = -1, g third dim = num_band0."""
    num_triplets, num_band0, num_band = 5, 3, 6
    cutoff = 0.05
    pp, freqs, triplets, weights, g, g_zero = _random_inputs(
        num_triplets, num_band0, num_band, g_dim=num_band0, seed=1
    )

    out_c = _run_c(
        num_band0, pp, triplets, weights, freqs, temperature_thz, g, g_zero, cutoff, -1
    )
    out_rust = np.zeros(num_band0, dtype="double")
    run_imag_self_energy_with_g_rust(
        out_rust,
        pp,
        triplets,
        weights,
        freqs,
        temperature_thz,
        g,
        g_zero,
        cutoff,
        frequency_point_index=-1,
    )
    np.testing.assert_allclose(out_rust, out_c, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("freq_point_idx", [0, 2])
@pytest.mark.parametrize("temperature_thz", [0.0, 5.0])
def test_imag_self_energy_rust_vs_c_freq_point_mode(
    freq_point_idx: int, temperature_thz: float
):
    """Frequency-point mode: g third dim = num_frequency_points."""
    num_triplets, num_band0, num_band = 4, 2, 5
    num_freq_points = 3
    cutoff = 0.05
    pp, freqs, triplets, weights, g, g_zero = _random_inputs(
        num_triplets, num_band0, num_band, g_dim=num_freq_points, seed=7
    )

    out_c = _run_c(
        num_band0,
        pp,
        triplets,
        weights,
        freqs,
        temperature_thz,
        g,
        g_zero,
        cutoff,
        freq_point_idx,
    )
    out_rust = np.zeros(num_band0, dtype="double")
    run_imag_self_energy_with_g_rust(
        out_rust,
        pp,
        triplets,
        weights,
        freqs,
        temperature_thz,
        g,
        g_zero,
        cutoff,
        frequency_point_index=freq_point_idx,
    )
    np.testing.assert_allclose(out_rust, out_c, rtol=1e-12, atol=1e-14)


def _build_ise(ph3: Phono3py, *, with_detail: bool, lang: str) -> ImagSelfEnergy:
    """Build an ImagSelfEnergy with the specified ISE backend.

    The underlying ``Interaction`` is always built with ``lang="C"`` so that
    ``pp_strength`` is identical between the two ISE backends being compared.
    Rust phonon-solver and Rust interaction are exercised in the
    ``test_interaction_rust`` tests separately.
    """
    ph3.mesh_numbers = [4, 4, 4]
    assert ph3.grid is not None
    from phono3py.phonon3.interaction import Interaction

    itr = Interaction(
        ph3.primitive,
        ph3.grid,
        ph3.primitive_symmetry,
        fc3=ph3.fc3,
        cutoff_frequency=1e-4,
    )
    itr.init_dynamical_matrix(ph3.fc2, ph3.phonon_supercell, ph3.phonon_primitive)
    itr.run_phonon_solver()
    ise = ImagSelfEnergy(itr, with_detail=with_detail, lang=lang)
    ise.set_grid_point(1)
    ise.temperature = 300.0
    return ise


def test_imag_self_energy_class_rust_vs_c_band(si_pbesol: Phono3py):
    """ImagSelfEnergy band-index dispatch: lang='Rust' matches lang='C'."""
    ise_c = _build_ise(si_pbesol, with_detail=False, lang="C")
    ise_c.run_interaction()
    ise_c.run_integration_weights()
    ise_c.run()

    ise_rust = _build_ise(si_pbesol, with_detail=False, lang="Rust")
    ise_rust.run_interaction()
    ise_rust.run_integration_weights()
    ise_rust.run()

    assert ise_c.imag_self_energy is not None
    assert ise_rust.imag_self_energy is not None
    np.testing.assert_allclose(
        ise_rust.imag_self_energy, ise_c.imag_self_energy, rtol=1e-10, atol=1e-14
    )


def test_imag_self_energy_class_rust_vs_c_detailed(si_pbesol: Phono3py):
    """ImagSelfEnergy detailed + band-index dispatch: lang='Rust' matches C."""
    ise_c = _build_ise(si_pbesol, with_detail=True, lang="C")
    ise_c.run_interaction()
    ise_c.run_integration_weights()
    ise_c.run()

    ise_rust = _build_ise(si_pbesol, with_detail=True, lang="Rust")
    ise_rust.run_interaction()
    ise_rust.run_integration_weights()
    ise_rust.run()

    assert ise_c.detailed_imag_self_energy is not None
    assert ise_rust.detailed_imag_self_energy is not None
    np.testing.assert_allclose(
        ise_rust.detailed_imag_self_energy,
        ise_c.detailed_imag_self_energy,
        rtol=1e-10,
        atol=1e-14,
    )
    ise_c_N, ise_c_U = ise_c.get_imag_self_energy_N_and_U()
    ise_rust_N, ise_rust_U = ise_rust.get_imag_self_energy_N_and_U()
    assert ise_c_N is not None and ise_c_U is not None
    assert ise_rust_N is not None and ise_rust_U is not None
    np.testing.assert_allclose(ise_rust_N, ise_c_N, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(ise_rust_U, ise_c_U, rtol=1e-10, atol=1e-14)


def test_imag_self_energy_class_rust_vs_c_freq_points(si_pbesol: Phono3py):
    """ImagSelfEnergy frequency-point dispatch: lang='Rust' matches C."""
    ise_c = _build_ise(si_pbesol, with_detail=False, lang="C")
    ise_c.frequency_points = np.array([2.0, 6.0, 10.0], dtype="double")
    ise_c.run_interaction()
    ise_c.run_integration_weights()
    ise_c.run()

    ise_rust = _build_ise(si_pbesol, with_detail=False, lang="Rust")
    ise_rust.frequency_points = np.array([2.0, 6.0, 10.0], dtype="double")
    ise_rust.run_interaction()
    ise_rust.run_integration_weights()
    ise_rust.run()

    assert ise_c.imag_self_energy is not None
    assert ise_rust.imag_self_energy is not None
    np.testing.assert_allclose(
        ise_rust.imag_self_energy, ise_c.imag_self_energy, rtol=1e-10, atol=1e-14
    )
