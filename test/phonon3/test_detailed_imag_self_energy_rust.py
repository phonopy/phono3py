"""Compare the Rust detailed_imag_self_energy_with_g backend against the C one."""

from __future__ import annotations

import numpy as np
import pytest

from phono3py.phonon3.imag_self_energy import (
    run_detailed_imag_self_energy_with_g_rust,
)

pytest.importorskip("phono3py_rs")


def _random_inputs(
    num_triplets: int,
    num_band0: int,
    num_band: int,
    num_grid: int,
    *,
    seed: int = 0,
    g_zero_density: float = 0.3,
    u_fraction: float = 0.5,
):
    """Build a consistent set of synthetic inputs for both backends.

    A fraction ``u_fraction`` of triplets are made Umklapp by placing
    one vertex at a non-zero BZ address; the rest stay Normal
    (addresses summing to zero on every axis).

    """
    rng = np.random.default_rng(seed)
    pp_strength = rng.standard_normal(
        (num_triplets, num_band0, num_band, num_band)
    ).astype("double")
    frequencies = (rng.random((num_grid, num_band)) * 5.0 + 0.01).astype("double")

    # Grid 0 at origin so any triplet (0, a, -a) is Normal.
    bz_grid_addresses = rng.integers(-2, 3, size=(num_grid, 3), dtype="int64")
    bz_grid_addresses[0] = 0

    triplets = np.zeros((num_triplets, 3), dtype="int64")
    for i in range(num_triplets):
        if rng.random() < u_fraction:
            triplets[i] = rng.integers(1, num_grid, size=3, dtype="int64")
        else:
            a = int(rng.integers(1, num_grid))
            # Append the inverse of grid a so the sum is zero.
            inv = np.zeros(3, dtype="int64")
            inv[:] = -bz_grid_addresses[a]
            # Find a grid point whose address equals inv; if none,
            # fall through to a random (likely Umklapp) triplet.
            matches = np.where((bz_grid_addresses == inv[None, :]).all(axis=1))[0]
            if matches.size == 0:
                triplets[i] = rng.integers(1, num_grid, size=3, dtype="int64")
            else:
                triplets[i] = [0, a, int(matches[0])]

    weights = rng.integers(1, 5, size=num_triplets, dtype="int64")
    g = rng.standard_normal((2, num_triplets, num_band0, num_band, num_band)).astype(
        "double"
    )
    g_zero = (
        rng.random((num_triplets, num_band0, num_band, num_band)) < g_zero_density
    ).astype("byte")
    return (
        pp_strength,
        frequencies,
        triplets,
        weights,
        bz_grid_addresses,
        g,
        g_zero,
    )


def _run_c(
    num_triplets: int,
    num_band0: int,
    num_band: int,
    pp_strength,
    triplets,
    weights,
    bz_grid_addresses,
    frequencies,
    temperature_thz: float,
    g,
    g_zero,
    cutoff_frequency: float,
):
    import phono3py._phono3py as phono3c  # type: ignore[import-untyped]

    detailed = np.zeros((num_triplets, num_band0, num_band, num_band), dtype="double")
    ise_N = np.zeros(num_band0, dtype="double")
    ise_U = np.zeros(num_band0, dtype="double")
    phono3c.detailed_imag_self_energy_with_g(
        detailed,
        ise_N,
        ise_U,
        pp_strength,
        triplets,
        weights,
        bz_grid_addresses,
        frequencies,
        temperature_thz,
        g,
        g_zero,
        cutoff_frequency,
    )
    return detailed, ise_N, ise_U


@pytest.mark.parametrize("temperature_thz", [0.0, 5.0])
def test_detailed_imag_self_energy_rust_vs_c(temperature_thz: float):
    """Detailed output and N/U splits match the C backend."""
    num_triplets, num_band0, num_band = 6, 3, 5
    num_grid = 9
    cutoff = 0.05
    pp, freqs, triplets, weights, addrs, g, g_zero = _random_inputs(
        num_triplets, num_band0, num_band, num_grid, seed=11
    )

    d_c, n_c, u_c = _run_c(
        num_triplets,
        num_band0,
        num_band,
        pp,
        triplets,
        weights,
        addrs,
        freqs,
        temperature_thz,
        g,
        g_zero,
        cutoff,
    )

    d_rust = np.zeros_like(d_c)
    n_rust = np.zeros_like(n_c)
    u_rust = np.zeros_like(u_c)
    run_detailed_imag_self_energy_with_g_rust(
        d_rust,
        n_rust,
        u_rust,
        pp,
        triplets,
        weights,
        addrs,
        freqs,
        temperature_thz,
        g,
        g_zero,
        cutoff,
    )
    np.testing.assert_allclose(d_rust, d_c, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(n_rust, n_c, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(u_rust, u_c, rtol=1e-12, atol=1e-14)
