"""Tests for phono3py-kdeplot script."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
from numpy.typing import NDArray

from phono3py.cui.kdeplot_script import (
    KdeplotMockArgs,
    collect_data,
    main,
    run_KDE,
)

cwd = pathlib.Path(__file__).parent
KAPPA_HDF5 = cwd / ".." / "kappa-m111111_si_pbesol.hdf5"

# ---------------------------------------------------------------------------
# collect_data
# ---------------------------------------------------------------------------


def test_collect_data_basic():
    """Positive tau values are returned; zero/negative gamma are excluded."""
    gamma = np.array([[1.0, 2.0, 0.0]])
    weights = np.array([1])
    frequencies = np.array([[1.0, 2.0, 3.0]])

    x, y = collect_data(gamma, weights, frequencies, cutoff=None, max_freq=None)

    assert x.shape == y.shape
    assert len(x) == 2  # gamma=0 excluded
    assert np.all(y > 0)


def test_collect_data_weight_repeats():
    """Weight multiplies data points."""
    gamma = np.array([[1.0]])
    weights = np.array([3])
    frequencies = np.array([[1.0]])

    x, y = collect_data(gamma, weights, frequencies, cutoff=None, max_freq=None)

    assert len(x) == 3
    assert np.allclose(x, 1.0)


def test_collect_data_cutoff():
    """Points with tau >= cutoff are excluded."""
    gamma = np.array([[0.001, 1.0]])
    weights = np.array([1])
    frequencies = np.array([[1.0, 2.0]])

    tau_large = 1.0 / (0.001 * 4 * np.pi)
    cutoff = tau_large / 2

    x, y = collect_data(gamma, weights, frequencies, cutoff=cutoff, max_freq=None)

    assert len(x) == 1
    assert np.all(y < cutoff)


def test_collect_data_max_freq():
    """Points with frequency >= max_freq are excluded."""
    gamma = np.array([[1.0, 1.0, 1.0]])
    weights = np.array([1])
    frequencies = np.array([[1.0, 5.0, 10.0]])

    x, y = collect_data(gamma, weights, frequencies, cutoff=None, max_freq=3.0)

    assert len(x) == 1
    assert x[0] == pytest.approx(1.0)


def test_collect_data_returns_arrays():
    """Return values are numpy arrays."""
    gamma = np.array([[1.0]])
    weights = np.array([1])
    frequencies = np.array([[2.0]])

    x, y = collect_data(gamma, weights, frequencies, cutoff=None, max_freq=None)

    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)


def test_collect_data_all_negative_gamma():
    """All-negative gamma results in empty output."""
    gamma = np.array([[-1.0, -2.0]])
    weights = np.array([1])
    frequencies = np.array([[1.0, 2.0]])

    x, y = collect_data(gamma, weights, frequencies, cutoff=None, max_freq=None)

    assert len(x) == 0
    assert len(y) == 0


# ---------------------------------------------------------------------------
# Regression tests: collect_data numerical values
# ---------------------------------------------------------------------------

# tau = 1 / (gamma * 4 * pi)
_TAU_G1 = 1.0 / (1.0 * 4 * np.pi)  # gamma=1.0 -> 0.07957747...
_TAU_G2 = 1.0 / (2.0 * 4 * np.pi)  # gamma=2.0 -> 0.03978874...


def test_regression_tau_values():
    """Lifetime tau = 1/(gamma * 4*pi) is computed correctly."""
    gamma = np.array([[1.0, 2.0]])
    weights = np.array([1])
    frequencies = np.array([[1.0, 3.0]])

    x, y = collect_data(gamma, weights, frequencies, cutoff=None, max_freq=None)

    assert x[0] == pytest.approx(1.0)
    assert x[1] == pytest.approx(3.0)
    assert y[0] == pytest.approx(_TAU_G1, rel=1e-7)
    assert y[1] == pytest.approx(_TAU_G2, rel=1e-7)


def test_regression_weight_multiplies_values():
    """Weight=2 appends the mode list twice: [f0, f1, f0, f1]."""
    gamma = np.array([[1.0, 2.0]])
    weights = np.array([2])
    frequencies = np.array([[1.0, 3.0]])

    x, y = collect_data(gamma, weights, frequencies, cutoff=None, max_freq=None)

    assert len(x) == 4
    assert np.allclose(x, [1.0, 3.0, 1.0, 3.0])
    assert np.allclose(y, [_TAU_G1, _TAU_G2, _TAU_G1, _TAU_G2])


@pytest.mark.skipif(not KAPPA_HDF5.exists(), reason="kappa HDF5 fixture not available")
def test_regression_collect_data_from_hdf5():
    """collect_data on Si kappa HDF5 at 300 K returns expected point count and means."""
    import h5py

    with h5py.File(KAPPA_HDF5, "r") as f:
        gamma = f["gamma"][2]  # T=300 K (index 2)
        weights = f["weight"][:]
        frequencies = f["frequency"][:]

    x, y = collect_data(gamma, weights, frequencies, cutoff=None, max_freq=None)

    assert len(x) == 7983
    assert x.mean() == pytest.approx(9.80539822, rel=1e-6)
    assert y.mean() == pytest.approx(12.98336259, rel=1e-6)


@pytest.mark.skipif(not KAPPA_HDF5.exists(), reason="kappa HDF5 fixture not available")
def test_regression_collect_data_with_cutoff():
    """cutoff=10 ps reduces the point count correctly."""
    import h5py

    with h5py.File(KAPPA_HDF5, "r") as f:
        gamma = f["gamma"][2]
        weights = f["weight"][:]
        frequencies = f["frequency"][:]

    x, y = collect_data(gamma, weights, frequencies, cutoff=10.0, max_freq=None)

    assert len(x) == 4933
    assert np.all(y < 10.0)


@pytest.mark.skipif(not KAPPA_HDF5.exists(), reason="kappa HDF5 fixture not available")
def test_regression_collect_data_with_max_freq():
    """max_freq=8 THz reduces the point count and caps frequencies correctly."""
    import h5py

    with h5py.File(KAPPA_HDF5, "r") as f:
        gamma = f["gamma"][2]
        weights = f["weight"][:]
        frequencies = f["frequency"][:]

    x, y = collect_data(gamma, weights, frequencies, cutoff=None, max_freq=8.0)

    assert len(x) == 2969
    assert x.mean() == pytest.approx(4.55907259, rel=1e-6)
    assert np.all(x < 8.0)


# ---------------------------------------------------------------------------
# run_KDE
# ---------------------------------------------------------------------------


def _sample_xy(
    n: int = 40, seed: int = 0
) -> tuple[NDArray[np.double], NDArray[np.double]]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.5, 5.0, n)
    y = rng.uniform(0.1, 50.0, n)
    return x, y


def test_run_KDE_shapes():
    """Output grids have correct shape (nbins × ynbins)."""
    x, y = _sample_xy()
    nbins = 10
    xi, yi, zi, short_nbinds = run_KDE(x, y, nbins)

    assert xi.shape == zi.shape
    assert yi.shape == zi.shape
    assert xi.shape[0] == nbins


def test_run_KDE_with_fixed_ranges():
    """x_max and y_max produce a fixed (nbins × nbins) grid."""
    x, y = _sample_xy()
    nbins = 10
    xi, yi, zi, short_nbinds = run_KDE(x, y, nbins, x_max=6.0, y_max=60.0)

    assert xi.shape == (nbins, nbins)
    assert yi.shape == (nbins, nbins)
    assert short_nbinds == nbins


def test_run_KDE_zi_nonnegative():
    """KDE density values are non-negative."""
    x, y = _sample_xy()
    _, _, zi, _ = run_KDE(x, y, nbins=10)

    assert np.all(zi >= 0)


def test_run_KDE_short_nbinds_le_nbins():
    """short_nbinds never exceeds nbins when y_max is None."""
    x, y = _sample_xy()
    nbins = 15
    _, _, _, short_nbinds = run_KDE(x, y, nbins)

    assert short_nbinds <= nbins


# ---------------------------------------------------------------------------
# Integration: main()
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not KAPPA_HDF5.exists(), reason="kappa HDF5 fixture not available")
def test_main_runs_and_saves(tmp_path):
    """main() reads HDF5, runs KDE, and saves a PNG without error."""
    out = tmp_path / "lifetime.png"
    args = KdeplotMockArgs(
        filenames=[str(KAPPA_HDF5)],
        temperature=300.0,
        nbins=20,
        output_filename=str(out),
    )
    main(args=args)

    assert out.exists()
    assert out.stat().st_size > 0


@pytest.mark.skipif(not KAPPA_HDF5.exists(), reason="kappa HDF5 fixture not available")
def test_main_flip(tmp_path):
    """main() with flip=True runs without error."""
    out = tmp_path / "lifetime_flip.png"
    args = KdeplotMockArgs(
        filenames=[str(KAPPA_HDF5)],
        temperature=300.0,
        nbins=20,
        flip=True,
        output_filename=str(out),
    )
    main(args=args)

    assert out.exists()


@pytest.mark.skipif(not KAPPA_HDF5.exists(), reason="kappa HDF5 fixture not available")
def test_main_each_temperature(tmp_path):
    """main() selects the correct temperature from the dataset."""
    for t in [100.0, 200.0, 300.0, 400.0, 500.0]:
        out = tmp_path / f"lifetime_{t:.0f}.png"
        args = KdeplotMockArgs(
            filenames=[str(KAPPA_HDF5)],
            temperature=t,
            nbins=20,
            output_filename=str(out),
        )
        main(args=args)
        assert out.exists(), f"Output missing for T={t}"
