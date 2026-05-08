"""Tests for phono3py-kaccum script."""

from __future__ import annotations

import os
import pathlib
import shutil

import numpy as np
import pytest

from phono3py.cui.kaccum_script import (
    KaccumMockArgs,
    _get_T_target_index,
    _show_scalar,
    _show_tensor,
    main,
)

cwd = pathlib.Path(__file__).parent
KAPPA_HDF5 = cwd / ".." / "kappa-m111111_si_pbesol.hdf5"
PHONO3PY_YAML = cwd / ".." / "phono3py_si_pbesol.yaml"
EXPECTED = cwd / "expected"

# Temperatures in kappa-m111111_si_pbesol.hdf5: [100, 200, 300, 400, 500] K
TEMPERATURES = [100.0, 200.0, 300.0, 400.0, 500.0]
T_DEFAULT = 300.0

# Reference cumulative kappa_avg (W/mK) at the last sampling point per temperature.
# Derived from kappa-m111111_si_pbesol.hdf5 with --average --temperature T.
REF_KAPPA_AVG = {
    100.0: 841.29491,
    200.0: 196.86981,
    300.0: 110.54180,
    400.0: 78.04883,
    500.0: 60.73343,
}
# Last frequency sampling point (THz) — same for all temperatures.
REF_LAST_FREQ = 15.26976


# ---------------------------------------------------------------------------
# Unit tests: _get_T_target_index
# ---------------------------------------------------------------------------


def test_get_T_target_index_returns_correct_index():
    """Return the index of the matching temperature."""
    temperatures = np.array([100.0, 200.0, 300.0])
    assert _get_T_target_index(temperatures, 100.0) == 0
    assert _get_T_target_index(temperatures, 200.0) == 1
    assert _get_T_target_index(temperatures, 300.0) == 2


def test_get_T_target_index_not_found():
    """Raise RuntimeError when temperature is not in dataset."""
    temperatures = np.array([100.0, 200.0])

    with pytest.raises(RuntimeError, match="500.0"):
        _get_T_target_index(temperatures, 500.0)


@pytest.mark.parametrize("T_target", TEMPERATURES)
def test_get_T_target_index_each_temperature(T_target):
    """_get_T_target_index returns the correct index for each temperature."""
    temperatures = np.array(TEMPERATURES)
    expected = TEMPERATURES.index(T_target)
    assert _get_T_target_index(temperatures, T_target) == expected


# ---------------------------------------------------------------------------
# Unit tests: _show_tensor / _show_scalar
# ---------------------------------------------------------------------------


def test_show_tensor_average(capsys):
    """_show_tensor with --average outputs 3 columns per data line."""
    n_temps, n_samp, n_elem = 1, 3, 6
    cumulative = np.zeros((n_temps, n_samp, n_elem), dtype="double")
    density = np.zeros((n_temps, n_samp, n_elem), dtype="double")
    temperatures = np.array([T_DEFAULT])
    sampling_points = np.linspace(0, 10, n_samp)

    args = KaccumMockArgs(average=True)
    _show_tensor(cumulative, density, temperatures, sampling_points, args)  # type: ignore[arg-type]

    captured = capsys.readouterr()
    data_lines = [line for line in captured.out.splitlines() if line.strip()]
    assert len(data_lines) == n_samp + 1  # +1 for "# 300 K" header


def test_show_scalar_default(capsys):
    """_show_scalar outputs one block per temperature."""
    n_temps, n_samp = 2, 4
    cumulative = np.zeros((n_temps, n_samp), dtype="double")
    density = np.zeros((n_temps, n_samp), dtype="double")
    temperatures = np.array([100.0, 200.0])
    sampling_points = np.linspace(0, 10, n_samp)

    args = KaccumMockArgs()
    _show_scalar(cumulative, density, temperatures, sampling_points, args)  # type: ignore[arg-type]

    captured = capsys.readouterr()
    lines = [line for line in captured.out.splitlines() if line.strip()]
    # 2 temps × (1 header + n_samp data lines)
    assert len(lines) == n_temps * (1 + n_samp)


# ---------------------------------------------------------------------------
# Integration tests: main()
# ---------------------------------------------------------------------------


@pytest.fixture()
def kaccum_env(tmp_path):
    """Set up temp directory with phono3py.yaml and kappa hdf5."""
    shutil.copy(PHONO3PY_YAML, tmp_path / "phono3py.yaml")
    shutil.copy(KAPPA_HDF5, tmp_path / "kappa.hdf5")
    original_cwd = pathlib.Path.cwd()
    os.chdir(tmp_path)
    yield tmp_path
    os.chdir(original_cwd)


def test_main_tensor(kaccum_env, capsys):
    """main() runs tensor (mode_kappa) path and produces output."""
    args = KaccumMockArgs(filenames=["kappa.hdf5"], temperature=T_DEFAULT)
    main(args=args)
    captured = capsys.readouterr()
    assert f"# {T_DEFAULT:.0f} K" in captured.out


def test_main_scalar_gamma(kaccum_env, capsys):
    """main() runs scalar gamma path and produces output."""
    args = KaccumMockArgs(filenames=["kappa.hdf5"], gamma=True, temperature=T_DEFAULT)
    main(args=args)
    captured = capsys.readouterr()
    assert captured.out.strip() != ""


def test_main_tensor_average(kaccum_env, capsys):
    """main() with --average outputs exactly 3 columns per data line."""
    args = KaccumMockArgs(filenames=["kappa.hdf5"], average=True, temperature=T_DEFAULT)
    main(args=args)
    captured = capsys.readouterr()
    data_lines = [
        line
        for line in captured.out.splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert len(data_lines) > 0
    assert all(len(line.split()) == 3 for line in data_lines)


@pytest.mark.parametrize("temperature", TEMPERATURES)
def test_main_tensor_each_temperature(kaccum_env, capsys, temperature):
    """main() with --temperature selects the correct temperature block."""
    args = KaccumMockArgs(filenames=["kappa.hdf5"], temperature=temperature)
    main(args=args)
    captured = capsys.readouterr()
    assert f"# {temperature:.0f} K" in captured.out
    for other in TEMPERATURES:
        if other != temperature:
            assert f"# {other:.0f} K" not in captured.out


# ---------------------------------------------------------------------------
# Regression tests: numerical values
# ---------------------------------------------------------------------------


def _last_data_line(out: str) -> list[float]:
    """Return float values from the last non-comment, non-empty output line."""
    lines = [
        line for line in out.splitlines() if line.strip() and not line.startswith("#")
    ]
    return [float(v) for v in lines[-1].split()]


def test_regression_tensor_average_at_300K(kaccum_env, capsys):
    """Cumulative kappa_avg at last sampling point matches reference at 300 K."""
    args = KaccumMockArgs(filenames=["kappa.hdf5"], average=True, temperature=T_DEFAULT)
    main(args=args)
    freq, kappa_avg, _ = _last_data_line(capsys.readouterr().out)
    assert freq == pytest.approx(REF_LAST_FREQ, rel=1e-5)
    assert kappa_avg == pytest.approx(REF_KAPPA_AVG[T_DEFAULT], rel=1e-5)


@pytest.mark.parametrize("temperature", TEMPERATURES)
def test_regression_tensor_average_all_temperatures(kaccum_env, capsys, temperature):
    """Cumulative kappa_avg at last sampling point matches reference for each T."""
    args = KaccumMockArgs(
        filenames=["kappa.hdf5"], average=True, temperature=temperature
    )
    main(args=args)
    freq, kappa_avg, _ = _last_data_line(capsys.readouterr().out)
    assert freq == pytest.approx(REF_LAST_FREQ, rel=1e-5)
    assert kappa_avg == pytest.approx(REF_KAPPA_AVG[temperature], rel=1e-5)


def test_regression_tensor_kappa_xx_at_300K(kaccum_env, capsys):
    """Cumulative kappa_xx at last sampling point matches reference at 300 K."""
    args = KaccumMockArgs(filenames=["kappa.hdf5"], temperature=T_DEFAULT)
    main(args=args)
    vals = _last_data_line(capsys.readouterr().out)
    # Column layout: freq, kappa_xx, kappa_yy, kappa_zz, kappa_yz, kappa_xz, kappa_xy,
    #                dos_xx, dos_yy, dos_zz, dos_yz, dos_xz, dos_xy
    freq, kappa_xx = vals[0], vals[1]
    assert freq == pytest.approx(REF_LAST_FREQ, rel=1e-5)
    assert kappa_xx == pytest.approx(REF_KAPPA_AVG[T_DEFAULT], rel=1e-5)


def test_regression_num_sampling_points(kaccum_env, capsys):
    """Default 100 sampling points are output."""
    args = KaccumMockArgs(filenames=["kappa.hdf5"], average=True, temperature=T_DEFAULT)
    main(args=args)
    data_lines = [
        line
        for line in capsys.readouterr().out.splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert len(data_lines) == 100


def test_main_too_many_filenames():
    """main() raises RuntimeError when more than one filename is given."""
    args = KaccumMockArgs(filenames=["a.hdf5", "b.hdf5"])
    with pytest.raises(RuntimeError):
        main(args=args)


# ---------------------------------------------------------------------------
# Regression tests: --mfp output (captured from pre-refactor run)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expected_name,kaccum_args",
    [
        ("kaccum_mfp_si_300K.txt", {"mfp": True, "temperature": T_DEFAULT}),
        ("kaccum_mfp_si_all_T.txt", {"mfp": True}),
        (
            "kaccum_mfp_si_300K_avg.txt",
            {"mfp": True, "average": True, "temperature": T_DEFAULT},
        ),
    ],
)
def test_main_mfp_matches_pre_refactor_output(
    kaccum_env, capsys, expected_name, kaccum_args
):
    """``phono3py-kaccum --mfp`` output is byte-identical to pre-refactor.

    The expected-output files were captured by running the script against
    ``kappa-m111111_si_pbesol.hdf5`` before the spectrum.py refactor.  Any
    divergence means the refactor changed user-visible behaviour for the
    ``--mfp`` path.

    """
    args = KaccumMockArgs(filenames=["kappa.hdf5"], **kaccum_args)
    main(args=args)
    actual = capsys.readouterr().out
    expected = (EXPECTED / expected_name).read_text()
    assert actual == expected
