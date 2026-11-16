"""Tests of inter-band transport variants with a part of bands selected.

Band indices are used to compute and store gamma band by band. The velocity
matrix and the heat capacity matrix need all bands, so they are not computed
unless all bands are selected in their order. In that case, the RTA
calculation is the same as that of the standard method.

"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

import phono3py
from phono3py import Phono3py

test_dir = pathlib.Path(__file__).parents[2]
mesh = [5, 5, 5]
num_band = 6


def _load(band_indices: list[int] | None = None, log_level: int = 0) -> Phono3py:
    """Return Phono3py instance of Si.

    Session fixtures are not used because band indices are changed.

    """
    ph3 = phono3py.load(
        test_dir / "phono3py_params_Si111.yaml", produce_fc=True, log_level=log_level
    )
    ph3.mesh_numbers = mesh
    if band_indices is not None:
        ph3.band_indices = [band_indices]
    ph3.init_phph_interaction()
    return ph3


@pytest.mark.parametrize("transport_type", ["NJC23", "IBDB19", "SMM19"])
@pytest.mark.parametrize("band_indices", [[4, 5], [3, 4, 5, 0, 1, 2]])
def test_interband_rta_with_band_indices(transport_type: str, band_indices: list[int]):
    """Test that the standard method is used unless all bands are selected."""
    ph3_std = _load(band_indices)
    ph3_std.run_thermal_conductivity(temperatures=[300])
    tc_std = ph3_std.thermal_conductivity

    ph3 = _load(band_indices)
    ph3.run_thermal_conductivity(temperatures=[300], transport_type=transport_type)
    tc = ph3.thermal_conductivity

    num_gp = len(tc.grid_points)
    assert tc.group_velocities.shape == (num_gp, len(band_indices), 3)
    assert np.abs(tc.group_velocities).max() > 1
    np.testing.assert_allclose(tc.group_velocities, tc_std.group_velocities, atol=1e-8)
    assert tc.gamma.shape == (1, 1, num_gp, len(band_indices))
    np.testing.assert_allclose(tc.gamma, tc_std.gamma, rtol=0, atol=1e-10)
    np.testing.assert_allclose(tc.kappa, tc_std.kappa, rtol=0, atol=1e-8)
    with pytest.raises(AttributeError):
        _ = tc.kappa_inter


@pytest.mark.parametrize(
    "band_indices,is_shown", [([4, 5], True), ([3, 4, 5, 0, 1, 2], True), (None, False)]
)
def test_interband_rta_message_with_band_indices(
    capsys: pytest.CaptureFixture[str], band_indices: list[int] | None, is_shown: bool
):
    """Test the message shown when the standard method is used."""
    ph3 = _load(band_indices, log_level=1)
    capsys.readouterr()
    ph3.run_thermal_conductivity(temperatures=[300], transport_type="NJC23")
    message = "Velocity matrix and heat capacity matrix of NJC23 are not computed"
    # Line breaks made by textwrap are replaced by spaces.
    out = " ".join(capsys.readouterr().out.split())
    assert (out.count(message) == 1) is is_shown


def test_interband_rta_with_all_bands_in_sets():
    """Test that all bands given in sets are treated as all bands."""
    ph3_ref = _load()
    ph3_ref.run_thermal_conductivity(temperatures=[300], transport_type="NJC23")
    tc_ref = ph3_ref.thermal_conductivity

    ph3 = _load()
    ph3.band_indices = [[0, 1, 2], [3, 4, 5]]
    ph3.init_phph_interaction()
    ph3.run_thermal_conductivity(temperatures=[300], transport_type="NJC23")
    tc = ph3.thermal_conductivity

    assert np.abs(tc_ref.kappa_inter).max() > 0.1
    np.testing.assert_allclose(tc.kappa_inter, tc_ref.kappa_inter, rtol=0, atol=1e-8)
    np.testing.assert_allclose(tc.kappa_intra, tc_ref.kappa_intra, rtol=0, atol=1e-8)


def test_interband_rta_gamma_band_by_band(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    """Test that gamma written band by band is read to compute kappa."""
    ph3_ref = _load()
    ph3_ref.run_thermal_conductivity(temperatures=[300], transport_type="NJC23")
    tc_ref = ph3_ref.thermal_conductivity

    monkeypatch.chdir(tmp_path)
    num_gp = len(tc_ref.grid_points)
    gamma_written = np.zeros((1, 1, num_gp, num_band), dtype="double")
    for band_index in range(num_band):
        ph3 = _load([band_index])
        ph3.run_thermal_conductivity(
            temperatures=[300], write_gamma=True, transport_type="NJC23"
        )
        gamma_written[:, :, :, band_index] = ph3.thermal_conductivity.gamma[:, :, :, 0]
    assert len(list(tmp_path.glob("kappa-m555-g*-b*.hdf5"))) == num_gp * num_band

    ph3 = _load()
    ph3.run_thermal_conductivity(
        temperatures=[300], read_gamma=True, transport_type="NJC23"
    )
    tc = ph3.thermal_conductivity

    assert np.abs(gamma_written).max() > 1e-3
    np.testing.assert_allclose(tc.gamma, gamma_written, rtol=0, atol=1e-10)
    # Gamma is not averaged over degenerate bands computed in different runs,
    # so kappa differs slightly from that computed with all bands at once.
    # xx, yy, and zz components are compared.
    np.testing.assert_allclose(
        tc.kappa_inter[0, 0, :3], tc_ref.kappa_inter[0, 0, :3], rtol=1e-3
    )
    np.testing.assert_allclose(
        tc.kappa_intra[0, 0, :3], tc_ref.kappa_intra[0, 0, :3], rtol=3e-2
    )
