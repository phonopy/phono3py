"""Regression tests for Gruneisen class."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from phono3py import Phono3py
from phono3py.phonon3.gruneisen import Gruneisen


@pytest.fixture
def gruneisen(si_pbesol_111: Phono3py) -> Gruneisen:
    """Return Gruneisen instance for Si 1x1x1."""
    ph3 = si_pbesol_111
    assert ph3.fc2 is not None
    assert ph3.fc3 is not None
    return Gruneisen(ph3.fc2, ph3.fc3, ph3.supercell, ph3.primitive)


def test_gruneisen_qpoints(gruneisen):
    """Test Gruneisen calculation at specific q-points."""
    qpoints = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 0.5, 0.0]])
    gruneisen.set_qpoints(qpoints)
    gruneisen.run()

    gp = gruneisen.gruneisen_parameters
    assert gp is not None
    assert gp.shape == (3, 6, 3, 3)
    assert gruneisen.frequencies is not None
    assert gruneisen.frequencies.shape == (3, 6)

    # Frequencies at X point (0.5, 0, 0)
    np.testing.assert_allclose(
        gruneisen.frequencies[1],
        [3.98406999, 3.98406999, 9.44271189, 12.03447142, 14.76890302, 14.76890302],
        rtol=1e-5,
    )

    # Gruneisen trace at X point (0.5, 0, 0)
    trace = gp[1, :, 0, 0] + gp[1, :, 1, 1] + gp[1, :, 2, 2]
    np.testing.assert_allclose(
        trace,
        [-3.95849219, -3.95849219, 0.59997949, 4.54031737, 3.54802433, 3.54802433],
        rtol=1e-5,
    )

    # Optical mode Gruneisen at Gamma: trace / 3 should be ~1.022
    trace_gamma_opt = gp[0, 3:, 0, 0] + gp[0, 3:, 1, 1] + gp[0, 3:, 2, 2]
    np.testing.assert_allclose(trace_gamma_opt, [3.06562808] * 3, rtol=1e-5)


def test_gruneisen_mesh(si_pbesol_111: Phono3py):
    """Test Gruneisen calculation on sampling mesh."""
    ph3 = si_pbesol_111
    assert ph3.fc2 is not None
    assert ph3.fc3 is not None
    g = Gruneisen(ph3.fc2, ph3.fc3, ph3.supercell, ph3.primitive)
    g.set_sampling_mesh(np.array([4, 4, 4], dtype="int64"))
    g.run()

    gp = g.gruneisen_parameters
    assert gp is not None
    assert isinstance(gp, np.ndarray)
    assert gp.shape == (36, 6, 3, 3)
    assert g.frequencies is not None
    assert isinstance(g.frequencies, np.ndarray)
    assert g.frequencies.shape == (36, 6)

    # Average Gruneisen parameter (trace/3 averaged over all q and bands)
    trace = gp[:, :, 0, 0] + gp[:, :, 1, 1] + gp[:, :, 2, 2]
    np.testing.assert_allclose(trace.mean(), 0.5679428, rtol=1e-5)


def test_gruneisen_mesh_with_symmetry(si_pbesol_111: Phono3py):
    """Test Gruneisen calculation on sampling mesh with primitive_symmetry."""
    ph3 = si_pbesol_111
    assert ph3.fc2 is not None
    assert ph3.fc3 is not None
    g = Gruneisen(ph3.fc2, ph3.fc3, ph3.supercell, ph3.primitive)
    g.set_sampling_mesh(np.array([4, 4, 4], dtype="int64"), ph3.primitive_symmetry)
    g.run()

    gp = g.gruneisen_parameters
    assert gp is not None
    assert isinstance(gp, np.ndarray)
    assert gp.shape == (8, 6, 3, 3)
    assert g.frequencies is not None
    assert isinstance(g.frequencies, np.ndarray)
    assert g.frequencies.shape == (8, 6)

    trace = gp[:, :, 0, 0] + gp[:, :, 1, 1] + gp[:, :, 2, 2]
    np.testing.assert_allclose(trace.mean(), 0.5799835, rtol=1e-5)


def test_gruneisen_band(si_pbesol_111: Phono3py):
    """Test Gruneisen calculation along a band path."""
    ph3 = si_pbesol_111
    assert ph3.fc2 is not None
    assert ph3.fc3 is not None
    g = Gruneisen(ph3.fc2, ph3.fc3, ph3.supercell, ph3.primitive)
    band_path = [
        np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 0.5, 0.0]]),
    ]
    g.set_band_structure(band_path)
    g.run()

    gp = g.gruneisen_parameters
    assert gp is not None
    assert len(gp) == 1
    assert gp[0].shape == (3, 6, 3, 3)
    assert g.frequencies is not None
    assert g.frequencies[0].shape == (3, 6)

    # Frequencies at middle point (0.5, 0, 0)
    np.testing.assert_allclose(
        g.frequencies[0][1],
        [3.98406999, 3.98406999, 9.44271189, 12.03447142, 14.76890302, 14.76890302],
        rtol=1e-5,
    )

    # Gruneisen trace at (0.5, 0, 0)
    trace = gp[0][1, :, 0, 0] + gp[0][1, :, 1, 1] + gp[0][1, :, 2, 2]
    np.testing.assert_allclose(
        trace,
        [-3.95849219, -3.95849219, 0.59997949, 4.54031737, 3.54802433, 3.54802433],
        rtol=1e-5,
    )


def test_gruneisen_write_qpoints(si_pbesol_111: Phono3py, tmp_path: Path):
    """Test write() in qpoints mode produces a valid yaml file."""
    ph3 = si_pbesol_111
    assert ph3.fc2 is not None
    assert ph3.fc3 is not None
    g = Gruneisen(ph3.fc2, ph3.fc3, ph3.supercell, ph3.primitive)
    g.set_qpoints(np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 0.5, 0.0]]))
    g.run()
    g.write(filename=str(tmp_path / "gruneisen"))

    yaml_file = tmp_path / "gruneisen.yaml"
    assert yaml_file.exists()

    with open(yaml_file) as f:
        data = yaml.safe_load(f)

    assert data["nqpoint"] == 3
    assert len(data["phonon"]) == 3
    # Frequency at X point (0.5, 0, 0), band 3
    assert data["phonon"][1]["band"][2]["frequency"] == pytest.approx(
        9.4427118925, rel=1e-5
    )
    # Scalar Gruneisen (trace/3) at X point, band 3
    assert data["phonon"][1]["band"][2]["gruneisen"] == pytest.approx(
        0.1999931623, rel=1e-5
    )


def test_gruneisen_write_band(si_pbesol_111: Phono3py, tmp_path: Path):
    """Test write() in band mode produces a valid yaml file."""
    ph3 = si_pbesol_111
    assert ph3.fc2 is not None
    assert ph3.fc3 is not None
    g = Gruneisen(ph3.fc2, ph3.fc3, ph3.supercell, ph3.primitive)
    g.set_band_structure(
        [np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 0.5, 0.0]])]
    )
    g.run()
    g.write(filename=str(tmp_path / "gruneisen"))

    yaml_file = tmp_path / "gruneisen.yaml"
    assert yaml_file.exists()

    with open(yaml_file) as f:
        data = yaml.safe_load(f)

    assert len(data["path"]) == 1
    assert data["path"][0]["nqpoint"] == 3
    # Frequency at X point (0.5, 0, 0), band 3
    assert data["path"][0]["phonon"][1]["band"][2]["frequency"] == pytest.approx(
        9.4427118925, rel=1e-5
    )
    # Scalar Gruneisen (trace/3) at X point, band 3
    assert data["path"][0]["phonon"][1]["band"][2]["gruneisen"] == pytest.approx(
        0.1999931623, rel=1e-5
    )


def test_gruneisen_write_mesh(si_pbesol_111: Phono3py, tmp_path: Path):
    """Test write() in mesh mode produces a valid hdf5 file."""
    import h5py

    ph3 = si_pbesol_111
    assert ph3.fc2 is not None
    assert ph3.fc3 is not None
    g = Gruneisen(ph3.fc2, ph3.fc3, ph3.supercell, ph3.primitive)
    g.set_sampling_mesh(np.array([4, 4, 4], dtype="int64"))
    g.run()
    g.write(filename=str(tmp_path / "gruneisen"))

    hdf5_file = tmp_path / "gruneisen.hdf5"
    assert hdf5_file.exists()

    with h5py.File(hdf5_file) as f:
        assert set(f.keys()) == {
            "frequency",
            "gruneisen",
            "gruneisen_tensor",
            "mesh",
            "qpoint",
            "weight",
        }
        assert f["gruneisen"].shape == (36, 6)  # type: ignore
        assert f["gruneisen_tensor"].shape == (36, 6, 3, 3)  # type: ignore
        assert f["frequency"].shape == (36, 6)  # type: ignore
        assert f["qpoint"].shape == (36, 3)  # type: ignore
        np.testing.assert_array_equal(f["mesh"][:], [4, 4, 4])  # type: ignore
        # Gamma-point optical mode Gruneisen (trace/3)
        np.testing.assert_allclose(
            f["gruneisen"][0, 3:],  # type: ignore
            [1.02187603] * 3,
            rtol=1e-5,
        )
