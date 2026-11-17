"""Tests for direct solution of LBTE."""

import pathlib

import numpy as np
import pytest
from phonopy.phonon.grid import get_ir_grid_points

import phono3py
from phono3py._lang import have_c_ext
from phono3py.api_phono3py import Phono3py
from phono3py.conductivity.exceptions import LBTEInputError


def _c_ext_with_lapacke() -> bool:
    """Return True when the C extension is available and built with lapacke."""
    if not have_c_ext():
        return False
    import phono3py._phono3py as phono3c

    return phono3c.include_lapacke()


@pytest.mark.skipif(
    not _c_ext_with_lapacke(), reason="test for phono3py compiled with lapacke"
)
@pytest.mark.parametrize("pinv_solver", [1, 2, 6])
def test_kappa_LBTE_126(si_pbesol: Phono3py, pinv_solver: int):
    """Test for symmetry reduced collision matrix."""
    _test_kappa_LBTE(si_pbesol, pinv_solver)


@pytest.mark.parametrize("pinv_solver", [3, 4, 5, 7])
def test_kappa_LBTE_3457(si_pbesol: Phono3py, pinv_solver: int):
    """Test for symmetry reduced collision matrix."""
    _test_kappa_LBTE(si_pbesol, pinv_solver)


def _test_kappa_LBTE(si_pbesol: Phono3py, pinv_solver: int):
    ref_kappa = [110.896, 110.896, 110.896, 0, 0, 0]
    si_pbesol.mesh_numbers = [9, 9, 9]
    si_pbesol.init_phph_interaction()
    si_pbesol.run_thermal_conductivity(
        is_LBTE=True,
        temperatures=[
            300,
        ],
        pinv_solver=pinv_solver,
    )
    kappa = si_pbesol.thermal_conductivity.kappa.ravel()
    np.testing.assert_allclose(ref_kappa, kappa, atol=0.3)


@pytest.mark.skipif(
    not have_c_ext() or _c_ext_with_lapacke(),
    reason="test for phono3py C extension compiled without lapacke",
)
@pytest.mark.parametrize("pinv_solver", [1, 2])
def test_kappa_LBTE_witout_lapacke(si_pbesol: Phono3py, pinv_solver: int):
    """Test for symmetry reduced collision matrix."""
    si_pbesol.mesh_numbers = [9, 9, 9]
    si_pbesol.init_phph_interaction()
    with pytest.raises(RuntimeError):
        si_pbesol.run_thermal_conductivity(
            is_LBTE=True,
            temperatures=[
                300,
            ],
            pinv_solver=pinv_solver,
        )


def test_kappa_LBTE_full_colmat(si_pbesol: Phono3py):
    """Test for full collision matrix."""
    ref_kappa = [62.415, 62.415, 62.415, 0, 0, 0]
    si_pbesol.mesh_numbers = [5, 5, 5]
    si_pbesol.init_phph_interaction()
    si_pbesol.run_thermal_conductivity(
        is_LBTE=True,
        temperatures=[
            300,
        ],
        is_reducible_collision_matrix=True,
    )
    kappa = si_pbesol.thermal_conductivity.kappa.ravel()
    np.testing.assert_allclose(ref_kappa, kappa, atol=0.5)


def test_kappa_LBTE_full_colmat_RTA_with_cutoff_frequency(si_pbesol: Phono3py):
    """Test kappa_RTA of full collision matrix with a high cutoff frequency.

    With the cutoff frequency of 4 THz, the cutoff applies to different modes
    at a grid point than at the grid point of the same index in the other
    grid indexing (GR or BZ), so taking the frequencies of the wrong grid
    point gives NaN.

    """
    ph3 = Phono3py(
        si_pbesol.unitcell,
        supercell_matrix=si_pbesol.supercell_matrix,
        primitive_matrix=si_pbesol.primitive_matrix,
        cutoff_frequency=4.0,
    )
    ph3.fc2 = si_pbesol.fc2
    ph3.fc3 = si_pbesol.fc3
    ph3.mesh_numbers = [5, 5, 5]
    ph3.init_phph_interaction()
    ph3.run_thermal_conductivity(
        is_LBTE=True,
        temperatures=[300],
        is_reducible_collision_matrix=True,
    )
    tc = ph3.thermal_conductivity
    assert np.isfinite(tc.kappa_RTA).all()
    assert np.isfinite(tc.mode_kappa_RTA).all()


def test_kappa_LBTE_full_colmat_RTA(si_pbesol: Phono3py):
    """Test kappa_RTA of full collision matrix against RTA kappa.

    kappa_RTA of the full collision matrix is computed from the diagonal
    elements of the collision matrix, not from the imaginary part of the
    self-energy, so it agrees with the RTA kappa only approximately.

    """
    si_pbesol.mesh_numbers = [5, 5, 5]
    si_pbesol.init_phph_interaction()
    si_pbesol.run_thermal_conductivity(temperatures=[300])
    kappa_rta = si_pbesol.thermal_conductivity.kappa.copy()
    si_pbesol.run_thermal_conductivity(
        is_LBTE=True,
        temperatures=[300],
        is_reducible_collision_matrix=True,
    )
    kappa_lbte_rta = si_pbesol.thermal_conductivity.kappa_RTA
    np.testing.assert_allclose(kappa_lbte_rta[0, 0, :3], kappa_rta[0, 0, :3], rtol=0.01)


@pytest.mark.parametrize("is_reducible_collision_matrix", [False, True])
def test_kappa_LBTE_si_boundary_mfp(
    si_pbesol: Phono3py, is_reducible_collision_matrix: bool
):
    """Test LBTE with boundary scattering by Si."""
    ref_kappa = [8.358, 8.358, 8.358, 0, 0, 0]
    si_pbesol.mesh_numbers = [5, 5, 5]
    si_pbesol.init_phph_interaction(
        exclude_gamma_acoustic=True, average_degenerate_weights=True
    )
    si_pbesol.run_thermal_conductivity(
        is_LBTE=True,
        temperatures=[300],
        boundary_mfp=0.01,
        is_reducible_collision_matrix=is_reducible_collision_matrix,
    )
    kappa = si_pbesol.thermal_conductivity.kappa.ravel()
    np.testing.assert_allclose(ref_kappa, kappa, atol=0.01)


def test_kappa_LBTE_no_kappa_star(si_pbesol: Phono3py):
    """Test LBTE with is_kappa_star=False iterates all grid points."""
    si_pbesol.mesh_numbers = [5, 5, 5]
    si_pbesol.init_phph_interaction()
    si_pbesol.run_thermal_conductivity(
        is_LBTE=True,
        temperatures=[1000],
        is_kappa_star=True,
    )
    kappa_star = si_pbesol.thermal_conductivity.kappa.ravel()
    si_pbesol.run_thermal_conductivity(
        is_LBTE=True,
        temperatures=[1000],
        is_kappa_star=False,
    )
    kappa_nostar = si_pbesol.thermal_conductivity.kappa.ravel()
    np.testing.assert_allclose(kappa_star[:3], kappa_nostar[:3], atol=0.5)


def test_kappa_LBTE_aln(aln_lda: Phono3py):
    """Test direct solution by AlN."""
    ref_kappa = [234.141, 234.141, 254.006, 0, 0, 0]
    aln_lda.mesh_numbers = [7, 7, 5]
    aln_lda.init_phph_interaction()
    aln_lda.run_thermal_conductivity(
        is_LBTE=True,
        temperatures=[
            300,
        ],
    )
    kappa = aln_lda.thermal_conductivity.kappa.ravel()
    # print(", ".join([f"{k:e}" for k in kappa]))
    np.testing.assert_allclose(ref_kappa, kappa, atol=0.5)


def test_kappa_LBTE_aln_with_sigma(aln_lda: Phono3py):
    """Test direct solution by AlN."""
    ref_kappa = [254.111, 254.111, 271.406, 0, 0, 0]
    aln_lda.sigmas = [
        0.1,
    ]
    aln_lda.sigma_cutoff = 3
    aln_lda.mesh_numbers = [7, 7, 5]
    aln_lda.init_phph_interaction()
    aln_lda.run_thermal_conductivity(
        is_LBTE=True,
        temperatures=[
            300,
        ],
    )
    kappa = aln_lda.thermal_conductivity.kappa.ravel()
    # np.testing.assert_allclose(aln_lda_kappa_RTA_with_sigmas, kappa, atol=0.5)
    aln_lda.sigmas = None
    aln_lda.sigma_cutoff = None
    # print(", ".join([f"{k:e}" for k in kappa]))
    np.testing.assert_allclose(ref_kappa, kappa, atol=0.3)


@pytest.mark.parametrize("is_reducible_collision_matrix", [False, True])
def test_kappa_LBTE_read_collision_per_gp(
    si_pbesol: Phono3py, tmp_path, monkeypatch, is_reducible_collision_matrix: bool
):
    """Write per-GP collision files, then --read-collision round-trip.

    Exercises the fallback to per-GP files (``collision-m*-g*.hdf5``) and
    the gv/cv/isotope recomputation path in ``set_kappa_at_sigmas``.

    """
    monkeypatch.chdir(tmp_path)
    si_pbesol.mesh_numbers = [5, 5, 5]
    si_pbesol.init_phph_interaction()

    si_pbesol.run_thermal_conductivity(
        is_LBTE=True,
        temperatures=[100, 300],
        is_reducible_collision_matrix=is_reducible_collision_matrix,
        write_collision=True,
    )
    ref_kappa = si_pbesol.thermal_conductivity.kappa.ravel().copy()
    assert list(tmp_path.glob("collision-m555-g*.hdf5"))

    si_pbesol.run_thermal_conductivity(
        is_LBTE=True,
        temperatures=[100, 300],
        is_reducible_collision_matrix=is_reducible_collision_matrix,
        read_collision="all",
    )
    read_kappa = si_pbesol.thermal_conductivity.kappa.ravel()

    np.testing.assert_allclose(ref_kappa, read_kappa, atol=1e-3)


@pytest.mark.parametrize("is_reducible_collision_matrix", [False, True])
def test_kappa_LBTE_read_full_collision_matrix(
    si_pbesol: Phono3py, tmp_path, monkeypatch, is_reducible_collision_matrix: bool
):
    """Write the full collision matrix, then --read-collision round-trip.

    The full collision matrix file (``collision-m*.hdf5``) written with
    ``write_LBTE_solution`` has to contain the collisions before the main
    diagonal is added and the matrix is solved.

    Two temperatures are used, because gamma at the first temperature was
    once used at all temperatures in reading the file.

    """
    monkeypatch.chdir(tmp_path)
    si_pbesol.mesh_numbers = [5, 5, 5]
    si_pbesol.init_phph_interaction()
    kwargs = {
        "is_LBTE": True,
        "temperatures": [100, 300],
        "is_isotope": True,
        "is_reducible_collision_matrix": is_reducible_collision_matrix,
    }

    si_pbesol.run_thermal_conductivity(write_LBTE_solution=True, **kwargs)
    ref_kappa = si_pbesol.thermal_conductivity.kappa.ravel().copy()
    assert (tmp_path / "collision-m555.hdf5").exists()

    si_pbesol.run_thermal_conductivity(read_collision="all", **kwargs)
    read_kappa = si_pbesol.thermal_conductivity.kappa.ravel()

    assert np.abs(ref_kappa).max() > 1
    np.testing.assert_allclose(ref_kappa, read_kappa, atol=1e-8)


@pytest.mark.parametrize(
    "write_kwargs", [{"write_collision": True}, {"write_LBTE_solution": True}]
)
def test_kappa_LBTE_read_collision_at_temperature_index(
    si_pbesol: Phono3py, tmp_path, monkeypatch, write_kwargs: dict
):
    """Read collisions at one temperature selected by its index.

    ``read_collision=[1]``, as ``--read-collision 1``, reads the data at the
    second temperature in per-grid-point files or the full collision matrix
    file. The calculation is done only at this temperature.

    """
    monkeypatch.chdir(tmp_path)
    si_pbesol.mesh_numbers = [5, 5, 5]
    si_pbesol.init_phph_interaction()

    si_pbesol.run_thermal_conductivity(
        is_LBTE=True, temperatures=[100, 300], **write_kwargs
    )
    ref_kappa = si_pbesol.thermal_conductivity.kappa[:, 1].copy()

    si_pbesol.run_thermal_conductivity(is_LBTE=True, read_collision=[1])
    tc = si_pbesol.thermal_conductivity

    np.testing.assert_allclose(tc.temperatures, [300])
    assert np.abs(ref_kappa).max() > 1
    np.testing.assert_allclose(ref_kappa, tc.kappa[:, 0], atol=1e-8)


@pytest.mark.parametrize("is_reducible_collision_matrix", [False, True])
def test_kappa_LBTE_write_collision_at_grid_points(
    si_pbesol: Phono3py, tmp_path, monkeypatch, is_reducible_collision_matrix: bool
):
    """Write collisions grid point by grid point, then read all of them.

    Collisions at each irreducible grid point are computed in a separate run
    with ``grid_points``, as in distributed calculations. Only the file of the
    specified grid point is written, and the collision matrix is stored for
    one grid point.

    """
    monkeypatch.chdir(tmp_path)
    si_pbesol.mesh_numbers = [5, 5, 5]
    si_pbesol.init_phph_interaction()
    kwargs = {
        "is_LBTE": True,
        "temperatures": [300],
        "is_isotope": True,
        "is_reducible_collision_matrix": is_reducible_collision_matrix,
    }

    si_pbesol.run_thermal_conductivity(**kwargs)
    ref_kappa = si_pbesol.thermal_conductivity.kappa.ravel().copy()
    ir_grid_points = si_pbesol.thermal_conductivity.grid_points.copy()
    assert len(ir_grid_points) > 1

    for count, gp in enumerate(ir_grid_points):
        si_pbesol.run_thermal_conductivity(
            grid_points=[gp], write_collision=True, **kwargs
        )
        collision_matrix = si_pbesol.thermal_conductivity.collision_matrix
        assert collision_matrix.shape[2] == 1
        filenames = sorted(p.name for p in tmp_path.glob("collision-m555-g*.hdf5"))
        assert len(filenames) == count + 1
        assert f"collision-m555-g{gp}.hdf5" in filenames

    si_pbesol.run_thermal_conductivity(read_collision="all", **kwargs)
    read_kappa = si_pbesol.thermal_conductivity.kappa.ravel()

    assert np.abs(ref_kappa).max() > 1
    np.testing.assert_allclose(ref_kappa, read_kappa, atol=1e-8)


def test_kappa_LBTE_not_irreducible_grid_point(si_pbesol: Phono3py):
    """Test that a grid point that is not irreducible is rejected."""
    si_pbesol.mesh_numbers = [5, 5, 5]
    si_pbesol.init_phph_interaction()
    bz_grid = si_pbesol.grid
    assert bz_grid is not None
    _, _, ir_map = get_ir_grid_points(bz_grid)
    not_ir_grg = np.where(ir_map != np.arange(len(ir_map)))[0][0]
    grid_point = int(bz_grid.grg2bzg[not_ir_grg])

    with pytest.raises(LBTEInputError, match="not an irreducible grid point"):
        si_pbesol.run_thermal_conductivity(
            is_LBTE=True, temperatures=[300], grid_points=[grid_point]
        )


def _load_si_111(band_indices: list[list[int]] | None = None) -> Phono3py:
    """Return Phono3py instance of Si.

    Session fixtures are not used because band indices are changed.

    """
    ph3 = phono3py.load(
        pathlib.Path(__file__).parent / ".." / "phono3py_params_Si111.yaml",
        produce_fc=True,
        log_level=0,
    )
    ph3.mesh_numbers = [5, 5, 5]
    if band_indices is not None:
        ph3.band_indices = band_indices
    ph3.init_phph_interaction()
    return ph3


@pytest.mark.parametrize("read_collision", [None, "all"])
def test_kappa_LBTE_with_band_indices_without_write_collision(
    read_collision: str | None,
):
    """Test that a part of bands needs write_collision without read_collision."""
    ph3 = _load_si_111([[4, 5]])
    with pytest.raises(LBTEInputError, match="needs all bands"):
        ph3.run_thermal_conductivity(
            is_LBTE=True,
            temperatures=[300],
            read_collision=read_collision,
            write_collision=read_collision is not None,
        )


def test_kappa_LBTE_write_collision_with_band_indices(tmp_path, monkeypatch):
    """Write collisions band by band, then read all of them.

    Degenerate bands are given in the same run. Two temperatures are used,
    because gamma at the first temperature was once used at all temperatures
    in reading the files.

    """
    monkeypatch.chdir(tmp_path)
    temperatures = [100, 300]
    ph3 = _load_si_111()
    ph3.run_thermal_conductivity(is_LBTE=True, temperatures=temperatures)
    ref_kappa = ph3.thermal_conductivity.kappa.ravel().copy()
    num_ir_grid_points = len(ph3.thermal_conductivity.grid_points)

    for band_indices in ([[0, 1, 2]], [[3, 4, 5]]):
        ph3 = _load_si_111(band_indices)
        ph3.run_thermal_conductivity(
            is_LBTE=True, temperatures=temperatures, write_collision=True
        )
        assert not list(tmp_path.glob("kappa-*.hdf5"))
    filenames = list(tmp_path.glob("collision-m555-g*-b*.hdf5"))
    assert len(filenames) == num_ir_grid_points * 6

    ph3 = _load_si_111()
    ph3.run_thermal_conductivity(
        is_LBTE=True, temperatures=temperatures, read_collision="all"
    )
    read_kappa = ph3.thermal_conductivity.kappa.ravel()

    assert np.abs(ref_kappa).max() > 1
    np.testing.assert_allclose(ref_kappa, read_kappa, atol=1e-8)
