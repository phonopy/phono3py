"""Create dynamical matrix and solve harmonic phonons on grid."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from phonopy.harmonic.dynamical_matrix import (
    DynamicalMatrix,
    DynamicalMatrixGL,
    DynamicalMatrixNAC,
)
from phonopy.physical_units import get_physical_units


def run_phonon_solver_c(
    dm: DynamicalMatrix,
    frequencies: NDArray[np.double],
    eigenvectors: NDArray[np.cdouble],
    phonon_done: NDArray[np.byte],
    grid_points: Sequence[int] | NDArray[np.int64],
    grid_address: NDArray[np.int64],
    QDinv: NDArray[np.double],
    frequency_conversion_factor: float | None = None,
    nac_q_direction: Sequence[float]
    | NDArray[np.double]
    | None = None,  # in reduced coordinates
    lapack_zheev_uplo: Literal["L", "U"] = "L",
) -> None:
    """Build and solve dynamical matrices on grid in C-API.

    Note
    ----
    When LAPACKE is linked in C, `phononcalc.phonons_at_gridpoints` constructs
    and solves dynamical matrices on grid points. Otherwise, it only constructs
    dynamical matrices and solves them in python.

    Parameters
    ----------
    dm : DynamicalMatrix
        DynamicalMatrix instance.
    frequencies, eigenvectors, phonon_done :
        See Interaction.get_phonons().
    grid_points : ndarray
        Grid point indices.
        shape=(grid_points, ), dtype='int64'
    grid_address : ndarray
        See BZGrid.addresses.
    QDinv : ndarray
        See BZGrid.QDinv.
    frequency_conversion_factor : float, optional
        Frequency conversion factor that is multiplied with sqrt or eigenvalue
        of dynamical matrix. Default is VaspToTHz.
    nac_q_direction : array_like, optional
        See Interaction.nac_q_direction. Default is None.
    lapack_zheev_uplo : str, optional
        'U' or 'L' for lapack zheev solver. Default is 'L'.

    """
    import phono3py_rs  # type: ignore[import-untyped]

    if frequency_conversion_factor is None:
        _frequency_conversion_factor = get_physical_units().DefaultToTHz
    else:
        _frequency_conversion_factor = frequency_conversion_factor

    assert lapack_zheev_uplo in ("L", "U")

    requested = np.asarray(grid_points, dtype="int64")
    undone = np.unique(requested[phonon_done[requested] == 0])
    if len(undone) == 0:
        return

    (
        svecs,
        multi,
        masses,
        rec_lattice,
        positions,
        born,
        nac_factor,
        dielectric,
    ) = _extract_params(dm)

    if isinstance(dm, DynamicalMatrixGL):
        if dm.short_range_force_constants is None:
            dm.make_Gonze_nac_dataset()
        (
            gonze_fc,
            dd_q0,
            _G_cutoff,
            G_list,
            Lambda,
        ) = dm.Gonze_nac_dataset
        assert Lambda is not None
        fc = gonze_fc
        use_GL_NAC = True
    else:
        use_GL_NAC = False
        fc = dm.force_constants

    _nac_q_direction = (
        np.ascontiguousarray(nac_q_direction, dtype="double")
        if nac_q_direction is not None
        else None
    )

    fc_p2s, fc_s2p = _get_fc_elements_mapping(dm, fc)
    grid_address_c = np.ascontiguousarray(grid_address, dtype="int64")
    QDinv_c = np.ascontiguousarray(QDinv, dtype="double")
    fc_c = np.ascontiguousarray(fc, dtype="double")
    svecs_c = np.ascontiguousarray(svecs, dtype="double")
    multi_c = np.ascontiguousarray(multi, dtype="int64")
    masses_c = np.ascontiguousarray(masses, dtype="double")
    fc_p2s_c = np.ascontiguousarray(fc_p2s, dtype="int64")
    fc_s2p_c = np.ascontiguousarray(fc_s2p, dtype="int64")
    rec_lattice_c = np.ascontiguousarray(rec_lattice, dtype="double")
    positions_c = np.ascontiguousarray(positions, dtype="double")

    if use_GL_NAC:
        phono3py_rs.dynamical_matrices_at_gridpoints_gonze(
            eigenvectors,
            undone,
            grid_address_c,
            QDinv_c,
            fc_c,
            svecs_c,
            multi_c,
            masses_c,
            fc_p2s_c,
            fc_s2p_c,
            np.ascontiguousarray(born, dtype="double"),
            np.ascontiguousarray(dielectric, dtype="double"),
            rec_lattice_c,
            positions_c,
            np.ascontiguousarray(dd_q0, dtype="cdouble"),
            np.ascontiguousarray(G_list, dtype="double"),
            float(nac_factor),
            float(Lambda),
            _nac_q_direction,
        )
    else:
        phono3py_rs.dynamical_matrices_at_gridpoints(
            eigenvectors,
            undone,
            grid_address_c,
            QDinv_c,
            fc_c,
            svecs_c,
            multi_c,
            masses_c,
            fc_p2s_c,
            fc_s2p_c,
            np.ascontiguousarray(born, dtype="double") if born is not None else None,
            np.ascontiguousarray(dielectric, dtype="double")
            if born is not None
            else None,
            rec_lattice_c if born is not None else None,
            _nac_q_direction if born is not None else None,
            float(nac_factor),
        )

    for gp in undone:
        eigvals, vecs = np.linalg.eigh(eigenvectors[gp], UPLO=lapack_zheev_uplo)
        frequencies[gp] = (
            np.sign(eigvals) * np.sqrt(np.abs(eigvals)) * _frequency_conversion_factor
        )
        eigenvectors[gp] = vecs
    phonon_done[undone] = 1


def run_phonon_solver_py(
    grid_point: int,
    phonon_done: NDArray[np.byte],
    frequencies: NDArray[np.double],
    eigenvectors: NDArray[np.cdouble],
    grid_address: NDArray[np.int64],
    QDinv: NDArray[np.double],
    dynamical_matrix: DynamicalMatrix,
    frequency_conversion_factor: float | None = None,
    lapack_zheev_uplo: Literal["L", "U"] = "L",
) -> None:
    """Build and solve dynamical matrices on grid in python."""
    if frequency_conversion_factor is None:
        _frequency_conversion_factor = get_physical_units().DefaultToTHz
    else:
        _frequency_conversion_factor = frequency_conversion_factor

    gp = grid_point
    if phonon_done[gp] == 0:
        phonon_done[gp] = 1
        q = np.dot(grid_address[gp], QDinv.T)
        dynamical_matrix.run(q)
        dm = dynamical_matrix.dynamical_matrix
        assert dm is not None
        eigvals, eigvecs = np.linalg.eigh(dm, UPLO=lapack_zheev_uplo)
        eigvals = eigvals.real  # type: ignore[no-untyped-call]
        frequencies[gp] = (
            np.sqrt(np.abs(eigvals)) * np.sign(eigvals) * _frequency_conversion_factor
        )
        eigenvectors[gp] = eigvecs


def _extract_params(
    dm: DynamicalMatrix | DynamicalMatrixNAC,
) -> tuple[
    NDArray[np.double],
    NDArray[np.int64],
    NDArray[np.double],
    NDArray[np.double],
    NDArray[np.double],
    NDArray[np.double] | None,
    float,
    NDArray[np.double] | None,
]:
    svecs, multi = dm.primitive.get_smallest_vectors()
    assert dm.primitive.store_dense_svecs

    masses = np.asarray(dm.primitive.masses, dtype="double")
    rec_lattice = np.asarray(
        np.linalg.inv(dm.primitive.cell), dtype="double", order="C"
    )
    positions = np.asarray(dm.primitive.positions, dtype="double", order="C")
    if isinstance(dm, DynamicalMatrixNAC):
        born = dm.born
        nac_factor = dm.nac_factor
        dielectric = dm.dielectric_constant
    else:
        born = None
        nac_factor = 0
        dielectric = None

    return (
        svecs,
        multi,
        masses,
        rec_lattice,
        positions,
        born,
        nac_factor,
        dielectric,
    )


def _get_fc_elements_mapping(
    dm: DynamicalMatrix, fc: NDArray[np.double]
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    p2s_map = dm.primitive.p2s_map
    s2p_map = dm.primitive.s2p_map
    if fc.shape[0] == fc.shape[1]:  # full fc
        fc_p2s = p2s_map
        fc_s2p = s2p_map
    else:  # compact fc
        primitive = dm.primitive
        p2p_map = primitive.p2p_map
        s2pp_map = np.array(
            [p2p_map[s2p_map[i]] for i in range(len(s2p_map))], dtype="int64"
        )
        fc_p2s = np.arange(len(p2s_map), dtype="int64")
        fc_s2p = s2pp_map

    return fc_p2s, fc_s2p
