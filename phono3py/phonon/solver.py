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
    import phono3py._phono3py as phono3c  # type: ignore[import-untyped]
    import phono3py._phononcalc as phononcalc  # type: ignore[import-untyped]

    if frequency_conversion_factor is None:
        _frequency_conversion_factor = get_physical_units().DefaultToTHz
    else:
        _frequency_conversion_factor = frequency_conversion_factor

    (
        svecs,
        multi,
        masses,
        rec_lattice,  # column vectors
        positions,
        born,
        nac_factor,
        dielectric,
    ) = _extract_params(dm)

    if isinstance(dm, DynamicalMatrixGL):
        if dm.short_range_force_constants is None:
            dm.make_Gonze_nac_dataset()

        (
            gonze_fc,  # fc where the dipole-diple contribution is removed.
            dd_q0,  # second term of dipole-dipole expression.
            G_cutoff,  # Cutoff radius in reciprocal space. This will not be used.
            G_list,  # List of G points where d-d interactions are integrated.
            Lambda,
        ) = dm.Gonze_nac_dataset  # Convergence parameter
        assert Lambda is not None
        fc = gonze_fc
        use_GL_NAC = True
    else:
        use_GL_NAC = False
        positions = np.zeros(3)  # dummy variable
        dd_q0 = np.zeros(2)  # type: ignore[assignment]  # dummy variable
        G_list = np.zeros(3)  # dummy variable
        Lambda = 0  # dummy variable
        if not isinstance(dm, DynamicalMatrixNAC):
            born = np.zeros((3, 3))  # dummy variable
            dielectric = np.zeros(3)  # dummy variable
        fc = dm.force_constants

    if nac_q_direction is None:
        is_nac_q_zero = False
        _nac_q_direction = np.zeros(3)  # dummy variable
    else:
        is_nac_q_zero = True
        _nac_q_direction = np.array(nac_q_direction, dtype="double")

    assert lapack_zheev_uplo in ("L", "U")

    if not phono3c.include_lapacke():
        # phonon_done is set even with phono3c.include_lapacke() == 0 for which
        # dynamical matrices are not diagonalized in
        # phononcalc.phonons_at_gridpoints.
        phonon_undone = np.where(phonon_done == 0)[0]

    fc_p2s, fc_s2p = _get_fc_elements_mapping(dm, fc)
    phononcalc.phonons_at_gridpoints(
        frequencies,
        eigenvectors,
        phonon_done,
        np.asarray(grid_points, dtype="int64"),
        np.asarray(grid_address, dtype="int64", order="C"),
        np.asarray(QDinv, dtype="double", order="C"),
        fc,
        svecs,
        multi,
        positions,
        masses,
        fc_p2s,
        fc_s2p,
        _frequency_conversion_factor,
        born,
        dielectric,
        rec_lattice,
        _nac_q_direction,
        float(nac_factor),
        dd_q0,
        G_list,
        float(Lambda),
        isinstance(dm, DynamicalMatrixNAC) * 1,
        is_nac_q_zero * 1,
        use_GL_NAC * 1,
        lapack_zheev_uplo,
    )

    if not phono3c.include_lapacke():
        # The variable `eigenvectors` contains dynamical matrices.
        # They are diagonalized in python as follows.
        for gp in phonon_undone:
            frequencies[gp], eigenvectors[gp] = np.linalg.eigh(
                eigenvectors[gp], UPLO=lapack_zheev_uplo
            )
        frequencies[phonon_undone] = (
            np.sign(frequencies[phonon_undone])
            * np.sqrt(np.abs(frequencies[phonon_undone]))
            * _frequency_conversion_factor
        )


def run_phonon_solver_rust(
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
    """Build and solve dynamical matrices on grid in Rust + python.

    Dynamical matrices are constructed in Rust via
    ``phonors.dynamical_matrices_at_gridpoints`` (and its Gonze NAC
    variant).  Eigen-decomposition is performed in python with
    ``numpy.linalg.eigh`` since LAPACK is not linked in Rust.

    See ``run_phonon_solver_c`` for parameter documentation.

    """
    import phonors  # type: ignore[import-untyped]

    from phono3py._lang import log_dispatch

    log_dispatch("Rust", "run_phonon_solver_rust")

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
        phonors.dynamical_matrices_at_gridpoints_gonze(
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
        phonors.dynamical_matrices_at_gridpoints(
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
