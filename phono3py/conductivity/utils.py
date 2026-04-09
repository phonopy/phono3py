"""Utilities for lattice thermal conductivity calculation."""

# Copyright (C) 2022 Atsushi Togo
# All rights reserved.
#
# This file is part of phono3py.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in
#   the documentation and/or other materials provided with the
#   distribution.
#
# * Neither the name of the phonopy project nor the names of its
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast

import numpy as np
from numpy.typing import NDArray
from phonopy.physical_units import get_physical_units

from phono3py.file_IO import write_pp_to_hdf5
from phono3py.phonon.grid import (
    BZGrid,
    get_grid_points_by_rotations,
    get_qpoints_from_bz_grid_points,
)
from phono3py.phonon3.interaction import Interaction
from phono3py.phonon3.triplets import get_all_triplets

if TYPE_CHECKING:
    from phono3py.conductivity.calculators import LBTECalculator, RTACalculator

_TOptions = TypeVar("_TOptions")

# Voigt notation index pairs for the six independent components of a
# symmetric 3x3 tensor: xx, yy, zz, yz, xz, xy.
VOIGT_INDEX_PAIRS: tuple[tuple[int, int], ...] = (
    (0, 0),
    (1, 1),
    (2, 2),
    (1, 2),
    (0, 2),
    (0, 1),
)


def get_kappa_star_operations(
    bz_grid: BZGrid, is_kappa_star: bool
) -> tuple[NDArray[np.int64], NDArray[np.double]]:
    """Return reciprocal operations and Cartesian rotations for kappa-star.

    When ``is_kappa_star`` is True, return the full point-group operations
    from the BZ grid.  Otherwise return a single identity operation.

    Returns
    -------
    reciprocal_operations : ndarray of int64, shape (num_ops, 3, 3)
    rotations_cartesian : ndarray of double, shape (num_ops, 3, 3)

    """
    if is_kappa_star:
        return bz_grid.reciprocal_operations, bz_grid.rotations_cartesian
    return (
        np.eye(3, dtype="int64", order="C").reshape(1, 3, 3),
        np.eye(3, dtype="double", order="C").reshape(1, 3, 3),
    )


def show_grid_point_header(
    bzgp: int,
    i_gp: int,
    num_gps: int,
    bz_grid: BZGrid,
    boundary_mfp: float | None = None,
    mass_variances: NDArray[np.double] | None = None,
) -> None:
    """Print grid point header for conductivity calculation progress.

    Parameters
    ----------
    bzgp : int
        BZ grid point index.
    i_gp : int
        0-based position in the irreducible grid point list.
    num_gps : int
        Total number of irreducible grid points.
    bz_grid : BZGrid
        Brillouin zone grid, used to obtain the q-point coordinates.
    boundary_mfp : float or None, optional
        Boundary mean free path in micrometers.  Printed when provided.
    mass_variances : ndarray or None, optional
        Mass variance parameters.  Printed when provided.
    """
    print(
        "======================= Grid point %d (%d/%d) "
        "=======================" % (bzgp, i_gp + 1, num_gps)
    )
    qpoint = get_qpoints_from_bz_grid_points(bzgp, bz_grid)
    print("q-point: (%5.2f %5.2f %5.2f)" % tuple(qpoint))
    if boundary_mfp is not None:
        if boundary_mfp > 1000:
            print(
                "Boundary mean free path (millimeter): %.3f" % (boundary_mfp / 1000.0)
            )
        else:
            print("Boundary mean free path (micrometer): %.5f" % boundary_mfp)
    if mass_variances is not None:
        print(
            ("Mass variance parameters: " + "%5.2e " * len(mass_variances))
            % tuple(mass_variances)
        )
    print(end="", flush=True)


def show_grid_point_frequencies_gv(
    frequencies: NDArray[np.double],
    gv: NDArray[np.double],
    gv_delta_q: float | None = None,
    ave_pp: NDArray[np.double] | None = None,
) -> None:
    """Print frequencies and group velocities at a grid point.

    Parameters
    ----------
    frequencies : ndarray of double, shape (num_band0,)
        Phonon frequencies in THz.
    gv : ndarray of double, shape (num_band0, 3)
        Group velocities.
    gv_delta_q : float or None, optional
        Finite-difference step used for group velocity; printed when provided.
    ave_pp : ndarray of double, shape (num_band0,), optional
        Averaged ph-ph interaction strength.  When provided, an extra Pqj
        column is printed.

    """
    text = "Frequency     group velocity (x, y, z)     |gv|"
    if ave_pp is not None:
        text += "       Pqj"
    if gv_delta_q is not None:
        text += "  (dq=%3.1e)" % gv_delta_q
    print(text)
    _print_freq_gv_rows(frequencies, gv, ave_pp)
    print("", end="", flush=True)


def show_grid_point_frequencies_gv_on_kstar(
    frequencies: NDArray[np.double],
    gv: NDArray[np.double],
    gp: int,
    bz_grid: BZGrid,
    point_operations: NDArray[np.int64],
    rotations_cartesian: NDArray[np.double],
    gv_delta_q: float | None = None,
    ave_pp: NDArray[np.double] | None = None,
) -> None:
    """Print frequencies and group velocities expanded over k-star arms.

    Parameters
    ----------
    frequencies : ndarray of double, shape (num_band0,)
        Phonon frequencies in THz.
    gv : ndarray of double, shape (num_band0, 3)
        Group velocities at the irreducible grid point.
    gp : int
        BZ grid point index.
    bz_grid : BZGrid
        Brillouin zone grid object.
    point_operations : ndarray of int64, shape (num_ops, 3, 3)
        Reciprocal-space point-group operations (integer).
    rotations_cartesian : ndarray of double, shape (num_ops, 3, 3)
        Cartesian rotation matrices.
    gv_delta_q : float or None, optional
        Finite-difference step used for group velocity; printed when provided.
    ave_pp : ndarray of double, shape (num_band0,), optional
        Averaged ph-ph interaction strength.

    """
    text = "Frequency     group velocity (x, y, z)     |gv|"
    if ave_pp is not None:
        text += "       Pqj"
    if gv_delta_q is not None:
        text += "  (dq=%3.1e)" % gv_delta_q
    print(text)

    q = get_qpoints_from_bz_grid_points(gp, bz_grid)
    rotation_map = get_grid_points_by_rotations(gp, bz_grid)
    for i, j in enumerate(np.unique(rotation_map)):
        for k, (rot, rot_c) in enumerate(
            zip(point_operations, rotations_cartesian, strict=True)
        ):
            if rotation_map[k] != j:
                continue
            q_rot = tuple(np.dot(rot, q))
            print(" k*%-2d (%5.2f %5.2f %5.2f)" % ((i + 1,) + q_rot))
            gv_rot = np.dot(rot_c, gv.T).T
            _print_freq_gv_rows(frequencies, gv_rot, ave_pp)
    print("", end="", flush=True)


def _print_freq_gv_rows(
    frequencies: NDArray[np.double],
    gv: NDArray[np.double],
    ave_pp: NDArray[np.double] | None = None,
) -> None:
    """Print frequency/velocity rows with optional Pqj column."""
    if ave_pp is not None:
        for f, v, pp in zip(frequencies, gv, ave_pp, strict=True):
            print(
                "%8.3f   (%8.3f %8.3f %8.3f) %8.3f %11.3e"
                % (f, v[0], v[1], v[2], np.linalg.norm(v), pp)
            )
    else:
        for f, v in zip(frequencies, gv, strict=True):
            print(
                "%8.3f   (%8.3f %8.3f %8.3f) %8.3f"
                % (f, v[0], v[1], v[2], np.linalg.norm(v))
            )


def log_sigma_header(sigma: float | None) -> None:
    """Print the sigma/tetrahedron banner line only."""
    text = "----------- Thermal conductivity (W/m-k) "
    if sigma:
        text += "for sigma=%s -----------" % sigma
    else:
        text += "with tetrahedron method -----------"
    print(text, flush=True)


def log_kappa_header(
    sigma: float | None,
    show_ipm: bool = False,
) -> None:
    """Print the kappa table header line for a given sigma."""
    log_sigma_header(sigma)
    if show_ipm:
        print(
            ("#%6s       " + " %-10s" * 6 + "#ipm")
            % ("T(K)", "xx", "yy", "zz", "yz", "xz", "xy")
        )
    else:
        print(
            ("#%6s       " + " %-10s" * 6)
            % ("T(K)", "xx", "yy", "zz", "yz", "xz", "xy")
        )


def log_kappa_row(
    label: str,
    temperature: float,
    kappa_row: NDArray[np.double],
    num_ignored: int | None = None,
    num_phonon_modes: int | None = None,
) -> None:
    """Print one row of the kappa table."""
    if num_ignored is not None and num_phonon_modes is not None:
        print(
            label
            + ("%7.1f" + " %10.3f" * 6 + "    %d/%d")
            % ((temperature,) + tuple(kappa_row) + (num_ignored, num_phonon_modes))
        )
    else:
        print(label + ("%7.1f " + " %10.3f" * 6) % ((temperature,) + tuple(kappa_row)))


def get_unit_to_WmK() -> float:
    """Return conversion factor to WmK."""
    unit_to_WmK = (
        (get_physical_units().THz * get_physical_units().Angstrom) ** 2
        / (get_physical_units().Angstrom ** 3)
        * get_physical_units().EV
        / get_physical_units().THz
        / (2 * np.pi)
    )  # 2pi comes from definition of lifetime.
    return unit_to_WmK


def build_options(_options_type: type[_TOptions], **kwargs: Any) -> _TOptions:
    """Return kwargs cast as typed options data.

    This is a tiny helper for constructing `TypedDict` option data in init
    modules without repeating explicit key-to-value dict literals.

    """
    return cast(_TOptions, kwargs)


def select_colmat_solver(pinv_solver: int) -> int:
    """Return collision matrix solver id."""
    try:
        import phono3py._phono3py as phono3c

        default_solver = phono3c.default_colmat_solver()
    except ImportError:
        print("Phono3py C-routine is not compiled correctly.")
        default_solver = 4

    if not phono3c.include_lapacke():
        if pinv_solver in (1, 2, 6):
            raise RuntimeError(
                "Use pinv-solver 3, 4, or 5 because "
                "phono3py is not compiled with LAPACKE."
            )

    solver_numbers = (1, 2, 3, 4, 5, 6, 7)

    solver = pinv_solver
    if solver == 6:  # 6 must return 3 for not transposing unitary matrix.
        solver = 3
    if solver == 0:  # default solver
        if default_solver in (3, 4, 5):
            try:
                import scipy.linalg  # noqa F401
            except ImportError:
                solver = 1
            else:
                solver = default_solver
        else:
            solver = default_solver
    elif solver not in solver_numbers:
        solver = default_solver

    return solver


def diagonalize_collision_matrix(
    collision_matrices: NDArray[np.double],
    i_sigma: int | None = None,
    i_temp: int | None = None,
    pinv_solver: int = 0,
    log_level: int = 0,
) -> NDArray[np.double] | None:
    """Diagonalize collision matrices.

    Note
    ----
    collision_matrices is overwritten by eigenvectors.

    Parameters
    ----------
    collision_matrices : ndarray
        Collision matrix. dtype='double', order='C'.
        Supported shapes:
            (sigmas, temperatures, prod(mesh), num_band, prod(mesh), num_band)
            (sigmas, temperatures, ir_grid_points, num_band, 3,
                                   ir_grid_points, num_band, 3)
            (size, size)
    i_sigma : int, optional
        Index of BZ integration method. Default is None.
    i_temp : int, optional
        Index of temperature. Default is None.
    pinv_solver : int, optional
        Diagonalization solver choice. Default is 0 (auto).
    log_level : int, optional
        Verbosity level. Default is 0.

    Returns
    -------
    w : ndarray or None
        Eigenvalues, shape=(size,), dtype='double'.
        None is returned when pinv_solver==7.

    """
    start = time.time()

    shape = collision_matrices.shape
    if len(shape) == 6:
        size = shape[2] * shape[3]
        assert size == shape[4] * shape[5]
    elif len(shape) == 8:
        size = np.prod(shape[2:5])
        assert size == np.prod(shape[5:8])
    elif len(shape) == 2:
        size = shape[0]
        assert size == shape[1]

    solver = select_colmat_solver(pinv_solver)
    trace = np.trace(collision_matrices[i_sigma, i_temp].reshape(size, size))

    if solver in [1, 2]:
        if log_level:
            routine = ["dsyev", "dsyevd"][solver - 1]
            print("Diagonalizing by lapacke %s ... " % routine, end="", flush=True)
        import phono3py._phono3py as phono3c

        w = np.zeros(size, dtype="double")
        _i_sigma = 0 if i_sigma is None else i_sigma
        _i_temp = 0 if i_temp is None else i_temp
        phono3c.diagonalize_collision_matrix(
            collision_matrices, w, _i_sigma, _i_temp, 0.0, (solver + 1) % 2, 0
        )
    elif solver == 3:
        if log_level:
            print("Diagonalize by np.linalg.eigh ", end="", flush=True)
        col_mat = collision_matrices[i_sigma, i_temp].reshape(size, size)
        w, col_mat[:] = np.linalg.eigh(col_mat)  # type: ignore[assignment]
    elif solver == 4:
        if log_level:
            print("Diagonalize by scipy.linalg.lapack.dsyev ", end="", flush=True)
        import scipy.linalg

        col_mat = collision_matrices[i_sigma, i_temp].reshape(size, size)
        w, _, info = scipy.linalg.lapack.dsyev(col_mat.T, overwrite_a=1)  # type: ignore
    elif solver == 5:
        if log_level:
            print("Diagnalize by scipy.linalg.lapack.dsyevd ", end="", flush=True)
        import scipy.linalg

        col_mat = collision_matrices[i_sigma, i_temp].reshape(size, size)
        w, _, info = scipy.linalg.lapack.dsyevd(col_mat.T, overwrite_a=1)  # type: ignore
    elif solver == 7:
        if log_level:
            print(
                "Pseudo inversion using np.linalg.pinv(a, hermitian=False) ",
                end="",
                flush=True,
            )
        col_mat = collision_matrices[i_sigma, i_temp].reshape(size, size)
        col_mat[:, :] = np.linalg.pinv(col_mat, hermitian=False)
        w = None

    if log_level:
        if w is not None:
            print(f"sum={w.sum():<.1e} d={trace - w.sum():<.1e} ", end="")
        print("[%.3fs]" % (time.time() - start), flush=True)

    return w


def write_pp_interaction(
    conductivity: RTACalculator | LBTECalculator,
    pp: Interaction,
    i: int,
    filename: str | os.PathLike | None = None,
    compression: Literal["gzip", "lzf"] | int | None = "gzip",
) -> None:
    """Write ph-ph interaction strength in hdf5 file."""
    grid_point = conductivity.grid_points[i]
    sigmas = conductivity.sigmas
    sigma_cutoff = conductivity.sigma_cutoff_width
    mesh = conductivity.mesh_numbers
    triplets, weights, _, _ = pp.get_triplets_at_q()
    all_triplets = get_all_triplets(grid_point, pp.bz_grid)

    if len(sigmas) > 1:
        print("Multiple smearing parameters were given. The last one in ")
        print("ph-ph interaction calculations was written in the file.")

    write_pp_to_hdf5(
        mesh,
        pp=pp.interaction_strength,
        g_zero=pp.zero_value_positions,
        grid_point=grid_point,
        triplet=triplets,
        weight=weights,
        triplet_all=all_triplets,
        sigma=sigmas[-1],
        sigma_cutoff=sigma_cutoff,
        filename=filename,
        compression=compression,
    )
