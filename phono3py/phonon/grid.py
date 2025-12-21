"""Regular grid tools."""

# Copyright (C) 2021 Atsushi Togo
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

import dataclasses
import warnings
from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from phonopy.structure.cells import (
    determinant,
    estimate_supercell_matrix,
    get_reduced_bases,
    is_primitive_cell,
)
from phonopy.structure.grid_points import extract_ir_grid_points, length2mesh
from phonopy.structure.symmetry import NosymDataset
from phonopy.utils import similarity_transformation
from spglib import SpglibDataset, SpglibMagneticDataset


@dataclasses.dataclass
class MockSymmetryDataset:
    """Mock class for symmetry dataset."""

    rotations: NDArray
    transformation_matrix: NDArray
    std_lattice: NDArray
    std_types: NDArray
    number: int


class BZGrid:
    """Data structure of BZ grid of primitive cell.

    Note when using SNF
    -------------------
    The grid lattice is generated against the conventional unit cell when using
    SNF. To make the grid lattice of the input cell commensurate with this generated
    grid lattice, the input cell is assumed to a primitive cell. When the input
    cell is not a primitive cell, it falls back to non-SNF grid generation.

    GR-grid and BZ-grid
    -------------------
    GR-grid address is defined by three integers of {0 <= m_i < D_diag[i]}.
    Therefore number of unique grid points represented by GR-grid is
    prod(D_diag).

    BZ-grid address is defined on GR-grid but to be closest to the origin
    in Cartesian coordinates of the reciprocal space in the periodic boundary
    condition of the reciprocal lattice. The translationally equivalent
    grid points on BZ surface can be equidistant from the origin.
    In this case, those all grid addresses are contained in the data structure
    of BZGrid. Therefore number of unique grid points represented by BZ-grid
    can be larger than prod(D_diag).

    The grid systems with (BZ-grid, BZG) and without (GR-grid, GRG) BZ surface
    are mutually related up to modulo D_diag. More precisely the conversion
    of grid addresses are performed as follows:

    From BZG to GRG
        gr_gp = get_grid_point_from_address(bz_grid.addresses[bz_gp], D_diag)
    and the shortcut is
        gr_gp = bz_grid.bzg2grg[bz_gp]

    From GRG to BZG
    When store_dense_gp_map=True,
        bz_gp = bz_grid.gp_map[gr_gp]
    When store_dense_gp_map=False,
        bz_gp = gr_gp
    The shortcut is
        bz_gp = bz_grid.grg2bzg[gr_gp]
    When translationally equivalent points exist on BZ surface, the one of them
    is chosen.

    Recovering reduced coordinates
    ------------------------------
    q-points with respect to the original reciprocal
    basis vectors are given by

    q = np.dot(Q, addresses[gp] / D_diag.astype('double'))

    for the Gamma centred grid. With shifted, where only half grid shifts
    that satisfy the symmetry are considered,

    q = np.dot(Q, (addresses[gp] + np.dot(P, s)) / D_diag.astype('double'))

    where s is the shift vectors that are 0 or 1/2. But it is more
    convenient to use the integer shift vectors S by 0 or 1, which gives

    q = (np.dot(Q, (2 * addresses[gp] + PS) / D_diag.astype('double') / 2))

    where PS = np.dot(P, s) * 2.

    Attributes
    ----------
    addresses : ndarray
    gp_map : ndarray
    bzg2grg : ndarray
    grg2bzg : ndarray
    store_dense_gp_map : bool, optional
    rotations : ndarray
    reciprocal_operations : ndarray
    D_diag : ndarray
    P : ndarray
    Q : ndarray
    PS : ndarray
    QDinv : ndarray
    grid_matrix : ndarray
    microzone_lattice : ndarray
    gp_Gamma : int

    """

    def __init__(
        self,
        mesh: float | ArrayLike,
        reciprocal_lattice: NDArray | Sequence[Sequence[float]] | None = None,
        lattice: NDArray | Sequence[Sequence[float]] | None = None,
        symmetry_dataset: SpglibDataset
        | SpglibMagneticDataset
        | NosymDataset
        | None = None,
        transformation_matrix: ArrayLike | None = None,
        is_shift: NDArray | Sequence[float] | None = None,
        is_time_reversal: bool = True,
        use_grg: bool = False,
        force_SNF: bool = False,
        SNF_coordinates: Literal["reciprocal", "direct"] = "reciprocal",
        store_dense_gp_map: bool = True,
    ):
        """Init method.

        mesh : array_like or float
            Mesh numbers or length. shape=(3,), dtype='int64'
        reciprocal_lattice : array_like
            Reciprocal primitive basis vectors given as column vectors shape=(3,
            3), dtype='double', order='C'
        lattice : array_like
            Direct primitive basis vectors given as row vectors shape=(3, 3),
            dtype='double', order='C'
        symmetry_dataset : SpglibDataset, SpglibMagneticDataset, NosymDataset, optional
            Symmetry dataset (Symmetry.dataset) searched for the primitive cell
            corresponding to ``reciprocal_lattice`` or ``lattice``.
        transformation_matrix : array_like, optional
            Transformation matrix equivalent to ``transformation_matrix`` in
            spglib-dataset. This is only used when ``use_grg=True`` and
            ``symmetry_dataset`` is unspecified. Default is None.
        is_shift : array_like or None, optional
            [0, 0, 0] (or [False, False, False]) gives Gamma center mesh and
            value 1 (or True) gives half mesh shift along the basis vectors.
            Default is None. dtype='int64', shape=(3,)
        is_time_reveral : bool, optional
            Inversion symmetry is included in reciprocal point group. Default is
            True.
        use_grg : bool, optional
            Use generalized regular grid. Default is False. ``symmetry_dataset``
            or ``transformation_matrix`` have to be specified when
            ``use_grg=True``.
        force_SNF : bool, optional
            Enforce Smith normal form even when grid lattice of GR-grid is the
            same as the traditional grid lattice. Default is False.
        SNF_coordinates : str, optional
            `reciprocal` or `direct`. Space of coordinates to generate grid
            generating matrix either in direct or reciprocal space. The default
            is `reciprocal`.
        store_dense_gp_map : bool, optional
            See the detail in the docstring of `_relocate_BZ_grid_address`.
            Default is True.

        """
        self._symmetry_dataset = symmetry_dataset
        self._transformation_matrix = transformation_matrix
        if is_shift is None:
            self._is_shift = None
        else:
            self._is_shift = [v * 1 for v in is_shift]
        self._is_time_reversal = is_time_reversal
        self._store_dense_gp_map = store_dense_gp_map
        self._addresses: NDArray
        self._gp_map: NDArray
        self._grid_matrix = None
        self._D_diag = np.ones(3, dtype="int64")
        self._Q = np.eye(3, dtype="int64", order="C")
        self._P = np.eye(3, dtype="int64", order="C")
        self._QDinv: NDArray
        self._microzone_lattice: NDArray
        self._rotations: NDArray
        self._reciprocal_operations: NDArray
        self._rotations_cartesian: NDArray
        self._gp_Gamma: int
        self._bzg2grg: NDArray
        self._grg2bzg: NDArray

        if reciprocal_lattice is not None:
            self._reciprocal_lattice = np.array(
                reciprocal_lattice, dtype="double", order="C"
            )
            self._lattice = np.array(
                np.linalg.inv(reciprocal_lattice), dtype="double", order="C"
            )
        if lattice is not None:
            self._lattice = np.array(lattice, dtype="double", order="C")
            self._reciprocal_lattice = np.array(
                np.linalg.inv(lattice), dtype="double", order="C"
            )

        self._generate_grid(
            mesh, use_grg=use_grg, force_SNF=force_SNF, SNF_coordinates=SNF_coordinates
        )

    @property
    def D_diag(self) -> NDArray:
        """Diagonal elements of diagonal matrix after SNF: D=PAQ.

        This corresponds to the mesh numbers in transformed reciprocal
        basis vectors.
        shape=(3,), dtype='int64'

        """
        return self._D_diag

    @property
    def P(self) -> NDArray:
        """Left unimodular matrix after SNF: D=PAQ.

        Left unimodular matrix after SNF: D=PAQ.
        shape=(3, 3), dtype='int64', order='C'.

        """
        return self._P

    @property
    def Q(self) -> NDArray:
        """Right unimodular matrix after SNF: D=PAQ.

        Right unimodular matrix after SNF: D=PAQ.
        shape=(3, 3), dtype='int64', order='C'.

        """
        return self._Q

    @property
    def QDinv(self) -> NDArray:
        """QD^-1.

        ndarray :
            shape=(3, 3), dtype='double', order='C'.

        """
        return self._QDinv

    @property
    def PS(self) -> NDArray:
        """Integer shift vectors of GRGrid."""
        if self._is_shift is None:
            return np.zeros(3, dtype="int64")
        else:
            return np.array(np.dot(self.P, self._is_shift), dtype="int64")

    @property
    def grid_matrix(self) -> NDArray | None:
        """Grid generating matrix to be represented by SNF.

        Grid generating matrix used for SNF.
        When SNF is used, ndarray, otherwise None.
        shape=(3, 3), dtype='int64', order='C'.

        """
        return self._grid_matrix

    @property
    def addresses(self) -> NDArray:
        """BZ-grid addresses.

        Integer grid address of the points in Brillouin zone including
        surface. There are two types of address order by either
        `store_dense_gp_map` is True or False.
        shape=(np.prod(D_diag) + some on surface, 3), dtype='int64', order='C'.

        """
        return self._addresses

    @property
    def gp_map(self) -> NDArray:
        """Definitions of grid index.

        Grid point mapping table containing BZ surface. There are two types of
        address order by either `store_dense_gp_map` is True or False. See more
        detail in `_relocate_BZ_grid_address` docstring.

        """
        return self._gp_map

    @property
    def gp_Gamma(self) -> int:
        """Return grid point index of Gamma-point."""
        return self._gp_Gamma

    @property
    def bzg2grg(self) -> NDArray:
        """Transform grid point indices from BZG to GRG.

        Grid index mapping table from BZGrid to GRgrid.
        shape=(len(addresses), ), dtype='int64'.

        Equivalent to
            get_grid_point_from_address(
                self._addresses[bz_grid_index], self._D_diag)

        """
        return self._bzg2grg

    @property
    def grg2bzg(self) -> NDArray:
        """Transform grid point indices from GRG to BZG.

        Grid index mapping table from GRgrid to BZGrid. Unique one
        of translationally equivalent grid points in BZGrid is chosen.
        shape=(prod(D_diag), ), dtype='int64'.

        """
        return self._grg2bzg

    @property
    def microzone_lattice(self) -> NDArray:
        """Basis vectors of microzone.

        Basis vectors of microzone of GR-grid in column vectors.
        shape=(3, 3), dtype='double', order='C'.

        """
        return self._microzone_lattice

    @property
    def reciprocal_lattice(self) -> NDArray:
        """Reciprocal basis vectors of primitive cell.

        Reciprocal basis vectors of primitive cell in column vectors.
        shape=(3, 3), dtype='double', order='C'.

        """
        return self._reciprocal_lattice

    @property
    def store_dense_gp_map(self) -> bool:
        """Return gp_map type.

        See the detail in the docstring of `_relocate_BZ_grid_address`.

        """
        return self._store_dense_gp_map

    @property
    def rotations(self) -> NDArray:
        """Return rotation matrices for grid points.

        Rotation matrices for GR-grid addresses (g) defined as g'=Rg. This can
        be different from ``reciprocal_operations`` when GR-grid is used because
        grid addresses are defined on an oblique lattice.
        shape=(rotations, 3, 3), dtype='int64', order='C'.

        """
        return self._rotations

    @property
    def rotations_cartesian(self) -> NDArray:
        """Return rotations in Cartesian coordinates."""
        return self._rotations_cartesian

    @property
    def reciprocal_operations(self) -> NDArray:
        """Return reciprocal rotations.

        Reciprocal space rotation matrices in fractional coordinates defined as
        q'=Rq.
        shape=(rotations, 3, 3), dtype='int64', order='C'.

        """
        return self._reciprocal_operations

    @property
    def symmetry_dataset(
        self,
    ) -> SpglibDataset | SpglibMagneticDataset | NosymDataset | None:
        """Return Symmetry.dataset."""
        return self._symmetry_dataset

    def get_indices_from_addresses(self, addresses: NDArray) -> int | NDArray:
        """Return BZ grid point indices from grid addresses.

        Parameters
        ----------
        addresses : ndarray
            Integer grid addresses.
            shape=(n, 3) or (3, ), where n is the number of grid points.

        Returns
        -------
        ndarray or int
            Grid point indices corresponding to the grid addresses. Each
            returned grid point index is one of those of the
            translationally equivalent grid points.
            shape=(n, ), dtype='int64' when multiple addresses are given.
            Otherwise one integer value is returned.

        """
        try:
            len(addresses[0])
        except TypeError:
            return int(
                self._grg2bzg[get_grid_point_from_address(addresses, self._D_diag)]
            )

        gps = [get_grid_point_from_address(adrs, self._D_diag) for adrs in addresses]
        return np.array(self._grg2bzg[gps], dtype="int64")

    def _generate_grid(
        self,
        mesh: ArrayLike,
        use_grg: bool = False,
        force_SNF: bool = False,
        SNF_coordinates: Literal["reciprocal", "direct"] = "reciprocal",
    ):
        gm = GridMatrix(
            mesh,
            self._lattice,
            symmetry_dataset=self._symmetry_dataset,
            transformation_matrix=self._transformation_matrix,
            use_grg=use_grg,
            force_SNF=force_SNF,
            SNF_coordinates=SNF_coordinates,
        )
        self._grid_matrix = gm.grid_matrix
        self._D_diag = gm.D_diag
        self._P = gm.P
        self._Q = gm.Q
        self._set_bz_grid()
        self._set_rotations()

    def _set_bz_grid(self):
        """Generate BZ grid addresses and grid point mapping table."""
        (self._addresses, self._gp_map, self._bzg2grg) = _relocate_BZ_grid_address(
            self._D_diag,
            self._Q,
            self._reciprocal_lattice,  # column vectors
            PS=self.PS,
            store_dense_gp_map=self._store_dense_gp_map,
        )
        if self._store_dense_gp_map:
            self._grg2bzg = np.array(self._gp_map[:-1], dtype="int64")
        else:
            self._grg2bzg = np.arange(np.prod(self._D_diag), dtype="int64")

        self._QDinv = np.array(
            self.Q * (1 / self.D_diag.astype("double")), dtype="double", order="C"
        )
        self._microzone_lattice = np.dot(
            self._reciprocal_lattice, np.dot(self._QDinv, self._P)
        )
        self._gp_Gamma = int(
            self._grg2bzg[get_grid_point_from_address([0, 0, 0], self._D_diag)]
        )

    def _set_rotations(self):
        """Rotation matrices are transformed those for non-diagonal D matrix.

        Terminate when symmetry of grid is broken.

        """
        import phono3py._recgrid as recgrid  # type: ignore

        if self._symmetry_dataset is None:
            direct_rotations = np.eye(3, dtype="int64", order="C").reshape(1, 3, 3)
        else:
            direct_rotations = np.array(
                self._symmetry_dataset.rotations, dtype="int64", order="C"
            )
        rec_rotations = np.zeros((48, 3, 3), dtype="int64", order="C")
        num_rec_rot = recgrid.reciprocal_rotations(
            rec_rotations, direct_rotations, self._is_time_reversal
        )
        self._reciprocal_operations = np.array(
            rec_rotations[:num_rec_rot], dtype="int64", order="C"
        )
        self._rotations = self._get_GRG_rotations()
        self._rotations_cartesian = np.array(
            [
                similarity_transformation(self._reciprocal_lattice, r)
                for r in self._reciprocal_operations
            ],
            dtype="double",
            order="C",
        )
        if self._is_shift is not None:
            check_grid_shift_symmetry(self._is_shift, self._rotations, self._P)

    def _get_GRG_rotations(self) -> NDArray:
        """Return rotation matrices in GR-grid."""
        import phono3py._recgrid as recgrid  # type: ignore

        rotations = np.zeros(
            self._reciprocal_operations.shape, dtype="int64", order="C"
        )
        if not recgrid.transform_rotations(
            rotations, self._reciprocal_operations, self._D_diag, self._Q
        ):
            msg = "Grid symmetry is broken. Use generalized regular grid."
            raise RuntimeError(msg)

        return rotations


class GridMatrix:
    """Class to generate regular grid in reciprocal space.

    Attributes
    ----------
    D_diag : ndarray
    P : ndarray
    Q : ndarray
    grid_matrix : ndarray

    """

    def __init__(
        self,
        mesh: ArrayLike,
        lattice: ArrayLike,
        symmetry_dataset: SpglibDataset
        | SpglibMagneticDataset
        | NosymDataset
        | None = None,
        transformation_matrix: ArrayLike | None = None,
        use_grg: bool = True,
        force_SNF: bool = False,
        SNF_coordinates: Literal["reciprocal", "direct"] = "reciprocal",
    ) -> None:
        """Init method.

        mesh : array_like or float
            Mesh numbers or length. With float number, either conventional or
            generalized regular grid is computed depending on the given flags
            (`use_grg`, `force_SNF`). Given ndarry with
                shape=(3,), dtype='int64': conventional regular grid shape=(3,
                3), dtype='int64': generalized regular grid
        lattice : array_like
            Primitive basis vectors in direct space given as row vectors.
            shape=(3, 3), dtype='double', order='C'
        symmetry_dataset : SpglibDataset, SpglibMagneticDataset, NosymDataset, optional
            Symmetry dataset of spglib (Symmetry.dataset) of primitive cell that
            has `lattice`. Default is None.
        transformation_matrix : array_like, optional
            Transformation matrix equivalent to ``transformation_matrix`` in
            spglib-dataset. This is only used when ``use_grg=True`` and
            ``symmetry_dataset`` is unspecified. Default is None.
        use_grg : bool, optional
            Use generalized regular grid. Default is False.
        force_SNF : bool, optional
            Enforce Smith normal form even when grid lattice of GR-grid is the
            same as the traditional grid lattice. Default is False.
        SNF_coordinates : str, optional
            `reciprocal` or `direct`. Space of coordinates to generate grid
            generating matrix either in direct or reciprocal space. The default
            is `reciprocal`.

        """
        self._mesh = mesh
        self._lattice = lattice
        self._grid_matrix = None
        self._D_diag = np.ones(3, dtype="int64")
        self._Q = np.eye(3, dtype="int64", order="C")
        self._P = np.eye(3, dtype="int64", order="C")

        self._set_mesh_numbers(
            mesh,
            use_grg=use_grg,
            symmetry_dataset=symmetry_dataset,
            transformation_matrix=transformation_matrix,
            force_SNF=force_SNF,
            coordinates=SNF_coordinates,
        )

    @property
    def grid_matrix(self) -> NDArray | None:
        """Grid generating matrix to be represented by SNF.

        Grid generating matrix used for SNF.
        When SNF is used, ndarray, otherwise None.
        shape=(3, 3), dtype='int64', order='C'.

        """
        return self._grid_matrix

    @property
    def D_diag(self) -> NDArray:
        """Diagonal elements of diagonal matrix after SNF: D=PAQ.

        This corresponds to the mesh numbers in transformed reciprocal
        basis vectors.
        shape=(3,), dtype='int64'

        """
        return self._D_diag

    @property
    def P(self) -> NDArray:
        """Left unimodular matrix after SNF: D=PAQ.

        Left unimodular matrix after SNF: D=PAQ.
        shape=(3, 3), dtype='int64', order='C'.

        """
        return self._P

    @property
    def Q(self) -> NDArray:
        """Right unimodular matrix after SNF: D=PAQ.

        Right unimodular matrix after SNF: D=PAQ.
        shape=(3, 3), dtype='int64', order='C'.

        """
        return self._Q

    def _set_mesh_numbers(
        self,
        mesh: ArrayLike,
        use_grg: bool = False,
        symmetry_dataset: SpglibDataset
        | SpglibMagneticDataset
        | NosymDataset
        | None = None,
        transformation_matrix: ArrayLike | None = None,
        force_SNF: bool = False,
        coordinates: Literal["reciprocal", "direct"] = "reciprocal",
    ):
        """Set mesh numbers from array or float value.

        self._grid_matrix and self._D_diag can be set.

        Four cases:
        1) Three integers are given.
            Use these numbers as regular grid.
        2) One number is given with no symmetry provided.
            Regular grid is computed from this value. Grid is generated so that
            distances in reciprocal space between neighboring grid points become
            similar.
        3) One number is given and use_grg=False.
            Regular grid is computed from this value and point group symmetry.
            Grid is generated so that distances in reciprocal space between
            neighboring grid points become similar.
        4) One number is given with symmetry provided and use_grg=True.
            Generalized regular grid is generated. However if the grid
            generating matrix is a diagonal matrix, use it as the D matrix
            of SNF and P and Q are set as identity matrices. Otherwise
            D, P, Q matrices are computed using SNF. Grid is generated so that
            basis vectors of supercell in direct space corresponding to this grid
            have similar lengths.

        """
        try:
            length = float(mesh)  # type: ignore
            if use_grg:
                found_grg = self._run_grg(
                    symmetry_dataset,
                    transformation_matrix,
                    length,
                    None,
                    force_SNF,
                    coordinates,
                )
            if not use_grg or not found_grg:
                if symmetry_dataset is None:
                    mesh_numbers = length2mesh(length, self._lattice)
                else:
                    mesh_numbers = length2mesh(
                        length, self._lattice, rotations=symmetry_dataset.rotations
                    )
                self._D_diag = np.array(mesh_numbers, dtype="int64")

        except (ValueError, TypeError):
            num_values = len(np.ravel(mesh))
            if num_values == 9:
                self._run_grg(
                    symmetry_dataset,
                    transformation_matrix,
                    None,
                    mesh,
                    force_SNF,
                    coordinates,
                )
            if num_values == 3:
                self._D_diag = np.array(mesh, dtype="int64")

        if symmetry_dataset is not None:
            check_grid_symmetry(symmetry_dataset.rotations, self._D_diag, self._Q)

    def _run_grg(
        self,
        symmetry_dataset: SpglibDataset | SpglibMagneticDataset | NosymDataset | None,
        transformation_matrix: ArrayLike | None,
        length: float | None,
        grid_matrix: ArrayLike | None,
        force_SNF: bool,
        coordinates: Literal["reciprocal", "direct"],
    ) -> bool:
        if symmetry_dataset is None or isinstance(symmetry_dataset, NosymDataset):
            if transformation_matrix is None:
                msg = "symmetry_dataset or transformation_matrix has to be specified."
                raise RuntimeError(msg)

            sym_dataset = self._get_mock_symmetry_dataset(
                np.array(transformation_matrix)
            )
        else:
            sym_dataset = symmetry_dataset

        if is_primitive_cell(sym_dataset.rotations):
            # self._D_diag or self._grid_matrix is set in this method.
            self._set_GRG_mesh(
                sym_dataset,
                length=length,
                grid_matrix=grid_matrix,
                force_SNF=force_SNF,
                coordinates=coordinates,
            )
            return True

        warnings.warn(
            "Non primitive cell input. Unable to use GR-grid.",
            RuntimeWarning,
            stacklevel=2,
        )
        return False

    def _get_mock_symmetry_dataset(
        self, transformation_matrix: NDArray
    ) -> MockSymmetryDataset:
        """Return mock symmetry_dataset containing transformation matrix.

        Assuming self._lattice as standardized cell, and inverse of
        transformation_matrix indicates original primitive lattice with respect
        to self._lattice.

        """
        tmat_inv = np.linalg.inv(transformation_matrix)
        tmat_inv_int = np.rint(tmat_inv).astype(int)
        if (tmat_inv - tmat_inv_int > 1e-8).all():
            msg = "Inverse of transformation matrix has to be an integer matrix."
            raise RuntimeError(msg)
        if determinant(tmat_inv_int) < 0:
            msg = "Determinant of transformation matrix has to be positive."
            raise RuntimeError(msg)
        if determinant(tmat_inv_int) < 1:
            msg = (
                "Determinant of inverse of transformation matrix has to "
                "be equal to or larger than 1."
            )
            raise RuntimeError(msg)

        sym_dataset = MockSymmetryDataset(
            rotations=np.eye(3, dtype="intc", order="C").reshape(1, 3, 3),
            transformation_matrix=transformation_matrix,
            std_lattice=np.array(self._lattice),
            std_types=np.array([1], dtype="intc"),
            number=1,
        )
        return sym_dataset

    def _set_GRG_mesh(
        self,
        sym_dataset: SpglibDataset | SpglibMagneticDataset | MockSymmetryDataset,
        length: float | None = None,
        grid_matrix: ArrayLike | None = None,
        force_SNF: bool = False,
        coordinates: Literal["reciprocal", "direct"] = "reciprocal",
    ):
        """Set grid_matrix or D_diag with generalized regular grid.

        Microzone is defined as the regular grid of a conventional
        unit cell. To find the conventional unit cell, symmetry
        information is used.

        """
        if length is not None:
            _grid_matrix = self._get_grid_matrix(
                sym_dataset, length, coordinates=coordinates
            )
        elif grid_matrix is not None:
            _grid_matrix = np.array(grid_matrix, dtype="int64", order="C")

        # If grid_matrix is a diagonal matrix, use it as D matrix.
        gm_diag = np.diagonal(_grid_matrix)
        if (np.diag(gm_diag) == _grid_matrix).all() and not force_SNF:
            self._D_diag = np.array(gm_diag, dtype="int64")
        else:
            import phono3py._recgrid as recgrid  # type: ignore

            if not recgrid.snf3x3(self._D_diag, self._P, self._Q, _grid_matrix):
                msg = "SNF3x3 failed."
                raise RuntimeError(msg)

            self._grid_matrix = _grid_matrix

    def _get_grid_matrix(
        self,
        sym_dataset: SpglibDataset | SpglibMagneticDataset | MockSymmetryDataset,
        length: float,
        coordinates: Literal["reciprocal", "direct"] = "reciprocal",
    ) -> NDArray:
        """Return grid matrix.

        Grid is generated by the distance `length`. `coordinates` is used either
        the grid is defined by supercell in real space or mesh grid in reciprocal
        space.

        Note
        ----
        It is assumed that self._lattice is a primitive cell basis vectors.

        Parameters
        ----------
        length : float
            Distance measure to generate grid.
        coordinates : str, optional
            `reciprocal` (default) or `direct`.

        """
        tmat = sym_dataset.transformation_matrix
        conv_lat = np.dot(np.linalg.inv(tmat).T, self._lattice)

        # GRG is wanted to be generated with respect to std_lattice if possible.
        if _can_use_std_lattice(
            conv_lat,
            tmat,
            sym_dataset.std_lattice,
            sym_dataset.rotations,
        ):
            conv_lat = sym_dataset.std_lattice
            tmat = np.dot(self._lattice, np.linalg.inv(conv_lat)).T

        if coordinates == "direct":
            num_cells = int(np.prod(length2mesh(length, conv_lat)))
            max_num_atoms = num_cells * len(sym_dataset.std_types)
            conv_mesh_numbers = estimate_supercell_matrix(
                sym_dataset, max_num_atoms=max_num_atoms, max_iter=200
            )
        elif coordinates == "reciprocal":
            conv_mesh_numbers = length2mesh(length, conv_lat)
        else:
            raise TypeError('Expect "direct" or "reciprocal" for coordinates.')

        inv_tmat = np.linalg.inv(tmat)
        inv_tmat_int = np.rint(inv_tmat).astype(int)
        assert (np.abs(inv_tmat - inv_tmat_int) < 1e-5).all()
        grid_matrix = np.array(
            (inv_tmat_int * conv_mesh_numbers).T, dtype="int64", order="C"
        )
        return grid_matrix


def check_grid_symmetry(
    direct_rotations: NDArray | Sequence,
    D_diag: NDArray | Sequence,
    Q: NDArray | Sequence,
) -> NDArray:
    """Check whether grid symmetry is satisfied.

    Return rotation matrices for test.

    """
    QDinv = np.array(Q) * (1 / np.array(D_diag, dtype="double"))
    rotations = []
    for r in direct_rotations:
        _r = np.linalg.inv(np.transpose(r) @ QDinv) @ QDinv
        rotations.append(np.rint(_r))
        if not np.allclose(_r, np.rint(_r), atol=1e-5):
            msg = "Grid symmetry is broken."
            raise RuntimeError(msg)
    return np.array(rotations, dtype="int64", order="C")


def check_grid_shift_symmetry(
    is_shift: NDArray | Sequence,
    grg_rotations: NDArray | Sequence,
    P: NDArray | Sequence,
):
    """Check whether given shift satisfies the symmetry."""
    Pinv = np.rint(np.linalg.inv(P)).astype(int)
    assert determinant(Pinv) == 1
    S = np.array(is_shift, dtype=int)
    for r in grg_rotations:
        _S = np.dot(np.dot(Pinv, np.dot(r, P)), S)
        if not np.array_equal((S - _S) % 2, [0, 0, 0]):
            msg = "Grid symmetry is broken by grid shift."
            raise RuntimeError(msg)


def get_qpoints_from_bz_grid_points(gps: int | NDArray, bz_grid: BZGrid) -> NDArray:
    """Return q-point(s) in reduced coordinates of grid point(s).

    Parameters
    ----------
    i_gps : int or ndarray
        BZ-grid index (int) or indices (ndarray).
    bz_grid : BZGrid
        BZ-grid instance.

    """
    return bz_grid.addresses[gps] @ bz_grid.QDinv.T


def get_grid_point_from_address_py(
    addresses: ArrayLike, D_diag: NDArray | Sequence
) -> NDArray:
    """Return GR-grid point index from addresses.

    Python version of get_grid_point_from_address.
    X runs first in XYZ
    In grid.c, Z first is possible with MACRO setting.

    addresses :
        shape=(..., 3)

    """
    return np.dot(np.mod(addresses, D_diag), [1, D_diag[0], D_diag[0] * D_diag[1]])


def get_grid_point_from_address(address: ArrayLike, D_diag: ArrayLike) -> NDArray:
    """Return GR grid-point indices of grid addresses.

    Parameters
    ----------
    address : array_like
        Grid address.
        shape=(3, ) or (n, 3), dtype='int64'
    D_diag : array_like
        This corresponds to mesh numbers. More precisely, this gives
        diagonal elements of diagonal matrix of Smith normal form of
        grid generating matrix. See the detail in the docstring of BZGrid.
        shape=(3,), dtype='int64'

    Returns
    -------
    int
        GR-grid point index.
    or

    ndarray
        GR-grid point indices.
        shape=(n, ), dtype='int64'

    """
    import phono3py._recgrid as recgrid  # type: ignore

    adrs_array = np.array(address, dtype="int64", order="C")
    mesh_array = np.array(D_diag, dtype="int64")

    if adrs_array.ndim == 1:
        return recgrid.grid_index_from_address(adrs_array, mesh_array)

    gps = np.zeros(adrs_array.shape[0], dtype="int64")
    for i, adrs in enumerate(adrs_array):
        gps[i] = recgrid.grid_index_from_address(adrs, mesh_array)
    return gps


def get_ir_grid_points(bz_grid: BZGrid) -> tuple[NDArray, NDArray, NDArray]:
    """Return ir-grid-points in generalized regular grid.

    bz_grid : BZGrid
        Data structure to represent BZ grid.

    Returns
    -------
    ir_grid_points : ndarray
        Irreducible grid point indices in GR-grid.
        shape=(num_ir_grid_points, ), dtype='int64'
    ir_grid_weights : ndarray
        Weights of irreducible grid points. Its sum is the number of
        grid points in GR-grid (prod(D_diag)).
        shape=(num_ir_grid_points, ), dtype='int64'
    ir_grid_map : ndarray
        Index mapping table to irreducible grid points from all grid points
        such as, [0, 0, 2, 3, 3, ...] in GR-grid.
        shape=(prod(D_diag), ), dtype='int64'

    """
    ir_grid_map = _get_ir_grid_map(bz_grid.D_diag, bz_grid.rotations, PS=bz_grid.PS)
    (ir_grid_points, ir_grid_weights) = extract_ir_grid_points(ir_grid_map)

    return ir_grid_points, ir_grid_weights, ir_grid_map


def get_grid_points_by_rotations(
    bz_gp: int,
    bz_grid: BZGrid,
    reciprocal_rotations: NDArray | None = None,
    with_surface: bool = False,
) -> NDArray:
    """Return BZ-grid point indices rotated from a BZ-grid point index.

    Parameters
    ----------
    bz_gp : int
        BZ-grid point index.
    bz_grid : BZGrid
        Data structure to represent BZ grid.
    reciprocal_rotations : array_like or None, optional
        Rotation matrices {R} with respect to basis vectors of GR-grid.
        Defined by g'=Rg, where g is the grid point address represented by
        three integers in BZ-grid.
        dtype='int64', shape=(rotations, 3, 3)
    with_surface : Bool, optional
        This parameter affects to how to treat grid points on BZ surface.
        When False, rotated BZ surface points are moved to representative
        ones among translationally equivalent points to hold one-to-one
        correspondence to GR grid points. With True, BZ grid point indices
        having the rotated grid addresses are returned. Default is False.

    Returns
    -------
    rot_grid_indices : ndarray
        BZ-grid point indices obtained after rotating a grid point index.
        dtype='int64', shape=(rotations,)

    """
    if reciprocal_rotations is not None:
        rec_rots = reciprocal_rotations
    else:
        rec_rots = bz_grid.rotations

    if with_surface:
        return _get_grid_points_by_bz_rotations(bz_gp, bz_grid, rec_rots)
    else:
        return _get_grid_points_by_rotations(bz_gp, bz_grid, rec_rots)


def _get_grid_points_by_rotations(
    bz_gp: int, bz_grid: BZGrid, rotations: ArrayLike
) -> NDArray:
    """Grid point rotations without surface treatment."""
    rot_adrs = np.dot(rotations, bz_grid.addresses[bz_gp])
    grgps = get_grid_point_from_address(rot_adrs, bz_grid.D_diag)
    return bz_grid.grg2bzg[grgps]


def _get_grid_points_by_bz_rotations(
    bz_gp: int, bz_grid: BZGrid, rotations: NDArray, lang: Literal["C", "Py"] = "C"
):
    """Grid point rotations with surface treatment."""
    if lang == "C":
        return _get_grid_points_by_bz_rotations_c(bz_gp, bz_grid, rotations)
    else:
        return _get_grid_points_by_bz_rotations_py(bz_gp, bz_grid, rotations)


def _get_grid_points_by_bz_rotations_c(bz_gp, bz_grid: BZGrid, rotations: NDArray):
    import phono3py._recgrid as recgrid  # type: ignore

    bzgps = np.zeros(len(rotations), dtype="int64")
    for i, r in enumerate(rotations):
        bzgps[i] = recgrid.rotate_bz_grid_index(
            bz_gp,
            r,
            bz_grid.addresses,
            bz_grid.gp_map,
            bz_grid.D_diag,
            bz_grid.PS,
            bz_grid.store_dense_gp_map * 1 + 1,
        )
    return bzgps


def _get_grid_points_by_bz_rotations_py(bz_gp, bz_grid: BZGrid, rotations: ArrayLike):
    """Return BZ-grid point indices generated by rotations.

    Rotated BZ-grid addresses are compared with translationally
    equivalent BZ-grid addresses to get the respective BZ-grid point
    indices.

    """
    rot_adrs = np.dot(rotations, bz_grid.addresses[bz_gp])
    grgps = get_grid_point_from_address(rot_adrs, bz_grid.D_diag)
    bzgps = np.zeros(len(grgps), dtype="int64")
    if bz_grid.store_dense_gp_map:
        for i, (gp, adrs) in enumerate(zip(grgps, rot_adrs, strict=True)):
            indices = np.where(
                (
                    bz_grid.addresses[bz_grid.gp_map[gp] : bz_grid.gp_map[gp + 1]]
                    == adrs
                ).all(axis=1)
            )[0]
            if len(indices) == 0:
                msg = "with_surface did not work properly."
                raise RuntimeError(msg)
            bzgps[i] = bz_grid.gp_map[gp] + indices[0]
    else:
        num_grgp = np.prod(bz_grid.D_diag)
        num_bzgp = num_grgp * 8
        for i, (gp, adrs) in enumerate(zip(grgps, rot_adrs, strict=True)):
            gps = (
                np.arange(
                    bz_grid.gp_map[num_bzgp + gp], bz_grid.gp_map[num_bzgp + gp + 1]
                )
                + num_grgp
            ).tolist()
            gps.insert(0, gp)
            indices = np.where((bz_grid.addresses[gps] == adrs).all(axis=1))[0]
            if len(indices) == 0:
                msg = "with_surface did not work properly."
                raise RuntimeError(msg)
            bzgps[i] = gps[indices[0]]

    return bzgps


def _get_grid_address(D_diag: NDArray | Sequence) -> NDArray:
    """Return generalized regular grid addresses.

    Parameters
    ----------
    D_diag : array_like
        Three integers that represent the generalized regular grid.
        shape=(3, ), dtype='int64'

    Returns
    -------
    gr_grid_addresses : ndarray
        Integer triplets that represents grid point addresses in
        generalized regular grid.
        shape=(prod(D_diag), 3), dtype='int64'

    """
    import phono3py._recgrid as recgrid  # type: ignore

    gr_grid_addresses = np.zeros((np.prod(D_diag), 3), dtype="int64")
    recgrid.gr_grid_addresses(gr_grid_addresses, np.array(D_diag, dtype="int64"))
    return gr_grid_addresses


def _relocate_BZ_grid_address(
    D_diag: NDArray | Sequence,
    Q: ArrayLike,
    reciprocal_lattice: ArrayLike,  # column vectors
    PS: ArrayLike | None = None,
    store_dense_gp_map: bool = False,
) -> tuple[NDArray, NDArray, NDArray]:
    """Grid addresses are relocated to be inside first Brillouin zone.

    Number of ir-grid-points inside Brillouin zone is returned.
    It is assumed that the following arrays have the shapes of
        bz_grid_address : (num_grid_points_in_FBZ, 3)

    Note that the shape of grid_address is (prod(mesh), 3) and the
    addresses in grid_address are arranged to be in parallelepiped
    made of reciprocal basis vectors. The addresses in bz_grid_address
    are inside the first Brillouin zone or on its surface. Each
    address in grid_address is mapped to one of those in
    bz_grid_address by a reciprocal lattice vector (including zero
    vector) with keeping element order. For those inside first BZ, the
    mapping is one-to-one. For those on the first BZ surface, more
    than one addresses in bz_grid_address that are equivalent by the
    reciprocal lattice translations are mapped to one address in
    grid_address. The bz_grid_address and bz_map are given in the
    following format depending on the choice of `store_dense_gp_map`.

    store_dense_gp_map = False
    --------------------------
    Those grid points on the BZ surface except for one of them are
    appended to the tail of this array, for which bz_grid_address has
    the following data storing:

    |------------------array size of bz_grid_address-------------------------|
    |--those equivalent to grid_address--|--those on surface except for one--|
    |-----array size of grid_address-----|

    Number of grid points stored in bz_grid_address is returned.
    bz_map is used to recover grid point index expanded to include BZ
    surface from grid address. The grid point indices are mapped to
    (mesh[0] * 2) x (mesh[1] * 2) x (mesh[2] * 2) space (bz_map).
    bz_map[(prod(mesh) * 8):(prod(mesh) * 9 + 1)] contains equivalent
    information to bz_map[:] with `store_dense_gp_map=True`.

    shape=(prod(mesh * 9) + 1, )

    store_dense_gp_map = True
    -------------------------
    The translationally equivalent grid points corresponding to one grid point
    on BZ surface are stored in continuously. If the multiplicity (number of
    equivalent grid points) is 1, 2, 1, 4, ... for the grid points,
    ``bz_map`` stores the multiplicities and the index positions of the first
    grid point of the equivalent grid points, i.e.,

    bz_map[:] = [0, 1, 3, 4, 8...]
    grid_address[0] -> bz_grid_address[0:1]
    grid_address[1] -> bz_grid_address[1:3]
    grid_address[2] -> bz_grid_address[3:4]
    grid_address[3] -> bz_grid_address[4:8]

    shape=(prod(mesh) + 1, )

    """
    import phono3py._recgrid as recgrid  # type: ignore

    if PS is None:
        _PS = np.zeros(3, dtype="int64")
    else:
        _PS = np.array(PS, dtype="int64")
    bz_grid_addresses = np.zeros((np.prod(D_diag) * 8, 3), dtype="int64", order="C")
    bzg2grg = np.zeros(len(bz_grid_addresses), dtype="int64")

    if store_dense_gp_map:
        bz_map = np.zeros(np.prod(D_diag) + 1, dtype="int64")
    else:
        bz_map = np.zeros(np.prod(D_diag) * 9 + 1, dtype="int64")

    reduced_basis, tmat_inv_int = get_reduced_bases_and_tmat_inv(reciprocal_lattice)
    num_gp = recgrid.bz_grid_addresses(
        bz_grid_addresses,
        bz_map,
        bzg2grg,
        np.array(D_diag, dtype="int64"),
        np.array(tmat_inv_int @ Q, dtype="int64", order="C"),
        _PS,
        reduced_basis,
        store_dense_gp_map * 1 + 1,
    )

    bz_grid_addresses = np.array(bz_grid_addresses[:num_gp], dtype="int64", order="C")
    bzg2grg = np.array(bzg2grg[:num_gp], dtype="int64")
    return bz_grid_addresses, bz_map, bzg2grg


def get_reduced_bases_and_tmat_inv(
    reciprocal_lattice: ArrayLike,
) -> tuple[NDArray, NDArray]:
    """Return reduced bases and inverse transformation matrix.

    Parameters
    ----------
    reciprocal_lattice : ArrayLike
        Reciprocal lattice vectors in column vectors.
        shape=(3, 3), dtype='double'

    Returns
    -------
    reduced_basis : ndarray
        Reduced basis vectors in column vectors.
        shape=(3, 3), dtype='double', order='C'
    tmat_inv_int : ndarray
        Inverse transformation matrix in integer.
        This is used to transform reciprocal lattice vectors to
        conventional lattice vectors.
        shape=(3, 3), dtype='int64'

    """
    # Mpr^-1 = Lr^-1 Lp
    reclat_T = np.array(np.transpose(reciprocal_lattice), dtype="double", order="C")
    reduced_basis = get_reduced_bases(reclat_T)
    assert reduced_basis is not None, "Reduced basis is not found."
    tmat_inv = np.linalg.inv(reduced_basis.T) @ reclat_T.T
    tmat_inv_int = np.rint(tmat_inv).astype("int64")
    assert (np.abs(tmat_inv - tmat_inv_int) < 1e-5).all()
    return np.array(reduced_basis.T, dtype="double", order="C"), tmat_inv_int


def _get_ir_grid_map(
    D_diag: NDArray | Sequence,
    grg_rotations: ArrayLike,
    PS: ArrayLike | None = None,
) -> NDArray:
    """Return mapping to irreducible grid points in GR-grid.

    Parameters
    ----------
    D_diag : array_like
        This corresponds to mesh numbers. More precisely, this gives
        diagonal elements of diagonal matrix of Smith normal form of
        grid generating matrix. See the detail in the docstring of BZGrid.
        shape=(3,), dtype='int64'
    grg_rotations : array_like
        GR-grid rotation matrices.
        dtype='int64', shape=(grg_rotations, 3)
    PS : array_like
        GR-grid shift defined.
        dtype='int64', shape=(3,)

    Returns
    -------
    ir_grid_map : ndarray
        Grid point mapping from all indices to ir-gird-point indices in GR-grid.
        dtype='int64', shape=(prod(mesh),)

    """
    import phono3py._recgrid as recgrid  # type: ignore

    ir_grid_map = np.zeros(np.prod(D_diag), dtype="int64")
    if PS is None:
        _PS = np.zeros(3, dtype="int64")
    else:
        _PS = np.array(PS, dtype="int64")

    num_ir = recgrid.ir_grid_map(
        ir_grid_map,
        np.array(D_diag, dtype="int64"),
        _PS,
        np.array(grg_rotations, dtype="int64", order="C"),
    )

    if num_ir > 0:
        return ir_grid_map
    else:
        raise RuntimeError("_get_ir_grid_map failed to find ir-grid-points.")


def _can_use_std_lattice(
    conv_lat: ArrayLike,
    tmat: ArrayLike,
    std_lattice: ArrayLike,
    rotations: NDArray | Sequence,
    symprec: float = 1e-5,
):
    """Inspect if std_lattice can be used as conv_lat.

    r_s is the rotation matrix of conv_lat.
    Return if conv_lat rotated by det(r_s)*r_s and std_lattice are equivalent.
    det(r_s) is necessary to make improper rotation to proper rotation.

    """
    for r in rotations:
        r_s = similarity_transformation(tmat, r)
        if np.allclose(
            np.linalg.det(r_s) * np.dot(np.transpose(conv_lat), r_s),
            np.transpose(std_lattice),
            atol=symprec,
        ):
            return True
    return False
