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

import numpy as np
from phonopy.harmonic.force_constants import similarity_transformation
from phonopy.structure.cells import (
    get_primitive_matrix_by_centring, estimate_supercell_matrix,
    get_reduced_bases)
from phonopy.structure.grid_points import length2mesh, extract_ir_grid_points


class BZGrid(object):
    """Data structure of BZ grid.

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
    of grid adresses are performed as follows:

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
    q-points with respect to the original recirpocal
    basis vectors are given by

    q = np.dot(Q, addresses[gp] / D_diag.astype('double'))

    for the Gamma cetnred grid. With shifted, where only half grid shifts
    that satisfy the symmetry are considered,

    q = np.dot(Q, (addresses[gp] + np.dot(P, s)) / D_diag.astype('double'))

    where s is the shift vectors that are 0 or 1/2. But it is more
    convenient to use the integer shift vectors S by 0 or 1, which gives

    q = (np.dot(Q, (2 * addresses[gp] + np.dot(P, S))
                    / D_diag.astype('double'))) / 2

    and this is the definition of PS in this class.

    Attributes
    ----------
    addresses : ndarray
        Integer grid address of the points in Brillouin zone including
        surface. There are two types of address order by either
        `store_dense_gp_map` is True or False.
        shape=(np.prod(D_diag) + some on surface, 3), dtype='int_', order='C'
    gp_map : ndarray
        Grid point mapping table containing BZ surface. There are two types of
        address order by either `store_dense_gp_map` is True or False. See more
        detail in `_relocate_BZ_grid_address` docstring.
    bzg2grg : ndarray
        Grid index mapping table from BZGrid to GRgrid.
        shape=(len(addresses), ), dtype='int_'
    grg2bzg : ndarray
        Grid index mapping table from GRgrid to BZGrid. Unique one
        of translationally equivalent grid points in BZGrid is chosen.
        shape=(prod(D_diag), ), dtype='int_'
    store_dense_gp_map : bool, optional
        See the detail in the docstring of `_relocate_BZ_grid_address`.
    rotations : ndarray
        Rotation matrices for GR-grid addresses (g) defined as g'=Rg.
        shape=(rotations, 3, 3), dtype='int_', order='C'.
    reciprocal_operations : ndarray
        Reciprocal space rotation matrices in fractional coordinates defined as
        q'=Rq.
        shape=(rotations, 3, 3), dtype='int_', order='C'.
    D_diag : ndarray
        This corresponds to the mesh numbers in transformed reciprocal
        basis vectors.
        shape=(3,), dtype='int_'
    P : ndarray
        Left unimodular matrix after SNF: D=PAQ.
        shape=(3, 3), dtype='int_', order='C'
    Q : ndarray
        Right unimodular matrix after SNF: D=PAQ.
        shape=(3, 3), dtype='int_', order='C'
    grid_matrix : ndarray
        Grid generating matrix used for SNF.
        shape=(3, 3), dtype='int_', order='C'
    microzone_lattice : ndarray
        Basis vectors of microzone of GR-grid.
        shape=(3, 3), dtype='double', order='C'

    """

    def __init__(self,
                 mesh,
                 reciprocal_lattice=None,
                 lattice=None,
                 symmetry_dataset=None,
                 is_shift=None,
                 is_time_reversal=True,
                 use_grg=False,
                 store_dense_gp_map=True):
        """Init method.

        mesh : array_like or float
            Mesh numbers or length.
            shape=(3,), dtype='int_'
        reciprocal_lattice : array_like
            Reciprocal primitive basis vectors given as column vectors
            shape=(3, 3), dtype='double', order='C'
        lattice : array_like
            Direct primitive basis vectors given as row vectors
            shape=(3, 3), dtype='double', order='C'
        symmetry_dataset : dict
            Symmetry dataset (Symmetry.dataset) searched for the primitive cell
            corresponding to ``reciprocal_lattice`` or ``lattice``.
        is_shift : array_like or None, optional
            [0, 0, 0] gives Gamma center mesh and value 1 gives half mesh shift
            along the basis vectors. Default is None.
            dtype='int_', shape=(3,)
        is_time_reveral : bool, optional
            Inversion symmetry is included in reciprocal point group.
            Default is True.
        use_grg : bool, optional
            Use generalized regular grid. Default is False.
        store_dense_gp_map : bool, optional
            See the detail in the docstring of `_relocate_BZ_grid_address`.
            Default is True.

        """
        if symmetry_dataset is None:
            identity = np.eye(3, dtype='int_', order='C')
            self._symmetry_dataset = {
                'rotations': identity.reshape(1, 3, 3),
                'translations': np.array([[0, 0, 0]], dtype='double')}
        else:
            self._symmetry_dataset = symmetry_dataset
        self._is_shift = is_shift
        self._is_time_reversal = is_time_reversal
        self._use_grg = use_grg
        self._store_dense_gp_map = store_dense_gp_map
        self._addresses = None
        self._gp_map = None
        self._grid_matrix = None
        self._D_diag = np.ones(3, dtype='int_')
        self._Q = np.eye(3, dtype='int_', order='C')
        self._P = np.eye(3, dtype='int_', order='C')
        self._QDinv = None
        self._microzone_lattice = None
        self._rotations = None
        self._reciprocal_operations = None

        if reciprocal_lattice is not None:
            self._reciprocal_lattice = np.array(
                reciprocal_lattice, dtype='double', order='C')
            self._lattice = np.array(
                np.linalg.inv(reciprocal_lattice), dtype='double', order='C')
        if lattice is not None:
            self._lattice = np.array(
                lattice, dtype='double', order='C')
            self._reciprocal_lattice = np.array(
                np.linalg.inv(lattice), dtype='double', order='C')

        self._generate_grid(mesh)

    @property
    def D_diag(self):
        """Diagonal elements of diagonal matrix after SNF: D=PAQ."""
        return self._D_diag

    @property
    def P(self):
        """Left unimodular matrix after SNF: D=PAQ."""
        return self._P

    @property
    def Q(self):
        """Right unimodular matrix after SNF: D=PAQ."""
        return self._Q

    @property
    def QDinv(self):
        """QD^-1."""
        return self._QDinv

    @property
    def PS(self):
        """Integer shift vectors of GRGrid."""
        if self._is_shift is None:
            return np.zeros(3, dtype='int_')
        else:
            return np.array(np.dot(self.P, self._is_shift), dtype='int_')

    @property
    def grid_matrix(self):
        """Grid generating matrix to be represented by SNF."""
        return self._grid_matrix

    @property
    def addresses(self):
        """BZ-grid addresses."""
        return self._addresses

    @property
    def gp_map(self):
        """Definitions of grid index."""
        return self._gp_map

    @property
    def bzg2grg(self):
        """Transform grid point indices from BZG to GRG.

        Equivalent to
            get_grid_point_from_address(
                self._addresses[bz_grid_index], self._D_diag)

        """
        return self._bzg2grg

    @property
    def grg2bzg(self):
        """Transform grid point indices from GRG to BZG."""
        return self._grg2bzg

    @property
    def microzone_lattice(self):
        """Basis vectors of microzone."""
        return self._microzone_lattice

    @property
    def store_dense_gp_map(self):
        """Return gp_map type."""
        return self._store_dense_gp_map

    @property
    def rotations(self):
        """Return rotations."""
        return self._rotations

    @property
    def rotations_cartesian(self):
        """Return rotations in Cartesian coordinates."""
        return self._rotations_cartesian

    @property
    def reciprocal_operations(self):
        """Return reciprocal rotations."""
        return self._reciprocal_operations

    @property
    def symmetry_dataset(self):
        """Return Symmetry.dataset."""
        return self._symmetry_dataset

    def get_indices_from_addresses(self, addresses):
        """Return BZ grid point indices from grid addresses.

        Parameters
        ----------
        addresses : array_like
            Integer grid addresses.
            shape=(n, 3) or (3, ), where n is the number of grid points.

        Returns
        -------
        ndarray or int
            Grid point indices corresponding to the grid addresses. Each
            returned grid point index is one of those of the
            translationally equivalent grid points.
            shape=(n, ), dtype='int_' when multiple addresses are given.
            Otherwise one integer value is returned.

        """
        try:
            len(addresses[0])
        except TypeError:
            return int(self._grg2bzg[get_grid_point_from_address(
                addresses, self._D_diag)])

        gps = [get_grid_point_from_address(adrs, self._D_diag)
               for adrs in addresses]
        return np.array(self._grg2bzg[gps], dtype='int_')

    def _set_bz_grid(self):
        """Generate BZ grid addresses and grid point mapping table."""
        (self._addresses,
         self._gp_map,
         self._bzg2grg) = _relocate_BZ_grid_address(
             self._D_diag,
             self._Q,
             self._reciprocal_lattice,  # column vectors
             is_shift=self._is_shift,
             store_dense_gp_map=self._store_dense_gp_map)
        if self._store_dense_gp_map:
            self._grg2bzg = self._gp_map[:-1]
        else:
            self._grg2bzg = np.arange(
                np.prod(self._D_diag), dtype='int_')

        self._QDinv = np.array(self.Q * (1 / self.D_diag.astype('double')),
                               dtype='double', order='C')
        self._microzone_lattice = np.dot(
            self._reciprocal_lattice, np.dot(self._QDinv, self._P))

    def _generate_grid(self, mesh, force_SNF=False):
        self._set_mesh_numbers(mesh, force_SNF=force_SNF)
        self._set_bz_grid()
        self._set_rotations()

    def _set_mesh_numbers(self, mesh, force_SNF=False):
        """Set mesh numbers from array or float value.

        Three case:
        1) Three integers are given.
            Use these numbers as regular grid.
        2) One number is given with no symmetry provided.
            Regular grid is computed from this value.
        3) One number is given and use_grg=True.
            Regular grid is computed from this value and point group symmetry.
        4) One number is given with symmetry provided and use_grg=True.
            Generalized regular grid is generated. However if the grid
            generating matrix is a diagonal matrix, use it as the D matrix
            of SNF and P and Q are set as identity matrices. Otherwise
            D, P, Q matrices are computed using SNF.

        """
        try:
            num_values = len(mesh)
            if num_values == 3:
                self._D_diag = np.array(mesh, dtype='int_')
        except TypeError:
            length = float(mesh)
            if self._use_grg:
                self._set_SNF(length, force_SNF=force_SNF)
            else:
                self._D_diag = length2mesh(length, self._lattice)

    def _set_SNF(self, length, force_SNF=False):
        """Calculate Smith normal form.

        Microzone is defined as the regular grid of a conventional
        unit cell. To find the conventional unit cell, symmetry
        information is used.

        """
        sym_dataset = self._symmetry_dataset
        tmat = sym_dataset['transformation_matrix']
        centring = sym_dataset['international'][0]
        pmat = get_primitive_matrix_by_centring(centring)
        conv_lat = np.dot(np.linalg.inv(tmat).T, self._lattice)
        num_cells = int(np.prod(length2mesh(length, conv_lat)))
        max_num_atoms = num_cells * len(sym_dataset['std_types'])
        conv_mesh_numbers = estimate_supercell_matrix(
            sym_dataset,
            max_num_atoms=max_num_atoms,
            max_iter=200)
        inv_pmat = np.linalg.inv(pmat)
        inv_pmat_int = np.rint(inv_pmat).astype(int)
        assert (np.abs(inv_pmat - inv_pmat_int) < 1e-5).all()
        # transpose in reciprocal space
        self._grid_matrix = np.array(
            (inv_pmat_int * conv_mesh_numbers).T, dtype='int_', order='C')
        # If grid_matrix is a diagonal matrix, use it as D matrix.
        gm_diag = np.diagonal(self._grid_matrix)
        if (np.diag(gm_diag) == self._grid_matrix).all() and not force_SNF:
            self._D_diag = np.array(gm_diag, dtype='int_')
        else:
            import phono3py._phono3py as phono3c
            if not phono3c.snf3x3(self._D_diag,
                                  self._P,
                                  self._Q,
                                  self._grid_matrix):
                msg = "SNF3x3 failed."
                raise RuntimeError(msg)

    def _set_rotations(self):
        """Rotation matrices are transformed those for non-diagonal D matrix.

        Terminate when symmetry of grid is broken.

        """
        import phono3py._phono3py as phono3c

        direct_rotations = np.array(self._symmetry_dataset['rotations'],
                                    dtype='int_', order='C')
        rec_rotations = np.zeros((48, 3, 3), dtype='int_', order='C')
        num_rec_rot = phono3c.reciprocal_rotations(rec_rotations,
                                                   direct_rotations,
                                                   self._is_time_reversal)
        self._reciprocal_operations = np.array(rec_rotations[:num_rec_rot],
                                               dtype='int_', order='C')
        self._rotations_cartesian = np.array(
            [similarity_transformation(self._reciprocal_lattice, r)
             for r in self._reciprocal_operations], dtype='double', order='C')
        self._rotations = np.zeros(self._reciprocal_operations.shape,
                                   dtype='int_', order='C')
        if not phono3c.transform_rotations(self._rotations,
                                           self._reciprocal_operations,
                                           self._D_diag,
                                           self._Q):
            msg = "Grid symmetry is broken. Use generalized regular grid."
            raise RuntimeError(msg)


def get_grid_point_from_address_py(address, mesh):
    """Python version of get_grid_point_from_address."""
    # X runs first in XYZ
    # (*In spglib, Z first is possible with MACRO setting.)
    m = mesh
    return (address[0] % m[0] +
            (address[1] % m[1]) * m[0] +
            (address[2] % m[2]) * m[0] * m[1])


def get_grid_point_from_address(address, D_diag):
    """Return GR grid-point indices of grid addresses.

    Parameters
    ----------
    address : array_like
        Grid address.
        shape=(3, ) or (n, 3), dtype='int_'
    D_diag : array_like
        This corresponds to mesh numbers. More precisely, this gives
        diagonal elements of diagonal matrix of Smith normal form of
        grid generating matrix. See the detail in the docstring of BZGrid.
        shape=(3,), dtype='int_'

    Returns
    -------
    int
        GR-grid point index.
    or

    ndarray
        GR-grid point indices.
        shape=(n, ), dtype='int_'

    """
    import phono3py._phono3py as phono3c

    adrs_array = np.array(address, dtype='int_', order='C')
    mesh_array = np.array(D_diag, dtype='int_')

    if adrs_array.ndim == 1:
        return phono3c.grid_index_from_address(adrs_array, mesh_array)

    gps = np.zeros(adrs_array.shape[0], dtype='int_')
    for i, adrs in enumerate(adrs_array):
        gps[i] = phono3c.grid_index_from_address(adrs, mesh_array)
    return gps


def get_ir_grid_points(bz_grid):
    """Return ir-grid-points in generalized regular grid.

    bz_grid : BZGrid
        Data structure to represent BZ grid.

    Returns
    -------
    ir_grid_points : ndarray
        Irreducible grid point indices in GR-grid.
        shape=(num_ir_grid_points, ), dtype='int_'
    ir_grid_weights : ndarray
        Weights of irreducible grid points. Its sum is the number of
        grid points in GR-grid (prod(D_diag)).
        shape=(num_ir_grid_points, ), dtype='int_'
    ir_grid_map : ndarray
        Index mapping table to irreducible grid points from all grid points
        such as, [0, 0, 2, 3, 3, ...].
        shape=(prod(D_diag), ), dtype='int_'

    """
    ir_grid_map = _get_ir_grid_map(
        bz_grid.D_diag,
        bz_grid.rotations,
        PS=bz_grid.PS)
    (ir_grid_points,
     ir_grid_weights) = extract_ir_grid_points(ir_grid_map)

    return ir_grid_points, ir_grid_weights, ir_grid_map


def get_grid_points_by_rotations(bz_gp,
                                 bz_grid,
                                 reciprocal_rotations=None,
                                 with_surface=False):
    """Return BZ-grid point indices rotated from a BZ-grid point index.

    Parameters
    ----------
    bz_gp : int
        BZ-grid point index.
    bz_grid : BZGrid
        Data structure to represent BZ grid.
    reciprocal_rotations : array_like or None, optional
        Rotation matrices {R} with respect to reciprocal basis vectors.
        Defined by q'=Rq.
        dtype='int_', shape=(rotations, 3, 3)
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
        dtype='int_', shape=(rotations,)

    """
    if reciprocal_rotations is not None:
        rec_rots = reciprocal_rotations
    else:
        rec_rots = bz_grid.rotations

    if with_surface:
        return _get_grid_points_by_bz_rotations(bz_gp, bz_grid, rec_rots)
    else:
        return _get_grid_points_by_rotations(bz_gp, bz_grid, rec_rots)


def _get_grid_points_by_rotations(bz_gp, bz_grid, rotations):
    """Grid point rotations without surface treatment."""
    rot_adrs = np.dot(rotations, bz_grid.addresses[bz_gp])
    grgps = get_grid_point_from_address(rot_adrs, bz_grid.D_diag)
    return bz_grid.grg2bzg[grgps]


def _get_grid_points_by_bz_rotations(bz_gp, bz_grid, rotations, lang='C'):
    """Grid point rotations with surface treatment."""
    if lang == 'C':
        return _get_grid_points_by_bz_rotations_c(bz_gp, bz_grid, rotations)
    else:
        return _get_grid_points_by_bz_rotations_py(bz_gp, bz_grid, rotations)


def _get_grid_points_by_bz_rotations_c(bz_gp, bz_grid, rotations):
    import phono3py._phono3py as phono3c

    bzgps = np.zeros(len(rotations), dtype='int_')
    for i, r in enumerate(rotations):
        bzgps[i] = phono3c.rotate_bz_grid_index(
            bz_gp,
            r,
            bz_grid.addresses,
            bz_grid.gp_map,
            bz_grid.D_diag,
            bz_grid.PS,
            bz_grid.store_dense_gp_map * 1 + 1)
    return bzgps


def _get_grid_points_by_bz_rotations_py(bz_gp, bz_grid, rotations):
    """Return BZ-grid point indices generated by rotations.

    Rotated BZ-grid addresses are compared with translationally
    equivalent BZ-grid addresses to get the respective BZ-grid point
    indices.

    """
    rot_adrs = np.dot(rotations, bz_grid.addresses[bz_gp])
    grgps = get_grid_point_from_address(rot_adrs, bz_grid.D_diag)
    bzgps = np.zeros(len(grgps), dtype='int_')
    if bz_grid.store_dense_gp_map:
        for i, (gp, adrs) in enumerate(zip(grgps, rot_adrs)):
            indices = np.where(
                (bz_grid.addresses[bz_grid.gp_map[gp]:bz_grid.gp_map[gp + 1]]
                 == adrs).all(axis=1))[0]
            if len(indices) == 0:
                msg = "with_surface did not work properly."
                raise RuntimeError(msg)
            bzgps[i] = bz_grid.gp_map[gp] + indices[0]
    else:
        num_grgp = np.prod(bz_grid.D_diag)
        num_bzgp = num_grgp * 8
        for i, (gp, adrs) in enumerate(zip(grgps, rot_adrs)):
            gps = (np.arange(
                bz_grid.gp_map[num_bzgp + gp],
                bz_grid.gp_map[num_bzgp + gp + 1]) + num_grgp).tolist()
            gps.insert(0, gp)
            indices = np.where((bz_grid.addresses[gps] == adrs).all(axis=1))[0]
            if len(indices) == 0:
                msg = "with_surface did not work properly."
                raise RuntimeError(msg)
            bzgps[i] = gps[indices[0]]

    return bzgps


def _get_grid_address(D_diag):
    """Return generalized regular grid addresses.

    Parameters
    ----------
    D_diag : array_like
        Three integers that represent the generalized regular grid.
        shape=(3, ), dtype='int_'

    Returns
    -------
    gr_grid_addresses : ndarray
        Integer triplets that represents grid point addresses in
        generalized regular grid.
        shape=(prod(D_diag), 3), dtype='int_'

    """
    import phono3py._phono3py as phono3c

    gr_grid_addresses = np.zeros((np.prod(D_diag), 3), dtype='int_')
    phono3c.gr_grid_addresses(gr_grid_addresses,
                              np.array(D_diag, dtype='int_'))
    return gr_grid_addresses


def _relocate_BZ_grid_address(D_diag,
                              Q,
                              reciprocal_lattice,  # column vectors
                              is_shift=None,
                              store_dense_gp_map=False):
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
    ``bz_map`` stores the multiplicites and the index positions of the first
    grid point of the equivalent grid points, i.e.,

    bz_map[:] = [0, 1, 3, 4, 8...]
    grid_address[0] -> bz_grid_address[0:1]
    grid_address[1] -> bz_grid_address[1:3]
    grid_address[2] -> bz_grid_address[3:4]
    grid_address[3] -> bz_grid_address[4:8]

    shape=(prod(mesh) + 1, )

    """
    import phono3py._phono3py as phono3c

    if is_shift is None:
        _is_shift = np.zeros(3, dtype='int_')
    else:
        _is_shift = np.array(is_shift, dtype='int_')
    bz_grid_addresses = np.zeros((np.prod(np.add(D_diag, 1)), 3),
                                 dtype='int_', order='C')
    bzg2grg = np.zeros(len(bz_grid_addresses), dtype='int_')

    if store_dense_gp_map:
        bz_map = np.zeros(np.prod(D_diag) + 1, dtype='int_')
    else:
        bz_map = np.zeros(np.prod(D_diag) * 9 + 1, dtype='int_')

    reclat_T = np.array(reciprocal_lattice.T, dtype='double', order='C')
    reduced_basis = get_reduced_bases(reclat_T)
    tmat_inv = np.dot(np.linalg.inv(reduced_basis.T), reclat_T.T)
    tmat_inv_int = np.rint(tmat_inv).astype('int_')
    assert (np.abs(tmat_inv - tmat_inv_int) < 1e-5).all()

    num_gp = phono3c.bz_grid_addresses(
        bz_grid_addresses,
        bz_map,
        bzg2grg,
        np.array(D_diag, dtype='int_'),
        np.array(np.dot(tmat_inv_int, Q), dtype='int_', order='C'),
        _is_shift,
        np.array(reduced_basis.T, dtype='double', order='C'),
        store_dense_gp_map * 1 + 1)

    bz_grid_addresses = np.array(bz_grid_addresses[:num_gp],
                                 dtype='int_', order='C')
    bzg2grg = np.array(bzg2grg[:num_gp], dtype='int_')
    return bz_grid_addresses, bz_map, bzg2grg


def _get_ir_grid_map(D_diag,
                     grg_rotations,
                     PS=None):
    """Irreducible k-points are searched.

    Parameters
    ----------
    D_diag : array_like
        This corresponds to mesh numbers. More precisely, this gives
        diagonal elements of diagonal matrix of Smith normal form of
        grid generating matrix. See the detail in the docstring of BZGrid.
        shape=(3,), dtype='int_'
    grg_rotations : array_like
        GR-grid rotation matrices.
        dtype='int_', shape=(grg_rotations, 3)
    PS : array_like
        GR-grid shift defined.
        dtype='int_', shape=(3,)

    Returns
    -------
    ir_grid_map : ndarray
        Grid point mapping table to ir-gird-points in gereralized
        regular grid.
        dtype='int_', shape=(prod(mesh),)

    """
    import phono3py._phono3py as phono3c

    ir_grid_map = np.zeros(np.prod(D_diag), dtype='int_')
    if PS is None:
        _PS = np.zeros(3, dtype='int_')
    else:
        _PS = np.array(PS, dtype='int_')

    num_ir = phono3c.ir_grid_map(
        ir_grid_map,
        np.array(D_diag, dtype='int_'),
        _PS,
        np.array(grg_rotations, dtype='int_', order='C'))

    if num_ir > 0:
        return ir_grid_map
    else:
        raise RuntimeError("_get_ir_grid_map failed to find ir-grid-points.")
