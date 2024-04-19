"""Calculated accumulated property with respect to other property."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional, Union

import numpy as np
from phonopy.phonon.dos import NormalDistribution

from phono3py.other.tetrahedron_method import get_integration_weights
from phono3py.phonon.grid import BZGrid

epsilon = 1.0e-8


class KappaDOSTHM:
    """Class to calculate DOS like spectram with tetrahedron method.

    To compute usual DOS,

    kappados = KappaDOSTHM(
        np.ones(freqs.shape)[None, :, :, None],
        freqs,
        bzgrid,
        ir_grid_points,
        ir_grid_weights=ir_weights,
        ir_grid_map=ir_grid_map,
        num_sampling_points=201
    )

    """

    def __init__(
        self,
        mode_kappa: np.ndarray,
        frequencies: np.ndarray,
        bz_grid: BZGrid,
        ir_grid_points: np.ndarray,
        ir_grid_weights: Optional[np.ndarray] = None,
        ir_grid_map: Optional[np.ndarray] = None,
        frequency_points: Optional[Union[np.ndarray, Sequence]] = None,
        num_sampling_points: int = 100,
    ):
        """Init method.

        Parameters
        ----------
        mode_kappa : ndarray
            Target value.
            shape=(temperatures, ir_grid_points, num_band, num_elem),
            dtype='double'
        frequencies : ndarray
            Frequencies at ir-grid points.
            shape=(ir_grid_points, num_band), dtype='double'
        bz_grid : BZGrid
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
        frequency_points : array_like, optional, default=None
            This is used as the frequency points. When None,
            frequency points are created from `num_sampling_points`.
        num_sampling_points : int, optional, default=100
            Number of uniform sampling points.

        """
        min_freq = min(frequencies.ravel())
        max_freq = max(frequencies.ravel()) + epsilon
        if frequency_points is None:
            self._frequency_points = np.linspace(
                min_freq, max_freq, num_sampling_points, dtype="double"
            )
        else:
            self._frequency_points = np.array(frequency_points, dtype="double")

        n_temp, _, _, n_elem = mode_kappa.shape
        self._kdos = np.zeros(
            (n_temp, len(self._frequency_points), 2, n_elem), dtype="double"
        )

        if ir_grid_map is None:
            bzgp2irgp_map = None
        else:
            bzgp2irgp_map = self._get_bzgp2irgp_map(bz_grid, ir_grid_map)
        if ir_grid_weights is None:
            grid_weights = np.ones(mode_kappa.shape[1])
        else:
            grid_weights = ir_grid_weights
        for j, function in enumerate(("J", "I")):
            iweights = get_integration_weights(
                self._frequency_points,
                frequencies,
                bz_grid,
                grid_points=bz_grid.grg2bzg[ir_grid_points],
                bzgp2irgp_map=bzgp2irgp_map,
                function=function,
            )
            for i, iw in enumerate(iweights):
                self._kdos[:, :, j] += np.transpose(
                    np.dot(iw, mode_kappa[:, i] * grid_weights[i]), axes=(1, 0, 2)
                )
        self._kdos /= np.prod(bz_grid.D_diag)

    def get_kdos(self):
        """Return thermal conductivity spectram.

        Returns
        -------
        tuple
            frequency_points : ndarray
                shape=(sampling_points, ), dtype='double'
            kdos : ndarray
                shape=(temperatures, sampling_points, 2 (J, I), num_elem),
                dtype='double', order='C'

        """
        return self._frequency_points, self._kdos

    def _get_bzgp2irgp_map(self, bz_grid, ir_grid_map):
        unique_gps = np.unique(ir_grid_map)
        gp_map = {j: i for i, j in enumerate(unique_gps)}
        bzgp2irgp_map = np.array(
            [gp_map[ir_grid_map[grgp]] for grgp in bz_grid.bzg2grg], dtype="int_"
        )
        return bzgp2irgp_map


class GammaDOSsmearing:
    """Class to calculate Gamma spectram by smearing method."""

    def __init__(
        self,
        gamma,
        frequencies,
        ir_grid_weights,
        sigma: Optional[float] = None,
        num_sampling_points: int = 200,
    ):
        """Init method.

        gamma : ndarray
            Target value.
            shape=(temperatures, ir_grid_points, num_band)
            dtype='double'
        frequencies : ndarray
            shape=(ir_grid_points, num_band), dtype='double'
        ir_grid_weights : ndarray
            Grid point weights at ir-grid points.
            shape=(ir_grid_points, ), dtype='int_'
        sigma : float
            Smearing width.
        num_sampling_points : int, optional, default=100
            Number of uniform sampling points.

        """
        self._gamma = gamma
        self._frequencies = frequencies
        self._ir_grid_weights = ir_grid_weights
        self._num_sampling_points = num_sampling_points
        self._set_frequency_points()
        self._gdos = np.zeros(
            (len(gamma), len(self._frequency_points), 2), dtype="double"
        )
        if sigma is None:
            self._sigma = (
                max(self._frequency_points) - min(self._frequency_points)
            ) / 100
        else:
            self._sigma = 0.1
        self._smearing_function = NormalDistribution(self._sigma)
        self._run_smearing_method()

    def get_gdos(self):
        """Return Gamma spectram.

        gdos[:, :, 0] is not used but eixts to be similar shape to kdos.

        """
        return self._frequency_points, self._gdos

    def _set_frequency_points(self):
        min_freq = np.min(self._frequencies)
        max_freq = np.max(self._frequencies) + epsilon
        self._frequency_points = np.linspace(
            min_freq, max_freq, self._num_sampling_points
        )

    def _run_smearing_method(self):
        self._dos = []
        num_gp = np.sum(self._ir_grid_weights)
        for i, f in enumerate(self._frequency_points):
            dos = self._smearing_function.calc(self._frequencies - f)
            for j, g_t in enumerate(self._gamma):
                self._gdos[j, i, 1] = (
                    np.sum(np.dot(self._ir_grid_weights, dos * g_t)) / num_gp
                )


def run_prop_dos(
    frequencies,
    mode_prop,
    ir_grid_map,
    ir_grid_points,
    num_sampling_points: int,
    bz_grid: BZGrid,
):
    """Run DOS-like calculation.

    This is a simple wrapper of KappsDOSTHM.

    Parameters
    ----------
    frequencies:
        Frequencies at ir-grid points.
    mode_prop:
        Properties at  ir-grid points.
    ir_grid_map:
        Obtained by get_ir_grid_points(bz_grid)[2].
    ir_grid_points:
        Obtained by get_ir_grid_points(bz_grid)[0].
    num_sampling_points:
        Number of sampling points in horizontal axis.
    bz_grid:
        BZ grid.

    """
    kappa_dos = KappaDOSTHM(
        mode_prop,
        frequencies,
        bz_grid,
        bz_grid.bzg2grg[ir_grid_points],
        ir_grid_map=ir_grid_map,
        num_sampling_points=num_sampling_points,
    )
    freq_points, kdos = kappa_dos.get_kdos()
    sampling_points = np.tile(freq_points, (len(kdos), 1))
    return kdos, sampling_points


def run_mfp_dos(
    mean_freepath,
    mode_prop,
    ir_grid_map,
    ir_grid_points,
    num_sampling_points: int,
    bz_grid: BZGrid,
):
    """Run DOS-like calculation for mean free path.

    mean_freepath : shape=(temperatures, ir_grid_points, 6)
    mode_prop : shape=(temperatures, ir_grid_points, 6, 6)

    """
    kdos = []
    sampling_points = []
    for i, _ in enumerate(mean_freepath):
        kappa_dos = KappaDOSTHM(
            mode_prop[i : i + 1, :, :],
            mean_freepath[i],
            bz_grid,
            bz_grid.bzg2grg[ir_grid_points],
            ir_grid_map=ir_grid_map,
            num_sampling_points=num_sampling_points,
        )
        sampling_points_at_T, kdos_at_T = kappa_dos.get_kdos()
        kdos.append(kdos_at_T[0])
        sampling_points.append(sampling_points_at_T)
    kdos = np.array(kdos)
    sampling_points = np.array(sampling_points)

    return kdos, sampling_points


def get_mfp(g, gv):
    """Calculate mean free path from inverse lifetime and group velocity."""
    g = np.where(g > 0, g, -1)
    gv_norm = np.sqrt((gv**2).sum(axis=2))
    mean_freepath = np.where(g > 0, gv_norm / (2 * 2 * np.pi * g), 0)
    return mean_freepath
