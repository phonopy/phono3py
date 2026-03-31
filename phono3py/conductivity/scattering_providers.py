"""Scattering provider building blocks for conductivity calculations."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from phonopy.physical_units import get_physical_units

from phono3py.conductivity.grid_point_data import GridPointInput, GridPointResult
from phono3py.other.isotope import Isotope
from phono3py.phonon3.imag_self_energy import ImagSelfEnergy, average_by_degeneracy
from phono3py.phonon3.interaction import Interaction


class RTAScatteringProvider:
    """Compute ph-ph linewidth (gamma) at a grid point using the RTA.

    This provider implements the ``ScatteringProvider`` protocol.  It wraps
    ``ImagSelfEnergy`` and handles the sigma loop, the temperature loop, and
    the various computation modes (read_pp, use_ave_pp, low-memory path).

    The returned ``GridPointResult.gamma`` has shape
    ``(num_sigma, num_temp, num_band0)`` and contains ph-ph contributions
    only.  Isotope and boundary contributions are handled by separate
    providers.

    Parameters
    ----------
    pp : Interaction
        Interaction instance.
    sigmas : sequence of float or None
        Smearing widths.  None selects the tetrahedron method.
    temperatures : ndarray of double, shape (num_temp,)
        Temperatures in Kelvin.
    sigma_cutoff : float or None, optional
        Cutoff width in units of sigma. Default None.
    is_full_pp : bool, optional
        Compute full ph-ph interaction matrix. Default False.
    use_ave_pp : bool, optional
        Use pre-stored averaged ph-ph interaction. Default False.
    read_pp : bool, optional
        Read ph-ph interaction from file. Default False.
    store_pp : bool, optional
        Store ph-ph interaction to file. Default False.
    pp_filename : str or None, optional
        Filename for reading/writing ph-ph interaction. Default None.
    is_N_U : bool, optional
        Decompose gamma into Normal and Umklapp parts. Default False.
    is_gamma_detail : bool, optional
        Store detailed per-triplet gamma. Default False.
    log_level : int, optional
        Verbosity level. Default 0.
    """

    def __init__(
        self,
        pp: Interaction,
        sigmas: Sequence[float | None],
        temperatures: NDArray[np.double],
        sigma_cutoff: float | None = None,
        is_full_pp: bool = False,
        use_ave_pp: bool = False,
        read_pp: bool = False,
        store_pp: bool = False,
        pp_filename: str | None = None,
        is_N_U: bool = False,
        is_gamma_detail: bool = False,
        log_level: int = 0,
    ):
        """Init method."""
        self._pp = pp
        self._sigmas = list(sigmas)
        self._temperatures = temperatures
        self._sigma_cutoff = sigma_cutoff
        self._is_full_pp = is_full_pp
        self._use_ave_pp = use_ave_pp
        self._use_const_ave_pp = pp.constant_averaged_interaction is not None
        self._read_pp = read_pp
        self._store_pp = store_pp
        self._pp_filename = pp_filename
        self._is_N_U = is_N_U
        self._is_gamma_detail = is_gamma_detail
        self._log_level = log_level

        self._collision = ImagSelfEnergy(pp, with_detail=(is_gamma_detail or is_N_U))

        # Per-grid-point state set during compute_gamma, accessible for output.
        self._gamma_N: NDArray[np.double] | None = None
        self._gamma_U: NDArray[np.double] | None = None
        self._gamma_detail_at_q: NDArray[np.double] | None = None

    @property
    def is_full_pp(self) -> bool:
        """Return True if averaged ph-ph interaction will be computed."""
        return self._is_full_pp or self._use_const_ave_pp

    @property
    def gamma_N(self) -> NDArray[np.double] | None:
        """Return Normal-process part of gamma from last compute_gamma call."""
        return self._gamma_N

    @property
    def gamma_U(self) -> NDArray[np.double] | None:
        """Return Umklapp-process part of gamma from last compute_gamma call."""
        return self._gamma_U

    @property
    def gamma_detail_at_q(self) -> NDArray[np.double] | None:
        """Return per-triplet gamma from last compute_gamma call."""
        return self._gamma_detail_at_q

    def compute_gamma(self, gp: GridPointInput) -> GridPointResult:
        """Compute ph-ph linewidth at a grid point.

        Parameters
        ----------
        gp : GridPointInput
            Per-grid-point phonon data.

        Returns
        -------
        GridPointResult
            ``gamma`` (num_sigma, num_temp, num_band0) is set.
            ``averaged_pp_interaction`` (num_band0) is set when applicable.
        """
        num_band0 = len(gp.band_indices)
        num_temp = len(self._temperatures)
        num_sigma = len(self._sigmas)

        gamma = np.zeros((num_sigma, num_temp, num_band0), dtype="double", order="C")
        if self._is_N_U:
            self._gamma_N = np.zeros_like(gamma)
            self._gamma_U = np.zeros_like(gamma)
        else:
            self._gamma_N = None
            self._gamma_U = None
        self._gamma_detail_at_q = None

        result = GridPointResult(input=gp)

        self._collision.set_grid_point(gp.grid_point)

        if self._log_level:
            triplets_at_q = self._pp.get_triplets_at_q()[0]
            assert triplets_at_q is not None
            print("Number of triplets: %d" % len(triplets_at_q), flush=True)

        if self._requires_full_gamma_path():
            self._run_sigmas(gp, gamma, result)
        else:
            self._run_sigmas_lowmem(gp, gamma)

        result.gamma = gamma
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _requires_full_gamma_path(self) -> bool:
        return (
            self._is_full_pp
            or self._read_pp
            or self._store_pp
            or self._use_ave_pp
            or self._use_const_ave_pp
            or self._is_gamma_detail
        )

    def _run_sigmas(
        self,
        gp: GridPointInput,
        gamma: NDArray[np.double],
        result: GridPointResult,
    ) -> None:
        for j, sigma in enumerate(self._sigmas):
            self._collision.set_sigma(sigma, sigma_cutoff=self._sigma_cutoff)
            self._collision.run_integration_weights()
            self._set_interaction_strength(gp, j, sigma, result)
            self._allocate_gamma_detail_if_needed()
            self._run_temperatures(gp, j, gamma)

    def _set_interaction_strength(
        self,
        gp: GridPointInput,
        i_sigma: int,
        sigma: float | None,
        result: GridPointResult,
    ) -> None:
        if self._read_pp:
            self._set_from_file(gp, sigma)
        elif self._use_ave_pp:
            assert result.averaged_pp_interaction is not None
            self._collision.set_averaged_pp_interaction(result.averaged_pp_interaction)
        elif self._use_const_ave_pp:
            if self._log_level:
                assert self._pp.constant_averaged_interaction is not None
                print(
                    "Constant ph-ph interaction of %6.3e is used."
                    % self._pp.constant_averaged_interaction
                )
            self._collision.run_interaction()
            result.averaged_pp_interaction = self._pp.averaged_interaction
        elif i_sigma != 0 and (self._is_full_pp or self._sigma_cutoff is None):
            if self._log_level:
                print("Existing ph-ph interaction is used.")
        else:
            if self._log_level:
                print("Calculating ph-ph interaction...")
            self._collision.run_interaction(is_full_pp=self._is_full_pp)
            if self._is_full_pp:
                result.averaged_pp_interaction = self._pp.averaged_interaction

    def _set_from_file(self, gp: GridPointInput, sigma: float | None) -> None:
        from phono3py.file_IO import read_pp_from_hdf5

        pp, _g_zero = read_pp_from_hdf5(
            self._pp.mesh_numbers,
            grid_point=gp.grid_point,
            sigma=sigma,
            sigma_cutoff=self._sigma_cutoff,
            filename=self._pp_filename,
            verbose=(self._log_level > 0),
        )
        _, g_zero = self._collision.get_integration_weights()
        if _g_zero is not None and (_g_zero != g_zero).any():
            raise ValueError("Inconsistency found in g_zero.")
        self._collision.set_interaction_strength(pp)

    def _allocate_gamma_detail_if_needed(self) -> None:
        if not self._is_gamma_detail:
            return
        assert self._pp.interaction_strength is not None
        num_temp = len(self._temperatures)
        self._gamma_detail_at_q = np.zeros(
            (num_temp,) + self._pp.interaction_strength.shape,
            dtype="double",
            order="C",
        )

    def _run_temperatures(
        self,
        gp: GridPointInput,
        i_sigma: int,
        gamma: NDArray[np.double],
    ) -> None:
        if self._log_level:
            print("Calculating collisions at temperatures...")
        for k, t in enumerate(self._temperatures):
            self._collision.temperature = t
            self._collision.run()
            gamma[i_sigma, k] = self._collision.imag_self_energy
            if self._is_N_U:
                g_N, g_U = self._collision.get_imag_self_energy_N_and_U()
                assert self._gamma_N is not None
                assert self._gamma_U is not None
                self._gamma_N[i_sigma, k] = g_N
                self._gamma_U[i_sigma, k] = g_U
            if self._is_gamma_detail:
                assert self._gamma_detail_at_q is not None
                self._gamma_detail_at_q[k] = self._collision.detailed_imag_self_energy

    def _run_sigmas_lowmem(
        self,
        gp: GridPointInput,
        gamma: NDArray[np.double],
    ) -> None:
        """Compute gamma without storing full ph-ph interaction strength."""
        num_band = len(self._pp.primitive) * 3
        band_indices = self._pp.band_indices
        svecs, multi = self._pp.primitive.get_smallest_vectors()
        p2s = self._pp.primitive.p2s_map
        s2p = self._pp.primitive.s2p_map
        masses = self._pp.primitive.masses
        triplets_at_q, weights_at_q, _, _ = self._pp.get_triplets_at_q()
        assert triplets_at_q is not None
        assert weights_at_q is not None

        frequencies, eigenvectors, _ = self._pp.get_phonons()
        assert frequencies is not None
        assert eigenvectors is not None

        temperatures_THz = np.array(
            self._temperatures * get_physical_units().KB / get_physical_units().THzToEv,
            dtype="double",
        )

        if None in self._sigmas:
            from phono3py.other.tetrahedron_method import (
                get_tetrahedra_relative_grid_address,
            )

            tetrahedra = get_tetrahedra_relative_grid_address(
                self._pp.bz_grid.microzone_lattice
            )

        if self._pp.openmp_per_triplets is None:
            openmp_per_triplets = len(triplets_at_q) > num_band
        else:
            openmp_per_triplets = self._pp.openmp_per_triplets

        import phono3py._phono3py as phono3c

        for j, sigma in enumerate(self._sigmas):
            self._collision.set_sigma(sigma)
            if self._is_N_U:
                collisions = np.zeros(
                    (2, len(self._temperatures), len(band_indices)),
                    dtype="double",
                    order="C",
                )
            else:
                collisions = np.zeros(
                    (len(self._temperatures), len(band_indices)),
                    dtype="double",
                    order="C",
                )

            if sigma is None:
                phono3c.pp_collision(
                    collisions,
                    np.array(
                        np.dot(tetrahedra, self._pp.bz_grid.P.T),
                        dtype="int64",
                        order="C",
                    ),
                    frequencies,
                    eigenvectors,
                    triplets_at_q,
                    weights_at_q,
                    self._pp.bz_grid.addresses,
                    self._pp.bz_grid.gp_map,
                    self._pp.bz_grid.store_dense_gp_map * 1 + 1,
                    self._pp.bz_grid.D_diag,
                    self._pp.bz_grid.Q,
                    self._pp.fc3,
                    self._pp.fc3_nonzero_indices,
                    svecs,
                    multi,
                    masses,
                    p2s,
                    s2p,
                    band_indices,
                    temperatures_THz,
                    self._is_N_U * 1,
                    self._pp.symmetrize_fc3q * 1,
                    self._pp.make_r0_average * 1,
                    self._pp.all_shortest,
                    self._pp.cutoff_frequency,
                    openmp_per_triplets * 1,
                )
            else:
                sigma_cutoff = (
                    -1.0 if self._sigma_cutoff is None else self._sigma_cutoff
                )
                phono3c.pp_collision_with_sigma(
                    collisions,
                    sigma,
                    sigma_cutoff,
                    frequencies,
                    eigenvectors,
                    triplets_at_q,
                    weights_at_q,
                    self._pp.bz_grid.addresses,
                    self._pp.bz_grid.D_diag,
                    self._pp.bz_grid.Q,
                    self._pp.fc3,
                    self._pp.fc3_nonzero_indices,
                    svecs,
                    multi,
                    masses,
                    p2s,
                    s2p,
                    band_indices,
                    temperatures_THz,
                    self._is_N_U * 1,
                    self._pp.symmetrize_fc3q * 1,
                    self._pp.make_r0_average * 1,
                    self._pp.all_shortest,
                    self._pp.cutoff_frequency,
                    openmp_per_triplets * 1,
                )

            col_unit_conv = self._collision.unit_conversion_factor
            pp_unit_conv = self._pp.unit_conversion_factor
            if self._is_N_U:
                col = collisions.sum(axis=0)
                col_N = collisions[0]
                col_U = collisions[1]
            else:
                col = collisions

            freq_at_gp = frequencies[gp.grid_point]
            for k in range(len(self._temperatures)):
                gamma[j, k] = average_by_degeneracy(
                    col[k] * col_unit_conv * pp_unit_conv,
                    band_indices,
                    freq_at_gp,
                )
                if self._is_N_U:
                    assert self._gamma_N is not None
                    assert self._gamma_U is not None
                    self._gamma_N[j, k] = average_by_degeneracy(
                        col_N[k] * col_unit_conv * pp_unit_conv,
                        band_indices,
                        freq_at_gp,
                    )
                    self._gamma_U[j, k] = average_by_degeneracy(
                        col_U[k] * col_unit_conv * pp_unit_conv,
                        band_indices,
                        freq_at_gp,
                    )


class IsotopeScatteringProvider:
    """Compute isotope scattering linewidth at a grid point.

    Sets ``GridPointResult.gamma_isotope`` with shape
    ``(num_sigma, num_band0)``.

    Parameters
    ----------
    isotope : Isotope
        Isotope instance, already initialised with phonons.
    sigmas : sequence of float or None
        Smearing widths matching those used for ph-ph gamma.
    log_level : int, optional
        Verbosity level. Default 0.
    """

    def __init__(
        self,
        isotope: Isotope,
        sigmas: Sequence[float | None],
        log_level: int = 0,
    ):
        """Init method."""
        self._isotope = isotope
        self._sigmas = list(sigmas)
        self._log_level = log_level

    @property
    def isotope(self) -> Isotope:
        """Return the wrapped Isotope instance."""
        return self._isotope

    def compute_gamma_isotope(self, gp: GridPointInput) -> GridPointResult:
        """Compute isotope linewidth at a grid point.

        Parameters
        ----------
        gp : GridPointInput
            Per-grid-point phonon data.

        Returns
        -------
        GridPointResult
            ``gamma_isotope`` (num_sigma, num_band0) is set.
        """
        gamma_iso = []
        for sigma in self._sigmas:
            if self._log_level:
                text = "Calculating Gamma of ph-isotope with "
                text += "tetrahedron method" if sigma is None else "sigma=%s" % sigma
                print(text)
            self._isotope.sigma = sigma
            self._isotope.set_grid_point(gp.grid_point)
            self._isotope.run()
            gamma_iso.append(self._isotope.gamma)

        result = GridPointResult(input=gp)
        result.gamma_isotope = np.array(gamma_iso, dtype="double", order="C")
        return result


class BoundaryScatteringProvider:
    """Compute boundary scattering linewidth at a grid point.

    Sets ``GridPointResult.gamma_boundary`` with shape ``(num_band0,)``.
    The formula is:

        gamma_boundary[s] = |v_s| * 1e6 * Angstrom / (4 * pi * boundary_mfp)

    where ``boundary_mfp`` is in micrometres and ``|v_s|`` is the group
    velocity magnitude in THz*Angstrom.

    Parameters
    ----------
    boundary_mfp : float
        Boundary mean free path in micrometres.
    """

    def __init__(self, boundary_mfp: float):
        """Init method."""
        self._boundary_mfp = boundary_mfp

    def compute_gamma_boundary(self, gp: GridPointInput) -> GridPointResult:
        """Compute boundary scattering linewidth at a grid point.

        Parameters
        ----------
        gp : GridPointInput
            Per-grid-point phonon data.  ``group_velocities`` must be set
            in the input (filled by a VelocityProvider beforehand).

        Returns
        -------
        GridPointResult
            ``gamma_boundary`` (num_band0,) is set.

        Notes
        -----
        This provider requires group velocities.  Call a VelocityProvider
        first and pass the resulting ``group_velocities`` via ``gp`` or
        provide them externally.
        """
        result = GridPointResult(input=gp)
        num_band0 = len(gp.band_indices)
        g_boundary = np.zeros(num_band0, dtype="double")
        result.gamma_boundary = g_boundary
        return result

    def compute_gamma_boundary_from_gv(
        self,
        gp: GridPointInput,
        group_velocities: NDArray[np.double],
    ) -> GridPointResult:
        """Compute boundary scattering linewidth using precomputed group velocities.

        Parameters
        ----------
        gp : GridPointInput
            Per-grid-point phonon data.
        group_velocities : ndarray, shape (num_band0, 3)
            Group velocities in THz*Angstrom.

        Returns
        -------
        GridPointResult
            ``gamma_boundary`` (num_band0,) is set.
        """
        result = GridPointResult(input=gp)
        num_band0 = len(gp.band_indices)
        g_boundary = np.zeros(num_band0, dtype="double")
        for ll in range(num_band0):
            g_boundary[ll] = (
                np.linalg.norm(group_velocities[ll])
                * get_physical_units().Angstrom
                * 1e6
                / (4 * np.pi * self._boundary_mfp)
            )
        result.gamma_boundary = g_boundary
        return result
