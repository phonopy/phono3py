"""Scattering solver building blocks for conductivity calculations."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from phonopy.physical_units import get_physical_units

from phono3py._lang import resolve_lang
from phono3py.conductivity.grid_point_data import ScatteringResult
from phono3py.other.isotope import Isotope
from phono3py.phonon3.imag_self_energy import ImagSelfEnergy, average_by_degeneracy
from phono3py.phonon3.interaction import Interaction
from phono3py.phonon3.triplets import get_triplets_at_q


def run_pp_collision_rust(
    collisions: NDArray[np.double],
    relative_grid_address: NDArray[np.int64],
    frequencies: NDArray[np.double],
    eigenvectors: NDArray[np.cdouble],
    triplets_at_q: NDArray[np.int64],
    weights_at_q: NDArray[np.int64],
    bz_grid_addresses: NDArray[np.int64],
    bz_map: NDArray[np.int64],
    bz_grid_type: int,
    d_diag: NDArray[np.int64],
    q_matrix: NDArray[np.int64],
    fc3: NDArray[np.double],
    fc3_nonzero_indices: NDArray[np.byte],
    svecs: NDArray[np.double],
    multiplicity: NDArray[np.int64],
    masses: NDArray[np.double],
    p2s_map: NDArray[np.int64],
    s2p_map: NDArray[np.int64],
    band_indices: NDArray[np.int64],
    temperatures_thz: NDArray[np.double],
    is_N_U: bool,
    symmetrize_fc3_q: bool,
    make_r0_average: bool,
    all_shortest: NDArray[np.byte],
    cutoff_frequency: float,
) -> None:
    """Compute low-memory collision with the tetrahedron method (Rust).

    Drop-in replacement for ``phono3c.pp_collision``.  Writes into
    ``collisions`` in place; the shape is ``(num_temps, num_band0)`` or
    ``(2, num_temps, num_band0)`` when ``is_N_U`` is True.

    """
    import phonors

    is_compact_fc3 = fc3.shape[0] != fc3.shape[1]

    phonors.pp_collision(
        collisions,
        np.ascontiguousarray(relative_grid_address, dtype="int64"),
        np.ascontiguousarray(frequencies, dtype="double"),
        np.ascontiguousarray(eigenvectors, dtype="complex128"),
        np.ascontiguousarray(triplets_at_q, dtype="int64"),
        np.ascontiguousarray(weights_at_q, dtype="int64"),
        np.ascontiguousarray(bz_grid_addresses, dtype="int64"),
        np.ascontiguousarray(bz_map, dtype="int64"),
        int(bz_grid_type),
        np.ascontiguousarray(d_diag, dtype="int64"),
        np.ascontiguousarray(q_matrix, dtype="int64"),
        np.ascontiguousarray(fc3, dtype="double"),
        np.ascontiguousarray(fc3_nonzero_indices, dtype="byte"),
        np.ascontiguousarray(svecs, dtype="double"),
        np.ascontiguousarray(multiplicity, dtype="int64"),
        np.ascontiguousarray(masses, dtype="double"),
        np.ascontiguousarray(p2s_map, dtype="int64"),
        np.ascontiguousarray(s2p_map, dtype="int64"),
        np.ascontiguousarray(band_indices, dtype="int64"),
        np.ascontiguousarray(temperatures_thz, dtype="double"),
        bool(is_N_U),
        bool(symmetrize_fc3_q),
        bool(make_r0_average),
        np.ascontiguousarray(all_shortest, dtype="byte"),
        float(cutoff_frequency),
        is_compact_fc3,
    )


def run_pp_collision_with_sigma_rust(
    collisions: NDArray[np.double],
    sigma: float,
    sigma_cutoff: float,
    frequencies: NDArray[np.double],
    eigenvectors: NDArray[np.cdouble],
    triplets_at_q: NDArray[np.int64],
    weights_at_q: NDArray[np.int64],
    bz_grid_addresses: NDArray[np.int64],
    d_diag: NDArray[np.int64],
    q_matrix: NDArray[np.int64],
    fc3: NDArray[np.double],
    fc3_nonzero_indices: NDArray[np.byte],
    svecs: NDArray[np.double],
    multiplicity: NDArray[np.int64],
    masses: NDArray[np.double],
    p2s_map: NDArray[np.int64],
    s2p_map: NDArray[np.int64],
    band_indices: NDArray[np.int64],
    temperatures_thz: NDArray[np.double],
    is_N_U: bool,
    symmetrize_fc3_q: bool,
    make_r0_average: bool,
    all_shortest: NDArray[np.byte],
    cutoff_frequency: float,
) -> None:
    """Compute low-memory collision with Gaussian smearing (Rust).

    Drop-in replacement for ``phono3c.pp_collision_with_sigma``.  The
    ``sigma_cutoff <= 0`` convention disables the cutoff-skip
    optimisation (matches the C backend).

    """
    import phonors

    is_compact_fc3 = fc3.shape[0] != fc3.shape[1]

    phonors.pp_collision_with_sigma(
        collisions,
        float(sigma),
        float(sigma_cutoff),
        np.ascontiguousarray(frequencies, dtype="double"),
        np.ascontiguousarray(eigenvectors, dtype="complex128"),
        np.ascontiguousarray(triplets_at_q, dtype="int64"),
        np.ascontiguousarray(weights_at_q, dtype="int64"),
        np.ascontiguousarray(bz_grid_addresses, dtype="int64"),
        np.ascontiguousarray(d_diag, dtype="int64"),
        np.ascontiguousarray(q_matrix, dtype="int64"),
        np.ascontiguousarray(fc3, dtype="double"),
        np.ascontiguousarray(fc3_nonzero_indices, dtype="byte"),
        np.ascontiguousarray(svecs, dtype="double"),
        np.ascontiguousarray(multiplicity, dtype="int64"),
        np.ascontiguousarray(masses, dtype="double"),
        np.ascontiguousarray(p2s_map, dtype="int64"),
        np.ascontiguousarray(s2p_map, dtype="int64"),
        np.ascontiguousarray(band_indices, dtype="int64"),
        np.ascontiguousarray(temperatures_thz, dtype="double"),
        bool(is_N_U),
        bool(symmetrize_fc3_q),
        bool(make_r0_average),
        np.ascontiguousarray(all_shortest, dtype="byte"),
        float(cutoff_frequency),
        is_compact_fc3,
    )


def run_collision_at_grid_points_batched_rust(
    collisions: NDArray[np.double],
    grid_points: NDArray[np.int64],
    sigmas: NDArray[np.double],
    sigma_cutoffs: NDArray[np.double],
    relative_grid_address: NDArray[np.int64],
    bzg2grg: NDArray[np.int64],
    reciprocal_rotations: NDArray[np.int64],
    is_time_reversal: bool,
    swappable: bool,
    is_mesh_symmetry: bool,
    reciprocal_lattice: NDArray[np.double],
    bz_triplets_q_mat: NDArray[np.int64],
    frequencies: NDArray[np.double],
    eigenvectors: NDArray[np.cdouble],
    bz_grid_addresses: NDArray[np.int64],
    bz_map: NDArray[np.int64],
    bz_grid_type: int,
    d_diag: NDArray[np.int64],
    q_matrix: NDArray[np.int64],
    fc3: NDArray[np.double],
    fc3_nonzero_indices: NDArray[np.byte],
    svecs: NDArray[np.double],
    multiplicity: NDArray[np.int64],
    masses: NDArray[np.double],
    p2s_map: NDArray[np.int64],
    s2p_map: NDArray[np.int64],
    band_indices: NDArray[np.int64],
    temperatures_thz: NDArray[np.double],
    is_N_U: bool,
    symmetrize_fc3_q: bool,
    make_r0_average: bool,
    all_shortest: NDArray[np.byte],
    cutoff_frequency: float,
) -> None:
    """Compute gamma for a batch of grid points in one Rust call.

    Collapses the nested rayon parallelism present in the single-gp
    path (``inner_par=true`` when ``num_triplets < num_threads``) by
    flattening the outer rayon loop across all ``(gp, triplet)``
    pairs in the batch.  Every per-triplet kernel then runs with
    ``inner_par=false``, eliminating the ~30% ``do_spin`` observed
    in single-gp profiles on many-core machines.

    ``collisions`` has shape
    ``(num_gp_batch, num_sigma, num_temps, num_band0)`` or
    ``(num_gp_batch, num_sigma, 2, num_temps, num_band0)`` when
    ``is_N_U``.  ``grid_points[i]`` is the BZ grid point for batch
    slot ``i``.  A NaN entry in ``sigmas`` selects the tetrahedron
    path for that slot; finite values pick Gaussian smearing.

    """
    import phonors

    is_compact_fc3 = fc3.shape[0] != fc3.shape[1]

    phonors.collision_at_grid_points_batched(
        collisions,
        np.ascontiguousarray(grid_points, dtype="int64"),
        np.ascontiguousarray(sigmas, dtype="double"),
        np.ascontiguousarray(sigma_cutoffs, dtype="double"),
        np.ascontiguousarray(relative_grid_address, dtype="int64"),
        np.ascontiguousarray(bzg2grg, dtype="int64"),
        np.ascontiguousarray(reciprocal_rotations, dtype="int64"),
        bool(is_time_reversal),
        bool(swappable),
        bool(is_mesh_symmetry),
        np.ascontiguousarray(reciprocal_lattice, dtype="double"),
        np.ascontiguousarray(bz_triplets_q_mat, dtype="int64"),
        np.ascontiguousarray(frequencies, dtype="double"),
        np.ascontiguousarray(eigenvectors, dtype="complex128"),
        np.ascontiguousarray(bz_grid_addresses, dtype="int64"),
        np.ascontiguousarray(bz_map, dtype="int64"),
        int(bz_grid_type),
        np.ascontiguousarray(d_diag, dtype="int64"),
        np.ascontiguousarray(q_matrix, dtype="int64"),
        np.ascontiguousarray(fc3, dtype="double"),
        np.ascontiguousarray(fc3_nonzero_indices, dtype="byte"),
        np.ascontiguousarray(svecs, dtype="double"),
        np.ascontiguousarray(multiplicity, dtype="int64"),
        np.ascontiguousarray(masses, dtype="double"),
        np.ascontiguousarray(p2s_map, dtype="int64"),
        np.ascontiguousarray(s2p_map, dtype="int64"),
        np.ascontiguousarray(band_indices, dtype="int64"),
        np.ascontiguousarray(temperatures_thz, dtype="double"),
        bool(is_N_U),
        bool(symmetrize_fc3_q),
        bool(make_r0_average),
        np.ascontiguousarray(all_shortest, dtype="byte"),
        float(cutoff_frequency),
        is_compact_fc3,
    )


def run_collision_at_grid_point_rust(
    collisions: NDArray[np.double],
    grid_point: int,
    sigmas: NDArray[np.double],
    sigma_cutoffs: NDArray[np.double],
    relative_grid_address: NDArray[np.int64],
    bzg2grg: NDArray[np.int64],
    reciprocal_rotations: NDArray[np.int64],
    is_time_reversal: bool,
    swappable: bool,
    is_mesh_symmetry: bool,
    reciprocal_lattice: NDArray[np.double],
    bz_triplets_q_mat: NDArray[np.int64],
    frequencies: NDArray[np.double],
    eigenvectors: NDArray[np.cdouble],
    bz_grid_addresses: NDArray[np.int64],
    bz_map: NDArray[np.int64],
    bz_grid_type: int,
    d_diag: NDArray[np.int64],
    q_matrix: NDArray[np.int64],
    fc3: NDArray[np.double],
    fc3_nonzero_indices: NDArray[np.byte],
    svecs: NDArray[np.double],
    multiplicity: NDArray[np.int64],
    masses: NDArray[np.double],
    p2s_map: NDArray[np.int64],
    s2p_map: NDArray[np.int64],
    band_indices: NDArray[np.int64],
    temperatures_thz: NDArray[np.double],
    is_N_U: bool,
    symmetrize_fc3_q: bool,
    make_r0_average: bool,
    all_shortest: NDArray[np.byte],
    cutoff_frequency: float,
) -> None:
    """Compute gamma for one grid point with multiple sigmas (Rust).

    Folds Interaction.set_grid_point + get_triplets_at_q + the
    per-sigma pp_collision loop into a single Rust call so that rayon
    workers stay warm across sigma iterations and the GIL is released
    only once per grid point.  ``collisions`` has shape
    ``(num_sigma, num_temps, num_band0)`` or
    ``(num_sigma, 2, num_temps, num_band0)`` when ``is_N_U``.  A NaN
    entry in ``sigmas`` selects the tetrahedron-method path for that
    slot; otherwise Gaussian smearing is used with
    ``sigma_cutoffs[i]`` (``<= 0`` disables cutoff-skip, matching C).

    """
    import phonors

    is_compact_fc3 = fc3.shape[0] != fc3.shape[1]

    phonors.collision_at_grid_point(
        collisions,
        int(grid_point),
        np.ascontiguousarray(sigmas, dtype="double"),
        np.ascontiguousarray(sigma_cutoffs, dtype="double"),
        np.ascontiguousarray(relative_grid_address, dtype="int64"),
        np.ascontiguousarray(bzg2grg, dtype="int64"),
        np.ascontiguousarray(reciprocal_rotations, dtype="int64"),
        bool(is_time_reversal),
        bool(swappable),
        bool(is_mesh_symmetry),
        np.ascontiguousarray(reciprocal_lattice, dtype="double"),
        np.ascontiguousarray(bz_triplets_q_mat, dtype="int64"),
        np.ascontiguousarray(frequencies, dtype="double"),
        np.ascontiguousarray(eigenvectors, dtype="complex128"),
        np.ascontiguousarray(bz_grid_addresses, dtype="int64"),
        np.ascontiguousarray(bz_map, dtype="int64"),
        int(bz_grid_type),
        np.ascontiguousarray(d_diag, dtype="int64"),
        np.ascontiguousarray(q_matrix, dtype="int64"),
        np.ascontiguousarray(fc3, dtype="double"),
        np.ascontiguousarray(fc3_nonzero_indices, dtype="byte"),
        np.ascontiguousarray(svecs, dtype="double"),
        np.ascontiguousarray(multiplicity, dtype="int64"),
        np.ascontiguousarray(masses, dtype="double"),
        np.ascontiguousarray(p2s_map, dtype="int64"),
        np.ascontiguousarray(s2p_map, dtype="int64"),
        np.ascontiguousarray(band_indices, dtype="int64"),
        np.ascontiguousarray(temperatures_thz, dtype="double"),
        bool(is_N_U),
        bool(symmetrize_fc3_q),
        bool(make_r0_average),
        np.ascontiguousarray(all_shortest, dtype="byte"),
        float(cutoff_frequency),
        is_compact_fc3,
    )


class RTAScatteringSolver:
    """Compute ph-ph linewidth (gamma) at a grid point using the RTA.

    This solver implements the ``ScatteringSolver`` protocol.  It wraps
    ``ImagSelfEnergy`` and handles the sigma loop, the temperature loop, and
    the various computation modes (read_pp, use_ave_pp, low-memory path).

    The returned ``ScatteringResult`` contains ``gamma`` with shape
    ``(num_sigma, num_temp, num_band0)`` (ph-ph contributions only).
    Isotope and boundary contributions are handled by separate providers.

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
    lang : str, optional
        Backend selection for the low-memory and ImagSelfEnergy paths.
        ``"C"`` (default) uses the C extension, ``"Python"`` the slow
        reference, and ``"Rust"`` the Rust backend.

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
        pp_filename: str | os.PathLike | None = None,
        is_N_U: bool = False,
        is_gamma_detail: bool = False,
        log_level: int = 0,
        lang: Literal["C", "Python", "Rust"] = "Rust",
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
        if lang in ("C", "Rust"):
            lang = resolve_lang(lang)
        self._lang: Literal["C", "Python", "Rust"] = lang

        self._collision = ImagSelfEnergy(
            pp, with_detail=(is_gamma_detail or is_N_U), lang=lang
        )

        # Per-grid-point state set during compute, accessible for output.
        self._gamma_N: NDArray[np.double] | None = None
        self._gamma_U: NDArray[np.double] | None = None
        self._gamma_detail_at_q: NDArray[np.double] | None = None

        # Cached loop-invariant buffers for the Rust fast path; built
        # lazily on the first Rust compute() call.
        self._rust_cache: dict | None = None

    @property
    def is_full_pp(self) -> bool:
        """Return True if averaged ph-ph interaction will be computed."""
        return self._is_full_pp or self._use_const_ave_pp

    @property
    def gamma_N(self) -> NDArray[np.double] | None:
        """Return Normal-process part of gamma from last compute call."""
        return self._gamma_N

    @property
    def gamma_U(self) -> NDArray[np.double] | None:
        """Return Umklapp-process part of gamma from last compute call."""
        return self._gamma_U

    @property
    def gamma_detail_at_q(self) -> NDArray[np.double] | None:
        """Return per-triplet gamma from last compute call."""
        return self._gamma_detail_at_q

    def compute(self, grid_point: int) -> ScatteringResult:
        """Compute ph-ph linewidth at a grid point.

        Parameters
        ----------
        grid_point : int
            BZ grid point index.

        Returns
        -------
        ScatteringResult
            ``gamma`` (num_sigma, num_temp, num_band0) is set.
            ``averaged_pp_interaction`` (num_band0) is set when applicable.
        """
        if self._lang == "Rust" and not self._requires_full_gamma_path():
            return self._compute_rust(grid_point)

        num_band0 = len(self._pp.band_indices)
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

        self._averaged_pp_interaction: NDArray[np.double] | None = None

        self._collision.set_grid_point(grid_point)

        if self._log_level:
            triplets_at_q = self._pp.get_triplets_at_q()[0]
            assert triplets_at_q is not None
            print("Number of triplets: %d" % len(triplets_at_q), flush=True)

        if self._requires_full_gamma_path():
            self._run_sigmas(grid_point, gamma)
        else:
            self._run_sigmas_lowmem(grid_point, gamma)

        return ScatteringResult(
            gamma=gamma,
            averaged_pp_interaction=self._averaged_pp_interaction,
        )

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
        grid_point: int,
        gamma: NDArray[np.double],
    ) -> None:
        for j, sigma in enumerate(self._sigmas):
            self._collision.set_sigma(sigma, sigma_cutoff=self._sigma_cutoff)
            self._collision.run_integration_weights()
            self._set_interaction_strength(grid_point, j, sigma)
            self._allocate_gamma_detail_if_needed()
            self._run_temperatures(j, gamma)

    def _set_interaction_strength(
        self,
        grid_point: int,
        i_sigma: int,
        sigma: float | None,
    ) -> None:
        if self._read_pp:
            self._set_from_file(grid_point, sigma)
        elif self._use_ave_pp:
            assert self._averaged_pp_interaction is not None
            self._collision.set_averaged_pp_interaction(self._averaged_pp_interaction)
        elif self._use_const_ave_pp:
            if self._log_level:
                assert self._pp.constant_averaged_interaction is not None
                print(
                    "Constant ph-ph interaction of %6.3e is used."
                    % self._pp.constant_averaged_interaction
                )
            self._collision.run_interaction()
            self._averaged_pp_interaction = self._pp.averaged_interaction
        elif i_sigma != 0 and (self._is_full_pp or self._sigma_cutoff is None):
            if self._log_level:
                print("Existing ph-ph interaction is used.")
        else:
            self._collision.run_interaction(is_full_pp=self._is_full_pp)
            if self._is_full_pp:
                self._averaged_pp_interaction = self._pp.averaged_interaction

    def _set_from_file(self, grid_point: int, sigma: float | None) -> None:
        from phono3py.file_IO import read_pp_from_hdf5

        pp, _g_zero = read_pp_from_hdf5(
            self._pp.mesh_numbers,
            grid_point=grid_point,
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
        grid_point: int,
        gamma: NDArray[np.double],
    ) -> None:
        """Compute gamma without storing full ph-ph interaction strength."""
        temperatures_THz = np.array(
            self._temperatures * get_physical_units().KB / get_physical_units().THzToEv,
            dtype="double",
        )

        tetrahedra: NDArray[np.int64] | None = None
        if None in self._sigmas:
            from phonopy.phonon.tetrahedron_method import (
                get_tetrahedra_relative_grid_address,
            )

            tetrahedra = get_tetrahedra_relative_grid_address(
                self._pp.bz_grid.microzone_lattice
            )

        if self._pp.openmp_per_triplets is None:
            triplets_at_q, _, _, _ = self._pp.get_triplets_at_q()
            num_band = len(self._pp.primitive) * 3
            openmp_per_triplets = len(triplets_at_q) > num_band
        else:
            openmp_per_triplets = self._pp.openmp_per_triplets

        for j, sigma in enumerate(self._sigmas):
            collisions = self._dispatch_lowmem_collision(
                sigma,
                temperatures_THz,
                tetrahedra,
                openmp_per_triplets,
            )
            self._store_lowmem_results(j, grid_point, gamma, collisions)

    def _dispatch_lowmem_collision(
        self,
        sigma: float | None,
        temperatures_THz: NDArray[np.double],
        tetrahedra: NDArray[np.int64] | None,
        openmp_per_triplets: bool,
    ) -> NDArray[np.double]:
        """Call C-extension for low-memory collision at one sigma."""
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

        self._collision.set_sigma(sigma)

        if sigma is None:
            assert tetrahedra is not None
            relative_grid_address = np.array(
                np.dot(tetrahedra, self._pp.bz_grid.P.T),
                dtype="int64",
                order="C",
            )
            if self._lang == "Rust":
                run_pp_collision_rust(
                    collisions,
                    relative_grid_address,
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
                    self._is_N_U,
                    self._pp.symmetrize_fc3q,
                    self._pp.make_r0_average,
                    self._pp.all_shortest,
                    self._pp.cutoff_frequency,
                )
            else:
                import phono3py._phono3py as phono3c

                phono3c.pp_collision(
                    collisions,
                    relative_grid_address,
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
            sigma_cutoff = -1.0 if self._sigma_cutoff is None else self._sigma_cutoff
            if self._lang == "Rust":
                run_pp_collision_with_sigma_rust(
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
                    self._is_N_U,
                    self._pp.symmetrize_fc3q,
                    self._pp.make_r0_average,
                    self._pp.all_shortest,
                    self._pp.cutoff_frequency,
                )
            else:
                import phono3py._phono3py as phono3c

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

        return collisions

    def _store_lowmem_results(
        self,
        i_sigma: int,
        grid_point: int,
        gamma: NDArray[np.double],
        collisions: NDArray[np.double],
    ) -> None:
        """Apply unit conversion and degeneracy averaging to collision results."""
        col_unit_conv = self._collision.unit_conversion_factor
        pp_unit_conv = self._pp.unit_conversion_factor
        band_indices = self._pp.band_indices
        frequencies, _, _ = self._pp.get_phonons()
        freq_at_gp = frequencies[grid_point]

        if self._is_N_U:
            col = collisions.sum(axis=0)
            col_N = collisions[0]
            col_U = collisions[1]
        else:
            col = collisions

        unit_conv = col_unit_conv * pp_unit_conv
        gamma[i_sigma] = average_by_degeneracy(
            col * unit_conv,
            band_indices,
            freq_at_gp,
        )
        if self._is_N_U:
            assert self._gamma_N is not None
            assert self._gamma_U is not None
            self._gamma_N[i_sigma] = average_by_degeneracy(
                col_N * unit_conv,
                band_indices,
                freq_at_gp,
            )
            self._gamma_U[i_sigma] = average_by_degeneracy(
                col_U * unit_conv,
                band_indices,
                freq_at_gp,
            )

    # ------------------------------------------------------------------
    # Rust fast path: set_grid_point + triplets + per-sigma pp_collision
    # fused into a single Rust call, with loop-invariant data cached.
    # ------------------------------------------------------------------

    def _compute_rust(self, grid_point: int) -> ScatteringResult:
        cache = self._get_rust_cache()

        num_band0 = len(self._pp.band_indices)
        num_temp = len(self._temperatures)
        num_sigma = len(self._sigmas)

        gamma = np.zeros((num_sigma, num_temp, num_band0), dtype="double", order="C")
        if self._is_N_U:
            self._gamma_N = np.zeros_like(gamma)
            self._gamma_U = np.zeros_like(gamma)
            collisions = np.zeros(
                (num_sigma, 2, num_temp, num_band0),
                dtype="double",
                order="C",
            )
        else:
            self._gamma_N = None
            self._gamma_U = None
            collisions = np.zeros(
                (num_sigma, num_temp, num_band0), dtype="double", order="C"
            )
        self._gamma_detail_at_q = None
        self._averaged_pp_interaction = None

        if self._log_level:
            triplets_at_q = get_triplets_at_q(grid_point, self._pp.bz_grid)[0]
            print(
                "Number of triplets: %d" % len(triplets_at_q),
                flush=True,
            )

        run_collision_at_grid_point_rust(
            collisions,
            grid_point,
            cache["sigmas"],
            cache["sigma_cutoffs"],
            cache["relative_grid_address"],
            cache["bzg2grg"],
            cache["reciprocal_rotations"],
            cache["is_time_reversal"],
            cache["swappable"],
            cache["is_mesh_symmetry"],
            cache["reciprocal_lattice"],
            cache["bz_triplets_q_mat"],
            cache["frequencies"],
            cache["eigenvectors"],
            cache["bz_grid_addresses"],
            cache["bz_map"],
            cache["bz_grid_type"],
            cache["d_diag"],
            cache["q_matrix"],
            cache["fc3"],
            cache["fc3_nonzero_indices"],
            cache["svecs"],
            cache["multiplicity"],
            cache["masses"],
            cache["p2s_map"],
            cache["s2p_map"],
            cache["band_indices"],
            cache["temperatures_thz"],
            self._is_N_U,
            cache["symmetrize_fc3q"],
            cache["make_r0_average"],
            cache["all_shortest"],
            cache["cutoff_frequency"],
        )

        for j in range(num_sigma):
            self._store_lowmem_results(j, grid_point, gamma, collisions[j])

        return ScatteringResult(
            gamma=gamma,
            averaged_pp_interaction=self._averaged_pp_interaction,
        )

    @property
    def supports_rust_batching(self) -> bool:
        """Whether ``compute_batched`` goes through the Rust batched path."""
        return self._lang == "Rust" and not self._requires_full_gamma_path()

    def compute_batched(self, grid_points: Sequence[int]) -> list[dict]:
        """Compute ph-ph linewidth at multiple grid points in one Rust call.

        Collapses the per-gp rayon nested parallelism by flattening the
        outer par over all ``(gp, triplet)`` pairs in the batch; useful
        on many-core machines where single-gp ``num_triplets`` is below
        the thread count.

        Falls back to a per-gp ``compute()`` loop if the Rust batched
        path is not available (e.g. non-Rust ``lang`` or full-gamma mode).

        Returns
        -------
        list of dict, one per grid point in ``grid_points``, each with
        keys ``"result"`` (``ScatteringResult``), ``"gamma_N"`` and
        ``"gamma_U"`` (``NDArray | None``).  Unlike ``compute()``, this
        method does not leave the last gp's N/U on ``self`` — the caller
        must read them from the returned list.

        """
        if not self.supports_rust_batching:
            fallback: list[dict] = []
            for gp in grid_points:
                result = self.compute(int(gp))
                fallback.append(
                    {
                        "result": result,
                        "gamma_N": None
                        if self._gamma_N is None
                        else self._gamma_N.copy(),
                        "gamma_U": None
                        if self._gamma_U is None
                        else self._gamma_U.copy(),
                    }
                )
            return fallback

        cache = self._get_rust_cache()

        num_gp_batch = len(grid_points)
        num_band0 = len(self._pp.band_indices)
        num_temp = len(self._temperatures)
        num_sigma = len(self._sigmas)

        if self._is_N_U:
            collisions = np.zeros(
                (num_gp_batch, num_sigma, 2, num_temp, num_band0),
                dtype="double",
                order="C",
            )
        else:
            collisions = np.zeros(
                (num_gp_batch, num_sigma, num_temp, num_band0),
                dtype="double",
                order="C",
            )

        grid_points_arr = np.ascontiguousarray(list(grid_points), dtype="int64")

        run_collision_at_grid_points_batched_rust(
            collisions,
            grid_points_arr,
            cache["sigmas"],
            cache["sigma_cutoffs"],
            cache["relative_grid_address"],
            cache["bzg2grg"],
            cache["reciprocal_rotations"],
            cache["is_time_reversal"],
            cache["swappable"],
            cache["is_mesh_symmetry"],
            cache["reciprocal_lattice"],
            cache["bz_triplets_q_mat"],
            cache["frequencies"],
            cache["eigenvectors"],
            cache["bz_grid_addresses"],
            cache["bz_map"],
            cache["bz_grid_type"],
            cache["d_diag"],
            cache["q_matrix"],
            cache["fc3"],
            cache["fc3_nonzero_indices"],
            cache["svecs"],
            cache["multiplicity"],
            cache["masses"],
            cache["p2s_map"],
            cache["s2p_map"],
            cache["band_indices"],
            cache["temperatures_thz"],
            self._is_N_U,
            cache["symmetrize_fc3q"],
            cache["make_r0_average"],
            cache["all_shortest"],
            cache["cutoff_frequency"],
        )

        out: list[dict] = []
        for i, gp in enumerate(grid_points):
            gamma_i = np.zeros(
                (num_sigma, num_temp, num_band0), dtype="double", order="C"
            )
            if self._is_N_U:
                self._gamma_N = np.zeros_like(gamma_i)
                self._gamma_U = np.zeros_like(gamma_i)
            else:
                self._gamma_N = None
                self._gamma_U = None
            self._gamma_detail_at_q = None
            self._averaged_pp_interaction = None
            for j in range(num_sigma):
                self._store_lowmem_results(j, int(gp), gamma_i, collisions[i, j])
            out.append(
                {
                    "result": ScatteringResult(
                        gamma=gamma_i,
                        averaged_pp_interaction=None,
                    ),
                    "gamma_N": None if self._gamma_N is None else self._gamma_N.copy(),
                    "gamma_U": None if self._gamma_U is None else self._gamma_U.copy(),
                }
            )
        return out

    def _get_rust_cache(self) -> dict:
        if self._rust_cache is not None:
            return self._rust_cache

        from phonopy.phonon.grid import get_reduced_bases_and_tmat_inv
        from phonopy.phonon.tetrahedron_method import (
            get_tetrahedra_relative_grid_address,
        )

        pp = self._pp
        svecs, multi = pp.primitive.get_smallest_vectors()
        frequencies, eigenvectors, _ = pp.get_phonons()
        assert frequencies is not None
        assert eigenvectors is not None

        tetrahedra = get_tetrahedra_relative_grid_address(pp.bz_grid.microzone_lattice)
        relative_grid_address = np.array(
            np.dot(tetrahedra, pp.bz_grid.P.T), dtype="int64", order="C"
        )

        reduced_basis, tmat_inv_int = get_reduced_bases_and_tmat_inv(
            pp.bz_grid.reciprocal_lattice
        )
        bz_triplets_q_mat = np.array(
            tmat_inv_int @ pp.bz_grid.Q, dtype="int64", order="C"
        )

        sigmas = np.array(
            [np.nan if s is None else float(s) for s in self._sigmas],
            dtype="double",
        )
        sigma_cutoff_val = (
            -1.0 if self._sigma_cutoff is None else float(self._sigma_cutoff)
        )
        sigma_cutoffs = np.full(len(self._sigmas), sigma_cutoff_val, dtype="double")

        temperatures_thz = np.array(
            self._temperatures * get_physical_units().KB / get_physical_units().THzToEv,
            dtype="double",
        )

        self._rust_cache = {
            "sigmas": sigmas,
            "sigma_cutoffs": sigma_cutoffs,
            "relative_grid_address": relative_grid_address,
            "bzg2grg": np.ascontiguousarray(pp.bz_grid.bzg2grg, dtype="int64"),
            "reciprocal_rotations": np.ascontiguousarray(
                pp.bz_grid.rotations, dtype="int64"
            ),
            "is_time_reversal": True,
            "swappable": True,
            "is_mesh_symmetry": bool(pp.is_mesh_symmetry),
            "reciprocal_lattice": np.ascontiguousarray(reduced_basis, dtype="double"),
            "bz_triplets_q_mat": bz_triplets_q_mat,
            "frequencies": np.ascontiguousarray(frequencies, dtype="double"),
            "eigenvectors": np.ascontiguousarray(eigenvectors, dtype="complex128"),
            "bz_grid_addresses": np.ascontiguousarray(
                pp.bz_grid.addresses, dtype="int64"
            ),
            "bz_map": np.ascontiguousarray(pp.bz_grid.gp_map, dtype="int64"),
            "bz_grid_type": int(pp.bz_grid.store_dense_gp_map) + 1,
            "d_diag": np.ascontiguousarray(pp.bz_grid.D_diag, dtype="int64"),
            "q_matrix": np.ascontiguousarray(pp.bz_grid.Q, dtype="int64"),
            "fc3": np.ascontiguousarray(pp.fc3, dtype="double"),
            "fc3_nonzero_indices": np.ascontiguousarray(
                pp.fc3_nonzero_indices, dtype="byte"
            ),
            "svecs": np.ascontiguousarray(svecs, dtype="double"),
            "multiplicity": np.ascontiguousarray(multi, dtype="int64"),
            "masses": np.ascontiguousarray(pp.primitive.masses, dtype="double"),
            "p2s_map": np.ascontiguousarray(pp.primitive.p2s_map, dtype="int64"),
            "s2p_map": np.ascontiguousarray(pp.primitive.s2p_map, dtype="int64"),
            "band_indices": np.ascontiguousarray(pp.band_indices, dtype="int64"),
            "temperatures_thz": temperatures_thz,
            "symmetrize_fc3q": bool(pp.symmetrize_fc3q),
            "make_r0_average": bool(pp.make_r0_average),
            "all_shortest": np.ascontiguousarray(pp.all_shortest, dtype="byte"),
            "cutoff_frequency": float(pp.cutoff_frequency),
        }
        return self._rust_cache


class IsotopeScatteringSolver:
    """Compute isotope scattering linewidth at a grid point.

    Returns gamma_isotope as NDArray with shape ``(num_sigma, num_band0)``.

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

    def compute(self, grid_point: int) -> NDArray[np.double]:
        """Compute isotope linewidth at a grid point.

        Parameters
        ----------
        grid_point : int
            BZ grid point index.

        Returns
        -------
        ndarray of double, shape (num_sigma, num_band)
            Isotope scattering linewidth for all bands.
        """
        gamma_iso = []
        for sigma in self._sigmas:
            self._isotope.sigma = sigma
            self._isotope.set_grid_point(grid_point)
            self._isotope.run()
            gamma_iso.append(self._isotope.gamma)

        return np.array(gamma_iso, dtype="double", order="C")


def compute_bulk_boundary_scattering(
    group_velocities: NDArray[np.double],
    boundary_mfp: float,
) -> NDArray[np.double]:
    """Compute boundary scattering linewidth for all grid points at once.

    The formula is:

        gamma_boundary[gp, s] = |v_s| * 1e6 * Angstrom / (4 * pi * boundary_mfp)

    where ``boundary_mfp`` is in micrometres and ``|v_s|`` is the group
    velocity magnitude in THz*Angstrom.

    Parameters
    ----------
    group_velocities : ndarray of double, shape (num_gp, num_band0, 3)
        Group velocities in THz*Angstrom.
    boundary_mfp : float
        Boundary mean free path in micrometres.

    Returns
    -------
    ndarray of double, shape (num_gp, num_band0)
        Boundary scattering linewidth.

    """
    return (
        np.linalg.norm(group_velocities, axis=-1)
        * get_physical_units().Angstrom
        * 1e6
        / (4 * np.pi * boundary_mfp)
    )
