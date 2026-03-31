"""WignerLBTECalculator: LBTE with Wigner coherence term."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray
from phonopy.physical_units import get_physical_units

from phono3py.conductivity.grid_point_data import GridPointInput, make_grid_point_input
from phono3py.conductivity.heat_capacity_providers import ModeHeatCapacityProvider
from phono3py.conductivity.lbte_collision_provider import LBTECollisionProvider
from phono3py.conductivity.lbte_kappa_accumulator import LBTEKappaAccumulator
from phono3py.conductivity.scattering_providers import IsotopeScatteringProvider
from phono3py.conductivity.utils import (
    show_grid_point_frequencies_gv,
    show_grid_point_header,
)
from phono3py.conductivity.wigner.formulas import DEGENERATE_FREQUENCY_THRESHOLD_THZ
from phono3py.conductivity.wigner.providers import VelocityOperatorProvider
from phono3py.other.isotope import Isotope
from phono3py.phonon.grid import (
    BZGrid,
    get_ir_grid_points,
    get_qpoints_from_bz_grid_points,
)
from phono3py.phonon3.interaction import Interaction


class WignerLBTECalculator:
    """LBTE thermal conductivity calculator with Wigner coherence (C) term.

    Extends the standard LBTE calculation (two-stage design) with the
    population (P) and coherence (C) decomposition from the Wigner Transport
    Equation.

    Stage 1 (per-grid-point): collision matrix rows, velocity operators, and
    heat capacities are accumulated.  Stage 2 (global): LBTEKappaAccumulator
    solves the collision matrix for the P-term kappa.  Stage 3: the C-term is
    computed from the stored velocity operator outer products and linewidths.

    Properties
    ----------
    kappa_P_exact : NDArray
        Full LBTE (Boltzmann) kappa, shape (num_sigma, num_temp, 6).
    kappa_P_RTA : NDArray
        RTA kappa, shape (num_sigma, num_temp, 6).
    kappa_C : NDArray
        Coherence kappa (real part), shape (num_sigma, num_temp, 6).
    mode_kappa_P_exact : NDArray
        Per-mode P-term LBTE kappa.
    mode_kappa_P_RTA : NDArray
        Per-mode P-term RTA kappa.
    mode_kappa_C : NDArray
        Per-mode C-term kappa (complex).

    Parameters
    ----------
    pp : Interaction
        Ph-ph interaction object.  init_dynamical_matrix must have been called.
    velocity_provider : VelocityOperatorProvider
        Provides group velocities (diagonal) and velocity operator outer
        product for each grid point.
    cv_provider : ModeHeatCapacityProvider
        Computes mode heat capacities at each grid point.
    collision_provider : LBTECollisionProvider
        Computes gamma and one collision matrix row per grid point.
    accumulator : LBTEKappaAccumulator
        Owns the global collision matrix and solves for the P-term kappa.
    temperatures : NDArray[np.double]
        Temperatures in Kelvin, shape (num_temp,).
    sigmas : list of float or None
        Smearing widths.  Empty list selects the tetrahedron method.
    conversion_factor_WTE : float
        Unit conversion factor for the WTE coherence term.
    is_isotope : bool, optional
        Include isotope scattering.  Default False.
    mass_variances : array-like or None, optional
        Mass variances for isotope scattering.  Default None.
    is_reducible_collision_matrix : bool, optional
        When True, coherence term is not computed (not implemented).
    is_full_pp : bool, optional
        Compute averaged ph-ph interaction.  Default False.
    log_level : int, optional
        Verbosity level.  Default 0.
    """

    def __init__(
        self,
        pp: Interaction,
        velocity_provider: VelocityOperatorProvider,
        cv_provider: ModeHeatCapacityProvider,
        collision_provider: LBTECollisionProvider,
        accumulator: LBTEKappaAccumulator,
        temperatures: NDArray[np.double],
        sigmas: list[float | None],
        conversion_factor_WTE: float,
        is_isotope: bool = False,
        mass_variances: Sequence[float] | NDArray[np.double] | None = None,
        is_reducible_collision_matrix: bool = False,
        is_full_pp: bool = False,
        log_level: int = 0,
    ) -> None:
        """Init method."""
        self._pp = pp
        self._velocity_provider = velocity_provider
        self._cv_provider = cv_provider
        self._collision_provider = collision_provider
        self._accumulator = accumulator
        self._temperatures = np.asarray(temperatures, dtype="double")
        self._sigmas = sigmas
        self._conversion_factor_WTE = conversion_factor_WTE
        self._is_reducible_collision_matrix = is_reducible_collision_matrix
        self._is_full_pp = is_full_pp
        self._log_level = log_level

        self._pp.nac_q_direction = None
        self._pp.run_phonon_solver_at_gamma()
        self._frequencies, self._eigenvectors, self._phonon_done = (
            self._pp.get_phonons()
        )
        if not self._pp.phonon_all_done:
            self._pp.run_phonon_solver()

        ir_gps, self._grid_weights = self._get_ir_grid_points()
        self._ir_grid_points = np.array(self._pp.bz_grid.grg2bzg[ir_gps], dtype="int64")

        self._isotope_provider: IsotopeScatteringProvider | None = None
        if is_isotope or mass_variances is not None:
            self._isotope_provider = self._build_isotope_provider(mass_variances)

        self._num_sampling_grid_points = 0
        self._grid_point_count = 0

        num_ir = len(self._ir_grid_points)
        num_band0 = len(pp.band_indices)
        num_band = len(pp.primitive) * 3
        num_temp = len(self._temperatures)

        # Per-grid-point storage for C-term (Stage 1).
        self._gv_by_gv_operator = np.zeros(
            (num_ir, num_band0, num_band, 6), dtype="complex128"
        )
        self._mode_cv = np.zeros((num_ir, num_temp, num_band0), dtype="double")

        # C-term output arrays (populated after Stage 3).
        self._kappa_C: NDArray[np.double] | None = None
        self._mode_kappa_C: NDArray[np.cdouble] | None = None

        self._accumulator.prepare(is_full_pp=self._is_full_pp)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        on_grid_point: Callable[[int], None] | None = None,
    ) -> None:
        """Run all stages and compute P-term and C-term kappa.

        Parameters
        ----------
        on_grid_point : callable or None, optional
            Called with the grid-point loop index after each grid point is
            processed.  Used for per-grid-point file writes.
        """
        if self._log_level:
            print(
                "==================== Lattice thermal conductivity (LBTE, Wigner) "
                "===================="
            )

        self._prepare_isotope_phonons()

        self._num_sampling_grid_points = 0
        self._grid_point_count = 0

        for i_gp in range(len(self._ir_grid_points)):
            self._run_at_grid_point(i_gp)
            self._grid_point_count = i_gp + 1
            if on_grid_point is not None:
                on_grid_point(i_gp)

        if self._log_level:
            print(
                "=================== End of collection of collisions "
                "==================="
            )

        self._accumulator.finalize(
            self._num_sampling_grid_points,
            suppress_kappa_log=bool(self._log_level),
        )
        self._compute_coherence_kappa()
        if self._log_level:
            self._log_wigner_kappa()

    def set_kappa_at_sigmas(self) -> None:
        """Finalize kappa from a pre-loaded collision matrix (read-from-file path).

        The P-term is computed by accumulator.finalize().  The C-term uses
        velocity operator data that must have been collected during Stage 1.
        If Stage 1 was not run (read-from-file without run()), the C-term
        will be zero.
        """
        self._accumulator.finalize(
            int(self._grid_weights.sum()),
            suppress_kappa_log=bool(self._log_level),
        )
        self._compute_coherence_kappa()
        if self._log_level:
            self._log_wigner_kappa()

    def delete_gp_collision_and_pp(self) -> None:
        """No-op: memory management compatibility method."""

    # ------------------------------------------------------------------
    # Properties — grid / phonon metadata
    # ------------------------------------------------------------------

    @property
    def mesh_numbers(self) -> NDArray[np.int64]:
        """Return BZ mesh numbers."""
        return self._pp.mesh_numbers

    @property
    def bz_grid(self) -> BZGrid:
        """Return BZ grid object."""
        return self._pp.bz_grid

    @property
    def grid_points(self) -> NDArray[np.int64]:
        """Return irreducible BZ grid point indices."""
        return self._ir_grid_points

    @property
    def qpoints(self) -> NDArray[np.double]:
        """Return q-point coordinates of the irreducible grid points."""
        return np.array(
            get_qpoints_from_bz_grid_points(self._ir_grid_points, self._pp.bz_grid),
            dtype="double",
            order="C",
        )

    @property
    def grid_weights(self) -> NDArray[np.int64]:
        """Return symmetry weights of the irreducible grid points."""
        return self._grid_weights

    @property
    def temperatures(self) -> NDArray[np.double]:
        """Return temperatures in Kelvin."""
        return self._temperatures

    @temperatures.setter
    def temperatures(self, value: Sequence[float] | NDArray[np.double]) -> None:
        """Set temperatures and re-allocate accumulator arrays."""
        self._temperatures = np.asarray(value, dtype="double")
        self._accumulator.temperatures = self._temperatures

    @property
    def sigmas(self) -> list[float | None]:
        """Return smearing widths."""
        return self._sigmas

    @property
    def sigma_cutoff_width(self) -> float | None:
        """Return smearing cutoff width."""
        return self._collision_provider._sigma_cutoff

    @property
    def frequencies(self) -> NDArray[np.double]:
        """Return phonon frequencies at irreducible grid points."""
        assert self._frequencies is not None
        return self._frequencies[self._ir_grid_points]

    @property
    def grid_point_count(self) -> int:
        """Return number of grid points processed so far."""
        return self._grid_point_count

    @property
    def number_of_sampling_grid_points(self) -> int:
        """Return total BZ grid points represented (sum of k-star orders)."""
        return self._num_sampling_grid_points

    # ------------------------------------------------------------------
    # Properties — P-term (delegated to accumulator)
    # ------------------------------------------------------------------

    @property
    def kappa_P_exact(self) -> NDArray[np.double]:
        """Return full LBTE kappa (P-term), shape (num_sigma, num_temp, 6)."""
        return self._accumulator.kappa

    @property
    def kappa_P_RTA(self) -> NDArray[np.double]:
        """Return RTA kappa (P-term), shape (num_sigma, num_temp, 6)."""
        return self._accumulator.kappa_RTA

    @property
    def mode_kappa_P_exact(self) -> NDArray[np.double]:
        """Return per-mode LBTE kappa (P-term)."""
        return self._accumulator.mode_kappa

    @property
    def mode_kappa_P_RTA(self) -> NDArray[np.double]:
        """Return per-mode RTA kappa (P-term)."""
        return self._accumulator.mode_kappa_RTA

    @property
    def gamma(self) -> NDArray[np.double]:
        """Return ph-ph gamma, shape (num_sigma, num_temp, num_gp, num_band0)."""
        return self._accumulator.gamma

    @gamma.setter
    def gamma(self, value: NDArray[np.double]) -> None:
        """Set gamma (for loading from file)."""
        self._accumulator.gamma = value

    @property
    def gamma_isotope(self) -> NDArray[np.double] | None:
        """Return isotope gamma, shape (num_sigma, num_gp, num_band0)."""
        return self._accumulator.gamma_iso

    @property
    def collision_matrix(self) -> NDArray[np.double] | None:
        """Return assembled collision matrix."""
        return self._accumulator.collision_matrix

    @collision_matrix.setter
    def collision_matrix(self, value: NDArray[np.double] | None) -> None:
        """Set collision matrix (for loading from file)."""
        self._accumulator.collision_matrix = value

    @property
    def collision_eigenvalues(self) -> NDArray[np.double] | None:
        """Return eigenvalues of collision matrix."""
        return self._accumulator.collision_eigenvalues

    @property
    def averaged_pp_interaction(self) -> NDArray[np.double] | None:
        """Return averaged ph-ph interaction."""
        return self._accumulator.averaged_pp_interaction

    @property
    def boundary_mfp(self) -> float | None:
        """Return boundary mean free path in micrometres."""
        return self._accumulator.boundary_mfp

    @property
    def mode_heat_capacities(self) -> NDArray[np.double]:
        """Return mode heat capacities, shape (num_temp, num_gp, num_band0)."""
        return self._accumulator.mode_heat_capacities

    @property
    def f_vectors(self) -> NDArray[np.double] | None:
        """Return f-vectors."""
        return self._accumulator.f_vectors

    @property
    def mfp(self) -> NDArray[np.double] | None:
        """Return mean free path."""
        return self._accumulator.mfp

    def get_frequencies_all(self) -> NDArray[np.double]:
        """Return phonon frequencies on the full BZ grid."""
        assert self._frequencies is not None
        return self._frequencies[self._pp.bz_grid.grg2bzg]

    # ------------------------------------------------------------------
    # Properties — C-term
    # ------------------------------------------------------------------

    @property
    def kappa_C(self) -> NDArray[np.double] | None:
        """Return coherence kappa, shape (num_sigma, num_temp, 6)."""
        return self._kappa_C

    @property
    def mode_kappa_C(self) -> NDArray[np.cdouble] | None:
        """Return per-mode coherence kappa (complex)."""
        return self._mode_kappa_C

    def get_extra_kappa_output(self) -> dict[str, NDArray[np.double] | None]:
        """Return Wigner-specific LBTE kappa arrays keyed by HDF5 dataset name.

        Each value has sigma as its first axis so that callers can slice by
        sigma index with ``value[i_sigma]``.

        Keys
        ----
        kappa_TOT_exact   : (num_sigma, num_temp, 6) -- P_exact + C
        kappa_TOT_RTA     : (num_sigma, num_temp, 6) -- P_RTA + C
        kappa_P_exact     : (num_sigma, num_temp, 6)
        kappa_P_RTA       : (num_sigma, num_temp, 6)
        kappa_C           : (num_sigma, num_temp, 6) or None
        mode_kappa_P_exact : (num_sigma, num_temp, num_gp, num_band0, 6)
        mode_kappa_P_RTA  : (num_sigma, num_temp, num_gp, num_band0, 6)
        mode_kappa_C      : (num_sigma, num_temp, num_gp, num_band0, num_band, 6)
                            or None

        """
        kappa_P_exact = self._accumulator.kappa
        kappa_P_RTA = self._accumulator.kappa_RTA
        kappa_C = self._kappa_C
        if kappa_C is not None:
            kappa_TOT_exact: NDArray[np.double] | None = kappa_P_exact + kappa_C
            kappa_TOT_RTA: NDArray[np.double] | None = kappa_P_RTA + kappa_C
        else:
            kappa_TOT_exact = kappa_P_exact
            kappa_TOT_RTA = kappa_P_RTA
        mode_kappa_C = self._mode_kappa_C
        return {
            "kappa_TOT_exact": kappa_TOT_exact,
            "kappa_TOT_RTA": kappa_TOT_RTA,
            "kappa_P_exact": kappa_P_exact,
            "kappa_P_RTA": kappa_P_RTA,
            "kappa_C": kappa_C,
            "mode_kappa_P_exact": self._accumulator.mode_kappa,
            "mode_kappa_P_RTA": self._accumulator.mode_kappa_RTA,
            "mode_kappa_C": None if mode_kappa_C is None else mode_kappa_C.real,
        }

    # ------------------------------------------------------------------
    # Private: grid
    # ------------------------------------------------------------------

    def _get_ir_grid_points(self) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        ir_gps, ir_weights, _ = get_ir_grid_points(self._pp.bz_grid)
        return np.array(ir_gps, dtype="int64"), np.array(ir_weights, dtype="int64")

    # ------------------------------------------------------------------
    # Private: isotope
    # ------------------------------------------------------------------

    def _build_isotope_provider(
        self,
        mass_variances: Sequence[float] | NDArray[np.double] | None,
    ) -> IsotopeScatteringProvider:
        isotope = Isotope(
            self._pp.mesh_numbers,
            self._pp.primitive,
            mass_variances=mass_variances,
            bz_grid=self._pp.bz_grid,
            frequency_factor_to_THz=self._pp.frequency_factor_to_THz,
            symprec=self._pp.primitive_symmetry.tolerance,
            cutoff_frequency=self._pp.cutoff_frequency,
            lapack_zheev_uplo=self._pp.lapack_zheev_uplo,
        )
        return IsotopeScatteringProvider(
            isotope, self._sigmas, log_level=self._log_level
        )

    def _prepare_isotope_phonons(self) -> None:
        if self._isotope_provider is None:
            return
        assert self._frequencies is not None
        assert self._eigenvectors is not None
        assert self._phonon_done is not None
        self._isotope_provider.isotope.set_phonons(
            self._frequencies,
            self._eigenvectors,
            self._phonon_done,
            dm=self._pp.dynamical_matrix,
        )

    # ------------------------------------------------------------------
    # Private: per-grid-point computation (Stage 1)
    # ------------------------------------------------------------------

    def _make_grid_point_input(self, i_gp: int) -> GridPointInput:
        assert self._frequencies is not None
        assert self._eigenvectors is not None
        return make_grid_point_input(
            grid_point=int(self._ir_grid_points[i_gp]),
            grid_weight=int(self._grid_weights[i_gp]),
            frequencies=self._frequencies,
            eigenvectors=self._eigenvectors,
            bz_grid=self._pp.bz_grid,
            band_indices=np.asarray(self._pp.band_indices, dtype="int64"),
        )

    def _show_log_header(self, i_gp: int) -> None:
        if not self._log_level:
            return
        mass_variances = (
            self._isotope_provider.isotope.mass_variances
            if self._isotope_provider is not None
            else None
        )
        show_grid_point_header(
            bzgp=self._ir_grid_points[i_gp],
            i_gp=i_gp,
            num_gps=len(self._ir_grid_points),
            bz_grid=self._pp.bz_grid,
            boundary_mfp=self._accumulator.boundary_mfp,
            mass_variances=mass_variances,
        )

    def _show_log(self, i_gp: int, gv: NDArray[np.double]) -> None:
        bz_gp = self._ir_grid_points[i_gp]
        assert self._frequencies is not None
        frequencies = self._frequencies[bz_gp][self._pp.band_indices]
        show_grid_point_frequencies_gv(
            frequencies,
            gv,
            gv_delta_q=getattr(self._velocity_provider, "gv_delta_q", None),
        )

    def _log_wigner_kappa(self) -> None:
        """Print Wigner LBTE kappa table (K_P_exact, K_P_RTA, K_C, K_TOT).

        Called after Stage 3 (_compute_coherence_kappa) so that K_C is
        available.  The sigma header and diagonalize lines have already been
        printed by LBTEKappaAccumulator.finalize(); this method only adds the
        kappa value rows.
        """
        kappa_P_exact = self._accumulator.kappa
        kappa_P_RTA = self._accumulator.kappa_RTA
        kappa_C = self._kappa_C

        for i_sigma in range(len(self._sigmas)):
            for i_temp, t in enumerate(self._temperatures):
                if t <= 0:
                    continue
                print(
                    ("#%6s       " + " %-10s" * 6)
                    % ("         \t\t  T(K)", "xx", "yy", "zz", "yz", "xz", "xy")
                )
                print(
                    "K_P_exact\t\t"
                    + ("%7.1f " + " %10.3f" * 6)
                    % ((t,) + tuple(kappa_P_exact[i_sigma, i_temp]))
                )
                print(
                    "(K_P_RTA)\t\t"
                    + ("%7.1f " + " %10.3f" * 6)
                    % ((t,) + tuple(kappa_P_RTA[i_sigma, i_temp]))
                )
                if kappa_C is not None:
                    print(
                        "K_C      \t\t"
                        + ("%7.1f " + " %10.3f" * 6)
                        % ((t,) + tuple(kappa_C[i_sigma, i_temp]))
                    )
                    print(" ")
                    kappa_tot = (
                        kappa_P_exact[i_sigma, i_temp] + kappa_C[i_sigma, i_temp]
                    )
                    print(
                        "K_TOT=K_P_exact+K_C\t"
                        + ("%7.1f " + " %10.3f" * 6) % ((t,) + tuple(kappa_tot))
                    )
                print("-" * 76, flush=True)

    def _run_at_grid_point(self, i_gp: int) -> None:
        self._show_log_header(i_gp)
        gp_input = self._make_grid_point_input(i_gp)

        # Velocity operator: provides group velocities (diagonal) and
        # gv_by_gv_operator (off-diagonal outer product) for C-term.
        vel_result = self._velocity_provider.compute(gp_input)
        assert vel_result.group_velocities is not None
        assert vel_result.velocity_product is not None
        gv = vel_result.group_velocities
        self._num_sampling_grid_points += vel_result.num_sampling_grid_points
        self._gv_by_gv_operator[i_gp] = vel_result.velocity_product

        # Mode heat capacities (needed for both P-term accumulation and C-term).
        cv_result = self._cv_provider.compute(gp_input, self._temperatures)
        assert cv_result.heat_capacities is not None
        cv = cv_result.heat_capacities  # (num_temp, num_band0)
        self._mode_cv[i_gp] = cv

        # Collision matrix row + gamma (Stage 1).
        collision_result = self._collision_provider.compute(gp_input)

        # Isotope scattering (optional diagonal contribution).
        if self._isotope_provider is not None:
            iso_result = self._isotope_provider.compute_gamma_isotope(gp_input)
            assert iso_result.gamma_isotope is not None
            self._accumulator.store_gamma_iso(
                i_gp,
                iso_result.gamma_isotope[:, self._pp.band_indices],
            )

        self._accumulator.accumulate(i_gp, collision_result, gv, cv)

        if self._log_level:
            self._show_log(i_gp, gv)

    # ------------------------------------------------------------------
    # Private: coherence term (Stage 3)
    # ------------------------------------------------------------------

    def _compute_coherence_kappa(self) -> None:
        """Compute the Wigner coherence (C) term of thermal conductivity.

        Uses the velocity operator outer product and linewidths from the
        collision matrix diagonal to evaluate the Lorentzian-broadened
        coherence contribution.

        For reducible collision matrix, the coherence term is not implemented
        and remains zero.
        """
        if self._is_reducible_collision_matrix:
            print(
                " WARNING: Coherences conductivity not implemented for "
                "is_reducible_collision_matrix=True"
            )
            return

        num_sigma = len(self._sigmas)
        num_temp = len(self._temperatures)
        num_ir = len(self._ir_grid_points)
        num_band0 = len(self._pp.band_indices)
        num_band = len(self._pp.primitive) * 3
        assert self._frequencies is not None

        mode_kappa_C = np.zeros(
            (num_sigma, num_temp, num_ir, num_band0, num_band, 6), dtype="complex128"
        )
        THzToEv = get_physical_units().THzToEv

        for i_sigma in range(num_sigma):
            for i_temp in range(num_temp):
                for i_gp in range(num_ir):
                    gp = int(self._ir_grid_points[i_gp])
                    # Total scattering rate (FWHM) at this grid point.
                    g = self._accumulator.get_main_diagonal(i_gp, i_sigma, i_temp) * 2.0
                    frequencies = self._frequencies[gp]
                    cv = self._mode_cv[i_gp, i_temp, :]  # (num_band0,)
                    # shape: (num_band0, num_band, 6)
                    gv_by_gv_op = self._gv_by_gv_operator[i_gp]

                    for s1 in range(num_band0):
                        for s2 in range(num_band):
                            contrib = self._compute_pair_contribution(
                                freq_s1=float(frequencies[s1]),
                                freq_s2=float(frequencies[s2]),
                                linewidth_s1=float(g[s1]),
                                linewidth_s2=float(g[s2]),
                                cv_s1=float(cv[s1]),
                                cv_s2=float(cv[s2]),
                                gv_by_gv_s1s2=gv_by_gv_op[s1, s2, :],
                                THzToEv=THzToEv,
                            )
                            if contrib is None:
                                continue
                            mode_kappa_C[i_sigma, i_temp, i_gp, s1, s2] = contrib

        self._mode_kappa_C = mode_kappa_C
        N = self._num_sampling_grid_points
        if N == 0:
            N = int(self._grid_weights.sum())
        self._kappa_C = (mode_kappa_C.sum(axis=(2, 3, 4)) / N).real

    def _compute_pair_contribution(
        self,
        *,
        freq_s1: float,
        freq_s2: float,
        linewidth_s1: float,
        linewidth_s2: float,
        cv_s1: float,
        cv_s2: float,
        gv_by_gv_s1s2: NDArray[np.cdouble],
        THzToEv: float,
    ) -> NDArray[np.cdouble] | None:
        """Return coherence contribution for a single (s1, s2) pair.

        Returns None when the pair is degenerate or below the cutoff
        frequency, so the caller can skip it without modifying the output
        array.
        """
        cutoff = self._pp.cutoff_frequency
        if freq_s1 <= cutoff or freq_s2 <= cutoff:
            return None
        if np.abs(freq_s1 - freq_s2) <= DEGENERATE_FREQUENCY_THRESHOLD_THZ:
            return None

        hbar_omega_s1 = freq_s1 * THzToEv
        hbar_omega_s2 = freq_s2 * THzToEv
        hbar_gamma_s1 = linewidth_s1 * THzToEv
        hbar_gamma_s2 = linewidth_s2 * THzToEv

        gamma_sum = hbar_gamma_s1 + hbar_gamma_s2
        delta_omega = hbar_omega_s1 - hbar_omega_s2
        lorentzian_div_hbar = (0.5 * gamma_sum) / (delta_omega**2 + 0.25 * gamma_sum**2)
        prefactor = (
            0.25
            * (hbar_omega_s1 + hbar_omega_s2)
            * (cv_s1 / hbar_omega_s1 + cv_s2 / hbar_omega_s2)
        )
        factor = lorentzian_div_hbar * self._conversion_factor_WTE
        return gv_by_gv_s1s2 * prefactor * factor
