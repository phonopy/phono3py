"""Kappa accumulator for the Wigner transport equation."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from phonopy.physical_units import get_physical_units

from phono3py.conductivity.collision_matrix_solver import CollisionMatrixSolver
from phono3py.conductivity.context import ConductivityContext
from phono3py.conductivity.grid_point_data import (
    GridPointAggregates,
    GridPointResult,
    compute_effective_gamma,
)
from phono3py.conductivity.lbte_collision_provider import LBTECollisionResult
from phono3py.conductivity.utils import log_kappa_header, log_kappa_row
from phono3py.conductivity.wigner.kappa_formulas import (
    DEGENERATE_FREQUENCY_THRESHOLD_THZ,
)


class WignerRTAKappaAccumulator:
    """Kappa accumulator for the Wigner transport equation (WTE).

    Decomposes kappa into a population term (kappa_P_RTA) and a coherence
    term (kappa_C).  The total kappa is kappa_TOT_RTA = kappa_P_RTA + kappa_C.

    The P arrays are pre-allocated in ``prepare()``.  The C arrays are
    allocated lazily on the first grid point because their shape depends on
    ``num_band`` (nat3), which is not known at prepare time.

    Parameters
    ----------
    context : ConductivityContext
        Shared computation metadata (grid, phonon, symmetry, configuration).
    conversion_factor_WTE : float
        Unit conversion factor to W/(m*K) for the WTE formula.
    log_level : int
        Verbosity level.

    """

    def __init__(
        self,
        context: ConductivityContext,
        conversion_factor_WTE: float,
        log_level: int = 0,
    ) -> None:
        """Init method."""
        self._context = context
        self._conversion_factor_WTE = conversion_factor_WTE
        self._log_level = log_level

        # Kappa arrays (allocated in prepare()).
        self._kappa_P: NDArray[np.double]
        self._mode_kappa_P: NDArray[np.double]
        self._kappa_C: NDArray[np.double]
        self._mode_kappa_C: NDArray[np.double]
        self._velocity_operator: NDArray[np.cdouble]

    def prepare(
        self,
        num_sigma: int,
        num_temp: int,
        num_gp: int,
        num_band0: int,
        *,
        num_band: int | None = None,
    ) -> None:
        """Allocate per-grid-point and kappa arrays."""
        if num_band is None:
            num_band = num_band0
        self._kappa_P = np.zeros((num_sigma, num_temp, 6), dtype="double", order="C")
        self._mode_kappa_P = np.zeros(
            (num_sigma, num_temp, num_gp, num_band0, 6), dtype="double", order="C"
        )
        self._mode_kappa_C = np.zeros(
            (num_sigma, num_temp, num_gp, num_band0, num_band, 6),
            dtype="double",
            order="C",
        )
        self._velocity_operator = np.zeros(
            (num_gp, num_band0, num_band, 3), dtype="complex128", order="C"
        )

    def accumulate(self, i_gp: int, result: GridPointResult) -> None:
        """Store per-grid-point data and compute mode kappa at ``i_gp``."""
        assert result.group_velocities is not None
        assert result.gv_by_gv is not None
        assert result.vm_by_vm is not None
        assert result.heat_capacities is not None
        assert result.gamma is not None

        # Store raw velocity operator for per-grid-point HDF5 output.
        vel_op = result.extra.get("velocity_operator")
        if vel_op is not None:
            self._velocity_operator[i_gp] = vel_op

        frequencies = result.input.frequencies  # (num_band,) all bands
        vm_by_vm = result.vm_by_vm  # (num_band0, num_band, 6) complex
        cv = result.heat_capacities  # (num_temp, num_band0)

        gamma = compute_effective_gamma(
            result.gamma,
            gamma_isotope=result.gamma_isotope,
            gamma_boundary=result.gamma_boundary,
            gamma_elph=result.gamma_elph,
        )  # (num_sigma, num_temp, num_band0)
        num_sigma, num_temp, num_band0 = gamma.shape
        num_band = len(frequencies)
        THzToEv = get_physical_units().THzToEv

        mode_kappa_P = np.zeros((num_sigma, num_temp, num_band0, 6), dtype="double")
        mode_kappa_C = np.zeros(
            (num_sigma, num_temp, num_band0, num_band, 6), dtype="double"
        )

        for j in range(num_sigma):
            for k in range(num_temp):
                g = gamma[j, k]  # (num_band0,)
                cv_k = cv[k]  # (num_band0,)
                for s1 in range(num_band0):
                    freq_s1 = frequencies[s1]
                    if freq_s1 <= self._context.cutoff_frequency:
                        continue
                    for s2 in range(num_band):
                        freq_s2 = frequencies[s2]
                        if freq_s2 <= self._context.cutoff_frequency:
                            continue
                        pair = self._get_pair_contribution(
                            freq_s1=freq_s1,
                            freq_s2=freq_s2,
                            g_s1=g[s1],
                            g_s2=g[s2],
                            cv_s1=cv_k[s1],
                            cv_s2=cv_k[s2],
                            gv_by_gv_s1s2=vm_by_vm[s1, s2],
                            THzToEv=THzToEv,
                        )
                        if pair is None:
                            continue
                        contribution, is_population = pair
                        if is_population:
                            mode_kappa_P[j, k, s1] += 0.5 * contribution
                            mode_kappa_P[j, k, s2] += 0.5 * contribution
                        else:
                            mode_kappa_C[j, k, s1, s2] += contribution

        self._mode_kappa_P[:, :, i_gp, :, :] = mode_kappa_P
        self._mode_kappa_C[:, :, i_gp, :, :, :] = mode_kappa_C

    def _get_pair_contribution(
        self,
        *,
        freq_s1: float,
        freq_s2: float,
        g_s1: float,
        g_s2: float,
        cv_s1: float,
        cv_s2: float,
        gv_by_gv_s1s2: NDArray[np.cdouble],
        THzToEv: float,
    ) -> tuple[NDArray[np.double], bool] | None:
        """Return the kappa contribution of a (s1, s2) mode pair.

        Returns None when the contribution is identically zero.

        Returns
        -------
        (contribution, is_population) or None
            contribution : (6,) real array
            is_population : True when the pair is treated as a population
                            term (|freq_s1 - freq_s2| <
                            DEGENERATE_FREQUENCY_THRESHOLD_THZ).

        """
        hbar_omega_s1 = freq_s1 * THzToEv
        hbar_omega_s2 = freq_s2 * THzToEv
        hbar_gamma_s1 = 2.0 * g_s1 * THzToEv
        hbar_gamma_s2 = 2.0 * g_s2 * THzToEv

        gamma_sum = hbar_gamma_s1 + hbar_gamma_s2
        delta_omega = hbar_omega_s1 - hbar_omega_s2
        denominator = delta_omega**2 + 0.25 * gamma_sum**2
        if denominator == 0.0:
            return None
        lorentzian_div_hbar = (0.5 * gamma_sum) / denominator

        prefactor = (
            0.25
            * (hbar_omega_s1 + hbar_omega_s2)
            * (cv_s1 / hbar_omega_s1 + cv_s2 / hbar_omega_s2)
        )
        contribution = (
            gv_by_gv_s1s2
            * prefactor
            * lorentzian_div_hbar
            * self._conversion_factor_WTE
        ).real
        is_population = abs(freq_s1 - freq_s2) < DEGENERATE_FREQUENCY_THRESHOLD_THZ
        return contribution, is_population

    def finalize(self, aggregates: GridPointAggregates) -> None:
        """Compute kappa_P and kappa_C from mode_kappa arrays."""
        num_sampling_grid_points = aggregates.num_sampling_grid_points
        if num_sampling_grid_points > 0:
            self._kappa_P = (
                np.sum(self._mode_kappa_P, axis=(2, 3)) / num_sampling_grid_points
            )
            self._kappa_C = (
                np.sum(self._mode_kappa_C, axis=(2, 3, 4)) / num_sampling_grid_points
            )

    @property
    def kappa(self) -> NDArray[np.double]:
        """Return total Wigner kappa (kappa_P + kappa_C)."""
        return self._kappa_P + self._kappa_C

    @property
    def kappa_TOT_RTA(self) -> NDArray[np.double]:
        """Return total Wigner kappa (same as ``kappa``)."""
        return self.kappa

    @property
    def kappa_P_RTA(self) -> NDArray[np.double]:
        """Return population kappa, shape (num_sigma, num_temp, 6)."""
        return self._kappa_P

    @property
    def kappa_C(self) -> NDArray[np.double]:
        """Return coherence kappa, shape (num_sigma, num_temp, 6)."""
        return self._kappa_C

    @property
    def mode_kappa(self) -> NDArray[np.double]:
        """Return population mode kappa.

        Shape: (num_sigma, num_temp, num_gp, num_band0, 6).

        """
        return self._mode_kappa_P

    @property
    def mode_kappa_C(self) -> NDArray[np.double]:
        """Return coherence mode kappa.

        Shape: (num_sigma, num_temp, num_gp, num_band0, num_band, 6).

        """
        return self._mode_kappa_C

    # ------------------------------------------------------------------
    # Properties — per-grid-point data
    # ------------------------------------------------------------------

    def get_extra_grid_point_output(self, i: int) -> dict[str, Any]:
        """Return per-grid-point extra data for HDF5 output.

        Returns the velocity operator at grid point ``i`` as
        ``{"velocity_operator": ...}``.

        """
        return {"velocity_operator": self._velocity_operator[i]}

    def log_kappa(
        self,
        num_ignored_phonon_modes: NDArray[np.int64] | None = None,
        num_phonon_modes: int | None = None,
    ) -> None:
        """Print K_P, K_C, K_T rows for the Wigner-RTA conductivity."""
        if not self._log_level:
            return

        kappa_tot = self.kappa
        show_ipm = (
            self._log_level > 1
            and num_ignored_phonon_modes is not None
            and num_phonon_modes is not None
        )

        for i, sigma in enumerate(self._context.sigmas):
            log_kappa_header(sigma, show_ipm=show_ipm)
            for j, t in enumerate(self._context.temperatures):
                ipm = (
                    int(num_ignored_phonon_modes[i, j])
                    if show_ipm and num_ignored_phonon_modes is not None
                    else None
                )
                log_kappa_row("K_P\t", t, self._kappa_P[i, j], ipm, num_phonon_modes)
            print(" ")
            for j, t in enumerate(self._context.temperatures):
                ipm = (
                    int(num_ignored_phonon_modes[i, j])
                    if show_ipm and num_ignored_phonon_modes is not None
                    else None
                )
                log_kappa_row("K_C\t", t, self._kappa_C[i, j], ipm, num_phonon_modes)
            print(" ")
            for j, t in enumerate(self._context.temperatures):
                ipm = (
                    int(num_ignored_phonon_modes[i, j])
                    if show_ipm and num_ignored_phonon_modes is not None
                    else None
                )
                log_kappa_row("K_T\t", t, kappa_tot[i, j], ipm, num_phonon_modes)
            print("", flush=True)

    def get_extra_kappa_output(self) -> dict[str, Any]:
        """Return Wigner-specific extra arrays keyed by HDF5 dataset name.

        Sigma-dependent values have sigma as their first axis; the caller
        slices them with ``value[i_sigma]``.  Sigma-independent values
        (e.g. velocity_operator) are written as-is for every sigma.

        """
        return {
            "kappa_TOT_RTA": self.kappa_TOT_RTA,
            "kappa_P_RTA": self._kappa_P,
            "kappa_C": self._kappa_C,
            "mode_kappa_P_RTA": self._mode_kappa_P,
            "mode_kappa_C": self._mode_kappa_C,
            "velocity_operator": self._velocity_operator,
        }


class WignerLBTEKappaAccumulator:
    """LBTE accumulator with added Wigner coherence (C) term.

    Composes a CollisionMatrixSolver for the standard LBTE solve (P-term) and
    adds the coherence (C) term from the stored velocity operator outer
    products and linewidths.

    Stage 1 (per-grid-point): accumulate() stores the velocity operator outer
    product and heat capacities, then delegates collision data to the solver.

    Stage 2 (global): finalize() calls solver.solve() for the P-term kappa,
    then computes the C-term from the stored outer products and linewidths.

    Parameters
    ----------
    solver : CollisionMatrixSolver
        Shared solver for the P-term (standard LBTE).
    context : ConductivityContext
        Shared computation metadata (grid, phonon, symmetry, configuration).
    conversion_factor_WTE : float
        Unit conversion factor for the coherence term.
    is_reducible_collision_matrix : bool, optional
        When True the C-term is not computed (not implemented).  Default False.
    log_level : int, optional
        Verbosity level.  Default 0.

    """

    def __init__(
        self,
        solver: CollisionMatrixSolver,
        context: ConductivityContext,
        conversion_factor_WTE: float,
        is_reducible_collision_matrix: bool = False,
        log_level: int = 0,
    ) -> None:
        """Init method."""
        self._solver = solver
        self._context = context
        self._conversion_factor_WTE = conversion_factor_WTE
        self._is_reducible = is_reducible_collision_matrix
        self._log_level = log_level

        # Per-grid-point storage (lazily allocated in accumulate()).
        self._velocity_operator: NDArray[np.cdouble] | None = None

        # C-term output arrays (populated in finalize()).
        self._kappa_C: NDArray[np.double] | None = None
        self._mode_kappa_C: NDArray[np.cdouble] | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def prepare(self) -> None:
        """Allocate accumulator arrays."""
        self._solver.prepare()

    def accumulate(
        self,
        i_gp: int,
        collision_result: LBTECollisionResult,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Store per-grid-point Stage 1 data and delegate to the solver.

        Parameters
        ----------
        i_gp : int
            Loop index over ir_grid_points (0-based).
        collision_result : LBTECollisionResult
            Result from LBTECollisionProvider.compute().
        extra : dict or None
            Plugin-specific data from the velocity provider.  Expected keys:
            ``velocity_operator`` (num_band0, nat3, 3) complex.

        """
        velocity_operator = extra.get("velocity_operator") if extra else None

        if velocity_operator is not None:
            if self._velocity_operator is None:
                num_ir = len(self._context.ir_grid_points)
                self._velocity_operator = np.zeros(
                    (num_ir,) + velocity_operator.shape, dtype="complex128"
                )
            self._velocity_operator[i_gp] = velocity_operator
        self._solver.store(i_gp, collision_result)

    def finalize(
        self,
        aggregates: GridPointAggregates,
        suppress_kappa_log: bool = False,
    ) -> None:
        """Finalize P-term via solver, then compute C-term.

        The standard LBTE kappa table is suppressed (replaced by the Wigner
        table printed at the end of this method) when log_level > 0.

        """
        self._solver.solve(
            aggregates,
            suppress_kappa_log=bool(self._log_level),
        )
        self._compute_coherence_kappa(aggregates)
        if self._log_level:
            self._log_wigner_kappa()

    # ------------------------------------------------------------------
    # Properties — delegated to solver (LBTE P-term results)
    # ------------------------------------------------------------------

    @property
    def kappa(self) -> NDArray[np.double]:
        """Return LBTE thermal conductivity, shape (num_sigma, num_temp, 6)."""
        return self._solver.kappa

    @property
    def kappa_RTA(self) -> NDArray[np.double]:
        """Return RTA thermal conductivity, shape (num_sigma, num_temp, 6)."""
        return self._solver.kappa_RTA

    @property
    def mode_kappa(self) -> NDArray[np.double]:
        """Return mode LBTE kappa."""
        return self._solver.mode_kappa

    @property
    def mode_kappa_RTA(self) -> NDArray[np.double]:
        """Return mode RTA kappa."""
        return self._solver.mode_kappa_RTA

    @property
    def collision_matrix(self) -> NDArray[np.double] | None:
        """Return assembled collision matrix."""
        return self._solver.collision_matrix

    @collision_matrix.setter
    def collision_matrix(self, value: NDArray[np.double] | None) -> None:
        """Set collision matrix (used when reading from file)."""
        self._solver.collision_matrix = value

    @property
    def collision_eigenvalues(self) -> NDArray[np.double] | None:
        """Return eigenvalues of collision matrix."""
        return self._solver.collision_eigenvalues

    @property
    def f_vectors(self) -> NDArray[np.double] | None:
        """Return f-vectors, shape (num_gp, num_band0, 3)."""
        return self._solver.f_vectors

    @property
    def mfp(self) -> NDArray[np.double] | None:
        """Return mean free path."""
        return self._solver.mfp

    def get_main_diagonal(
        self, i_gp: int, i_sigma: int, i_temp: int
    ) -> NDArray[np.double]:
        """Return total scattering rate at a grid point."""
        return self._solver.get_main_diagonal(i_gp, i_sigma, i_temp)

    # ------------------------------------------------------------------
    # Properties — Wigner-specific aliases
    # ------------------------------------------------------------------

    @property
    def kappa_P_exact(self) -> NDArray[np.double]:
        """Return LBTE kappa (P-exact term) — alias for kappa."""
        return self._solver.kappa

    @property
    def kappa_P_RTA(self) -> NDArray[np.double]:
        """Return RTA kappa (P-RTA term) — alias for kappa_RTA."""
        return self._solver.kappa_RTA

    @property
    def mode_kappa_P_exact(self) -> NDArray[np.double]:
        """Return mode LBTE kappa (P-exact term) — alias for mode_kappa."""
        return self._solver.mode_kappa

    @property
    def mode_kappa_P_RTA(self) -> NDArray[np.double]:
        """Return mode RTA kappa (P-RTA term) — alias for mode_kappa_RTA."""
        return self._solver.mode_kappa_RTA

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

    def get_extra_kappa_output(self) -> dict[str, NDArray | None]:
        """Return Wigner-specific LBTE kappa arrays keyed by HDF5 dataset name.

        Keys
        ----
        kappa_TOT_exact   : (num_sigma, num_temp, 6)
        kappa_TOT_RTA     : (num_sigma, num_temp, 6)
        kappa_P_exact     : (num_sigma, num_temp, 6)
        kappa_P_RTA       : (num_sigma, num_temp, 6)
        kappa_C           : (num_sigma, num_temp, 6) or None
        mode_kappa_P_exact : (num_sigma, num_temp, num_gp, num_band0, 6)
        mode_kappa_P_RTA  : (num_sigma, num_temp, num_gp, num_band0, 6)
        mode_kappa_C      : (num_sigma, num_temp, num_gp, num_band0, num_band, 6)
                            or None

        """
        kappa_P_exact = self._solver.kappa
        kappa_P_RTA = self._solver.kappa_RTA
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
            "mode_kappa_P_exact": self._solver.mode_kappa,
            "mode_kappa_P_RTA": self._solver.mode_kappa_RTA,
            "mode_kappa_C": None if mode_kappa_C is None else mode_kappa_C.real,
            "velocity_operator": self._velocity_operator,
        }

    # ------------------------------------------------------------------
    # Private: C-term computation
    # ------------------------------------------------------------------

    def _compute_coherence_kappa(self, aggregates: GridPointAggregates) -> None:
        """Compute the Wigner coherence (C) term of thermal conductivity."""
        if self._is_reducible:
            print(
                " WARNING: Coherences conductivity not implemented for "
                "is_reducible_collision_matrix=True"
            )
            return
        vm_by_vm = aggregates.vm_by_vm
        mode_cv = aggregates.mode_heat_capacities
        if vm_by_vm is None or mode_cv is None:
            return

        THzToEv = get_physical_units().THzToEv
        num_sigma = len(self._context.sigmas)
        num_temp = len(self._context.temperatures)
        num_ir = len(self._context.ir_grid_points)
        num_band0 = len(self._context.band_indices)
        num_band = self._context.frequencies.shape[1]

        mode_kappa_C = np.zeros(
            (num_sigma, num_temp, num_ir, num_band0, num_band, 6), dtype="complex128"
        )

        for i_sigma in range(num_sigma):
            for i_temp in range(num_temp):
                for i_gp in range(num_ir):
                    gp = int(self._context.ir_grid_points[i_gp])
                    g = self._solver.get_main_diagonal(i_gp, i_sigma, i_temp) * 2.0
                    frequencies = self._context.frequencies[gp]
                    cv = mode_cv[i_temp, i_gp, :]
                    gv_by_gv_op = vm_by_vm[i_gp]

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
        self._kappa_C = (
            mode_kappa_C.sum(axis=(2, 3, 4)) / aggregates.num_sampling_grid_points
        ).real

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

        Returns None when the pair is degenerate or below the cutoff frequency.

        """
        cutoff = self._context.cutoff_frequency
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
        denominator = delta_omega**2 + 0.25 * gamma_sum**2
        lorentzian_div_hbar = (0.5 * gamma_sum) / denominator
        prefactor = (
            0.25
            * (hbar_omega_s1 + hbar_omega_s2)
            * (cv_s1 / hbar_omega_s1 + cv_s2 / hbar_omega_s2)
        )
        factor = lorentzian_div_hbar * self._conversion_factor_WTE
        return gv_by_gv_s1s2 * prefactor * factor

    def _log_wigner_kappa(self) -> None:
        """Print Wigner LBTE kappa table (K_P_exact, K_P_RTA, K_C, K_TOT)."""
        kappa_P_exact = self._solver.kappa
        kappa_P_RTA = self._solver.kappa_RTA
        kappa_C = self._kappa_C

        for i_sigma in range(len(self._context.sigmas)):
            for i_temp, t in enumerate(self._context.temperatures):
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
