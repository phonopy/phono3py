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
    compute_effective_gamma,
)
from phono3py.conductivity.lbte_collision_provider import LBTECollisionResult
from phono3py.conductivity.utils import log_kappa_header, log_kappa_row

# Threshold in THz below which two modes are considered degenerate and treated
# as a population (diagonal) term instead of a coherence (off-diagonal) term.
DEGENERATE_FREQUENCY_THRESHOLD_THZ = 1e-4


def _get_conversion_factor_WTE(volume: float) -> float:
    """Return unit conversion factor for Wigner transport equation kappa.

    Parameters
    ----------
    volume : float
        Primitive-cell volume in Angstrom^3.

    Returns
    -------
    float
        Conversion factor in W/(m*K).

    """
    u = get_physical_units()
    return (u.THz * u.Angstrom) ** 2 * u.EV * u.Hbar / (volume * u.Angstrom**3)


class WignerRTAKappaAccumulator:
    """Kappa accumulator for the Wigner transport equation (WTE).

    Decomposes kappa into a population term (kappa_P_RTA) and a coherence
    term (kappa_C).  The total kappa is kappa_TOT_RTA = kappa_P_RTA + kappa_C.

    The P and C arrays are allocated in ``finalize()``.

    Parameters
    ----------
    context : ConductivityContext
        Shared computation metadata (grid, phonon, symmetry, configuration).
    volume : float
        Primitive-cell volume in Angstrom^3.
    log_level : int
        Verbosity level.

    """

    def __init__(
        self,
        context: ConductivityContext,
        volume: float,
        log_level: int = 0,
    ) -> None:
        """Init method."""
        self._context = context
        self._conversion_factor_WTE = _get_conversion_factor_WTE(volume)
        self._log_level = log_level

        self._kappa_P: NDArray[np.double] | None = None
        self._mode_kappa_P: NDArray[np.double] | None = None
        self._kappa_C: NDArray[np.double] | None = None
        self._mode_kappa_C: NDArray[np.double] | None = None
        self._velocity_operator: NDArray[np.cdouble] | None = None

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
        """Compute mode kappa (P + C) and kappa from aggregated data."""
        assert aggregates.vm_by_vm is not None

        # Store velocity operator from aggregates.extra.
        vel_op = aggregates.extra.get("velocity_operator")
        if vel_op is not None:
            self._velocity_operator = vel_op

        num_sigma, num_temp, num_gp, num_band = aggregates.gamma.shape
        self._kappa_P = np.zeros((num_sigma, num_temp, 6), dtype="double", order="C")
        self._mode_kappa_P = np.zeros(
            (num_sigma, num_temp, num_gp, num_band, 6), dtype="double", order="C"
        )
        self._mode_kappa_C = np.zeros(
            (num_sigma, num_temp, num_gp, num_band, num_band, 6),
            dtype="double",
            order="C",
        )

        self._compute_mode_kappa(aggregates)

        num_sampling_grid_points = aggregates.num_sampling_grid_points
        if num_sampling_grid_points > 0:
            self._kappa_P = (
                np.sum(self._mode_kappa_P, axis=(2, 3)) / num_sampling_grid_points
            )
            self._kappa_C = (
                np.sum(self._mode_kappa_C, axis=(2, 3, 4)) / num_sampling_grid_points
            )

    def _compute_mode_kappa(self, aggregates: GridPointAggregates) -> None:
        """Compute mode kappa P and C at all grid points."""
        assert aggregates.vm_by_vm is not None

        vm_by_vm = aggregates.vm_by_vm
        cv = aggregates.mode_heat_capacities
        gamma_eff = compute_effective_gamma(aggregates)
        num_sigma, num_temp, num_gp, num_band = gamma_eff.shape
        THzToEv = get_physical_units().THzToEv

        for i_gp in range(num_gp):
            gp = self._context.grid_points[i_gp]
            frequencies = self._context.frequencies[gp]
            num_band = len(frequencies)

            mode_kappa_P = np.zeros((num_sigma, num_temp, num_band, 6), dtype="double")
            mode_kappa_C = np.zeros(
                (num_sigma, num_temp, num_band, num_band, 6), dtype="double"
            )

            for j in range(num_sigma):
                for k in range(num_temp):
                    g = gamma_eff[j, k, i_gp]
                    cv_k = cv[k, i_gp]
                    for s1 in range(num_band):
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
                                gv_by_gv_s1s2=vm_by_vm[i_gp, s1, s2],
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

    @property
    def kappa(self) -> NDArray[np.double] | None:
        """Return total Wigner kappa (kappa_P + kappa_C)."""
        if self._kappa_P is None or self._kappa_C is None:
            return None

        return self._kappa_P + self._kappa_C

    @property
    def kappa_TOT_RTA(self) -> NDArray[np.double] | None:
        """Return total Wigner kappa (same as ``kappa``)."""
        return self.kappa

    @property
    def kappa_P_RTA(self) -> NDArray[np.double] | None:
        """Return population kappa, shape (num_sigma, num_temp, 6)."""
        return self._kappa_P

    @property
    def mode_kappa(self) -> NDArray[np.double] | None:
        """Return population mode kappa.

        Shape: (num_sigma, num_temp, num_gp, num_band0, 6).

        """
        return self._mode_kappa_P

    @property
    def kappa_C(self) -> NDArray[np.double] | None:
        """Return coherence kappa, shape (num_sigma, num_temp, 6)."""
        return self._kappa_C

    @property
    def mode_kappa_C(self) -> NDArray[np.double] | None:
        """Return coherence mode kappa.

        Shape: (num_sigma, num_temp, num_gp, num_band0, num_band, 6).

        """
        return self._mode_kappa_C

    # ------------------------------------------------------------------
    # Properties — per-grid-point data
    # ------------------------------------------------------------------

    def get_extra_grid_point_output(self) -> dict[str, Any]:
        """Return per-grid-point extra arrays for HDF5 output.

        Returns the full velocity operator array. The caller is responsible
        for slicing by grid-point index.

        """
        return {"velocity_operator": self._velocity_operator}

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
            "mode_kappa": self._mode_kappa_P,
            "mode_kappa_P_RTA": self._mode_kappa_P,
            "mode_kappa_C": self._mode_kappa_C,
            "velocity_operator": self._velocity_operator,
        }


class WignerLBTEKappaAccumulator:
    """LBTE accumulator with added Wigner coherence (C) term.

    Composes a CollisionMatrixSolver for the standard LBTE solve (P-term) and
    adds the coherence (C) term from the stored velocity operator outer
    products and linewidths.

    Stage 1 (per-grid-point): store() delegates collision data to the solver.

    Stage 2 (global): finalize() calls solver.solve() for the P-term kappa,
    then computes the C-term from the stored outer products and linewidths.

    Parameters
    ----------
    solver : CollisionMatrixSolver
        Shared solver for the P-term (standard LBTE).
    context : ConductivityContext
        Shared computation metadata (grid, phonon, symmetry, configuration).
    volume : float
        Primitive-cell volume in Angstrom^3.
    is_reducible_collision_matrix : bool, optional
        When True the C-term is not computed (not implemented).  Default False.
    log_level : int, optional
        Verbosity level.  Default 0.

    """

    def __init__(
        self,
        solver: CollisionMatrixSolver,
        context: ConductivityContext,
        volume: float,
        is_reducible_collision_matrix: bool = False,
        log_level: int = 0,
    ) -> None:
        """Init method."""
        self._solver = solver
        self._context = context
        self._conversion_factor_WTE = _get_conversion_factor_WTE(volume)
        self._is_reducible = is_reducible_collision_matrix
        self._log_level = log_level

        # Per-grid-point storage (set in finalize() from aggregates.extra).
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

    def store(
        self,
        i_gp: int,
        collision_result: LBTECollisionResult,
    ) -> None:
        """Store collision matrix row for this grid point.

        Parameters
        ----------
        i_gp : int
            Loop index over ir_grid_points (0-based).
        collision_result : LBTECollisionResult
            Result from LBTECollisionProvider.compute().

        """
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
        # Store velocity operator from aggregates.extra.
        vel_op = aggregates.extra.get("velocity_operator")
        if vel_op is not None:
            self._velocity_operator = vel_op

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
    def solver(self) -> CollisionMatrixSolver:
        """Return the underlying CollisionMatrixSolver."""
        return self._solver

    @property
    def kappa(self) -> NDArray[np.double]:
        """Return LBTE thermal conductivity, shape (num_sigma, num_temp, 6)."""
        return self._solver.kappa

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

    @property
    def mode_kappa(self) -> NDArray[np.double]:
        """Return mode LBTE kappa."""
        return self._solver.mode_kappa

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
            "mode_kappa": self._solver.mode_kappa,
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
