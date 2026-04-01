"""Kappa accumulator for the Wigner transport equation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from phonopy.physical_units import get_physical_units

from phono3py.conductivity.grid_point_data import GridPointResult
from phono3py.conductivity.lbte_collision_provider import LBTECollisionResult
from phono3py.conductivity.lbte_kappa_accumulator import LBTEKappaAccumulator
from phono3py.conductivity.wigner.kappa_formulas import (
    DEGENERATE_FREQUENCY_THRESHOLD_THZ,
    WignerKappaFormula,
)


class WignerKappaAccumulator:
    """Kappa accumulator for the Wigner transport equation (WTE).

    Decomposes kappa into a population term (kappa_P_RTA) and a coherence
    term (kappa_C).  The total kappa is kappa_TOT_RTA = kappa_P_RTA + kappa_C.

    The P arrays are pre-allocated in ``prepare()``.  The C arrays are
    allocated lazily on the first grid point because their shape depends on
    ``num_band`` (nat3), which is not known until the first formula call.

    Parameters
    ----------
    formula : WignerKappaFormula
        Formula instance used to compute mode kappa at each grid point.

    """

    def __init__(self, formula: WignerKappaFormula) -> None:
        """Init method."""
        self._formula = formula
        self._kappa_P: NDArray[np.double]
        self._mode_kappa_P: NDArray[np.double]
        self._kappa_C: NDArray[np.double] | None = None
        self._mode_kappa_C: NDArray[np.double] | None = None

    def prepare(
        self,
        num_sigma: int,
        num_temp: int,
        num_gp: int,
        num_band0: int,
    ) -> None:
        """Allocate population-term arrays; coherence arrays are lazy."""
        self._kappa_P = np.zeros((num_sigma, num_temp, 6), dtype="double", order="C")
        self._mode_kappa_P = np.zeros(
            (num_sigma, num_temp, num_gp, num_band0, 6), dtype="double", order="C"
        )

    def accumulate(self, i_gp: int, result: GridPointResult) -> None:
        """Compute and accumulate population and coherence kappa at ``i_gp``."""
        mode_kappa_P = self._formula.compute(
            result
        )  # sets result.extra["wigner_mode_kappa_C"]
        self._mode_kappa_P[:, :, i_gp, :, :] = mode_kappa_P
        self._kappa_P += np.sum(mode_kappa_P, axis=2)

        mode_kappa_C = result.extra.get("wigner_mode_kappa_C")
        if mode_kappa_C is not None:
            if self._mode_kappa_C is None:
                # Lazy allocation: shape depends on num_band (nat3).
                ns, nt, nb0, nb, _ = mode_kappa_C.shape
                num_gp = self._mode_kappa_P.shape[2]
                self._mode_kappa_C = np.zeros(
                    (ns, nt, num_gp, nb0, nb, 6), dtype="double", order="C"
                )
                self._kappa_C = np.zeros((ns, nt, 6), dtype="double", order="C")
            self._mode_kappa_C[:, :, i_gp, :, :, :] = mode_kappa_C
            assert self._kappa_C is not None
            self._kappa_C += np.sum(mode_kappa_C, axis=(2, 3))

    def finalize(self, num_sampling_grid_points: int) -> None:
        """Normalise both population and coherence terms."""
        if num_sampling_grid_points > 0:
            self._kappa_P /= num_sampling_grid_points
            if self._kappa_C is not None:
                self._kappa_C /= num_sampling_grid_points

    @property
    def kappa(self) -> NDArray[np.double]:
        """Return total Wigner kappa (kappa_P + kappa_C)."""
        if self._kappa_C is not None:
            return self._kappa_P + self._kappa_C
        return self._kappa_P

    @property
    def kappa_TOT_RTA(self) -> NDArray[np.double]:
        """Return total Wigner kappa (same as ``kappa``)."""
        return self.kappa

    @property
    def kappa_P_RTA(self) -> NDArray[np.double]:
        """Return population kappa, shape (num_sigma, num_temp, 6)."""
        return self._kappa_P

    @property
    def kappa_C(self) -> NDArray[np.double] | None:
        """Return coherence kappa, shape (num_sigma, num_temp, 6).

        None if not computed.

        """
        return self._kappa_C

    @property
    def mode_kappa(self) -> NDArray[np.double]:
        """Return population mode kappa.

        Shape: (num_sigma, num_temp, num_gp, num_band0, 6).

        """
        return self._mode_kappa_P

    @property
    def mode_kappa_C(self) -> NDArray[np.double] | None:
        """Return coherence mode kappa; None if not computed.

        Shape: (num_sigma, num_temp, num_gp, num_band0, num_band, 6).

        """
        return self._mode_kappa_C

    def show_rta_progress(self, br: object, log_level: int) -> None:
        """Print K_P, K_C, K_T rows for the Wigner-RTA conductivity.

        Called via duck typing from ShowCalcProgress.kappa_RTA so that
        all Wigner-specific display logic stays in this subpackage.

        Parameters
        ----------
        br :
            ConductivityCalculator instance (typed as object to avoid
            circular imports).
        log_level :
            Verbosity level.

        """

        def _req(v: object, name: str) -> "NDArray[np.double]":
            assert v is not None, f"{name} must not be None"
            return v  # type: ignore[return-value]

        temperatures = _req(getattr(br, "temperatures", None), "temperatures")
        sigmas = br.sigmas
        kappa_tot = _req(getattr(br, "kappa", None), "kappa")
        num_ignored = _req(
            getattr(br, "number_of_ignored_phonon_modes", None),
            "number_of_ignored_phonon_modes",
        )
        num_band = br.frequencies.shape[1]
        num_phonon_modes = br.number_of_sampling_grid_points * num_band

        kappa_P_RTA = _req(self._kappa_P, "kappa_P")
        kappa_C = _req(self._kappa_C, "kappa_C")

        for i, sigma in enumerate(sigmas):
            kappa_p_i = kappa_P_RTA[i]
            kappa_c_i = kappa_C[i]
            kappa_tot_i = kappa_tot[i]
            text = "----------- Thermal conductivity (W/m-k) "
            if sigma:
                text += "for sigma=%s -----------" % sigma
            else:
                text += "with tetrahedron method -----------"
            print(text)
            if log_level > 1:
                print(
                    ("#%6s       " + " %-10s" * 6 + "#ipm")
                    % ("      \t   T(K)", "xx", "yy", "zz", "yz", "xz", "xy")
                )
                for j, (t, k) in enumerate(zip(temperatures, kappa_p_i, strict=True)):
                    print(
                        "K_P\t"
                        + ("%7.1f" + " %10.3f" * 6 + " %d/%d")
                        % ((t,) + tuple(k) + (num_ignored[i, j], num_phonon_modes))
                    )
                print(" ")
                for j, (t, k) in enumerate(zip(temperatures, kappa_c_i, strict=True)):
                    print(
                        "K_C\t"
                        + ("%7.1f" + " %10.3f" * 6 + " %d/%d")
                        % ((t,) + tuple(k) + (num_ignored[i, j], num_phonon_modes))
                    )
                print(" ")
                for j, (t, k) in enumerate(zip(temperatures, kappa_tot_i, strict=True)):
                    print(
                        "K_T\t"
                        + ("%7.1f" + " %10.3f" * 6 + " %d/%d")
                        % ((t,) + tuple(k) + (num_ignored[i, j], num_phonon_modes))
                    )
            else:
                print(
                    ("#%6s       " + " %-10s" * 6)
                    % ("      \t   T(K)", "xx", "yy", "zz", "yz", "xz", "xy")
                )
                for t, k in zip(temperatures, kappa_p_i, strict=True):
                    print("K_P\t" + ("%7.1f " + " %10.3f" * 6) % ((t,) + tuple(k)))
                print(" ")
                for t, k in zip(temperatures, kappa_c_i, strict=True):
                    print("K_C\t" + ("%7.1f " + " %10.3f" * 6) % ((t,) + tuple(k)))
                print(" ")
                for t, k in zip(temperatures, kappa_tot_i, strict=True):
                    print("K_T\t" + ("%7.1f " + " %10.3f" * 6) % ((t,) + tuple(k)))
            print("", flush=True)

    def get_extra_kappa_output(self) -> dict[str, NDArray[np.double] | None]:
        """Return Wigner-specific kappa arrays keyed by HDF5 dataset name.

        Each value has sigma as its first axis so that callers can slice by
        sigma index with ``value[i_sigma]``.

        Keys
        ----
        kappa_TOT_RTA : (num_sigma, num_temp, 6)
        kappa_P_RTA   : (num_sigma, num_temp, 6)
        kappa_C       : (num_sigma, num_temp, 6) or None
        mode_kappa_P_RTA : (num_sigma, num_temp, num_gp, num_band0, 6)
        mode_kappa_C  : (num_sigma, num_temp, num_gp, num_band0, num_band, 6) or None

        """
        return {
            "kappa_TOT_RTA": self.kappa_TOT_RTA,
            "kappa_P_RTA": self._kappa_P,
            "kappa_C": self._kappa_C,
            "mode_kappa_P_RTA": self._mode_kappa_P,
            "mode_kappa_C": self._mode_kappa_C,
        }


class WignerLBTEAccumulator:
    """Wraps LBTEKappaAccumulator and adds the Wigner coherence (C) term.

    Stage 1 (per-grid-point): accumulate() stores the velocity operator outer
    product and heat capacities in addition to delegating to the inner
    LBTEKappaAccumulator.

    Stage 2 (global): finalize() calls inner.finalize() for the P-term kappa,
    then computes the C-term from the stored outer products and linewidths.

    Parameters
    ----------
    inner : LBTEKappaAccumulator
        Inner accumulator for the P-term (standard LBTE).
    ir_grid_points : NDArray[np.int64]
        BZ grid point indices of the irreducible grid points, shape (num_ir,).
    frequencies : NDArray[np.double]
        Phonon frequencies on the full BZ grid, shape (n_bz, n_band).
    band_indices : NDArray[np.int64]
        Band indices used for the calculation, shape (num_band0,).
    cutoff_frequency : float
        Cutoff frequency in THz.
    conversion_factor_WTE : float
        Unit conversion factor for the coherence term.
    temperatures : NDArray[np.double]
        Temperatures in Kelvin, shape (num_temp,).
    sigmas : list of float or None
        Smearing widths.
    is_reducible_collision_matrix : bool, optional
        When True the C-term is not computed (not implemented).  Default False.
    log_level : int, optional
        Verbosity level.  Default 0.

    """

    def __init__(
        self,
        inner: LBTEKappaAccumulator,
        ir_grid_points: NDArray[np.int64],
        frequencies: NDArray[np.double],
        band_indices: NDArray[np.int64],
        cutoff_frequency: float,
        conversion_factor_WTE: float,
        temperatures: NDArray[np.double],
        sigmas: list[float | None],
        is_reducible_collision_matrix: bool = False,
        log_level: int = 0,
    ) -> None:
        """Init method."""
        self._inner = inner
        self._ir_grid_points = ir_grid_points
        self._frequencies = frequencies
        self._band_indices = band_indices
        self._cutoff_frequency = cutoff_frequency
        self._conversion_factor_WTE = conversion_factor_WTE
        self._temperatures = temperatures
        self._sigmas = sigmas
        self._is_reducible = is_reducible_collision_matrix
        self._log_level = log_level

        # Per-grid-point storage for the C-term (lazily allocated in accumulate()).
        self._gv_by_gv_operator: NDArray[np.cdouble] | None = None
        self._mode_cv: NDArray[np.double] | None = None

        # C-term output arrays (populated in finalize()).
        self._kappa_C: NDArray[np.double] | None = None
        self._mode_kappa_C: NDArray[np.cdouble] | None = None

    # ------------------------------------------------------------------
    # Interface delegated to inner LBTEKappaAccumulator
    # ------------------------------------------------------------------

    def prepare(self, is_full_pp: bool = False) -> None:
        """Allocate accumulator arrays."""
        self._inner.prepare(is_full_pp=is_full_pp)

    def store_gamma_iso(self, i_gp: int, gamma_iso: NDArray[np.double]) -> None:
        """Store isotope scattering rate for grid point i_gp."""
        self._inner.store_gamma_iso(i_gp, gamma_iso)

    def accumulate(
        self,
        i_gp: int,
        collision_result: LBTECollisionResult,
        group_velocities: NDArray[np.double],
        heat_capacities: NDArray[np.double],
        velocity_product: NDArray[np.cdouble] | None = None,
    ) -> None:
        """Store per-grid-point Stage 1 data and delegate to the inner accumulator.

        Parameters
        ----------
        i_gp : int
            Loop index over ir_grid_points (0-based).
        collision_result : LBTECollisionResult
            Result from LBTECollisionProvider.compute().
        group_velocities : NDArray[np.double]
            Group velocities, shape (num_band0, 3).
        heat_capacities : NDArray[np.double]
            Mode heat capacities, shape (num_temp, num_band0).
        velocity_product : NDArray[np.cdouble] or None
            Velocity operator outer product, shape (num_band0, num_band, 6).

        """
        if velocity_product is not None and self._gv_by_gv_operator is None:
            num_ir = len(self._ir_grid_points)
            self._gv_by_gv_operator = np.zeros(
                (num_ir,) + velocity_product.shape, dtype="complex128"
            )
            self._mode_cv = np.zeros((num_ir,) + heat_capacities.shape, dtype="double")
        if velocity_product is not None and self._gv_by_gv_operator is not None:
            self._gv_by_gv_operator[i_gp] = velocity_product
        if self._mode_cv is not None:
            self._mode_cv[i_gp] = heat_capacities
        self._inner.accumulate(
            i_gp, collision_result, group_velocities, heat_capacities
        )

    def finalize(
        self,
        num_sampling_grid_points: int,
        suppress_kappa_log: bool = False,
    ) -> None:
        """Finalize P-term via inner accumulator, then compute C-term.

        The standard LBTE kappa table is suppressed (replaced by the Wigner
        table printed at the end of this method) when log_level > 0.

        """
        self._inner.finalize(
            num_sampling_grid_points,
            suppress_kappa_log=bool(self._log_level),
        )
        self._compute_coherence_kappa(num_sampling_grid_points)
        if self._log_level:
            self._log_wigner_kappa()

    # ------------------------------------------------------------------
    # Properties — P-term (delegated to inner)
    # ------------------------------------------------------------------

    @property
    def kappa(self) -> NDArray[np.double]:
        """Return LBTE kappa (P-exact term), shape (num_sigma, num_temp, 6)."""
        return self._inner.kappa

    @property
    def kappa_P_exact(self) -> NDArray[np.double]:
        """Return LBTE kappa (P-exact term) — Wigner-specific alias for kappa."""
        return self._inner.kappa

    @property
    def kappa_RTA(self) -> NDArray[np.double]:
        """Return RTA kappa (P-RTA term), shape (num_sigma, num_temp, 6)."""
        return self._inner.kappa_RTA

    @property
    def kappa_P_RTA(self) -> NDArray[np.double]:
        """Return RTA kappa (P-RTA term) — Wigner-specific alias for kappa_RTA."""
        return self._inner.kappa_RTA

    @property
    def mode_kappa(self) -> NDArray[np.double]:
        """Return mode LBTE kappa (P-exact term)."""
        return self._inner.mode_kappa

    @property
    def mode_kappa_P_exact(self) -> NDArray[np.double]:
        """Return mode LBTE kappa (P-exact term) — Wigner-specific alias."""
        return self._inner.mode_kappa

    @property
    def mode_kappa_RTA(self) -> NDArray[np.double]:
        """Return mode RTA kappa (P-RTA term)."""
        return self._inner.mode_kappa_RTA

    @property
    def mode_kappa_P_RTA(self) -> NDArray[np.double]:
        """Return mode RTA kappa (P-RTA term) — Wigner-specific alias."""
        return self._inner.mode_kappa_RTA

    @property
    def gamma(self) -> NDArray[np.double]:
        """Return ph-ph gamma."""
        return self._inner.gamma

    @gamma.setter
    def gamma(self, value: NDArray[np.double]) -> None:
        self._inner.gamma = value

    @property
    def gamma_iso(self) -> NDArray[np.double] | None:
        """Return isotope gamma."""
        return self._inner.gamma_iso

    @property
    def collision_matrix(self) -> NDArray[np.double] | None:
        """Return assembled collision matrix."""
        return self._inner.collision_matrix

    @collision_matrix.setter
    def collision_matrix(self, value: NDArray[np.double] | None) -> None:
        self._inner.collision_matrix = value

    @property
    def collision_eigenvalues(self) -> NDArray[np.double] | None:
        """Return eigenvalues of the collision matrix."""
        return self._inner.collision_eigenvalues

    @property
    def averaged_pp_interaction(self) -> NDArray[np.double] | None:
        """Return averaged ph-ph interaction."""
        return self._inner.averaged_pp_interaction

    @property
    def boundary_mfp(self) -> float | None:
        """Return boundary mean free path in micrometres."""
        return self._inner.boundary_mfp

    @property
    def mode_heat_capacities(self) -> NDArray[np.double]:
        """Return mode heat capacities."""
        return self._inner.mode_heat_capacities

    @property
    def f_vectors(self) -> NDArray[np.double] | None:
        """Return f-vectors."""
        return self._inner.f_vectors

    @property
    def mfp(self) -> NDArray[np.double] | None:
        """Return mean free path."""
        return self._inner.mfp

    @property
    def temperatures(self) -> NDArray[np.double]:
        """Return temperatures in Kelvin."""
        return self._inner.temperatures

    @temperatures.setter
    def temperatures(self, value: NDArray[np.double]) -> None:
        self._inner.temperatures = value
        self._temperatures = np.asarray(value, dtype="double")

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
        kappa_P_exact = self._inner.kappa
        kappa_P_RTA = self._inner.kappa_RTA
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
            "mode_kappa_P_exact": self._inner.mode_kappa,
            "mode_kappa_P_RTA": self._inner.mode_kappa_RTA,
            "mode_kappa_C": None if mode_kappa_C is None else mode_kappa_C.real,
        }

    # ------------------------------------------------------------------
    # Private: C-term computation
    # ------------------------------------------------------------------

    def _compute_coherence_kappa(self, num_sampling_grid_points: int) -> None:
        """Compute the Wigner coherence (C) term of thermal conductivity."""
        if self._is_reducible:
            print(
                " WARNING: Coherences conductivity not implemented for "
                "is_reducible_collision_matrix=True"
            )
            return
        if self._gv_by_gv_operator is None or self._mode_cv is None:
            return

        THzToEv = get_physical_units().THzToEv
        num_sigma = len(self._sigmas)
        num_temp = len(self._temperatures)
        num_ir = len(self._ir_grid_points)
        num_band0 = len(self._band_indices)
        num_band = self._frequencies.shape[1]

        mode_kappa_C = np.zeros(
            (num_sigma, num_temp, num_ir, num_band0, num_band, 6), dtype="complex128"
        )

        for i_sigma in range(num_sigma):
            for i_temp in range(num_temp):
                for i_gp in range(num_ir):
                    gp = int(self._ir_grid_points[i_gp])
                    g = self._inner.get_main_diagonal(i_gp, i_sigma, i_temp) * 2.0
                    frequencies = self._frequencies[gp]
                    cv = self._mode_cv[i_gp, i_temp, :]
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
        self._kappa_C = (
            mode_kappa_C.sum(axis=(2, 3, 4)) / num_sampling_grid_points
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
        if freq_s1 <= self._cutoff_frequency or freq_s2 <= self._cutoff_frequency:
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

    def _log_wigner_kappa(self) -> None:
        """Print Wigner LBTE kappa table (K_P_exact, K_P_RTA, K_C, K_TOT)."""
        kappa_P_exact = self._inner.kappa
        kappa_P_RTA = self._inner.kappa_RTA
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
