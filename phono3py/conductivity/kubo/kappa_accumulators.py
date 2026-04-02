"""Kappa accumulators for the Green-Kubo method."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from phono3py.conductivity.grid_point_data import GridPointResult
from phono3py.conductivity.kappa_accumulators import _log_kappa_header, _log_kappa_row
from phono3py.conductivity.kubo.kappa_formulas import (
    KuboKappaFormula,
    compute_kubo_mode_kappa_mat,
)
from phono3py.conductivity.lbte_collision_provider import LBTECollisionResult
from phono3py.conductivity.lbte_kappa_accumulator import LBTEKappaAccumulator


class KuboRTAKappaAccumulator:
    """Kappa accumulator for the Green-Kubo formula.

    Accumulates the full band-pair kappa matrix ``mode_kappa_inter`` returned by
    ``KuboKappaFormula``.  The ``kappa`` property sums over all band pairs.

    Parameters
    ----------
    formula : KuboKappaFormula
        Formula instance used to compute mode kappa at each grid point.
    temperatures : array-like or None
        Temperature values in Kelvin.
    sigmas : sequence or None
        Smearing widths (None for tetrahedron method).
    log_level : int
        Verbosity level.

    """

    def __init__(
        self,
        formula: KuboKappaFormula,
        temperatures: Sequence[float] | NDArray[np.double] | None = None,
        sigmas: Sequence[float | None] | None = None,
        log_level: int = 0,
    ) -> None:
        """Init method."""
        self._formula = formula
        self._temperatures = temperatures
        self._sigmas: list[float | None] = [] if sigmas is None else list(sigmas)
        self._log_level = log_level
        self._kappa: NDArray[np.double]
        # Full band-pair matrix; allocated lazily on first accumulate().
        self._mode_kappa_inter: NDArray[np.double] | None = None

    def prepare(
        self,
        num_sigma: int,
        num_temp: int,
        num_gp: int,
        num_band0: int,
    ) -> None:
        """Allocate kappa array; mode_kappa_inter is allocated lazily."""
        self._kappa = np.zeros((num_sigma, num_temp, 6), dtype="double", order="C")
        self._kappa_intra = np.zeros(
            (num_sigma, num_temp, 6), dtype="double", order="C"
        )
        self._num_gp = num_gp

    def accumulate(self, i_gp: int, result: GridPointResult) -> None:
        """Compute and accumulate Kubo mode-kappa at grid point ``i_gp``."""
        mkm = self._formula.compute(result)
        # mkm: (num_sigma, num_temp, num_band0, num_band, 6)
        if self._mode_kappa_inter is None:
            self._mode_kappa_inter = np.zeros(
                (self._num_gp,) + mkm.shape, dtype="double", order="C"
            )
        self._mode_kappa_inter[i_gp] = mkm
        # Sum over both band indices to accumulate kappa.
        self._kappa += mkm.sum(axis=(2, 3))
        # Diagonal (intra-band) sum: mkm[:, :, i, i, :] for all i.
        diag = np.diagonal(mkm, axis1=2, axis2=3)  # (num_sigma, num_temp, 6, num_band)
        self._kappa_intra += diag.sum(axis=-1)

    def finalize(self, num_sampling_grid_points: int) -> None:
        """Normalise accumulated kappa by the total number of sampling points."""
        if num_sampling_grid_points > 0:
            self._kappa /= num_sampling_grid_points
            self._kappa_intra /= num_sampling_grid_points

    @property
    def kappa(self) -> NDArray[np.double]:
        """Return total kappa (kappa_intra + kappa_inter).

        Shape: (num_sigma, num_temp, 6).

        """
        return self._kappa

    @property
    def kappa_intra(self) -> NDArray[np.double]:
        """Return intra-band (diagonal) kappa.

        Shape: (num_sigma, num_temp, 6).

        """
        return self._kappa_intra

    @property
    def kappa_inter(self) -> NDArray[np.double]:
        """Return inter-band (off-diagonal) kappa.

        Shape: (num_sigma, num_temp, 6).

        """
        return self._kappa - self._kappa_intra

    @property
    def mode_kappa(self) -> NDArray[np.double]:
        """Return mode kappa summed over j_band.

        Shape: (num_sigma, num_temp, num_gp, num_band0, 6).

        """
        if self._mode_kappa_inter is None:
            raise RuntimeError("mode_kappa_inter has not been computed yet.")
        # Sum over j_band axis (axis=-2 of the per-gp array, axis=4 in stored array).
        return self._mode_kappa_inter.sum(axis=4)

    @property
    def mode_kappa_inter(self) -> NDArray[np.double] | None:
        """Return full band-pair kappa matrix.

        Shape: (num_gp, num_sigma, num_temp, num_band0, num_band, 6).

        """
        return self._mode_kappa_inter

    def log_kappa(
        self,
        num_ignored_phonon_modes: NDArray[np.int64] | None = None,
        num_phonon_modes: int | None = None,
    ) -> None:
        """Print K_intra, K_inter, K_TOT rows for the Kubo-RTA conductivity."""
        if not self._log_level or self._temperatures is None:
            return

        kappa_inter = self._kappa - self._kappa_intra
        show_ipm = (
            self._log_level > 1
            and num_ignored_phonon_modes is not None
            and num_phonon_modes is not None
        )

        for i, sigma in enumerate(self._sigmas):
            _log_kappa_header(sigma, show_ipm=show_ipm)
            for j, t in enumerate(self._temperatures):
                ipm = (
                    int(num_ignored_phonon_modes[i, j])
                    if show_ipm and num_ignored_phonon_modes is not None
                    else None
                )
                _log_kappa_row(
                    "K_intra", t, self._kappa_intra[i, j], ipm, num_phonon_modes
                )
            print(" ")
            for j, t in enumerate(self._temperatures):
                ipm = (
                    int(num_ignored_phonon_modes[i, j])
                    if show_ipm and num_ignored_phonon_modes is not None
                    else None
                )
                _log_kappa_row(
                    "K_inter", t, kappa_inter[i, j], ipm, num_phonon_modes
                )
            print(" ")
            for j, t in enumerate(self._temperatures):
                ipm = (
                    int(num_ignored_phonon_modes[i, j])
                    if show_ipm and num_ignored_phonon_modes is not None
                    else None
                )
                _log_kappa_row(
                    "K_TOT", t, self._kappa[i, j], ipm, num_phonon_modes
                )
            print("", flush=True)


class KuboLBTEKappaAccumulator:
    """Wraps LBTEKappaAccumulator and computes Kubo kappa with LBTE linewidths.

    Stage 1 (per-grid-point): accumulate() stores the velocity product and
    heat capacity matrix in addition to delegating to the inner
    LBTEKappaAccumulator.

    Stage 2 (global): finalize() calls inner.finalize() for the standard LBTE
    kappa (P-term), then computes Kubo kappa using the collision matrix
    diagonal as effective linewidths.

    Parameters
    ----------
    inner : LBTEKappaAccumulator
        Inner accumulator for the standard LBTE solve.
    ir_grid_points : NDArray[np.int64]
        BZ grid point indices of the irreducible grid points, shape (num_ir,).
    frequencies : NDArray[np.double]
        Phonon frequencies on the full BZ grid, shape (n_bz, n_band).
    cutoff_frequency : float
        Cutoff frequency in THz.
    conversion_factor : float
        Unit conversion factor to W/(m*K).
    sigmas : list of float or None
        Smearing widths.
    log_level : int, optional
        Verbosity level. Default 0.

    """

    def __init__(
        self,
        inner: LBTEKappaAccumulator,
        ir_grid_points: NDArray[np.int64],
        frequencies: NDArray[np.double],
        cutoff_frequency: float,
        conversion_factor: float,
        sigmas: list[float | None],
        log_level: int = 0,
    ) -> None:
        """Init method."""
        self._inner = inner
        self._ir_grid_points = ir_grid_points
        self._frequencies = frequencies
        self._cutoff_frequency = cutoff_frequency
        self._conversion_factor = conversion_factor
        self._sigmas = sigmas
        self._log_level = log_level

        # Per-grid-point storage (lazily allocated in accumulate).
        self._velocity_product: NDArray[np.cdouble] | None = None
        self._heat_capacity_matrix: NDArray[np.double] | None = None

        # Output arrays (populated in finalize).
        self._kappa_inter: NDArray[np.double] | None = None
        self._mode_kappa_inter: NDArray[np.double] | None = None

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
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Store per-grid-point data and delegate to the inner accumulator.

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
        extra : dict or None
            Plugin-specific data. Expected keys:
            ``velocity_product`` (num_band0, num_band, 6) complex,
            ``heat_capacity_matrix`` (num_temp, num_band0, num_band).

        """
        velocity_product = extra.get("velocity_product") if extra else None
        heat_capacity_matrix = extra.get("heat_capacity_matrix") if extra else None

        num_ir = len(self._ir_grid_points)

        if velocity_product is not None:
            if self._velocity_product is None:
                self._velocity_product = np.zeros(
                    (num_ir,) + velocity_product.shape, dtype="complex128"
                )
            self._velocity_product[i_gp] = velocity_product

        if heat_capacity_matrix is not None:
            if self._heat_capacity_matrix is None:
                self._heat_capacity_matrix = np.zeros(
                    (num_ir,) + heat_capacity_matrix.shape, dtype="double"
                )
            self._heat_capacity_matrix[i_gp] = heat_capacity_matrix

        self._inner.accumulate(
            i_gp, collision_result, group_velocities, heat_capacities
        )

    def finalize(
        self,
        num_sampling_grid_points: int,
        suppress_kappa_log: bool = False,
    ) -> None:
        """Finalize LBTE solve, then compute Kubo kappa with LBTE linewidths."""
        self._inner.finalize(
            num_sampling_grid_points,
            suppress_kappa_log=bool(self._log_level),
        )
        self._compute_kubo_kappa(num_sampling_grid_points)
        if self._log_level:
            self._log_kubo_kappa()

    # ------------------------------------------------------------------
    # Attribute delegation
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        """Fall back to the inner LBTEKappaAccumulator for missing attributes."""
        return getattr(self._inner, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Route attribute assignment.

        gamma, collision_matrix, and temperatures are forwarded to
        self._inner. All other attributes are set on self.

        """
        if name in ("gamma", "collision_matrix", "temperatures"):
            setattr(self._inner, name, value)
        else:
            super().__setattr__(name, value)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def kappa(self) -> NDArray[np.double]:
        """Return total kappa (kappa_intra_exact + kappa_inter).

        Shape: (num_sigma, num_temp, 6).

        """
        if self._kappa_inter is not None:
            return self._inner.kappa + self._kappa_inter
        return self._inner.kappa

    @property
    def kappa_intra_exact(self) -> NDArray[np.double]:
        """Return intra-band kappa from LBTE exact solve.

        Shape: (num_sigma, num_temp, 6).

        """
        return self._inner.kappa

    @property
    def kappa_intra_RTA(self) -> NDArray[np.double]:
        """Return intra-band kappa from LBTE RTA.

        Shape: (num_sigma, num_temp, 6).

        """
        return self._inner.kappa_RTA

    @property
    def kappa_inter(self) -> NDArray[np.double] | None:
        """Return inter-band (off-diagonal) kappa from Kubo formula.

        Shape: (num_sigma, num_temp, 6).

        """
        return self._kappa_inter

    @property
    def mode_kappa_intra_exact(self) -> NDArray[np.double]:
        """Return intra-band mode kappa from LBTE exact solve.

        Shape: (num_sigma, num_temp, num_gp, num_band0, 6).

        """
        return self._inner.mode_kappa

    @property
    def mode_kappa_intra_RTA(self) -> NDArray[np.double]:
        """Return intra-band mode kappa from LBTE RTA.

        Shape: (num_sigma, num_temp, num_gp, num_band0, 6).

        """
        return self._inner.mode_kappa_RTA

    @property
    def mode_kappa(self) -> NDArray[np.double]:
        """Return intra-band mode kappa from LBTE exact solve.

        Shape: (num_sigma, num_temp, num_gp, num_band0, 6).

        """
        return self._inner.mode_kappa

    @property
    def mode_kappa_inter(self) -> NDArray[np.double] | None:
        """Return full band-pair Kubo kappa matrix.

        Shape: (num_gp, num_sigma, num_temp, num_band0, num_band, 6).

        """
        return self._mode_kappa_inter

    def get_extra_kappa_output(self) -> dict[str, NDArray[np.double] | None]:
        """Return Kubo LBTE kappa arrays keyed by HDF5 dataset name."""
        return {
            "kappa_inter": self._kappa_inter,
            "kappa_intra_exact": self._inner.kappa,
            "kappa_intra_RTA": self._inner.kappa_RTA,
            "mode_kappa_intra_exact": self._inner.mode_kappa,
            "mode_kappa_intra_RTA": self._inner.mode_kappa_RTA,
        }

    # ------------------------------------------------------------------
    # Private: Kubo kappa computation
    # ------------------------------------------------------------------

    def _compute_kubo_kappa(self, num_sampling_grid_points: int) -> None:
        """Compute inter-band kappa using LBTE collision matrix diagonal."""
        if self._velocity_product is None or self._heat_capacity_matrix is None:
            return

        num_sigma = len(self._sigmas)
        num_temp = len(self._inner.temperatures)
        num_ir = len(self._ir_grid_points)

        kappa_inter = np.zeros((num_sigma, num_temp, 6), dtype="double")
        mode_kappa_inter_list: list[NDArray[np.double]] = []

        for i_gp in range(num_ir):
            gp = int(self._ir_grid_points[i_gp])
            frequencies = self._frequencies[gp]
            num_band = len(frequencies)

            # Build gamma from LBTE collision matrix diagonal.
            gamma = np.zeros((num_sigma, num_temp, num_band), dtype="double")
            for i_sigma in range(num_sigma):
                for i_temp in range(num_temp):
                    gamma[i_sigma, i_temp] = self._inner.get_main_diagonal(
                        i_gp, i_sigma, i_temp
                    )

            mkm = compute_kubo_mode_kappa_mat(
                frequencies=frequencies,
                gamma=gamma,
                heat_capacity_matrix=self._heat_capacity_matrix[i_gp],
                velocity_product=self._velocity_product[i_gp],
                cutoff_frequency=self._cutoff_frequency,
                conversion_factor=self._conversion_factor,
                grid_point=gp,
            )
            # mkm: (num_sigma, num_temp, num_band0, num_band, 6)
            mode_kappa_inter_list.append(mkm)

            # Off-diagonal (inter-band) sum only.
            all_sum = mkm.sum(axis=(2, 3))
            diag = np.diagonal(mkm, axis1=2, axis2=3)  # (..., 6, num_band)
            kappa_inter += all_sum - diag.sum(axis=-1)

        if num_sampling_grid_points > 0:
            kappa_inter /= num_sampling_grid_points

        self._kappa_inter = kappa_inter
        self._mode_kappa_inter = np.array(mode_kappa_inter_list)

    def _log_kubo_kappa(self) -> None:
        """Print Kubo LBTE kappa table."""
        kappa_intra_exact = self._inner.kappa
        kappa_intra_RTA = self._inner.kappa_RTA
        kappa_inter = self._kappa_inter

        for i_sigma in range(len(self._sigmas)):
            for i_temp, t in enumerate(self._inner.temperatures):
                if t <= 0:
                    continue
                print(
                    ("#%6s       " + " %-10s" * 6)
                    % ("           T(K)", "xx", "yy", "zz", "yz", "xz", "xy")
                )
                print(
                    "K_intra_exact"
                    + ("%7.1f " + " %10.3f" * 6)
                    % ((t,) + tuple(kappa_intra_exact[i_sigma, i_temp]))
                )
                print(
                    "(K_intra_RTA)"
                    + ("%7.1f " + " %10.3f" * 6)
                    % ((t,) + tuple(kappa_intra_RTA[i_sigma, i_temp]))
                )
                if kappa_inter is not None:
                    print(
                        "K_inter      "
                        + ("%7.1f " + " %10.3f" * 6)
                        % ((t,) + tuple(kappa_inter[i_sigma, i_temp]))
                    )
                    print(" ")
                    kappa_tot = (
                        kappa_intra_exact[i_sigma, i_temp]
                        + kappa_inter[i_sigma, i_temp]
                    )
                    print(
                        "K_TOT=K_intra+K_inter"
                        + ("%7.1f " + " %10.3f" * 6) % ((t,) + tuple(kappa_tot))
                    )
                print("-" * 76, flush=True)
