"""Kappa accumulators for the standard BTE-RTA and LBTE methods."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from phono3py.conductivity.collision_matrix_solver import CollisionMatrixSolver
from phono3py.conductivity.context import ConductivityContext
from phono3py.conductivity.grid_point_data import (
    GridPointAggregates,
    compute_effective_gamma,
)
from phono3py.conductivity.lbte_collision_provider import LBTECollisionResult
from phono3py.conductivity.utils import (
    log_kappa_header,
    log_kappa_row,
    log_sigma_header,
)


class RTAKappaAccumulator:
    """Kappa accumulator for the standard BTE diagonal formula.

    Computes mode kappa using the standard Boltzmann transport equation:

        kappa_mode = Cv * (v x v) * tau / 2
                   = Cv * gv_by_gv / (2 * gamma_eff) * unit_conversion

    where gamma_eff is the total effective linewidth (phonon-phonon + isotope
    + electron-phonon + boundary scattering).

    Parameters
    ----------
    context : ConductivityContext
        Shared computation metadata (grid, phonon, symmetry, configuration).
    conversion_factor : float
        Unit conversion factor to W/(m*K).
    log_level : int
        Verbosity level.

    """

    def __init__(
        self,
        context: ConductivityContext,
        conversion_factor: float,
        log_level: int = 0,
    ) -> None:
        """Init method."""
        self._context = context
        self._conversion_factor = conversion_factor
        self._log_level = log_level

        self._kappa: NDArray[np.double]
        self._mode_kappa: NDArray[np.double]

    def finalize(self, aggregates: GridPointAggregates) -> None:
        """Compute mode kappa and kappa from aggregated data."""
        self._compute_mode_kappa(aggregates)
        num_sampling_grid_points = aggregates.num_sampling_grid_points
        if num_sampling_grid_points > 0:
            self._kappa = (
                np.sum(self._mode_kappa, axis=(2, 3)) / num_sampling_grid_points
            )

    def _compute_mode_kappa(self, aggregates: GridPointAggregates) -> None:
        """Compute mode kappa at all grid points."""
        assert aggregates.gv_by_gv is not None
        assert aggregates.gamma is not None

        gv_by_gv = aggregates.gv_by_gv
        cv = aggregates.mode_heat_capacities
        gamma_eff = compute_effective_gamma(aggregates)
        num_sigma, num_temp, num_gp, num_band0 = gamma_eff.shape

        self._mode_kappa = np.zeros(
            (num_sigma, num_temp, num_gp, num_band0, 6), dtype="double", order="C"
        )

        for i_gp in range(num_gp):
            gp = self._context.grid_points[i_gp]
            frequencies = self._context.frequencies[gp][self._context.band_indices]
            for ll in range(num_band0):
                if frequencies[ll] < self._context.cutoff_frequency:
                    continue
                for j in range(num_sigma):
                    for k in range(num_temp):
                        g = gamma_eff[j, k, i_gp, ll]
                        old_settings = np.seterr(all="raise")
                        try:
                            self._mode_kappa[j, k, i_gp, ll] = (
                                gv_by_gv[i_gp, ll]
                                * cv[k, i_gp, ll]
                                / (g * 2)
                                * self._conversion_factor
                            )
                        except FloatingPointError:
                            pass
                        except Exception:
                            print("=" * 26 + " Warning " + "=" * 26)
                            print(
                                " Unexpected physical condition"
                                " of ph-ph interaction"
                                " calculation was found."
                            )
                            print(
                                " g=%f at gp=%d, band=%d,"
                                " freq=%f" % (g, gp, ll + 1, frequencies[ll])
                            )
                            print("=" * 61)
                        finally:
                            np.seterr(**old_settings)

    def log_kappa(
        self,
        num_ignored_phonon_modes: NDArray[np.int64] | None = None,
        num_phonon_modes: int | None = None,
    ) -> None:
        """Print kappa table after finalization."""
        if not self._log_level:
            return
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
                log_kappa_row("", t, self._kappa[i, j], ipm, num_phonon_modes)
            print("", flush=True)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def kappa(self) -> NDArray[np.double] | None:
        """Return kappa tensor, shape (num_sigma, num_temp, 6)."""
        return self._kappa

    @property
    def mode_kappa(self) -> NDArray[np.double]:
        """Return mode kappa, shape (num_sigma, num_temp, num_gp, num_band0, 6)."""
        return self._mode_kappa

    def get_extra_kappa_output(self) -> dict[str, NDArray[np.double] | None]:
        """Return extra kappa arrays keyed by HDF5 dataset name."""
        return {"mode_kappa": self._mode_kappa}


class LBTEKappaAccumulator:
    """Assemble global collision matrix and compute LBTE thermal conductivity.

    This is Stage 2 of the two-stage LBTE design.  Stage 1 (per-grid-point)
    is handled by LBTECollisionProvider.  LBTECalculator calls store()
    once per irreducible grid point and then finalize() to assemble the full
    collision matrix, solve it, and compute kappa and kappa_RTA.

    Internally delegates all work to CollisionMatrixSolver.

    Parameters
    ----------
    solver : CollisionMatrixSolver
        Pre-configured collision matrix solver.
    context : ConductivityContext
        Shared computation metadata (grid, phonon, symmetry, configuration).
    log_level : int, optional
        Verbosity level. Default 0.

    """

    def __init__(
        self,
        solver: CollisionMatrixSolver,
        context: ConductivityContext,
        log_level: int = 0,
    ) -> None:
        """Init method."""
        self._solver = solver
        self._context = context
        self._log_level = log_level

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def prepare(self) -> None:
        """Allocate global arrays before the grid-point accumulation loop."""
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
    ) -> None:
        """Assemble collision matrix and compute LBTE thermal conductivity.

        Stage 2: combine diagonals, apply weights, symmetrize, solve for kappa.

        Parameters
        ----------
        aggregates : GridPointAggregates
            Aggregated per-grid-point data from the calculator.

        """
        n = aggregates.num_sampling_grid_points
        prev_sigma = -1
        for i_sigma, i_temp in self._solver.solve_iter(aggregates):
            if self._log_level:
                if i_sigma != prev_sigma:
                    log_sigma_header(self._context.sigmas[i_sigma])
                    prev_sigma = i_sigma
                self._log_kappa_at(i_sigma, i_temp, n)

    def _log_kappa_at(
        self,
        i_sigma: int,
        i_temp: int,
        num_sampling_grid_points: int,
    ) -> None:
        """Print standard LBTE kappa for one (sigma, temperature) pair."""
        t = self._context.temperatures[i_temp]
        if t <= 0:
            return

        n = num_sampling_grid_points if num_sampling_grid_points > 0 else 1
        print(
            ("#%6s       " + " %-10s" * 6)
            % ("T(K)", "xx", "yy", "zz", "yz", "xz", "xy")
        )
        print(
            ("%7.1f " + " %10.3f" * 6)
            % ((t,) + tuple(self._solver._kappa[i_sigma, i_temp] / n))
        )
        print(
            (" %6s " + " %10.3f" * 6)
            % (("(RTA)",) + tuple(self._solver._kappa_RTA[i_sigma, i_temp] / n))
        )
        print("-" * 76, flush=True)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def solver(self) -> CollisionMatrixSolver:
        """Return the underlying CollisionMatrixSolver."""
        return self._solver
