"""RTACalculator: composition-based RTA lattice thermal conductivity."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from phono3py.conductivity.build_components import KappaSettings
from phono3py.conductivity.grid_point_data import (
    GridPointAggregates,
    compute_effective_gamma,
)
from phono3py.conductivity.heat_capacity_solvers import ModeHeatCapacitySolver
from phono3py.conductivity.kappa_solvers import RTAKappaSolver
from phono3py.conductivity.scattering_solvers import (
    IsotopeScatteringSolver,
    RTAScatteringSolver,
    compute_bulk_boundary_scattering,
)
from phono3py.conductivity.utils import (
    get_kappa_star_operations,
    show_grid_point_frequencies_gv,
    show_grid_point_frequencies_gv_on_kstar,
    show_grid_point_header,
)
from phono3py.conductivity.velocity_solvers import GroupVelocitySolver
from phono3py.other.isotope import Isotope
from phono3py.phonon.grid import (
    BZGrid,
    get_qpoints_from_bz_grid_points,
)
from phono3py.phonon3.interaction import Interaction


class RTACalculator:
    """RTA lattice thermal conductivity calculator using composed building blocks.

    This class replaces the ``ConductivityBase`` / ``ConductivityRTABase`` /
    ``ConductivityRTA`` inheritance hierarchy.  Physical building blocks
    (velocity, heat-capacity, scattering, kappa solver) are injected via
    the constructor rather than inherited.

    Variant-specific output properties (``kappa_P_RTA``, ``kappa_C``, etc.)
    are not defined on this class.  They are forwarded transparently to the
    kappa solver via ``__getattr__``.

    Usage
    -----
    Create via ``conductivity_calculator`` (recommended) or directly::

        calc = RTACalculator(pp, vel, cv, scat, kappa_solver, kappa_settings, ...)
        calc.run(on_grid_point=lambda i: writer.write_gamma(calc, i))
        # results available via calc.kappa, calc.mode_kappa, etc.

    Parameters
    ----------
    pp : Interaction
        Interaction instance. ``init_dynamical_matrix`` must have been called.
    velocity_solver : GroupVelocitySolver
        Computes group velocities and v-outer-v at each grid point.
    cv_solver : ModeHeatCapacitySolver
        Computes mode heat capacities at each grid point.
    scattering_solver : RTAScatteringSolver
        Computes ph-ph linewidths at each grid point.
    kappa_solver
        Owns the kappa computation and BZ-summation arrays; exposes ``kappa``,
        ``mode_kappa``, and any variant-specific output properties.
    kappa_settings : KappaSettings
        Shared computation metadata (grid, phonon, symmetry, configuration).
    is_isotope : bool, optional
        Include isotope scattering. Default False.
    mass_variances : array-like or None, optional
        Mass variances for isotope scattering. Default None.
    is_N_U : bool, optional
        Decompose gamma into Normal and Umklapp. Default False.
    is_gamma_detail : bool, optional
        Store per-triplet gamma. Default False.
    log_level : int, optional
        Verbosity. Default 0.

    """

    def __init__(
        self,
        pp: Interaction,
        velocity_solver: GroupVelocitySolver,
        cv_solver: ModeHeatCapacitySolver,
        scattering_solver: RTAScatteringSolver,
        kappa_solver: RTAKappaSolver,
        kappa_settings: KappaSettings,
        frequencies: NDArray[np.double],
        *,
        is_isotope: bool = False,
        mass_variances: Sequence[float] | NDArray[np.double] | None = None,
        is_N_U: bool = False,
        is_gamma_detail: bool = False,
        sigma_cutoff_width: float | None = None,
        log_level: int = 0,
    ):
        """Init method."""
        self._pp = pp
        self._velocity_solver = velocity_solver
        self._cv_solver = cv_solver
        self._scattering_solver = scattering_solver
        self._kappa_solver = kappa_solver
        self._kappa_settings = kappa_settings
        self._frequencies = frequencies
        self._is_N_U = is_N_U
        self._is_gamma_detail = is_gamma_detail
        self._sigma_cutoff_width = sigma_cutoff_width
        self._log_level = log_level

        self._grid_point_count = 0

        # Isotope solver.
        self._is_isotope = is_isotope or (mass_variances is not None)
        self._isotope_solver: IsotopeScatteringSolver | None = None
        if self._is_isotope:
            self._isotope_solver = self._build_isotope_solver(mass_variances)

        # Read flags (set via property setters when gamma is loaded from file).
        self._read_gamma = False
        self._read_gamma_iso = False

        # Declare arrays; allocated lazily when temperatures are set.
        self._gamma: NDArray[np.double] | None = None
        self._gv: NDArray[np.double] | None = None
        self._gv_by_gv: NDArray[np.double] | None = None
        self._cv: NDArray[np.double] | None = None
        self._vm_by_vm: NDArray[np.cdouble] | None = None
        self._heat_capacity_matrix: NDArray[np.double] | None = None
        self._gamma_iso: NDArray[np.double] | None = None
        self._averaged_pp_interaction: NDArray[np.double] | None = None
        self._gamma_N: NDArray[np.double] | None = None
        self._gamma_U: NDArray[np.double] | None = None
        self._gamma_elph: NDArray[np.double] | None = None
        self._gamma_boundary: NDArray[np.double] | None = None
        self._num_ignored_phonon_modes: NDArray[np.int64] | None = None
        self._gamma_detail_at_q: NDArray[np.double] | None = None

        if self._kappa_settings.temperatures is not None:
            self._allocate_values()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        on_grid_point: Callable[[int], None] | None = None,
    ) -> None:
        """Run all grid points and compute kappa.

        The computation is organized in separate phases:

        1. Bulk heat capacity (vectorized, single call).
        2. Velocity loop (per-GP, no interaction state dependency).
        3. Isotope loop (per-GP, no interaction state dependency).
        4. Main gamma loop (per-GP, requires Interaction.set_grid_point).
        5. Bulk boundary scattering (vectorized, from accumulated gv).
        6. Finalize (aggregate and compute kappa).

        Parameters
        ----------
        on_grid_point : callable or None, optional
            Called with the grid-point count (0-based index) after each grid
            point is processed in the gamma loop. Used for per-grid-point
            file writes.

        """
        if self._kappa_settings.temperatures is None:
            raise RuntimeError("Set temperatures before calling run().")

        self._prepare_isotope_phonons()

        # (1) Bulk heat capacity.
        if self._log_level:
            print("Running heat capacity calculations...")
        self._compute_bulk_heat_capacities()

        # (2) Velocity loop.
        if self._log_level:
            print("Running velocity calculations...")
        self._compute_all_velocities()

        # (3) Bulk boundary scattering.
        if self._kappa_settings.boundary_mfp is not None:
            self._gamma_boundary = compute_bulk_boundary_scattering(
                self._gv, self._kappa_settings.boundary_mfp
            )

        # (4) Isotope loop.
        if self._is_isotope and not self._read_gamma_iso:
            if self._log_level:
                for sigma in self._kappa_settings.sigmas:
                    print("Running isotope scattering calculations ", end="")
                    print(
                        "with tetrahedron method..."
                        if sigma is None
                        else f"sigma={sigma}..."
                    )
            self._compute_all_isotope()

        # (5) Main gamma loop.
        self._grid_point_count = 0
        for i_gp in range(len(self._kappa_settings.grid_points)):
            self._compute_gamma_at_grid_point(i_gp)
            self._grid_point_count = i_gp + 1
            if on_grid_point is not None:
                on_grid_point(i_gp)

        if self._log_level:
            print(
                "=================== End of collection of collisions "
                "==================="
            )

        # (6) Finalize.
        aggregates = self._build_grid_point_aggregates()
        self._count_ignored_modes(aggregates)
        self._kappa_solver.finalize(aggregates)

    # ------------------------------------------------------------------
    # Properties -- kappa settings access
    # ------------------------------------------------------------------

    @property
    def kappa_settings(self) -> KappaSettings:
        """Return the kappa settings."""
        return self._kappa_settings

    # ------------------------------------------------------------------
    # Properties -- grid / phonon metadata
    # ------------------------------------------------------------------

    @property
    def mesh_numbers(self) -> NDArray[np.int64]:
        """Return BZ mesh numbers."""
        return self._kappa_settings.mesh_numbers

    @property
    def bz_grid(self) -> BZGrid:
        """Return BZ grid object."""
        return self._kappa_settings.bz_grid

    @property
    def frequencies(self) -> NDArray[np.double]:
        """Return phonon frequencies at the iterated grid points."""
        return self._frequencies[self._kappa_settings.grid_points]

    @property
    def qpoints(self) -> NDArray[np.double]:
        """Return q-point coordinates of the iterated grid points."""
        return np.array(
            get_qpoints_from_bz_grid_points(
                self._kappa_settings.grid_points, self._kappa_settings.bz_grid
            ),
            dtype="double",
            order="C",
        )

    @property
    def grid_points(self) -> NDArray[np.int64]:
        """Return BZ grid point indices that were iterated."""
        return self._kappa_settings.grid_points

    @property
    def grid_weights(self) -> NDArray[np.int64]:
        """Return symmetry weights of the iterated grid points."""
        return self._kappa_settings.grid_weights

    @property
    def temperatures(self) -> NDArray[np.double] | None:
        """Return temperatures in Kelvin."""
        return self._kappa_settings.temperatures

    @property
    def sigmas(self) -> list[float | None]:
        """Return smearing widths."""
        return self._kappa_settings.sigmas

    @property
    def sigma_cutoff_width(self) -> float | None:
        """Return smearing cutoff width."""
        return self._sigma_cutoff_width

    @property
    def boundary_mfp(self) -> float | None:
        """Return boundary mean free path in micrometres."""
        return self._kappa_settings.boundary_mfp

    @property
    def grid_point_count(self) -> int:
        """Return number of grid points processed so far."""
        return self._grid_point_count

    # ------------------------------------------------------------------
    # Properties -- computed physical quantities
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> object:
        """Delegate unknown attribute lookups to the kappa solver.

        This allows variant-specific properties (kappa_P_RTA, kappa_C,
        kappa_TOT_RTA, mode_kappa_C, ...) to be accessed directly on the
        calculator without hard-coding them here.

        """
        # Avoid infinite recursion if _kappa_solver is not yet set.
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            acc = object.__getattribute__(self, "_kappa_solver")
        except AttributeError:
            raise AttributeError(name) from None
        try:
            return getattr(acc, name)
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            ) from None

    def get_extra_kappa_output(self) -> dict[str, Any] | None:
        """Return variant-specific kappa output from the kappa solver.

        Called by output writers to obtain plugin-defined quantities that
        are written to the hdf5 file via
        ``write_kappa_to_hdf5(extra_datasets=...)``.

        Returns None when the kappa solver does not implement this method
        (standard RTA and Kubo).

        """
        fn = getattr(self._kappa_solver, "get_extra_kappa_output", None)
        return fn() if callable(fn) else None

    def get_extra_grid_point_output(self) -> dict[str, Any] | None:
        """Return per-grid-point extra arrays stored by the calculator.

        Called by output writers to obtain plugin-defined per-grid-point
        arrays that are written to the hdf5 file via
        ``write_kappa_to_hdf5(extra_datasets=...)``.
        The caller is responsible for slicing by grid-point index.

        Returns None when no extra data has been stored.

        """
        return None

    def log_kappa(self) -> None:
        """Delegate kappa logging to the kappa solver.

        Called by rta_init after run() when full kappa is available.

        """
        fn = getattr(self._kappa_solver, "log_kappa", None)
        if callable(fn):
            num_band = self._frequencies.shape[1]
            num_mesh_points = int(np.prod(self._kappa_settings.mesh_numbers))
            num_phonon_modes = num_mesh_points * num_band
            fn(
                num_ignored_phonon_modes=self._num_ignored_phonon_modes,
                num_phonon_modes=num_phonon_modes,
            )

    @property
    def number_of_ignored_phonon_modes(self) -> NDArray[np.int64] | None:
        """Return count of ignored modes, shape (num_sigma, num_temp)."""
        return self._num_ignored_phonon_modes

    # ------------------------------------------------------------------
    # Properties -- scattering arrays (with setters for file reads)
    # ------------------------------------------------------------------
    # kappa, mode_kappa are owned by the kappa solver and accessed via
    # __getattr__ delegation.  All per-grid-point arrays (gamma,
    # group_velocities, gv_by_gv, mode_heat_capacities, gamma_isotope,
    # averaged_pp_interaction, vm_by_vm, heat_capacity_matrix) are
    # owned by the calculator.

    @property
    def gamma(self) -> NDArray[np.double] | None:
        """Return ph-ph gamma, shape (num_sigma, num_temp, num_gp, num_band0)."""
        return self._gamma

    @gamma.setter
    def gamma(self, gamma: NDArray[np.double]) -> None:
        self._gamma = gamma
        self._read_gamma = True

    @property
    def group_velocities(self) -> NDArray[np.double] | None:
        """Return group velocities, shape (num_gp, num_band0, 3)."""
        return self._gv

    @property
    def gv_by_gv(self) -> NDArray[np.double] | None:
        """Return symmetrised v-outer-v, shape (num_gp, num_band0, 6)."""
        return self._gv_by_gv

    @property
    def mode_heat_capacities(self) -> NDArray[np.double] | None:
        """Return mode heat capacities, shape (num_temp, num_gp, num_band0)."""
        return self._cv

    @property
    def gamma_isotope(self) -> NDArray[np.double] | None:
        """Return isotope gamma, shape (num_sigma, num_gp, num_band0)."""
        return self._gamma_iso

    @gamma_isotope.setter
    def gamma_isotope(self, gamma_iso: NDArray[np.double] | None) -> None:
        self._gamma_iso = gamma_iso
        self._read_gamma_iso = gamma_iso is not None

    @property
    def gamma_elph(self) -> NDArray[np.double] | None:
        """Return el-ph gamma, shape (num_sigma, num_temp, num_gp, num_band0)."""
        return self._gamma_elph

    @gamma_elph.setter
    def gamma_elph(self, gamma_elph: NDArray[np.double] | None) -> None:
        self._gamma_elph = gamma_elph

    @property
    def averaged_pp_interaction(self) -> NDArray[np.double] | None:
        """Return averaged ph-ph interaction, shape (num_gp, num_band0)."""
        return self._averaged_pp_interaction

    def set_averaged_pp_interaction(self, ave_pp: NDArray[np.double]) -> None:
        """Set averaged ph-ph interaction from outside."""
        self._averaged_pp_interaction = ave_pp

    def get_gamma_N_U(
        self,
    ) -> tuple[NDArray[np.double] | None, NDArray[np.double] | None]:
        """Return Normal and Umklapp parts of gamma."""
        return self._gamma_N, self._gamma_U

    def set_gamma_N_U(
        self,
        gamma_N: NDArray[np.double],
        gamma_U: NDArray[np.double],
    ) -> None:
        """Set Normal and Umklapp parts of gamma."""
        self._gamma_N = gamma_N
        self._gamma_U = gamma_U

    def get_gamma_detail_at_q(self) -> NDArray[np.double] | None:
        """Return per-triplet gamma at the last processed q-point."""
        return self._gamma_detail_at_q

    # ------------------------------------------------------------------
    # Private: array allocation
    # ------------------------------------------------------------------

    def _allocate_values(self) -> None:
        assert self._kappa_settings.temperatures is not None
        num_sigma = len(self._kappa_settings.sigmas)
        num_temp = len(self._kappa_settings.temperatures)
        num_gp = len(self._kappa_settings.grid_points)
        num_band0 = len(self._kappa_settings.band_indices)
        num_band = self._frequencies.shape[1]

        self._gamma = np.zeros(
            (num_sigma, num_temp, num_gp, num_band0), order="C", dtype="double"
        )
        self._gv = np.zeros((num_gp, num_band0, 3), order="C", dtype="double")
        if self._velocity_solver.produces_gv_by_gv:
            self._gv_by_gv = np.zeros((num_gp, num_band0, 6), order="C", dtype="double")
        self._cv = np.zeros((num_temp, num_gp, num_band0), order="C", dtype="double")
        if not self._read_gamma:
            if self._is_N_U or self._is_gamma_detail:
                shape = (num_sigma, num_temp, num_gp, num_band0)
                self._gamma_N = np.zeros(shape, order="C", dtype="double")
                self._gamma_U = np.zeros(shape, order="C", dtype="double")
        if self._is_isotope:
            self._gamma_iso = np.zeros(
                (num_sigma, num_gp, num_band0), order="C", dtype="double"
            )
        if self._scattering_solver.is_full_pp:
            self._averaged_pp_interaction = np.zeros(
                (num_gp, num_band0), order="C", dtype="double"
            )
        if self._velocity_solver.produces_vm_by_vm:
            self._vm_by_vm = np.zeros(
                (num_gp, num_band0, num_band, 6), order="C", dtype="complex128"
            )
        if self._cv_solver.produces_heat_capacity_matrix:
            self._heat_capacity_matrix = np.zeros(
                (num_temp, num_gp, num_band0, num_band), order="C", dtype="double"
            )
        self._num_ignored_phonon_modes = np.zeros(
            (num_sigma, num_temp), order="C", dtype="int64"
        )

    # ------------------------------------------------------------------
    # Private: isotope
    # ------------------------------------------------------------------

    def _build_isotope_solver(
        self,
        mass_variances: Sequence[float] | NDArray[np.double] | None,
    ) -> IsotopeScatteringSolver:
        isotope = Isotope(
            self._kappa_settings.mesh_numbers,
            self._pp.primitive,
            mass_variances=mass_variances,
            bz_grid=self._kappa_settings.bz_grid,
            frequency_factor_to_THz=self._pp.frequency_factor_to_THz,
            symprec=self._pp.primitive_symmetry.tolerance,
            cutoff_frequency=self._kappa_settings.cutoff_frequency,
            lapack_zheev_uplo=self._pp.lapack_zheev_uplo,
        )
        return IsotopeScatteringSolver(
            isotope, self._kappa_settings.sigmas, log_level=self._log_level
        )

    def _prepare_isotope_phonons(self) -> None:
        if self._isotope_solver is None:
            return
        frequencies, eigenvectors, phonon_done = self._pp.get_phonons()
        self._isotope_solver.isotope.set_phonons(
            frequencies,
            eigenvectors,
            phonon_done,
            dm=self._pp.dynamical_matrix,
        )

    # ------------------------------------------------------------------
    # Private: per-grid-point computation
    # ------------------------------------------------------------------

    def _show_log_header(self, i_gp: int) -> None:
        if not self._log_level:
            return
        mass_variances = (
            self._isotope_solver.isotope.mass_variances
            if self._is_isotope and self._isotope_solver is not None
            else None
        )
        show_grid_point_header(
            bzgp=self._kappa_settings.grid_points[i_gp],
            i_gp=i_gp,
            num_gps=len(self._kappa_settings.grid_points),
            bz_grid=self._kappa_settings.bz_grid,
            boundary_mfp=self._kappa_settings.boundary_mfp,
            mass_variances=mass_variances,
        )

    def _compute_bulk_heat_capacities(self) -> None:
        """Compute heat capacities for all grid points at once."""
        assert self._kappa_settings.temperatures is not None
        cv_result = self._cv_solver.compute(self._kappa_settings.grid_points)
        self._cv = cv_result.heat_capacities
        if cv_result.heat_capacity_matrix is not None:
            self._heat_capacity_matrix = cv_result.heat_capacity_matrix

    def _compute_all_velocities(self) -> None:
        """Compute velocities for all grid points."""
        for i_gp in range(len(self._kappa_settings.grid_points)):
            grid_point = int(self._kappa_settings.grid_points[i_gp])
            vel_result = self._velocity_solver.compute(grid_point)
            self._gv[i_gp] = vel_result.group_velocities
            if vel_result.gv_by_gv is not None:
                self._gv_by_gv[i_gp] = vel_result.gv_by_gv
            if vel_result.vm_by_vm is not None:
                self._vm_by_vm[i_gp] = vel_result.vm_by_vm

    def _compute_all_isotope(self) -> None:
        """Compute isotope scattering for all grid points."""
        assert self._isotope_solver is not None
        for i_gp in range(len(self._kappa_settings.grid_points)):
            grid_point = int(self._kappa_settings.grid_points[i_gp])
            gamma_iso = self._isotope_solver.compute(grid_point)
            self._gamma_iso[:, i_gp, :] = gamma_iso[
                :, self._kappa_settings.band_indices
            ]

    def _compute_gamma_at_grid_point(self, i_gp: int) -> None:
        """Compute ph-ph scattering gamma at a single grid point."""
        self._show_log_header(i_gp)

        if not self._read_gamma:
            grid_point = int(self._kappa_settings.grid_points[i_gp])
            scat_result = self._scattering_solver.compute(grid_point)
            gamma = scat_result.gamma
            ave_pp = scat_result.averaged_pp_interaction
            if self._is_N_U or self._is_gamma_detail:
                g_N = self._scattering_solver.gamma_N
                g_U = self._scattering_solver.gamma_U
                if g_N is not None and self._gamma_N is not None:
                    self._gamma_N[:, :, i_gp, :] = g_N
                if g_U is not None and self._gamma_U is not None:
                    self._gamma_U[:, :, i_gp, :] = g_U
            self._gamma_detail_at_q = self._scattering_solver.gamma_detail_at_q
            self._gamma[:, :, i_gp, :] = gamma
            if ave_pp is not None and self._averaged_pp_interaction is not None:
                self._averaged_pp_interaction[i_gp] = ave_pp
        else:
            ave_pp = None
            if self._log_level:
                print("  Gamma is read from file.")

        if self._log_level:
            self._show_log(i_gp, self._gv[i_gp], ave_pp)

    def _build_grid_point_aggregates(self) -> GridPointAggregates:
        """Build GridPointAggregates for kappa_solver.finalize()."""
        return GridPointAggregates(
            group_velocities=self._gv,
            mode_heat_capacities=self._cv,
            gv_by_gv=self._gv_by_gv,
            gamma=self._gamma,
            gamma_isotope=self._gamma_iso,
            gamma_boundary=self._gamma_boundary,
            gamma_elph=self._gamma_elph,
            vm_by_vm=self._vm_by_vm,
            heat_capacity_matrix=self._heat_capacity_matrix,
        )

    def _show_log(
        self,
        i_gp: int,
        gv: NDArray[np.double],
        ave_pp: NDArray[np.double] | None,
    ) -> None:
        gp = self._kappa_settings.grid_points[i_gp]
        frequencies = self._frequencies[gp][self._kappa_settings.band_indices]
        pp = ave_pp if self._scattering_solver.is_full_pp else None
        gv_delta_q = getattr(self._velocity_solver, "gv_delta_q", None)
        if self._log_level > 2:
            point_ops, rot_cart = get_kappa_star_operations(
                self._kappa_settings.bz_grid, self._kappa_settings.is_kappa_star
            )
            show_grid_point_frequencies_gv_on_kstar(
                frequencies,
                gv,
                gp,
                self._kappa_settings.bz_grid,
                point_ops,
                rot_cart,
                gv_delta_q=gv_delta_q,
                ave_pp=pp,
            )
        else:
            show_grid_point_frequencies_gv(
                frequencies, gv, gv_delta_q=gv_delta_q, ave_pp=pp
            )

    def _count_ignored_modes(self, aggregates: GridPointAggregates) -> None:
        """Count modes below cutoff or with negative effective gamma."""
        assert self._num_ignored_phonon_modes is not None
        gamma_eff = compute_effective_gamma(aggregates)
        num_sigma, num_temp, num_gp, num_band0 = gamma_eff.shape
        for i_gp in range(num_gp):
            weight = int(self._kappa_settings.grid_weights[i_gp])
            gp = self._kappa_settings.grid_points[i_gp]
            freq = self._frequencies[gp][self._kappa_settings.band_indices]
            for j in range(num_sigma):
                for k in range(num_temp):
                    for ll in range(num_band0):
                        if freq[ll] < self._kappa_settings.cutoff_frequency:
                            self._num_ignored_phonon_modes[j, k] += weight
                        elif gamma_eff[j, k, i_gp, ll] < 0:
                            self._num_ignored_phonon_modes[j, k] += weight
