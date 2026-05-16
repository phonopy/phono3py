"""RTACalculator and LBTECalculator: composition-based thermal conductivity.

Both calculators inherit from ConductivityCalculatorBase, which provides the
shared pipeline (heat capacity -> velocity -> isotope -> main loop -> finalize),
common properties, and array management.  Subclass-specific behaviour is
injected via hook methods (_pre_run_check, _pre_main_loop, _compute_at_grid_point,
_post_main_loop, _finalize).

"""

from __future__ import annotations

import abc
import os
from collections.abc import Callable, Sequence
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from phonopy.phonon.grid import (
    BZGrid,
    get_qpoints_from_bz_grid_points,
)

from phono3py.conductivity.build_components import KappaSettings
from phono3py.conductivity.grid_point_data import (
    GridPointAggregates,
    ScatteringResult,
    compute_effective_gamma,
)
from phono3py.conductivity.heat_capacity_solvers import ModeHeatCapacitySolver
from phono3py.conductivity.kappa_solvers import LBTEKappaSolver, RTAKappaSolver
from phono3py.conductivity.lbte_collision_solver import LBTECollisionSolver
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
from phono3py.phonon3.interaction import Interaction
from phono3py.phonon3.triplets import get_triplets_at_q

# ------------------------------------------------------------------
# Shared helper functions
# ------------------------------------------------------------------


def _build_isotope_solver(
    pp: Interaction,
    kappa_settings: KappaSettings,
    log_level: int,
    mass_variances: Sequence[float] | NDArray[np.double] | None,
    lang: Literal["C", "Python", "Rust"] = "Rust",
) -> IsotopeScatteringSolver:
    """Build an IsotopeScatteringSolver from Interaction and KappaSettings."""
    isotope = Isotope(
        kappa_settings.mesh_numbers,
        pp.primitive,
        mass_variances=mass_variances,
        bz_grid=kappa_settings.bz_grid,
        frequency_factor_to_THz=pp.frequency_factor_to_THz,
        symprec=pp.primitive_symmetry.tolerance,
        cutoff_frequency=kappa_settings.cutoff_frequency,
        lapack_zheev_uplo=pp.lapack_zheev_uplo,
        lang=lang,
    )
    return IsotopeScatteringSolver(isotope, kappa_settings.sigmas, log_level=log_level)


def _prepare_isotope_phonons(
    pp: Interaction,
    isotope_solver: IsotopeScatteringSolver | None,
) -> None:
    """Set phonon data on the isotope solver's Isotope instance."""
    if isotope_solver is None:
        return
    frequencies, eigenvectors, phonon_done = pp.get_phonons()
    isotope_solver.isotope.set_phonons(
        frequencies,
        eigenvectors,
        phonon_done,
        dm=pp.dynamical_matrix,
    )


def _show_log_header(
    i_gp: int,
    kappa_settings: KappaSettings,
    isotope_solver: IsotopeScatteringSolver | None,
    log_level: int,
) -> None:
    """Print per-grid-point header line."""
    if not log_level:
        return
    mass_variances = (
        isotope_solver.isotope.mass_variances if isotope_solver is not None else None
    )
    show_grid_point_header(
        bzgp=kappa_settings.grid_points[i_gp],
        i_gp=i_gp,
        num_gps=len(kappa_settings.grid_points),
        bz_grid=kappa_settings.bz_grid,
        boundary_mfp=kappa_settings.boundary_mfp,
        mass_variances=mass_variances,
    )


def _compute_all_isotope(
    isotope_solver: IsotopeScatteringSolver,
    kappa_settings: KappaSettings,
    gamma_iso: NDArray[np.double],
) -> None:
    """Compute isotope scattering for all grid points (in-place)."""
    for i_gp in range(len(kappa_settings.grid_points)):
        grid_point = int(kappa_settings.grid_points[i_gp])
        result = isotope_solver.compute(grid_point)
        gamma_iso[:, i_gp, :] = result[:, kappa_settings.band_indices]


def _compute_bulk_heat_capacities(
    cv_solver: ModeHeatCapacitySolver,
    grid_points: NDArray[np.int64],
) -> tuple[NDArray[np.double], NDArray[np.double] | None]:
    """Compute heat capacities for all grid points.

    Returns
    -------
    heat_capacities : NDArray[np.double]
    heat_capacity_matrix : NDArray[np.double] | None

    """
    cv_result = cv_solver.compute(grid_points)
    return cv_result.heat_capacities, cv_result.heat_capacity_matrix


# ------------------------------------------------------------------
# ConductivityCalculatorBase
# ------------------------------------------------------------------


class ConductivityCalculatorBase(abc.ABC):
    """Shared base for RTACalculator and LBTECalculator.

    Provides the common pipeline (heat capacity -> velocity -> isotope
    -> main loop -> finalize), shared properties, and array management.

    Subclasses must implement:
    - ``_allocate_values``
    - ``_compute_at_grid_point``
    - ``_finalize``

    Subclasses may override:
    - ``_pre_run_check`` (default: no-op)
    - ``_pre_main_loop`` (default: no-op)
    - ``_post_main_loop`` (default: no-op)
    - ``_show_log_at_grid_point`` (default: basic frequency/gv printout)

    """

    def __init__(
        self,
        pp: Interaction,
        velocity_solver: GroupVelocitySolver,
        cv_solver: ModeHeatCapacitySolver,
        kappa_settings: KappaSettings,
        *,
        is_isotope: bool = False,
        mass_variances: Sequence[float] | NDArray[np.double] | None = None,
        sigma_cutoff_width: float | None = None,
        log_level: int = 0,
        lang: Literal["C", "Python", "Rust"] = "Rust",
    ):
        self._pp = pp
        self._velocity_solver = velocity_solver
        self._cv_solver = cv_solver
        self._kappa_settings = kappa_settings
        self._frequencies: NDArray[np.double] = pp.get_phonons()[0]
        self._sigma_cutoff_width = sigma_cutoff_width
        self._log_level = log_level
        self._grid_point_count = 0

        # Isotope solver (optional).
        self._isotope_solver: IsotopeScatteringSolver | None = None
        if is_isotope or mass_variances is not None:
            self._isotope_solver = _build_isotope_solver(
                pp, kappa_settings, log_level, mass_variances, lang=lang
            )

        # Shared per-grid-point arrays (allocated by subclass _allocate_values).
        self._gamma: NDArray[np.double] | None = None
        self._gv: NDArray[np.double] | None = None
        self._gv_by_gv: NDArray[np.double] | None = None
        self._cv: NDArray[np.double] | None = None
        self._vm_by_vm: NDArray[np.cdouble] | None = None
        self._heat_capacity_matrix: NDArray[np.double] | None = None
        self._gamma_iso: NDArray[np.double] | None = None
        self._averaged_pp_interaction: NDArray[np.double] | None = None
        self._gamma_elph: NDArray[np.double] | None = None
        self._gamma_boundary: NDArray[np.double] | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        on_grid_point: Callable[[int], None] | None = None,
    ) -> None:
        """Run all grid points and compute kappa.

        Pipeline:
        1. Pre-run check (subclass hook).
        2. Bulk heat capacity (vectorized, single call).
        3. Velocity loop (per-GP).
        4. Pre-main-loop hook (subclass hook).
        5. Isotope loop (per-GP).
        6. Main loop -> _compute_at_grid_point (abstract).
        7. Post-main-loop hook (subclass hook).
        8. Finalize (abstract).

        Parameters
        ----------
        on_grid_point : callable or None, optional
            Called with the grid-point loop index after each grid point is
            processed.  Used for per-grid-point file writes.

        """
        self._pre_run_check()

        _prepare_isotope_phonons(self._pp, self._isotope_solver)

        # (1) Bulk heat capacity.
        if self._log_level:
            print("Running heat capacity calculations...")
        self._cv, hcm = _compute_bulk_heat_capacities(
            self._cv_solver, self._kappa_settings.grid_points
        )
        if hcm is not None:
            self._heat_capacity_matrix = hcm

        # (2) Velocity loop.
        if self._log_level:
            print("Running velocity calculations...")
        self._compute_all_velocities()

        # (3) Pre-main-loop hook.
        self._pre_main_loop()

        # (4) Isotope loop.
        self._compute_isotope_if_needed()

        # (5) Main loop.
        self._grid_point_count = 0
        self._iterate_grid_points(on_grid_point)

        if self._log_level:
            print(
                "=================== End of collection of collisions "
                "==================="
            )

        # (6) Post-main-loop hook.
        self._post_main_loop()

        # (7) Finalize.
        self._finalize()

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
    # Properties -- computed physical quantities (shared)
    # ------------------------------------------------------------------

    @property
    def gamma(self) -> NDArray[np.double] | None:
        """Return ph-ph gamma, shape (num_sigma, num_temp, num_gp, num_band0)."""
        return self._gamma

    @gamma.setter
    def gamma(self, value: NDArray[np.double]) -> None:
        self._gamma = value

    @property
    def gamma_isotope(self) -> NDArray[np.double] | None:
        """Return isotope gamma, shape (num_sigma, num_gp, num_band0)."""
        return self._gamma_iso

    @property
    def group_velocities(self) -> NDArray[np.double] | None:
        """Return group velocities, shape (num_gp, num_band0, 3)."""
        return self._gv

    @property
    def mode_heat_capacities(self) -> NDArray[np.double] | None:
        """Return mode heat capacities, shape (num_temp, num_gp, num_band0)."""
        return self._cv

    @property
    def averaged_pp_interaction(self) -> NDArray[np.double] | None:
        """Return averaged ph-ph interaction, shape (num_gp, num_band0)."""
        return self._averaged_pp_interaction

    def __getattr__(self, name: str) -> object:
        """Delegate unknown attribute lookups to the kappa solver.

        This allows variant-specific properties (kappa_P_RTA, kappa_C,
        kappa_TOT_RTA, mode_kappa_C, ...) to be accessed directly on the
        calculator without hard-coding them here.

        """
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

        Returns None when the kappa solver does not implement this method.

        """
        fn = getattr(self._kappa_solver, "get_extra_kappa_output", None)
        return fn() if callable(fn) else None

    # ------------------------------------------------------------------
    # Private: shared computation helpers
    # ------------------------------------------------------------------

    def _compute_all_velocities(self) -> None:
        """Compute velocities for all grid points."""
        for i_gp in range(len(self._kappa_settings.grid_points)):
            grid_point = int(self._kappa_settings.grid_points[i_gp])
            vel_result = self._velocity_solver.compute(grid_point)
            self._gv[i_gp] = vel_result.group_velocities
            if vel_result.gv_by_gv is not None and self._gv_by_gv is not None:
                self._gv_by_gv[i_gp] = vel_result.gv_by_gv
            if vel_result.vm_by_vm is not None and self._vm_by_vm is not None:
                self._vm_by_vm[i_gp] = vel_result.vm_by_vm

    def _compute_isotope_if_needed(self) -> None:
        """Compute isotope scattering for all grid points if applicable."""
        if self._isotope_solver is None or self._should_skip_isotope():
            return
        if self._log_level:
            for sigma in self._kappa_settings.sigmas:
                print("Running isotope scattering calculations ", end="")
                print(
                    "with tetrahedron method..."
                    if sigma is None
                    else f"sigma={sigma}..."
                )
        _compute_all_isotope(
            self._isotope_solver, self._kappa_settings, self._gamma_iso
        )

    def _should_skip_isotope(self) -> bool:
        """Return True to skip isotope computation (overridden in RTA)."""
        return False

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

    # ------------------------------------------------------------------
    # Subclass hooks (default: no-op)
    # ------------------------------------------------------------------

    def _pre_run_check(self) -> None:  # noqa: B027
        """Check preconditions before the pipeline starts."""

    def _pre_main_loop(self) -> None:  # noqa: B027
        """Run pre-main-loop computation (e.g. boundary scattering)."""

    def _post_main_loop(self) -> None:  # noqa: B027
        """Run post-main-loop computation (e.g. counting ignored modes)."""

    # ------------------------------------------------------------------
    # Main-loop iteration hook
    # ------------------------------------------------------------------

    def _iterate_grid_points(self, on_grid_point: Callable[[int], None] | None) -> None:
        """Default per-gp iteration; subclasses may override for batching."""
        for i_gp in range(len(self._kappa_settings.grid_points)):
            self._compute_at_grid_point(i_gp)
            self._grid_point_count = i_gp + 1
            if on_grid_point is not None:
                on_grid_point(i_gp)

    # ------------------------------------------------------------------
    # Abstract methods
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _allocate_values(self) -> None:
        """Allocate per-grid-point arrays."""

    @abc.abstractmethod
    def _compute_at_grid_point(self, i_gp: int) -> None:
        """Process one grid point in the main loop."""

    @abc.abstractmethod
    def _finalize(self) -> None:
        """Aggregate results and compute kappa."""


# ------------------------------------------------------------------
# RTACalculator
# ------------------------------------------------------------------


class RTACalculator(ConductivityCalculatorBase):
    """RTA lattice thermal conductivity calculator.

    Variant-specific output properties (``kappa_P_RTA``, ``kappa_C``, etc.)
    are forwarded transparently to the kappa solver via ``__getattr__``.

    """

    def __init__(
        self,
        pp: Interaction,
        velocity_solver: GroupVelocitySolver,
        cv_solver: ModeHeatCapacitySolver,
        scattering_solver: RTAScatteringSolver,
        kappa_solver: RTAKappaSolver,
        kappa_settings: KappaSettings,
        *,
        is_isotope: bool = False,
        mass_variances: Sequence[float] | NDArray[np.double] | None = None,
        is_N_U: bool = False,
        is_gamma_detail: bool = False,
        sigma_cutoff_width: float | None = None,
        log_level: int = 0,
        lang: Literal["C", "Python", "Rust"] = "Rust",
        rust_gp_batch_size: int | None = None,
    ):
        """Init method."""
        self._scattering_solver = scattering_solver
        self._kappa_solver = kappa_solver
        self._is_N_U = is_N_U
        self._is_gamma_detail = is_gamma_detail
        self._rust_gp_batch_size = rust_gp_batch_size

        # Read flags (set via property setters when gamma is loaded from file).
        self._read_gamma = False
        self._read_gamma_iso = False

        # RTA-specific arrays (allocated in _allocate_values).
        self._gamma_N: NDArray[np.double] | None = None
        self._gamma_U: NDArray[np.double] | None = None
        self._num_ignored_phonon_modes: NDArray[np.int64] | None = None
        self._gamma_detail_at_q: NDArray[np.double] | None = None

        super().__init__(
            pp,
            velocity_solver,
            cv_solver,
            kappa_settings,
            is_isotope=is_isotope,
            mass_variances=mass_variances,
            sigma_cutoff_width=sigma_cutoff_width,
            log_level=log_level,
            lang=lang,
        )

        if self._kappa_settings.temperatures is not None:
            self._allocate_values()

    # ------------------------------------------------------------------
    # RTA-specific properties
    # ------------------------------------------------------------------

    @property
    def gv_by_gv(self) -> NDArray[np.double] | None:
        """Return symmetrised v-outer-v, shape (num_gp, num_band0, 6)."""
        return self._gv_by_gv

    @ConductivityCalculatorBase.gamma.setter  # type: ignore[attr-defined]
    def gamma(self, gamma: NDArray[np.double]) -> None:
        """Set gamma and mark as read from file."""
        self._gamma = gamma
        self._read_gamma = True

    @property
    def gamma_elph(self) -> NDArray[np.double] | None:
        """Return el-ph gamma, shape (num_sigma, num_temp, num_gp, num_band0)."""
        return self._gamma_elph

    @gamma_elph.setter
    def gamma_elph(self, gamma_elph: NDArray[np.double] | None) -> None:
        self._gamma_elph = gamma_elph

    @property
    def gamma_isotope(self) -> NDArray[np.double] | None:
        """Return isotope gamma, shape (num_sigma, num_gp, num_band0)."""
        return self._gamma_iso

    @gamma_isotope.setter
    def gamma_isotope(self, gamma_iso: NDArray[np.double] | None) -> None:
        self._gamma_iso = gamma_iso
        self._read_gamma_iso = gamma_iso is not None

    def get_extra_grid_point_output(self) -> dict[str, Any] | None:
        """Return per-grid-point extra arrays stored by the calculator."""
        return None

    def log_kappa(self) -> None:
        """Delegate kappa logging to the kappa solver."""
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

    @property
    def kappa(self) -> NDArray[np.double] | None:
        """Return kappa tensor, shape (num_sigma, num_temp, 6)."""
        return self._kappa_solver.kappa

    @property
    def mode_kappa(self) -> NDArray[np.double]:
        """Return mode kappa, shape (num_sigma, num_temp, num_gp, num_band0, 6)."""
        return self._kappa_solver.mode_kappa

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
    # Subclass hook overrides
    # ------------------------------------------------------------------

    def _pre_run_check(self) -> None:
        if self._kappa_settings.temperatures is None:
            raise RuntimeError("Set temperatures before calling run().")

    def _pre_main_loop(self) -> None:
        if self._kappa_settings.boundary_mfp is not None:
            self._gamma_boundary = compute_bulk_boundary_scattering(
                self._gv, self._kappa_settings.boundary_mfp
            )

    def _post_main_loop(self) -> None:
        aggregates = self._build_grid_point_aggregates()
        self._count_ignored_modes(aggregates)

    def _should_skip_isotope(self) -> bool:
        return self._read_gamma_iso

    # ------------------------------------------------------------------
    # Abstract method implementations
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
        if self._isotope_solver is not None:
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

    def _compute_at_grid_point(self, i_gp: int) -> None:
        """Compute ph-ph scattering gamma at a single grid point."""
        _show_log_header(
            i_gp, self._kappa_settings, self._isotope_solver, self._log_level
        )

        if not self._read_gamma:
            grid_point = int(self._kappa_settings.grid_points[i_gp])
            scat_result = self._scattering_solver.compute(grid_point)
            self._store_scattering_result(
                i_gp,
                scat_result,
                self._scattering_solver.gamma_N,
                self._scattering_solver.gamma_U,
            )
            self._gamma_detail_at_q = self._scattering_solver.gamma_detail_at_q
            ave_pp = scat_result.averaged_pp_interaction
        else:
            ave_pp = None

        if self._log_level:
            self._show_log(i_gp, self._gv[i_gp], ave_pp)

    def _store_scattering_result(
        self,
        i_gp: int,
        scat_result: ScatteringResult,
        gamma_N: NDArray[np.double] | None,
        gamma_U: NDArray[np.double] | None,
    ) -> None:
        """Store a precomputed per-gp scattering result into per-gp arrays."""
        if self._is_N_U or self._is_gamma_detail:
            if gamma_N is not None and self._gamma_N is not None:
                self._gamma_N[:, :, i_gp, :] = gamma_N
            if gamma_U is not None and self._gamma_U is not None:
                self._gamma_U[:, :, i_gp, :] = gamma_U
        self._gamma[:, :, i_gp, :] = scat_result.gamma
        ave_pp = scat_result.averaged_pp_interaction
        if ave_pp is not None and self._averaged_pp_interaction is not None:
            self._averaged_pp_interaction[i_gp] = ave_pp

    def _iterate_grid_points(self, on_grid_point: Callable[[int], None] | None) -> None:
        """Batched main loop when the Rust low-memory path is active.

        Batch size is resolved in this order:
        1. ``rust_gp_batch_size`` constructor argument (if not None).
        2. ``PHONO3PY_RUST_GP_BATCH_SIZE`` env var (default ``0``).

        A value of ``0`` (or negative) falls back to the per-gp
        ``_compute_at_grid_point`` loop; a positive integer enables
        batched ``compute_batched`` calls of that size.
        """
        if self._rust_gp_batch_size is not None:
            batch_size_resolved = self._rust_gp_batch_size
        else:
            batch_size_resolved = int(
                os.environ.get("PHONO3PY_RUST_GP_BATCH_SIZE", "0")
            )
        if (
            batch_size_resolved <= 0
            or self._read_gamma
            or not getattr(self._scattering_solver, "supports_rust_batching", False)
        ):
            super()._iterate_grid_points(on_grid_point)
            return

        batch_size = max(1, batch_size_resolved)
        n_gp = len(self._kappa_settings.grid_points)
        i = 0
        while i < n_gp:
            end = min(i + batch_size, n_gp)
            i_gp_list = list(range(i, end))
            gp_list = [int(self._kappa_settings.grid_points[j]) for j in i_gp_list]
            num_triplets_list: list[int] = []
            if self._log_level:
                num_triplets_list = [
                    len(get_triplets_at_q(gp, self._kappa_settings.bz_grid)[0])
                    for gp in gp_list
                ]
                print(
                    "Batch: %d grid points, %d triplets."
                    % (len(gp_list), sum(num_triplets_list)),
                    flush=True,
                )
            batched = self._scattering_solver.compute_batched(gp_list)
            for idx, i_gp in enumerate(i_gp_list):
                payload = batched[idx]
                if self._log_level:
                    _show_log_header(
                        i_gp,
                        self._kappa_settings,
                        self._isotope_solver,
                        self._log_level,
                    )
                    print(
                        "Number of triplets: %d" % num_triplets_list[idx],
                        flush=True,
                    )
                self._store_scattering_result(
                    i_gp,
                    payload["result"],
                    payload["gamma_N"],
                    payload["gamma_U"],
                )
                self._gamma_detail_at_q = None
                if self._log_level:
                    self._show_log(i_gp, self._gv[i_gp], None)
                self._grid_point_count = i_gp + 1
                if on_grid_point is not None:
                    on_grid_point(i_gp)
            i = end

    def _finalize(self) -> None:
        aggregates = self._build_grid_point_aggregates()
        self._kappa_solver.finalize(aggregates)

    # ------------------------------------------------------------------
    # Private: RTA-specific helpers
    # ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# LBTECalculator
# ------------------------------------------------------------------


class LBTECalculator(ConductivityCalculatorBase):
    """LBTE thermal conductivity calculator.

    Two-stage design:

    Stage 1 (per-grid-point): for each irreducible grid point compute
    the collision matrix row via LBTECollisionSolver and store velocities
    and heat capacities via the standard providers.

    Stage 2 (global): LBTEKappaSolver.finalize() assembles the full
    collision matrix, symmetrizes, diagonalizes/inverts, and computes kappa.

    """

    def __init__(
        self,
        pp: Interaction,
        velocity_solver: GroupVelocitySolver,
        cv_solver: ModeHeatCapacitySolver,
        collision_solver: LBTECollisionSolver,
        kappa_solver: LBTEKappaSolver,
        kappa_settings: KappaSettings,
        *,
        is_isotope: bool = False,
        mass_variances: Sequence[float] | NDArray[np.double] | None = None,
        is_full_pp: bool = False,
        sigma_cutoff_width: float | None = None,
        log_level: int = 0,
        lang: Literal["C", "Python", "Rust"] = "Rust",
    ) -> None:
        """Init method."""
        self._collision_solver = collision_solver
        self._kappa_solver = kappa_solver
        self._is_full_pp = is_full_pp

        super().__init__(
            pp,
            velocity_solver,
            cv_solver,
            kappa_settings,
            is_isotope=is_isotope,
            mass_variances=mass_variances,
            sigma_cutoff_width=sigma_cutoff_width,
            log_level=log_level,
            lang=lang,
        )

        # Allocate arrays.
        self._allocate_values()

    # ------------------------------------------------------------------
    # LBTE-specific public interface
    # ------------------------------------------------------------------

    def set_kappa_at_sigmas(self) -> None:
        """Finalize kappa from a pre-loaded collision matrix (read-from-file path).

        Mirrors the run() pipeline but replaces the per-grid-point collision
        computation with the gamma/collision_matrix loaded from file.  Group
        velocities, heat capacities, isotope and boundary scattering are not
        stored in the collision file and must be computed here.

        """
        self._pre_run_check()
        _prepare_isotope_phonons(self._pp, self._isotope_solver)

        if self._log_level:
            print("Running heat capacity calculations...")
        self._cv, hcm = _compute_bulk_heat_capacities(
            self._cv_solver, self._kappa_settings.grid_points
        )
        if hcm is not None:
            self._heat_capacity_matrix = hcm

        if self._log_level:
            print("Running velocity calculations...")
        self._compute_all_velocities()

        self._pre_main_loop()
        self._compute_isotope_if_needed()

        if self._log_level:
            for i_gp in range(len(self._kappa_settings.grid_points)):
                _show_log_header(
                    i_gp,
                    self._kappa_settings,
                    self._isotope_solver,
                    self._log_level,
                )
                self._show_log(i_gp, self._gv[i_gp])
            print(
                "=================== End of collection of collisions "
                "==================="
            )

        self._post_main_loop()
        self._finalize()

    def delete_gp_collision_and_pp(self) -> None:
        """No-op: memory management compatibility method."""

    # ------------------------------------------------------------------
    # LBTE-specific properties
    # ------------------------------------------------------------------

    @property
    def temperatures(self) -> NDArray[np.double]:
        """Return temperatures in Kelvin."""
        assert self._kappa_settings.temperatures is not None
        return self._kappa_settings.temperatures

    @property
    def gamma(self) -> NDArray[np.double]:
        """Return ph-ph gamma, shape (num_sigma, num_temp, num_gp, num_band0)."""
        assert self._gamma is not None
        return self._gamma

    @gamma.setter
    def gamma(self, value: NDArray[np.double]) -> None:
        """Set gamma (for loading from file)."""
        self._gamma = value

    @property
    def group_velocities(self) -> NDArray[np.double]:
        """Return group velocities, shape (num_gp, num_band0, 3)."""
        assert self._gv is not None
        return self._gv

    @property
    def mode_heat_capacities(self) -> NDArray[np.double]:
        """Return mode heat capacities, shape (num_temp, num_gp, num_band0)."""
        assert self._cv is not None
        return self._cv

    @property
    def kappa(self) -> NDArray[np.double]:
        """Return LBTE thermal conductivity, shape (num_sigma, num_temp, 6)."""
        return self._kappa_solver.solver.kappa

    @property
    def kappa_RTA(self) -> NDArray[np.double]:
        """Return RTA thermal conductivity, shape (num_sigma, num_temp, 6)."""
        return self._kappa_solver.solver.kappa_RTA

    @property
    def mode_kappa(self) -> NDArray[np.double]:
        """Return mode LBTE kappa, shape (num_sigma, num_temp, num_gp, num_band0, 6)."""
        return self._kappa_solver.solver.mode_kappa

    @property
    def mode_kappa_RTA(self) -> NDArray[np.double]:
        """Return mode RTA kappa, shape (num_sigma, num_temp, num_gp, num_band0, 6)."""
        return self._kappa_solver.solver.mode_kappa_RTA

    @property
    def collision_matrix(self) -> NDArray[np.double] | None:
        """Return assembled collision matrix."""
        return self._kappa_solver.solver.collision_matrix

    @collision_matrix.setter
    def collision_matrix(self, value: NDArray[np.double] | None) -> None:
        """Set collision matrix (for loading from file)."""
        self._kappa_solver.solver.collision_matrix = value

    @property
    def collision_eigenvalues(self) -> NDArray[np.double] | None:
        """Return eigenvalues of collision matrix."""
        return self._kappa_solver.solver.collision_eigenvalues

    @property
    def f_vectors(self) -> NDArray[np.double] | None:
        """Return f-vectors, shape (num_gp, num_band0, 3)."""
        return self._kappa_solver.solver.f_vectors

    @property
    def mfp(self) -> NDArray[np.double] | None:
        """Return mean free path, shape (num_sigma, num_temp, num_gp, num_band0, 3)."""
        return self._kappa_solver.solver.mfp

    def get_frequencies_all(self) -> NDArray[np.double]:
        """Return phonon frequencies on the full BZ grid."""
        return self._frequencies[self._kappa_settings.bz_grid.grg2bzg]

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def _allocate_values(self) -> None:
        """Allocate per-grid-point arrays."""
        num_sigma = len(self._kappa_settings.sigmas)
        num_temp = len(self._kappa_settings.temperatures)
        num_gp = len(self._kappa_settings.grid_points)
        num_band0 = len(self._kappa_settings.band_indices)
        num_band = self._frequencies.shape[1]

        self._gamma = np.zeros(
            (num_sigma, num_temp, num_gp, num_band0), order="C", dtype="double"
        )
        self._gv = np.zeros((num_gp, num_band0, 3), order="C", dtype="double")
        self._cv = np.zeros((num_temp, num_gp, num_band0), order="C", dtype="double")
        if self._isotope_solver is not None:
            self._gamma_iso = np.zeros(
                (num_sigma, num_gp, num_band0), order="C", dtype="double"
            )
        if self._is_full_pp:
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

    def _compute_at_grid_point(self, i_gp: int) -> None:
        """Compute collision matrix row and gamma at a single grid point."""
        _show_log_header(
            i_gp, self._kappa_settings, self._isotope_solver, self._log_level
        )
        grid_point = int(self._kappa_settings.grid_points[i_gp])

        collision_result = self._collision_solver.compute(grid_point)
        self._gamma[:, :, i_gp, :] = collision_result.gamma

        if self._is_full_pp and collision_result.averaged_pp is not None:
            self._averaged_pp_interaction[i_gp] = collision_result.averaged_pp

        self._kappa_solver.store(i_gp, collision_result)

        if self._log_level:
            self._show_log(i_gp, self._gv[i_gp])

    def _finalize(self) -> None:
        self._kappa_solver.finalize(self._build_grid_point_aggregates())

    # ------------------------------------------------------------------
    # Private: LBTE-specific helpers
    # ------------------------------------------------------------------

    def _show_log(self, i_gp: int, gv: NDArray[np.double]) -> None:
        bz_gp = self._kappa_settings.grid_points[i_gp]
        frequencies = self._frequencies[bz_gp][self._kappa_settings.band_indices]
        show_grid_point_frequencies_gv(
            frequencies,
            gv,
            gv_delta_q=getattr(self._velocity_solver, "gv_delta_q", None),
        )
