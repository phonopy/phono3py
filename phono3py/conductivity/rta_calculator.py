"""ConductivityCalculator: composition-based lattice thermal conductivity."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from phono3py.conductivity.grid_point_data import (
    GridPointInput,
    GridPointResult,
    make_grid_point_input,
)
from phono3py.conductivity.heat_capacity_providers import ModeHeatCapacityProvider
from phono3py.conductivity.kappa_accumulators import KappaAccumulator
from phono3py.conductivity.scattering_providers import (
    BoundaryScatteringProvider,
    IsotopeScatteringProvider,
    RTAScatteringProvider,
)
from phono3py.conductivity.utils import get_unit_to_WmK, show_grid_point_header
from phono3py.conductivity.velocity_providers import GroupVelocityProvider
from phono3py.other.isotope import Isotope
from phono3py.phonon.grid import (
    BZGrid,
    get_grid_points_by_rotations,
    get_ir_grid_points,
    get_qpoints_from_bz_grid_points,
)
from phono3py.phonon3.interaction import Interaction


class ConductivityCalculator:
    """Lattice thermal conductivity calculator using composed building blocks.

    This class replaces the ``ConductivityBase`` / ``ConductivityRTABase`` /
    ``ConductivityRTA`` inheritance hierarchy.  Physical building blocks
    (velocity, heat-capacity, scattering, kappa accumulator) are injected via
    the constructor rather than inherited.

    Variant-specific output properties (``kappa_P_RTA``, ``kappa_C``, etc.)
    are not defined on this class.  They are forwarded transparently to the
    accumulator via ``__getattr__``.

    Usage
    -----
    Create via ``make_conductivity_calculator`` (recommended) or directly::

        calc = ConductivityCalculator(pp, vel, cv, scat, accumulator, ...)
        calc.run(on_grid_point=lambda i: writer.write_gamma(calc, i))
        # results available via calc.kappa, calc.mode_kappa, etc.

    Parameters
    ----------
    pp : Interaction
        Interaction instance. ``init_dynamical_matrix`` must have been called.
    velocity_provider : GroupVelocityProvider
        Computes group velocities and v-outer-v at each grid point.
    cv_provider : ModeHeatCapacityProvider
        Computes mode heat capacities at each grid point.
    scattering_provider : RTAScatteringProvider
        Computes ph-ph linewidths at each grid point.
    accumulator : KappaAccumulator
        Owns the kappa formula and BZ-summation arrays; exposes ``kappa``,
        ``mode_kappa``, and any variant-specific output properties.
    grid_points : array-like or None, optional
        BZ grid point indices to iterate over. None uses irreducible grid
        points. Default None.
    temperatures : array-like or None, optional
        Temperatures in Kelvin. Default None.
    sigmas : sequence or None, optional
        Smearing widths. None selects the tetrahedron method.
    sigma_cutoff : float or None, optional
        Smearing cutoff in units of sigma. Default None.
    is_isotope : bool, optional
        Include isotope scattering. Default False.
    mass_variances : array-like or None, optional
        Mass variances for isotope scattering. Default None.
    boundary_mfp : float or None, optional
        Boundary mean free path in micrometres. Default None.
    is_kappa_star : bool, optional
        Use k-star symmetry. Default True.
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
        velocity_provider: GroupVelocityProvider,
        cv_provider: ModeHeatCapacityProvider,
        scattering_provider: RTAScatteringProvider,
        accumulator: KappaAccumulator,
        grid_points: Sequence[int] | NDArray[np.int64] | None = None,
        temperatures: Sequence[float] | NDArray[np.double] | None = None,
        sigmas: Sequence[float | None] | None = None,
        sigma_cutoff: float | None = None,
        is_isotope: bool = False,
        mass_variances: Sequence[float] | NDArray[np.double] | None = None,
        boundary_mfp: float | None = None,
        is_kappa_star: bool = True,
        is_N_U: bool = False,
        is_gamma_detail: bool = False,
        log_level: int = 0,
    ):
        """Init method."""
        self._pp = pp
        self._velocity_provider = velocity_provider
        self._cv_provider = cv_provider
        self._scattering_provider = scattering_provider
        self._accumulator = accumulator
        self._sigmas: list[float | None] = [] if sigmas is None else list(sigmas)
        self._sigma_cutoff = sigma_cutoff
        self._boundary_mfp = boundary_mfp
        self._is_kappa_star = is_kappa_star
        self._is_N_U = is_N_U
        self._is_gamma_detail = is_gamma_detail
        self._log_level = log_level

        # Ensure phonons are solved.
        self._pp.nac_q_direction = None
        self._pp.run_phonon_solver_at_gamma()
        self._frequencies, self._eigenvectors, self._phonon_done = (
            self._pp.get_phonons()
        )
        if not self._pp.phonon_all_done:
            self._pp.run_phonon_solver()

        # Grid setup.
        self._point_operations, self._rotations_cartesian = (
            self._build_point_operations()
        )
        self._grid_points, self._ir_grid_points, self._grid_weights = (
            self._build_grid_info(grid_points)
        )
        self._grid_point_count = 0

        self._temperatures: NDArray[np.double] | None = (
            np.asarray(temperatures, dtype="double")
            if temperatures is not None
            else None
        )

        # Isotope and boundary providers.
        self._is_isotope = is_isotope or (mass_variances is not None)
        self._isotope_provider: IsotopeScatteringProvider | None = None
        if self._is_isotope:
            self._isotope_provider = self._build_isotope_provider(mass_variances)
        self._boundary_provider: BoundaryScatteringProvider | None = (
            BoundaryScatteringProvider(boundary_mfp)
            if boundary_mfp is not None
            else None
        )

        # Read flags (set via property setters when gamma is loaded from file).
        self._read_gamma = False
        self._read_gamma_iso = False

        # Declare arrays; allocated lazily when temperatures are set.
        self._gamma: NDArray[np.double]
        self._gamma_N: NDArray[np.double] | None = None
        self._gamma_U: NDArray[np.double] | None = None
        self._gamma_iso: NDArray[np.double] | None = None
        self._gamma_elph: NDArray[np.double] | None = None
        self._gamma_boundary: NDArray[np.double] | None = None
        self._averaged_pp_interaction: NDArray[np.double] | None = None
        self._gv: NDArray[np.double]
        self._gv_by_gv: NDArray[np.double]
        # Wigner/Kubo: operator/matrix outer product, complex.
        # Shape: (num_gp, num_band0, num_band, 6).
        # Allocated lazily on first encounter; None for standard RTA.
        self._gv_operator_sum2: NDArray[np.cdouble] | None = None
        # Kubo: heat capacity matrix (num_temp, num_gp, num_band0, num_band).
        # Allocated lazily; None for standard RTA and Wigner.
        self._cv_mat: NDArray[np.double] | None = None
        self._cv: NDArray[np.double]
        self._num_sampling_grid_points = 0
        self._num_ignored_phonon_modes: NDArray[np.int64] | None = None
        self._gamma_detail_at_q: NDArray[np.double] | None = None

        self._conversion_factor = get_unit_to_WmK() / self._pp.primitive.volume

        if self._temperatures is not None:
            self._allocate_values()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        on_grid_point: Callable[[int], None] | None = None,
    ) -> None:
        """Run all grid points and compute kappa.

        Parameters
        ----------
        on_grid_point : callable or None, optional
            Called with the grid-point count (0-based index) after each grid
            point is processed. Used for per-grid-point file writes.

        """
        if self._temperatures is None:
            raise RuntimeError("Set temperatures before calling run().")

        if self._log_level:
            print(
                "==================== Lattice thermal conductivity (RTA) "
                "===================="
            )

        self._prepare_isotope_phonons()

        self._num_sampling_grid_points = 0
        self._grid_point_count = 0

        for i_gp in range(len(self._grid_points)):
            self._run_at_grid_point(i_gp)
            self._grid_point_count = i_gp + 1
            if on_grid_point is not None:
                on_grid_point(i_gp)

        if self._log_level:
            print(
                "=================== End of collection of collisions "
                "==================="
            )

        self._accumulator.finalize(self._num_sampling_grid_points)

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
    def frequencies(self) -> NDArray[np.double]:
        """Return phonon frequencies at the iterated grid points."""
        assert self._frequencies is not None
        return self._frequencies[self._grid_points]

    @property
    def qpoints(self) -> NDArray[np.double]:
        """Return q-point coordinates of the iterated grid points."""
        return np.array(
            get_qpoints_from_bz_grid_points(self._grid_points, self._pp.bz_grid),
            dtype="double",
            order="C",
        )

    @property
    def grid_points(self) -> NDArray[np.int64]:
        """Return BZ grid point indices that were iterated."""
        return self._grid_points

    @property
    def grid_weights(self) -> NDArray[np.int64]:
        """Return symmetry weights of the iterated grid points."""
        return self._grid_weights

    @property
    def temperatures(self) -> NDArray[np.double] | None:
        """Return temperatures in Kelvin."""
        return self._temperatures

    @temperatures.setter
    def temperatures(self, temperatures: Sequence[float] | NDArray[np.double]) -> None:
        self._temperatures = np.asarray(temperatures, dtype="double")
        self._allocate_values()

    @property
    def sigmas(self) -> list[float | None]:
        """Return smearing widths."""
        return self._sigmas

    @property
    def sigma_cutoff_width(self) -> float | None:
        """Return smearing cutoff width."""
        return self._sigma_cutoff

    @property
    def boundary_mfp(self) -> float | None:
        """Return boundary mean free path in micrometres."""
        return self._boundary_mfp

    @property
    def grid_point_count(self) -> int:
        """Return number of grid points processed so far."""
        return self._grid_point_count

    @property
    def number_of_sampling_grid_points(self) -> int:
        """Return total number of sampling grid points (sum of k-star orders)."""
        return self._num_sampling_grid_points

    # ------------------------------------------------------------------
    # Properties — computed physical quantities
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> object:
        """Delegate unknown attribute lookups to the accumulator.

        This allows variant-specific properties (kappa_P_RTA, kappa_C,
        kappa_TOT_RTA, mode_kappa_C, ...) to be accessed directly on the
        calculator without hard-coding them here.

        """
        # Avoid infinite recursion if _accumulator is not yet set.
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            acc = object.__getattribute__(self, "_accumulator")
        except AttributeError:
            raise AttributeError(name) from None
        try:
            return getattr(acc, name)
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            ) from None

    def get_extra_kappa_output(self) -> dict[str, Any] | None:
        """Return variant-specific kappa output from the accumulator.

        Called by output writers to obtain plugin-defined quantities (e.g.
        Wigner kappa_P_RTA, kappa_C) that are written to the hdf5 file via
        ``write_kappa_to_hdf5(extra_datasets=...)``.

        Returns None when the accumulator does not implement this method
        (standard RTA and Kubo).

        """
        fn = getattr(self._accumulator, "get_extra_kappa_output", None)
        return fn() if callable(fn) else None

    def show_rta_progress(self, log_level: int) -> bool:
        """Show variant-specific RTA kappa progress if the accumulator supports it.

        Delegates to ``accumulator.show_rta_progress(self, log_level)`` when
        available (e.g. WignerKappaAccumulator).

        Returns
        -------
        bool
            True when the accumulator handled the display; False otherwise
            (caller should fall back to the standard progress display).

        """
        fn = getattr(self._accumulator, "show_rta_progress", None)
        if callable(fn):
            fn(self, log_level)
            return True
        return False

    @property
    def kappa(self) -> NDArray[np.double]:
        """Return total kappa tensor.

        shape (num_sigma, num_temp, 6).

        """
        return self._accumulator.kappa

    @property
    def mode_kappa(self) -> NDArray[np.double]:
        """Return mode-resolved kappa.

        shape (num_sigma, num_temp, num_gp, num_band0, 6).

        """
        return self._accumulator.mode_kappa

    @property
    def group_velocities(self) -> NDArray[np.double]:
        """Return group velocities.

        shape (num_gp, num_band0, 3).

        """
        return self._gv

    @property
    def gv_by_gv(self) -> NDArray[np.double]:
        """Return symmetrised v-outer-v.

        shape (num_gp, num_band0, 6).

        """
        return self._gv_by_gv

    @property
    def mode_heat_capacities(self) -> NDArray[np.double]:
        """Return mode heat capacities.

        shape (num_temp, num_gp, num_band0).

        """
        return self._cv

    @property
    def number_of_ignored_phonon_modes(self) -> NDArray[np.int64] | None:
        """Return count of ignored modes.

        shape (num_sigma, num_temp).

        """
        return self._num_ignored_phonon_modes

    # ------------------------------------------------------------------
    # Properties — scattering arrays (with setters for file reads)
    # ------------------------------------------------------------------

    @property
    def gamma(self) -> NDArray[np.double]:
        """Return ph-ph gamma.

        shape (num_sigma, num_temp, num_gp, num_band0).

        """
        return self._gamma

    @gamma.setter
    def gamma(self, gamma: NDArray[np.double]) -> None:
        self._gamma = gamma
        self._read_gamma = True

    @property
    def gamma_isotope(self) -> NDArray[np.double] | None:
        """Return isotope gamma.

        shape (num_sigma, num_gp, num_band0).

        """
        return self._gamma_iso

    @gamma_isotope.setter
    def gamma_isotope(self, gamma_iso: NDArray[np.double] | None) -> None:
        self._gamma_iso = gamma_iso
        self._read_gamma_iso = gamma_iso is not None

    @property
    def gamma_elph(self) -> NDArray[np.double] | None:
        """Return el-ph gamma.

        shape (num_sigma, num_temp, num_gp, num_band0).

        """
        return self._gamma_elph

    @gamma_elph.setter
    def gamma_elph(self, gamma_elph: NDArray[np.double] | None) -> None:
        self._gamma_elph = gamma_elph

    @property
    def averaged_pp_interaction(self) -> NDArray[np.double] | None:
        """Return averaged ph-ph interaction.

        shape (num_gp, num_band0).

        """
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
    # Private: grid setup
    # ------------------------------------------------------------------

    def _build_point_operations(
        self,
    ) -> tuple[NDArray[np.int64], NDArray[np.double]]:
        if not self._is_kappa_star:
            point_ops = np.eye(3, dtype="int64", order="C").reshape(1, 3, 3)
            rot_cart = np.eye(3, dtype="double", order="C").reshape(1, 3, 3)
        else:
            point_ops = self._pp.bz_grid.reciprocal_operations
            rot_cart = self._pp.bz_grid.rotations_cartesian
        return point_ops, rot_cart

    def _build_grid_info(
        self,
        grid_points: Sequence[int] | NDArray[np.int64] | None,
    ) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
        ir_gps, ir_weights = self._get_ir_grid_points(grid_points)
        if grid_points is not None:
            gps = np.array(grid_points, dtype="int64")
            return gps, ir_gps, ir_weights
        if not self._is_kappa_star:
            all_gps = self._pp.bz_grid.grg2bzg
            return all_gps, all_gps, np.ones(len(all_gps), dtype="int64")
        return ir_gps, ir_gps, ir_weights

    def _get_ir_grid_points(
        self,
        grid_points: Sequence[int] | NDArray[np.int64] | None,
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        ir_gps, ir_weights, ir_map = get_ir_grid_points(self._pp.bz_grid)
        ir_gps = np.array(self._pp.bz_grid.grg2bzg[ir_gps], dtype="int64")
        if grid_points is None:
            return ir_gps, ir_weights
        weights = np.zeros_like(ir_map)
        for gp in ir_map:
            weights[gp] += 1
        gp_weights = np.array(
            weights[ir_map[self._pp.bz_grid.bzg2grg[grid_points]]], dtype="int64"
        )
        return ir_gps, gp_weights

    # ------------------------------------------------------------------
    # Private: array allocation
    # ------------------------------------------------------------------

    def _allocate_values(self) -> None:
        assert self._temperatures is not None
        num_sigma = len(self._sigmas)
        num_temp = len(self._temperatures)
        num_gp = len(self._grid_points)
        num_band0 = len(self._pp.band_indices)

        if not self._read_gamma:
            self._gamma = np.zeros(
                (num_sigma, num_temp, num_gp, num_band0), order="C", dtype="double"
            )
            if self._is_N_U or self._is_gamma_detail:
                self._gamma_N = np.zeros_like(self._gamma)
                self._gamma_U = np.zeros_like(self._gamma)
        if self._is_isotope:
            self._gamma_iso = np.zeros(
                (num_sigma, num_gp, num_band0), order="C", dtype="double"
            )
        if self._scattering_provider.is_full_pp:
            self._averaged_pp_interaction = np.zeros(
                (num_gp, num_band0), order="C", dtype="double"
            )
        if self._boundary_mfp is not None:
            self._gamma_boundary = np.zeros(
                (num_gp, num_band0), order="C", dtype="double"
            )
        self._gv = np.zeros((num_gp, num_band0, 3), order="C", dtype="double")
        self._gv_by_gv = np.zeros((num_gp, num_band0, 6), order="C", dtype="double")
        self._cv = np.zeros((num_temp, num_gp, num_band0), order="C", dtype="double")
        self._accumulator.prepare(num_sigma, num_temp, num_gp, num_band0)
        self._num_ignored_phonon_modes = np.zeros(
            (num_sigma, num_temp), order="C", dtype="int64"
        )

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
    # Private: per-grid-point computation
    # ------------------------------------------------------------------

    def _make_grid_point_input(self, i_gp: int) -> GridPointInput:
        assert self._frequencies is not None
        assert self._eigenvectors is not None
        return make_grid_point_input(
            grid_point=int(self._grid_points[i_gp]),
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
            if self._is_isotope and self._isotope_provider is not None
            else None
        )
        show_grid_point_header(
            bzgp=self._grid_points[i_gp],
            i_gp=i_gp,
            num_gps=len(self._grid_points),
            bz_grid=self._pp.bz_grid,
            boundary_mfp=self._boundary_mfp,
            mass_variances=mass_variances,
        )

    def _run_at_grid_point(self, i_gp: int) -> None:
        self._show_log_header(i_gp)
        gp_input = self._make_grid_point_input(i_gp)

        # Velocity.
        vel_result = self._velocity_provider.compute(gp_input)
        assert vel_result.group_velocities is not None
        assert vel_result.velocity_product is not None
        self._gv[i_gp] = vel_result.group_velocities
        vp = vel_result.velocity_product
        if vp.ndim == 2:
            # Standard RTA: (num_band0, 6) real.
            self._gv_by_gv[i_gp] = vp
        else:
            # Wigner: (num_band0, num_band, 6) complex — lazy allocation.
            if self._gv_operator_sum2 is None:
                self._gv_operator_sum2 = np.zeros(
                    (len(self._grid_points),) + vp.shape,
                    dtype="complex128",
                    order="C",
                )
            self._gv_operator_sum2[i_gp] = vp
        self._num_sampling_grid_points += vel_result.num_sampling_grid_points

        # Heat capacity.
        assert self._temperatures is not None
        cv_result = self._cv_provider.compute(gp_input, self._temperatures)
        assert cv_result.heat_capacities is not None
        self._cv[:, i_gp, :] = cv_result.heat_capacities
        if cv_result.heat_capacity_matrix is not None:
            if self._cv_mat is None:
                self._cv_mat = np.zeros(
                    (len(self._temperatures), len(self._grid_points))
                    + cv_result.heat_capacity_matrix.shape[1:],
                    dtype="double",
                    order="C",
                )
            self._cv_mat[:, i_gp, :, :] = cv_result.heat_capacity_matrix

        # ph-ph scattering.
        if not self._read_gamma:
            scat_result = self._scattering_provider.compute_gamma(gp_input)
            assert scat_result.gamma is not None
            self._gamma[:, :, i_gp, :] = scat_result.gamma
            if scat_result.averaged_pp_interaction is not None:
                assert self._averaged_pp_interaction is not None
                self._averaged_pp_interaction[i_gp] = (
                    scat_result.averaged_pp_interaction
                )
            if self._is_N_U or self._is_gamma_detail:
                g_N = self._scattering_provider.gamma_N
                g_U = self._scattering_provider.gamma_U
                if g_N is not None and self._gamma_N is not None:
                    self._gamma_N[:, :, i_gp, :] = g_N
                if g_U is not None and self._gamma_U is not None:
                    self._gamma_U[:, :, i_gp, :] = g_U
            self._gamma_detail_at_q = self._scattering_provider.gamma_detail_at_q
        elif self._log_level:
            print("  Gamma is read from file.")

        # Isotope scattering.
        if self._is_isotope and not self._read_gamma_iso:
            assert self._isotope_provider is not None
            iso_result = self._isotope_provider.compute_gamma_isotope(gp_input)
            assert iso_result.gamma_isotope is not None
            assert self._gamma_iso is not None
            self._gamma_iso[:, i_gp, :] = iso_result.gamma_isotope[
                :, self._pp.band_indices
            ]

        # Boundary scattering (needs group velocities computed above).
        if self._boundary_provider is not None:
            bd_result = self._boundary_provider.compute_gamma_boundary_from_gv(
                gp_input, self._gv[i_gp]
            )
            assert bd_result.gamma_boundary is not None
            assert self._gamma_boundary is not None
            self._gamma_boundary[i_gp] = bd_result.gamma_boundary

        if self._log_level:
            self._show_log(i_gp)

        result = self._make_grid_point_result(i_gp, gp_input)
        self._accumulator.accumulate(i_gp, result)
        self._count_ignored_modes(i_gp, result)

    def _show_log(self, i_gp: int) -> None:
        gp = self._grid_points[i_gp]
        qpoint = get_qpoints_from_bz_grid_points(gp, self._pp.bz_grid)
        assert self._frequencies is not None
        frequencies = self._frequencies[gp][self._pp.band_indices]
        gv = self._gv[i_gp]
        ave_pp = (
            self._averaged_pp_interaction[i_gp]
            if self._averaged_pp_interaction is not None
            else None
        )
        self._show_log_value_names()
        if self._log_level > 2:
            self._show_log_values_on_kstar(frequencies, gv, ave_pp, gp, qpoint)
        else:
            self._show_log_values(frequencies, gv, ave_pp)
        print("", end="", flush=True)

    def _show_log_values(
        self,
        frequencies: NDArray[np.double],
        gv: NDArray[np.double],
        ave_pp: NDArray[np.double] | None,
    ) -> None:
        if self._scattering_provider.is_full_pp:
            assert ave_pp is not None
            for f, v, pp in zip(frequencies, gv, ave_pp, strict=True):
                print(
                    "%8.3f   (%8.3f %8.3f %8.3f) %8.3f %11.3e"
                    % (f, v[0], v[1], v[2], np.linalg.norm(v), pp)
                )
        else:
            for f, v in zip(frequencies, gv, strict=True):
                print(
                    "%8.3f   (%8.3f %8.3f %8.3f) %8.3f"
                    % (f, v[0], v[1], v[2], np.linalg.norm(v))
                )

    def _show_log_values_on_kstar(
        self,
        frequencies: NDArray[np.double],
        gv: NDArray[np.double],
        ave_pp: NDArray[np.double] | None,
        gp: int,
        q: NDArray[np.double],
    ) -> None:
        rotation_map = get_grid_points_by_rotations(gp, self._pp.bz_grid)
        for i, j in enumerate(np.unique(rotation_map)):
            for k, (rot, rot_c) in enumerate(
                zip(self._point_operations, self._rotations_cartesian, strict=True)
            ):
                if rotation_map[k] != j:
                    continue
                q_rot = tuple(np.dot(rot, q))
                print(" k*%-2d (%5.2f %5.2f %5.2f)" % ((i + 1,) + q_rot))
                if self._scattering_provider.is_full_pp:
                    assert ave_pp is not None
                    for f, v, pp in zip(
                        frequencies, np.dot(rot_c, gv.T).T, ave_pp, strict=True
                    ):
                        print(
                            "%8.3f   (%8.3f %8.3f %8.3f) %8.3f %11.3e"
                            % (f, v[0], v[1], v[2], np.linalg.norm(v), pp)
                        )
                else:
                    for f, v in zip(frequencies, np.dot(rot_c, gv.T).T, strict=True):
                        print(
                            "%8.3f   (%8.3f %8.3f %8.3f) %8.3f"
                            % (f, v[0], v[1], v[2], np.linalg.norm(v))
                        )
        print("")

    def _show_log_value_names(self) -> None:
        if self._scattering_provider.is_full_pp:
            text = "Frequency     group velocity (x, y, z)     |gv|       Pqj"
        else:
            text = "Frequency     group velocity (x, y, z)     |gv|"
        gv_delta_q = getattr(self._velocity_provider, "gv_delta_q", None)
        if gv_delta_q is not None:
            text += "  (dq=%3.1e)" % gv_delta_q
        print(text)

    # ------------------------------------------------------------------
    # Private: kappa computation helpers
    # ------------------------------------------------------------------

    def _make_grid_point_result(
        self, i_gp: int, gp_input: GridPointInput
    ) -> GridPointResult:
        """Build GridPointResult for grid point i_gp from stored arrays."""
        result = GridPointResult(input=gp_input)
        result.group_velocities = self._gv[i_gp]
        if self._gv_operator_sum2 is not None:
            result.velocity_product = self._gv_operator_sum2[i_gp]
        else:
            result.velocity_product = self._gv_by_gv[i_gp]
        result.heat_capacities = self._cv[:, i_gp, :]
        if self._cv_mat is not None:
            result.heat_capacity_matrix = self._cv_mat[:, i_gp, :, :]
        result.gamma = self._gamma[:, :, i_gp, :]
        if self._gamma_iso is not None:
            result.gamma_isotope = self._gamma_iso[:, i_gp, :]
        if self._gamma_elph is not None:
            result.gamma_elph = self._gamma_elph[:, :, i_gp, :]
        if self._gamma_boundary is not None:
            result.gamma_boundary = self._gamma_boundary[i_gp]
        return result

    def _count_ignored_modes(self, i_gp: int, result: GridPointResult) -> None:
        """Increment ignored-mode counter for modes below cutoff or negative gamma."""
        assert self._num_ignored_phonon_modes is not None
        assert self._temperatures is not None
        weight = int(self._grid_weights[i_gp])
        freq = self._frequencies[self._grid_points[i_gp]]  # type: ignore
        for j in range(len(self._sigmas)):
            for k in range(len(self._temperatures)):
                g_eff = result.gamma[j, k].copy()
                if result.gamma_isotope is not None:
                    g_eff += result.gamma_isotope[j]
                if result.gamma_elph is not None:
                    g_eff += result.gamma_elph[j, k]
                if result.gamma_boundary is not None:
                    g_eff += result.gamma_boundary
                for ll, f in enumerate(freq[self._pp.band_indices]):
                    if f < self._pp.cutoff_frequency or g_eff[ll] < 0:
                        self._num_ignored_phonon_modes[j, k] += weight
