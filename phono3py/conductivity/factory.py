"""Factory for creating conductivity calculator instances.

Built-in methods
----------------
"rta"
    Standard BTE in the relaxation time approximation.
"lbte"
    Standard BTE via direct LBTE solution.
"wigner-rta"
    Wigner transport equation in RTA (registered plugin, can be overridden).
"wigner-lbte"
    Wigner transport equation via LBTE (registered plugin, can be overridden).
"kubo-rta"
    Green-Kubo formula in RTA (registered plugin, can be overridden).

External plugins
----------------
Register a factory callable via register_calculator():

    from phono3py.conductivity import register_calculator

    def my_factory(interaction, **kwargs):
        ...
        return MyCalculator(...)

    register_calculator("my-method", my_factory)

The factory receives the Interaction object as the first positional argument
and all keyword arguments that were passed to make_conductivity_calculator().

"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any, Callable, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from phono3py.conductivity.lbte_calculator import LBTECalculator
from phono3py.conductivity.rta_calculator import ConductivityCalculator
from phono3py.phonon3.interaction import Interaction

# ---------------------------------------------------------------------------
# Plugin registry
# ---------------------------------------------------------------------------

CalculatorFactory: TypeAlias = Callable[..., Any]

# "rta" and "lbte" are registered at module load and cannot be overridden.
# All other built-in methods (wigner-*, kubo-*) are registered below and can
# be overridden by external plugins.
_BUILTIN_METHODS: frozenset[str] = frozenset({"rta", "lbte"})

# Populated at module load with all built-in factories.
# External plugins are added via register_calculator().
_REGISTRY: dict[str, CalculatorFactory] = {}


def register_calculator(method: str, factory: CalculatorFactory) -> None:
    """Register an external calculator factory for a new method name.

    The factory is called as::

        calculator = factory(interaction, **kwargs)

    where ``kwargs`` are all keyword arguments passed to
    ``make_conductivity_calculator()`` except ``method`` itself.

    Parameters
    ----------
    method : str
        Method name used as the ``method`` argument of
        ``make_conductivity_calculator()``.  Must not conflict with
        built-in method names.
    factory : callable
        Factory function.  Signature: ``(interaction, **kwargs) -> Any``.
        May accept only the kwargs it needs and ignore the rest.

    Raises
    ------
    ValueError
        If ``method`` conflicts with a built-in method name.

    Examples
    --------
    Register a custom RTA-based calculator::

        from phono3py.conductivity import register_calculator

        def my_factory(interaction, *, temperatures=None, sigmas=None, **kwargs):
            return MyCalculator(interaction, temperatures=temperatures, ...)

        register_calculator("my-rta", my_factory)

    """
    if method in _BUILTIN_METHODS:
        raise ValueError(
            f"'{method}' is a built-in method and cannot be overridden. "
            f"Built-in methods: {sorted(_BUILTIN_METHODS)}."
        )
    _REGISTRY[method] = factory


def make_conductivity_calculator(
    interaction: Interaction,
    method: str = "rta",
    grid_points: Sequence[int] | NDArray[np.int64] | None = None,
    temperatures: Sequence[float] | NDArray[np.double] | None = None,
    sigmas: Sequence[float | None] | None = None,
    sigma_cutoff: float | None = None,
    is_isotope: bool = False,
    mass_variances: Sequence[float] | NDArray[np.double] | None = None,
    boundary_mfp: float | None = None,
    use_ave_pp: bool = False,
    is_kappa_star: bool = True,
    gv_delta_q: float | None = None,
    is_full_pp: bool = False,
    read_pp: bool = False,
    store_pp: bool = False,
    pp_filename: str | os.PathLike | None = None,
    is_N_U: bool = False,
    is_gamma_detail: bool = False,
    is_reducible_collision_matrix: bool = False,
    solve_collective_phonon: bool = False,
    pinv_cutoff: float = 1.0e-8,
    pinv_solver: int = 0,
    pinv_method: int = 0,
    lang: Literal["C", "Python"] = "C",
    log_level: int = 0,
) -> ConductivityCalculator | LBTECalculator:
    """Create a conductivity calculator with the appropriate building blocks.

    Parameters
    ----------
    interaction : Interaction
        Interaction instance.  init_dynamical_matrix must have been called.
    method : str, optional
        Calculation method.  Built-in: "rta", "lbte", "wigner-rta",
        "wigner-lbte", "kubo-rta".  Default "rta".
    grid_points : array-like or None, optional
        BZ grid point indices.  None uses irreducible grid points.  Default None.
    temperatures : array-like or None, optional
        Temperatures in Kelvin.  Default None.
    sigmas : sequence or None, optional
        Smearing widths.  None selects the tetrahedron method.
    sigma_cutoff : float or None, optional
        Smearing cutoff in units of sigma.  Default None.
    is_isotope : bool, optional
        Include isotope scattering.  Default False.
    mass_variances : array-like or None, optional
        Mass variances for isotope scattering.  Default None.
    boundary_mfp : float or None, optional
        Boundary mean free path in micrometres.  Default None.
    use_ave_pp : bool, optional
        Use pre-averaged ph-ph interaction (RTA only).  Default False.
    is_kappa_star : bool, optional
        Use k-star symmetry.  Default True.
    gv_delta_q : float or None, optional
        Finite-difference step for group velocity (NAC).  Default None.
    is_full_pp : bool, optional
        Compute full ph-ph interaction matrix.  Default False.
    read_pp : bool, optional
        Read ph-ph interaction from file.  Default False.
    store_pp : bool, optional
        Store ph-ph interaction to file (RTA only).  Default False.
    pp_filename : str or path or None, optional
        Filename for ph-ph interaction I/O.  Default None.
    is_N_U : bool, optional
        Decompose gamma into Normal and Umklapp (RTA only).  Default False.
    is_gamma_detail : bool, optional
        Store per-triplet gamma (RTA only).  Default False.
    is_reducible_collision_matrix : bool, optional
        Use full reducible collision matrix (LBTE only).  Default False.
    solve_collective_phonon : bool, optional
        Use Chaput collective-phonon method (LBTE only).  Default False.
    pinv_cutoff : float, optional
        Eigenvalue cutoff for pseudo-inversion (LBTE only).  Default 1e-8.
    pinv_solver : int, optional
        Solver selection index (LBTE only).  Default 0.
    pinv_method : int, optional
        Pseudo-inverse criterion (LBTE only).  Default 0.
    lang : {"C", "Python"}, optional
        Backend for C-extension operations (LBTE only).  Default "C".
    log_level : int, optional
        Verbosity.  Default 0.

    Returns
    -------
    ConductivityCalculator or LBTECalculator
        For plugin-registered methods the return type is determined by the
        registered factory.

    """
    if method not in _REGISTRY:
        raise NotImplementedError(
            f"method='{method}' is not implemented. "
            f"Built-in methods: {sorted(_BUILTIN_METHODS)}. "
            "Registered methods: "
            f"{sorted(_REGISTRY)}. "
            "Use register_calculator() to add custom methods."
        )

    return _REGISTRY[method](
        interaction,
        grid_points=grid_points,
        temperatures=temperatures,
        sigmas=sigmas,
        sigma_cutoff=sigma_cutoff,
        is_isotope=is_isotope,
        mass_variances=mass_variances,
        boundary_mfp=boundary_mfp,
        use_ave_pp=use_ave_pp,
        is_kappa_star=is_kappa_star,
        gv_delta_q=gv_delta_q,
        is_full_pp=is_full_pp,
        read_pp=read_pp,
        store_pp=store_pp,
        pp_filename=pp_filename,
        is_N_U=is_N_U,
        is_gamma_detail=is_gamma_detail,
        is_reducible_collision_matrix=is_reducible_collision_matrix,
        solve_collective_phonon=solve_collective_phonon,
        pinv_cutoff=pinv_cutoff,
        pinv_solver=pinv_solver,
        pinv_method=pinv_method,
        lang=lang,
        log_level=log_level,
    )


# ---------------------------------------------------------------------------
# Register all built-in factories.
# "rta" and "lbte" are protected by _BUILTIN_METHODS.
# "wigner-*" and "kubo-*" can be overridden by external plugins.
# The try/except allows each to be installed as a standalone package via
# namespace packages; if absent, the method is simply not available.
# ---------------------------------------------------------------------------

from phono3py.conductivity.calculator_factory import (  # noqa: E402
    make_lbte_calculator,
    make_rta_calculator,
)

_REGISTRY["rta"] = make_rta_calculator
_REGISTRY["lbte"] = make_lbte_calculator

try:
    import phono3py.conductivity.kubo  # noqa: F401, E402
except ImportError:
    pass

try:
    import phono3py.conductivity.wigner  # noqa: F401, E402
except ImportError:
    pass
