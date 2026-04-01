"""Wigner transport equation plugin for phono3py conductivity.

Importing this package registers the ``"wigner-rta"`` and ``"wigner-lbte"``
methods with the conductivity factory so that
``make_conductivity_calculator("wigner-rta", ...)`` works out of the box.

This package can also be installed as a standalone ``phono3py-wigner`` package
via namespace packages, in which case the factory auto-discovery still works
because ``factory.py`` does ``try: import phono3py.conductivity.wigner``.

"""

from phono3py.conductivity.factory import register_calculator
from phono3py.conductivity.wigner.calculator_factory import (
    make_wigner_lbte_calculator,
    make_wigner_rta_calculator,
)

register_calculator("wigner-rta", make_wigner_rta_calculator)
register_calculator("wigner-lbte", make_wigner_lbte_calculator)
