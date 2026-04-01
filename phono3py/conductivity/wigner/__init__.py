"""Wigner transport equation plugin for phono3py conductivity.

Importing this package registers the ``"SMM19-rta"`` and ``"SMM19-lbte"``
methods with the conductivity factory so that
``make_conductivity_calculator("SMM19-rta", ...)`` works out of the box.

This package can also be installed as a standalone ``phono3py-wigner`` package
via namespace packages, in which case the factory auto-discovery still works
because ``factory.py`` does ``try: import phono3py.conductivity.wigner``.

"""

from phono3py.conductivity.factory import register_calculator
from phono3py.conductivity.wigner.calculator_factory import (
    make_wigner_lbte_calculator,
    make_wigner_rta_calculator,
)

register_calculator("SMM19-rta".lower(), make_wigner_rta_calculator)
register_calculator("SMM19-lbte".lower(), make_wigner_lbte_calculator)
