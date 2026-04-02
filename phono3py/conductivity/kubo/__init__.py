"""Green-Kubo plugin for phono3py conductivity.

Importing this package registers the ``"kubo-rta"`` and ``"kubo-lbte"``
methods with the conductivity factory so that
``make_conductivity_calculator("kubo-rta", ...)`` and
``make_conductivity_calculator("kubo-lbte", ...)`` work out of the box.

"""

from phono3py.conductivity.factory import register_calculator
from phono3py.conductivity.kubo.calculator_factory import (
    make_kubo_lbte_calculator,
    make_kubo_rta_calculator,
)

register_calculator("kubo-rta", make_kubo_rta_calculator)
register_calculator("kubo-lbte", make_kubo_lbte_calculator)
