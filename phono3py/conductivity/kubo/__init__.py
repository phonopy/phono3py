"""Green-Kubo RTA plugin for phono3py conductivity.

Importing this package registers the ``"kubo-rta"`` method with the
conductivity factory so that
``make_conductivity_calculator("kubo-rta", ...)`` works out of the box.

"""

from phono3py.conductivity.factory import register_calculator
from phono3py.conductivity.kubo.calculator_factory import make_kubo_rta_calculator

register_calculator("kubo-rta", make_kubo_rta_calculator)
