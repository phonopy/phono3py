# SPDX-License-Identifier: BSD-3-Clause
"""Shortcuts to important classes, methods, and variables."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version

try:
    __version__ = _package_version("phono3py")
except PackageNotFoundError:  # running from a source tree without an install
    try:
        from phono3py._version import __version__
    except ImportError:
        __version__ = "0.0.0"

from phono3py.api_isotope import Phono3pyIsotope
from phono3py.api_jointdos import Phono3pyJointDos
from phono3py.api_phono3py import Phono3py
from phono3py.cui.load import load

__all__ = [
    "Phono3pyIsotope",
    "Phono3pyJointDos",
    "Phono3py",
    "load",
    "__version__",
]
