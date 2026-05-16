"""Dispatch logging and backend resolution for the C/Rust switch.

The dedicated ``phono3py.lang`` logger records which backend is selected
when a lang-aware routine is entered.  It stays silent by default.

To see the messages, either:

* set the environment variable ``PHONO3PY_TRACE_LANG=1`` before running
  (this configures a ``StreamHandler`` to stderr at DEBUG level), or
* configure ``logging.getLogger("phono3py.lang")`` from client code.

"""

from __future__ import annotations

import logging
import os
from typing import Literal

_logger = logging.getLogger("phono3py.lang")

if os.environ.get("PHONO3PY_TRACE_LANG"):
    if not _logger.handlers:
        _handler = logging.StreamHandler()
        _handler.setFormatter(logging.Formatter("[phono3py.lang] %(message)s"))
        _logger.addHandler(_handler)
    _logger.setLevel(logging.DEBUG)


def log_dispatch(lang: str, name: str) -> None:
    """Emit a dispatch-level trace line for a lang-aware call site."""
    _logger.debug("dispatch name=%s lang=%s", name, lang)


def have_c_ext() -> bool:
    """Return True when the ``phono3py._phono3py`` C extension is importable."""
    try:
        import phono3py._phono3py  # noqa: F401  # type: ignore[import-untyped]
    except ImportError:
        return False
    return True


def have_phonors() -> bool:
    """Return True when the ``phonors`` Rust extension is importable."""
    try:
        import phonors  # noqa: F401  # type: ignore[import-untyped]
    except ImportError:
        return False
    return True


def c_omp_max_threads() -> int:
    """Return ``phono3c.omp_max_threads()`` or ``0`` if C extension is absent."""
    try:
        import phono3py._phono3py as phono3c  # type: ignore[import-untyped]
    except ImportError:
        return 0
    return int(phono3c.omp_max_threads())


def c_include_lapacke() -> bool:
    """Return ``phono3c.include_lapacke()`` or ``False`` if C extension is absent."""
    try:
        import phono3py._phono3py as phono3c  # type: ignore[import-untyped]
    except ImportError:
        return False
    return bool(phono3c.include_lapacke())


def c_default_colmat_solver() -> int:
    """Return ``phono3c.default_colmat_solver()`` or ``4`` if C extension is absent.

    ``4`` is the historical fallback used when the C extension cannot be
    queried (numpy.linalg.eigh path).

    """
    try:
        import phono3py._phono3py as phono3c  # type: ignore[import-untyped]
    except ImportError:
        return 4
    return int(phono3c.default_colmat_solver())


_fallback_warned = False


def resolve_lang(lang: Literal["C", "Rust"]) -> Literal["C", "Rust"]:
    """Pick the best available backend, falling back from C to Rust.

    When ``lang == "C"`` but the C extension is not installed (e.g. a
    ``PHONO3PY_NO_C_EXT=1`` build), flip to ``"Rust"`` and emit a one-time
    informational message.  Raise ``ImportError`` if neither backend is
    available, or if Rust is requested without ``phonors`` installed.

    """
    global _fallback_warned

    if lang == "Rust":
        if not have_phonors():
            raise ImportError(
                "lang='Rust' was requested but the `phonors` package is not "
                "installed.  Install it (e.g. `pip install phonors`) or use "
                "lang='C'."
            )
        return "Rust"

    if have_c_ext():
        return "C"

    if have_phonors():
        if not _fallback_warned:
            print(
                "[phono3py] C extension `phono3py._phono3py` is not available; "
                "falling back to lang='Rust' via the `phonors` package."
            )
            _fallback_warned = True
        return "Rust"

    raise ImportError(
        "Neither the `phono3py._phono3py` C extension nor the `phonors` Rust "
        "package is importable.  Reinstall phono3py with C support, or "
        "install `phonors`."
    )
