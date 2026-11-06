# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from phono3py.cui.phono3py_script import main


def run() -> None:
    """Run phono3py-load script."""
    argparse_control = {
        "load_phono3py_yaml": True,
        "mode": "run",
        "deprecated_command": "phono3py-load",
    }
    main(**argparse_control)
