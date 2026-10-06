# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from phono3py.cui.phono3py_script import main


def run() -> None:
    """Run phono3py-init script."""
    argparse_control = {
        "load_phono3py_yaml": False,
        "mode": "init",
    }
    main(**argparse_control)
