# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from phono3py.cui.phono3py_script import main


def run() -> None:
    """Run phono3py script."""
    argparse_control = {
        "load_phono3py_yaml": True,
        "mode": "run",
    }
    main(**argparse_control)
