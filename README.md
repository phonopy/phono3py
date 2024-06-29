[![Version Badge](https://anaconda.org/conda-forge/phono3py/badges/version.svg)](https://anaconda.org/conda-forge/phono3py)
[![Downloads Badge](https://anaconda.org/conda-forge/phono3py/badges/downloads.svg)](https://anaconda.org/conda-forge/phono3py)
[![PyPI version](https://badge.fury.io/py/phono3py.svg)](https://badge.fury.io/py/phono3py)
[![PyPI](https://img.shields.io/pypi/dm/phono3py.svg?maxAge=2592000)](https://pypi.python.org/pypi/phono3py)
[![codecov](https://codecov.io/gh/phonopy/phono3py/branch/develop/graph/badge.svg)](https://codecov.io/gh/phonopy/phono3py)

# phono3py

A simulation package of phonon-phonon interaction related properties mainly
written in python. Phono3py user documentation is found at
http://phonopy.github.io/phono3py/.

## Mailing list for questions

Usual phono3py questions should be sent to phonopy mailing list
(https://sourceforge.net/p/phonopy/mailman/).

## Dependency

See `requirements.txt`. Optionally `symfc` and `scipy` are required
for using additional features.

## Development

The development of phono3py is managed on the `develop` branch of github
phono3py repository.

- Github issues is the place to discuss about phono3py issues.
- Github pull request is the place to request merging source code.
- Formatting rules are found in `pyproject.toml`.
- Not strictly, but VSCode's `settings.json` may be written like below

  ```json
  "ruff.lint.args": [
      "--config=${workspaceFolder}/pyproject.toml",
  ],
  "[python]": {
      "editor.defaultFormatter": "charliermarsh.ruff",
      "editor.codeActionsOnSave": {
          "source.organizeImports": "explicit"
      }
  },
  ```

- Use of pre-commit (https://pre-commit.com/) is encouraged.
  - Installed by `pip install pre-commit`, `conda install pre_commit` or see
    https://pre-commit.com/#install.
  - pre-commit hook is installed by `pre-commit install`.
  - pre-commit hook is run by `pre-commit run --all-files`.

## Documentation

Phono3py user documentation is written using python sphinx. The source files are
stored in `doc` directory. Please see how to write the documentation at
`doc/README.md`.

## How to run tests

Tests are written using pytest. To run tests, pytest has to be installed. The
tests can be run by

```bash
% pytest
```
