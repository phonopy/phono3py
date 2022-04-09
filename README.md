[![Version Badge](https://anaconda.org/conda-forge/phono3py/badges/version.svg)](https://anaconda.org/conda-forge/phono3py)
[![Downloads Badge](https://anaconda.org/conda-forge/phono3py/badges/downloads.svg)](https://anaconda.org/conda-forge/phono3py)
[![PyPI version](https://badge.fury.io/py/phono3py.svg)](https://badge.fury.io/py/phono3py)
[![PyPI](https://img.shields.io/pypi/dm/phono3py.svg?maxAge=2592000)](https://pypi.python.org/pypi/phono3py)
[![codecov](https://codecov.io/gh/phonopy/phono3py/branch/develop/graph/badge.svg)](https://codecov.io/gh/phonopy/phono3py)

# phono3py

A simulation package of phonon-phonon interaction related properties. Phono3py
user documentation is found at http://phonopy.github.io/phono3py/.

## Mailing list for questions

Usual phono3py questions should be sent to phonopy mailing list
(https://sourceforge.net/p/phonopy/mailman/).

## Development

The development of phono3py is managed on the `develop` branch of github
phono3py repository.

- Github issues is the place to discuss about phono3py issues.
- Github pull request is the place to request merging source code.
- Python 3.7 is the minimum requirement.
- Formatting is written in `pyproject.toml`.
- Not strictly, but VSCode's `settings.json` may be written like

  ```json
  "python.linting.flake8Enabled": true,
  "python.linting.flake8Args": ["--max-line-length=88", "--ignore=E203,W503"],
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": false,
  "python.linting.mypyEnabled": true,
  "python.linting.pycodestyleEnabled": false,
  "python.linting.pydocstyleEnabled": true,
  "python.formatting.provider": "black",
  "python.formatting.blackArgs": ["--line-length=88"],
  "python.sortImports.args": ["--profile", "black"],
  "[python]": {
      "editor.codeActionsOnSave": {
      "source.organizeImports": true
    },
  }
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

You need pytest. At home directory of phono3py after setup,

```bash
% pytest
```
