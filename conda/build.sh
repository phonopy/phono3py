#!/bin/bash

# export PATH=${HOME}/.miniconda/bin:${PATH}
# export C_INCLUDE_PATH=${HOME}/.miniconda/include:${C_INCLUDE_PATH}
# export LD_LIBRARY_PATH=${HOME}/.miniconda/lib:${LD_LIBRARY_PATH}
# conda activate travis

git checkout $GIT_BRANCH

./get_nanoversion.sh

$PYTHON setup.py install

# Add more build steps here, if they are necessary.

# See
# http://docs.continuum.io/conda/build.html
# for a list of environment variables that are set during the build process.
