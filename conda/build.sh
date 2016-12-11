#!/bin/bash

git describe --tags --dirty | sed -e 's/-\(.*\)-g.*/+\1/' -e 's/^[vr]//g' > __conda_version__.txt

# echo "1.11.3" > __conda_version__.txt

./get_nanoversion.sh

export C_INCLUDE_PATH=${HOME}/miniconda/include:${C_INCLUDE_PATH}
export LD_LIBRARY_PATH=${HOME}/miniconda/lib:${LD_LIBRARY_PATH}
echo "extra_link_args_lapacke = ['-lopenblas']" |tee libopenblas.py

$PYTHON setup.py install

# Add more build steps here, if they are necessary.

# See
# http://docs.continuum.io/conda/build.html
# for a list of environment variables that are set during the build process.
