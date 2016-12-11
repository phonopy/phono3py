#!/bin/bash

export LBL=$1
export TKN=$2
export GIT_BRANCH=$3

echo "extra_link_args_lapacke = ['-lopenblas']" |tee libopenblas.py
export C_INCLUDE_PATH=${HOME}/miniconda/include:${C_INCLUDE_PATH}
export LD_LIBRARY_PATH=${HOME}/miniconda/lib:${LD_LIBRARY_PATH}
conda install conda-build anaconda-client --yes
conda config --add channels atztogo
conda build conda --no-anaconda-upload
TRG=`conda build conda --output |sed -e 's/--/-*-/'`
echo "Uploading: $TRG"
anaconda --token $TKN upload --label $LBL $TRG

