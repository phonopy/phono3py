"""Phono3py setup.py.

The build steps are as follows:

mkdir _build && cd _build
cmake -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
    -DPHONONMOD=on -DPHONO3PY=on -DCMAKE_INSTALL_PREFIX="" ..
cmake --build . -v
cd ..
python setup.py build
pip install -e .

"""
import os
import pathlib
import shutil
import subprocess

import numpy
import setuptools


def _run_cmake(build_dir):
    build_dir.mkdir()
    args = [
        "cmake",
        "-S",
        ".",
        "-B",
        "_build",
        "-DPHONONMOD=on",
        "-DPHONO3PY=on",
        "-DCMAKE_INSTALL_PREFIX=.",
    ]
    # if "CONDA_PREFIX" in os.environ:
    #     args.append("-DUSE_CONDA_PATH=on")
    # if "CC" in os.environ:
    #     args.append(f'-DCMAKE_C_COMPILER={os.environ["CC"]}')

    cmake_output = subprocess.check_output(args)
    print(cmake_output.decode("utf-8"))
    subprocess.check_call(["cmake", "--build", "_build", "-v"])
    return cmake_output


def _clean_cmake(build_dir):
    shutil.rmtree(build_dir)


build_dir = pathlib.Path.cwd() / "_build"
found_extra_link_args = []
found_extra_compile_args = []
define_macros = []
use_mkl_lapacke = False
if build_dir.exists():
    pass
else:
    cmake_output = _run_cmake(build_dir)
    found_flags = {}
    found_libs = {}
    for line in cmake_output.decode("utf-8").split("\n"):
        for key in ["BLAS", "LAPACK", "OpenMP"]:
            if f"{key} libs" in line and len(line.split()) > 3:
                found_libs[key] = line.split()[3].split(";")
            if f"{key} flags" in line and len(line.split()) > 3:
                found_flags[key] = line.split()[3].split(";")
    for key, value in found_libs.items():
        found_extra_link_args += value
        for element in value:
            if "libmkl" in element:
                use_mkl_lapacke = True
    for key, value in found_flags.items():
        found_extra_compile_args += value
    if use_mkl_lapacke:
        define_macros.append(("MKL_LAPACKE", None))
    print("=============================================")
    print(found_extra_compile_args)
    print(found_extra_link_args)
    print(define_macros)
    print("=============================================")

git_num = None
include_dirs = ["c", numpy.get_include()]
# if "CONDA_PREFIX" in os.environ:
#     include_dirs.append(os.environ["CONDA_PREFIX"] + "/include")

# Different extensions
extensions = []

# Define the modules
extensions.append(
    setuptools.Extension(
        "phono3py._phono3py",
        sources=["c/_phono3py.c"],
        extra_link_args=["_build/libph3py.a"] + found_extra_link_args,
        include_dirs=include_dirs,
        extra_compile_args=found_extra_compile_args,
        define_macros=define_macros,
    )
)

extensions.append(
    setuptools.Extension(
        "phono3py._phononmod",
        sources=["c/_phononmod.c"],
        extra_link_args=["_build/libphmod.a"] + found_extra_link_args,
        include_dirs=include_dirs,
        extra_compile_args=found_extra_compile_args,
        define_macros=define_macros,
    )
)

packages_phono3py = [
    "phono3py",
    "phono3py.conductivity",
    "phono3py.cui",
    "phono3py.interface",
    "phono3py.other",
    "phono3py.phonon",
    "phono3py.phonon3",
    "phono3py.sscha",
]
scripts_phono3py = [
    "scripts/phono3py",
    "scripts/phono3py-load",
    "scripts/phono3py-kaccum",
    "scripts/phono3py-kdeplot",
    "scripts/phono3py-coleigplot",
]

if __name__ == "__main__":
    version_nums = [None, None, None]
    with open("phono3py/version.py") as w:
        for line in w:
            if "__version__" in line:
                for i, num in enumerate(line.split()[2].strip('"').split(".")):
                    version_nums[i] = num
                break

    # To deploy to pypi by travis-CI
    if os.path.isfile("__nanoversion__.txt"):
        nanoversion = 0
        with open("__nanoversion__.txt") as nv:
            try:
                for line in nv:
                    nanoversion = int(line.strip())
                    break
            except ValueError:
                nanoversion = 0
        if nanoversion != 0:
            version_nums.append(nanoversion)
    elif git_num:
        version_nums.append(git_num)

    if None in version_nums:
        print("Failed to get version number in setup.py.")
        raise

    version = ".".join(["%s" % n for n in version_nums[:3]])
    if len(version_nums) > 3:
        version += "-%s" % version_nums[3]

    setuptools.setup(
        name="phono3py",
        version=version,
        description="This is the phono3py module.",
        author="Atsushi Togo",
        author_email="atz.togo@gmail.com",
        url="http://phonopy.github.io/phono3py/",
        packages=packages_phono3py,
        python_requires=">=3.7",
        install_requires=[
            "numpy>=1.15.0",
            "scipy",
            "PyYAML",
            "matplotlib>=2.2.2",
            "h5py",
            "spglib",
            "phonopy>=2.15,<2.16",
        ],
        provides=["phono3py"],
        scripts=scripts_phono3py,
        ext_modules=extensions,
    )

    _clean_cmake(build_dir)
