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
    if build_dir.exists():
        shutil.rmtree(build_dir)


def _get_extensions(build_dir):
    # Initialization of parameters
    define_macros = []
    extra_link_args = []
    extra_compile_args = []
    include_dirs = []

    # Libraray search
    found_extra_link_args = []
    found_extra_compile_args = []
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
        print("extra_compile_args: ", found_extra_compile_args)
        print("extra_link_args: ", found_extra_link_args)
        print("define_macros: ", define_macros)
        print("=============================================")
        print()

    # Build ext_modules
    extensions = []
    extra_link_args += found_extra_link_args
    extra_compile_args += found_extra_link_args
    define_macros.append(("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"))
    include_dirs += ["c", numpy.get_include()]

    extra_link_args_ph3py = []
    libph3py = list((pathlib.Path.cwd() / "_build").glob("*ph3py.*"))
    if libph3py:
        print("=============================================")
        print(f"Phono3py library: {libph3py[0]}")
        print("=============================================")
        extra_link_args_ph3py += [str(libph3py[0])]

    extensions.append(
        setuptools.Extension(
            "phono3py._phono3py",
            sources=["c/_phono3py.c"],
            extra_link_args=extra_link_args + extra_link_args_ph3py,
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
            define_macros=define_macros,
        )
    )

    extra_link_args_phmod = []
    libphmod = list((pathlib.Path.cwd() / "_build").glob("*phmod.*"))
    if libphmod:
        print("=============================================")
        print(f"Phonon library: {libphmod[0]}")
        print("=============================================")
        extra_link_args_phmod += [str(libphmod[0])]

    extensions.append(
        setuptools.Extension(
            "phono3py._phononmod",
            sources=["c/_phononmod.c"],
            extra_link_args=extra_link_args + extra_link_args_phmod,
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
            define_macros=define_macros,
        )
    )
    return extensions


def _get_version() -> str:
    git_num = None
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
    return version


def main(build_dir):
    """Run setuptools."""
    version = _get_version()

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
        ext_modules=_get_extensions(build_dir),
    )
    _clean_cmake(build_dir)


if __name__ == "__main__":
    build_dir = pathlib.Path.cwd() / "_build"
    main(build_dir)
