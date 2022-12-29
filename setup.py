"""Phono3py setup.py.

Cmake handles automatic finding of library.
Custom settings should be written in site.cfg.

To fully customize using site.cfg,
set PHONO3PY_USE_CMAKE=false to avoid running cmake.

"""
import os
import pathlib
import shutil
import subprocess

import numpy
import setuptools

if (
    "PHONO3PY_USE_CMAKE" in os.environ
    and os.environ["PHONO3PY_USE_CMAKE"].lower() == "false"
):
    use_cmake = False
else:
    use_cmake = True


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


def _get_params_from_site_cfg():
    """Read extra_compile_args and extra_link_args.

    Examples
    --------
    # For macOS
    extra_compile_args = -fopenmp=libomp
    extra_link_args = -lomp -lopenblas

    # For linux
    extra_compile_args = -fopenmp
    extra_link_args = -lgomp  -lopenblas -lpthread

    """
    params = {
        "define_macros": [],
        "extra_link_args": [],
        "extra_compile_args": [],
        "extra_objects": [],
        "include_dirs": [],
    }
    use_mkl_lapacke = False
    use_threaded_blas = False

    site_cfg_file = pathlib.Path.cwd() / "site.cfg"
    if not site_cfg_file.exists():
        return params

    with open(site_cfg_file) as f:
        lines = [line.strip().split("=", maxsplit=1) for line in f]

        for line in lines:
            if len(line) < 2:
                continue
            key = line[0].strip()
            val = line[1].strip()
            if key not in params:
                continue
            if key == "define_macros":
                pair = val.split(maxsplit=1)
                if pair[1].lower() == "none":
                    pair[1] = None
                params[key].append(tuple(pair))
            else:
                if "mkl" in val:
                    use_mkl_lapacke = True
                if "openblas" in val:
                    use_threaded_blas = True
                params[key] += val.split()

    if use_mkl_lapacke:
        params["define_macros"].append(("MKL_LAPACKE", None))
    if use_threaded_blas:
        params["define_macros"].append(("MULTITHREADED_BLAS", None))
    if "THM_EPSILON" not in [macro[0] for macro in params["define_macros"]]:
        params["define_macros"].append(("THM_EPSILON", "1e-10"))

    print("=============================================")
    print("Parameters found in site.cfg")
    for key, val in params.items():
        print(f"{key}: {val}")
    print("=============================================")
    return params


def _get_extensions(build_dir):
    """Return python extension setting.

    User customization by site.cfg file
    -----------------------------------
    See _get_params_from_site_cfg().

    Automatic search using cmake
    ----------------------------
    Invoked by environment variable unless PHONO3PY_USE_CMAKE=false.

    """
    params = _get_params_from_site_cfg()
    extra_objects_ph3py = []
    extra_objects_phmod = []

    if not use_cmake or not shutil.which("cmake"):
        print("** Setup without using cmake **")
        sources_ph3py = [
            "c/_phono3py.c",
            "c/bzgrid.c",
            "c/collision_matrix.c",
            "c/fc3.c",
            "c/grgrid.c",
            "c/imag_self_energy_with_g.c",
            "c/interaction.c",
            "c/isotope.c",
            "c/lagrid.c",
            "c/lapack_wrapper.c",
            "c/phono3py.c",
            "c/phonoc_utils.c",
            "c/pp_collision.c",
            "c/real_self_energy.c",
            "c/real_to_reciprocal.c",
            "c/reciprocal_to_normal.c",
            "c/snf3x3.c",
            "c/tetrahedron_method.c",
            "c/triplet.c",
            "c/triplet_grid.c",
            "c/triplet_iw.c",
        ]
        sources_phmod = [
            "c/_phononmod.c",
            "c/dynmat.c",
            "c/lapack_wrapper.c",
            "c/phonon.c",
            "c/phononmod.c",
        ]
    else:
        print("** Setup using cmake **")
        use_mkl_lapacke = False
        found_extra_link_args = []
        found_extra_compile_args = []
        sources_ph3py = ["c/_phono3py.c"]
        sources_phmod = ["c/_phononmod.c"]
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
            params["define_macros"].append(("MKL_LAPACKE", None))

        libph3py = list((pathlib.Path.cwd() / "_build").glob("*ph3py.*"))
        if libph3py:
            print("=============================================")
            print(f"Phono3py library: {libph3py[0]}")
            print("=============================================")
            extra_objects_ph3py += [str(libph3py[0])]

        libphmod = list((pathlib.Path.cwd() / "_build").glob("*phmod.*"))
        if libphmod:
            print("=============================================")
            print(f"Phonon library: {libphmod[0]}")
            print("=============================================")
            extra_objects_phmod += [str(libphmod[0])]

        params["extra_link_args"] += found_extra_link_args
        params["extra_compile_args"] += found_extra_compile_args

        print("=============================================")
        print("Parameters found by cmake")
        print("extra_compile_args: ", found_extra_compile_args)
        print("extra_link_args: ", found_extra_link_args)
        print("define_macros: ", params["define_macros"])
        print("=============================================")
        print()

    extensions = []
    params["define_macros"].append(("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"))
    params["include_dirs"] += ["c", numpy.get_include()]

    extensions.append(
        setuptools.Extension(
            "phono3py._phono3py",
            sources=sources_ph3py,
            extra_link_args=params["extra_link_args"],
            include_dirs=params["include_dirs"],
            extra_compile_args=params["extra_compile_args"],
            extra_objects=params["extra_objects"] + extra_objects_ph3py,
            define_macros=params["define_macros"],
        )
    )

    extensions.append(
        setuptools.Extension(
            "phono3py._phononmod",
            sources=sources_phmod,
            extra_link_args=params["extra_link_args"],
            include_dirs=params["include_dirs"],
            extra_compile_args=params["extra_compile_args"],
            extra_objects=params["extra_objects"] + extra_objects_phmod,
            define_macros=params["define_macros"],
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
            "phonopy>=2.17,<2.18",
        ],
        provides=["phono3py"],
        scripts=scripts_phono3py,
        ext_modules=_get_extensions(build_dir),
    )
    _clean_cmake(build_dir)


if __name__ == "__main__":
    build_dir = pathlib.Path.cwd() / "_build"
    main(build_dir)
