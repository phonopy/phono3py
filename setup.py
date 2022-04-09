"""Phono3py setup.py."""
import os

import numpy
import setuptools

# Ensure that 'site.cfg' exists.
if not os.path.exists("site.cfg"):
    msg_list = [
        '"site.cfg" file is needed to run setup.py.',
        "See about installation at https://phonopy.github.io/phono3py/install.html.",
        "A minimum setting of site.cfg to build with openmp support is:",
        "# ------------------------------",
        "[phono3py]",
        "extra_compile_args = -fopenmp",
        "# ------------------------------",
        "Please create an emply site.cfg (no-openmp support) to run setup.py",
        "unless any custom setting is needed, although this is considered unusual.",
    ]

    raise FileNotFoundError("\n".join(msg_list))

# Retrieve the default flags from the numpy installation
# This also means one can override this with a site.cfg
# configuration file
from numpy.distutils.system_info import dict_append, get_info, system_info

git_num = None

# use flags defined in numpy
all_info_d = get_info("ALL")
lapack_info_d = get_info("lapack_opt")


class phono3py_info(system_info):
    """See system_info in numpy."""

    section = "phono3py"

    def calc_info(self):
        """Read in *all* options in the [phono3py] section of site.cfg."""
        info = self.calc_libraries_info()
        dict_append(info, **self.calc_extra_info())
        dict_append(info, include_dirs=self.get_include_dirs())
        self.set_info(**info)


macros = []

# in numpy>=1.16.0, silence build warnings about deprecated API usage
macros.append(("NPY_NO_DEPRECATED_API", "0"))

# Avoid divergence in tetrahedron method by ensuring denominator > 1e-10.
# macros.append(("THM_EPSILON", "1e-10"))

with_threaded_blas = False
with_mkl = False

# define options
# these are the basic definitions for all extensions
opts = lapack_info_d.copy()
if "mkl" in opts.get("libraries", ""):
    with_mkl = True

if with_mkl:
    with_threaded_blas = True
    # generally this should not be needed since the numpy distutils
    # finding of MKL creates the SCIPY_MKL_H flag
    macros.append(("MKL_LAPACKE", None))

if with_threaded_blas:
    macros.append(("MULTITHREADED_BLAS", None))

# Create the dictionary for compiling the codes
dict_append(opts, **all_info_d)
dict_append(opts, include_dirs=["c"])
dict_append(opts, define_macros=macros)
# Add numpy's headers
include_dirs = numpy.get_include()
if include_dirs is not None:
    dict_append(opts, include_dirs=[include_dirs])

# Add any phono3py manual flags from here
add_opts = phono3py_info().get_info()
dict_append(opts, **add_opts)

# Different extensions
extensions = []

# Define the modules
sources_phono3py = [
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
extensions.append(
    setuptools.Extension("phono3py._phono3py", sources=sources_phono3py, **opts)
)


sources_phononmod = [
    "c/_phononmod.c",
    "c/dynmat.c",
    "c/lapack_wrapper.c",
    "c/phonon.c",
    "c/phononmod.c",
]
extensions.append(
    setuptools.Extension("phono3py._phononmod", sources=sources_phononmod, **opts)
)

sources_lapackepy = ["c/_lapackepy.c", "c/lapack_wrapper.c"]
extensions.append(
    setuptools.Extension("phono3py._lapackepy", sources=sources_lapackepy, **opts)
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
            "phonopy>=2.14,<2.15",
        ],
        provides=["phono3py"],
        scripts=scripts_phono3py,
        ext_modules=extensions,
    )
