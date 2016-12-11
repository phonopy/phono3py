import numpy
import platform
import os

try:
    from setuptools import setup, Extension
    use_setuptools = True
    print("setuptools is used.")
except ImportError:
    from distutils.core import setup, Extension
    use_setuptools = False
    print("distutils is used.")

include_dirs_numpy = [numpy.get_include()]
extra_link_args = []
cc = None
if 'CC' in os.environ:
    if 'clang' in os.environ['CC']:
        cc = 'clang'
    if 'gcc' in os.environ['CC']:
        cc = 'gcc'
if cc == 'gcc' or cc is None:
    extra_link_args.append('-lgomp')

# Workaround Python issue 21121
import sysconfig
config_var = sysconfig.get_config_var("CFLAGS")
if config_var is not None and "-Werror=declaration-after-statement" in config_var:
    os.environ['CFLAGS'] = config_var.replace(
        "-Werror=declaration-after-statement", "")    

sources = ['c/_phono3py.c',
           'c/harmonic/dynmat.c',
           'c/harmonic/lapack_wrapper.c',
           'c/harmonic/phonoc_array.c',
           'c/harmonic/phonoc_utils.c',
           'c/anharmonic/phonon3/fc3.c',
           'c/anharmonic/phonon3/frequency_shift.c',
           'c/anharmonic/phonon3/interaction.c',
           'c/anharmonic/phonon3/real_to_reciprocal.c',
           'c/anharmonic/phonon3/reciprocal_to_normal.c',
           'c/anharmonic/phonon3/imag_self_energy.c',
           'c/anharmonic/phonon3/imag_self_energy_with_g.c',
           'c/anharmonic/phonon3/collision_matrix.c',
           'c/anharmonic/other/isotope.c',
           'c/anharmonic/triplet/triplet.c',
           'c/anharmonic/triplet/triplet_kpoint.c',
           'c/anharmonic/triplet/triplet_iw.c',
           'c/spglib/mathfunc.c',
           'c/spglib/kpoint.c',
           'c/kspclib/kgrid.c',
           'c/kspclib/tetrahedron_method.c']

extra_compile_args = ['-fopenmp',]
include_dirs = (['c/harmonic_h',
                 'c/anharmonic_h',
                 'c/spglib_h',
                 'c/kspclib_h'] +
                include_dirs_numpy)
define_macros = []

## Modify extra_link_args_lapacke depending on systems
if os.path.isfile("libopenblas.py"):
    # This is for travis-CI.
    from libopenblas import extra_link_args_lapacke
else:
    # This is when lapacke is installed on system
    extra_link_args_lapacke = ['-llapacke', '-llapack', '-lblas']

# For MacPort
# % sudo port install gcc6
# % sudo port select --set gcc mp-gcc
# % sudo port install OpenBLAS +gcc6
if platform.system() == 'Darwin':
    include_dirs += ['/opt/local/include']
    extra_link_args_lapacke = ['/opt/local/lib/libopenblas.a']

## Uncomment below to measure reciprocal_to_normal_squared_openmp performance
# define_macros = [('MEASURE_R2N', None)]

extra_link_args += extra_link_args_lapacke
extension_phono3py = Extension(
    'phono3py._phono3py',
    include_dirs=include_dirs,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    define_macros=define_macros,
    sources=sources)

packages_phono3py = ['phono3py',
                     'phono3py.cui',
                     'phono3py.other',
                     'phono3py.phonon',
                     'phono3py.phonon3']
scripts_phono3py = ['scripts/phono3py',
                    'scripts/kaccum',
                    'scripts/kdeplot']

########################
# _lapackepy extension #
########################
include_dirs_lapackepy = ['c/harmonic_h',] + include_dirs_numpy
sources_lapackepy = ['c/_lapackepy.c',
                     'c/harmonic/dynmat.c',
                     'c/harmonic/phonon.c',
                     'c/harmonic/phonoc_array.c',
                     'c/harmonic/phonoc_utils.c',
                     'c/harmonic/lapack_wrapper.c']
extension_lapackepy = Extension(
    'phono3py._lapackepy',
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    include_dirs=include_dirs,
    sources=sources_lapackepy)

if __name__ == '__main__':
    version_nums = [None, None, None]
    with open("phono3py/version.py") as w:
        for line in w:
            if "__version__" in line:
                for i, num in enumerate(line.split()[2].strip('\"').split('.')):
                    version_nums[i] = int(num)

    # To deploy to pypi by travis-CI
    if os.path.isfile("__nanoversion__.txt"):
        with open('__nanoversion__.txt') as nv:
            nanoversion = 0
            try :
                for line in nv:
                    nanoversion = int(line.strip())
                    break
            except ValueError :
                nanoversion = 0
            if nanoversion:
                version_nums.append(nanoversion)

    if None in version_nums:
        print("Failed to get version number in setup.py.")
        raise

    version_number = ".".join(["%d" % n for n in version_nums])
    if use_setuptools:
        setup(name='phono3py',
              version=version_number,
              description='This is the phono3py module.',
              author='Atsushi Togo',
              author_email='atz.togo@gmail.com',
              url='http://atztogo.github.io/phono3py/',
              packages=packages_phono3py,
              requires=['numpy', 'PyYAML', 'matplotlib', 'h5py', 'phonopy'],
              provides=['phono3py'],
              scripts=scripts_phono3py,
              ext_modules=[extension_lapackepy, extension_phono3py])
    else:
        setup(name='phono3py',
              version=version_number,
              description='This is the phono3py module.',
              author='Atsushi Togo',
              author_email='atz.togo@gmail.com',
              url='http://atztogo.github.io/phono3py/',
              packages=packages_phono3py,
              requires=['numpy', 'PyYAML', 'matplotlib', 'h5py', 'phonopy'],
              provides=['phono3py'],
              scripts=scripts_phono3py,
              ext_modules=[extension_lapackepy, extension_phono3py])
