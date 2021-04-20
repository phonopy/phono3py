cmake_minimum_required(VERSION 3.0)
cmake_policy(SET CMP0007 NEW)

project(phono3py C)
set(CMAKE_MACOSX_RPATH 1)
set(CMAKE_C_FLAGS_RELEASE "-Wall -O2")
set(CMAKE_C_FLAGS_DEBUG "-g -DLAGWARNING -DTHMWARNING")
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif(NOT CMAKE_BUILD_TYPE)
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

include(GNUInstallDirs)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)


# Version numbers
file(READ ${PROJECT_SOURCE_DIR}/phono3py/version.py version_file)
string(REGEX MATCH "__version__.*([0-9]+)[.]([0-9]+)[.]([0-9]+)" phono3py_version ${version_file})
set(phono3py_major_version ${CMAKE_MATCH_1})
set(phono3py_minor_version ${CMAKE_MATCH_2})
set(phono3py_micro_version ${CMAKE_MATCH_3})
set(serial "${phono3py_major_version}.${phono3py_minor_version}.${phono3py_micro_version}")
set(soserial "1")

find_package(OpenMP)
if (OPENMP_FOUND)
  set(openmp "-lgomp")
else()
  set(openmp "-lomp")
endif()

include_directories($ENV{CONDA_PREFIX}/include)
link_directories($ENV{CONDA_PREFIX}/lib)
set(LAPACK_LIBRARIES "-llapack")
if (PHONO3PY OR PHONONMOD)
  if (USE_MKL)
    message(STATUS "BLAS library: MKL")
    set(BLAS_LIBRARIES "-lmkl_rt")
    add_compile_definitions(MKL_LAPACKE)
  else()
    message(STATUS "BLAS library: OpenBLAS")
    set(BLAS_LIBRARIES "-lopenblas")
  endif()
endif()

############
# phono3py #
############
if (PHONO3PY)
  # Source code
  include_directories("${PROJECT_SOURCE_DIR}/c")
  set(SOURCES_PHONO3PY
    ${PROJECT_SOURCE_DIR}/c/bzgrid.c
    ${PROJECT_SOURCE_DIR}/c/collision_matrix.c
    ${PROJECT_SOURCE_DIR}/c/fc3.c
    ${PROJECT_SOURCE_DIR}/c/grgrid.c
    ${PROJECT_SOURCE_DIR}/c/imag_self_energy_with_g.c
    ${PROJECT_SOURCE_DIR}/c/interaction.c
    ${PROJECT_SOURCE_DIR}/c/isotope.c
    ${PROJECT_SOURCE_DIR}/c/lagrid.c
    ${PROJECT_SOURCE_DIR}/c/lapack_wrapper.c
    ${PROJECT_SOURCE_DIR}/c/phono3py.c
    ${PROJECT_SOURCE_DIR}/c/phonoc_utils.c
    ${PROJECT_SOURCE_DIR}/c/pp_collision.c
    ${PROJECT_SOURCE_DIR}/c/real_self_energy.c
    ${PROJECT_SOURCE_DIR}/c/real_to_reciprocal.c
    ${PROJECT_SOURCE_DIR}/c/reciprocal_to_normal.c
    ${PROJECT_SOURCE_DIR}/c/snf3x3.c
    ${PROJECT_SOURCE_DIR}/c/tetrahedron_method.c
    ${PROJECT_SOURCE_DIR}/c/triplet.c
    ${PROJECT_SOURCE_DIR}/c/triplet_grid.c
    ${PROJECT_SOURCE_DIR}/c/triplet_iw.c)

  # Shared library
  add_library(ph3py SHARED ${SOURCES_PHONO3PY})

  if(NOT MSVC)
    target_link_libraries(ph3py m ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} ${openmp})
  endif()
  set_property(TARGET ph3py PROPERTY VERSION ${serial})
  set_property(TARGET ph3py PROPERTY SOVERSION ${soserial})
  install(TARGETS ph3py LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

  # Static link library
  add_library(ph3py_static STATIC ${SOURCES_PHONO3PY})
  set_property(TARGET ph3py_static PROPERTY VERSION ${serial})
  set_property(TARGET ph3py_static PROPERTY SOVERSION ${soserial})
  set_property(TARGET ph3py_static PROPERTY OUTPUT_NAME ph3py)
  install(TARGETS ph3py_static ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

  # Header file
  install(FILES ${PROJECT_SOURCE_DIR}/c/phono3py.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()

#############
# phononmod #
#############
if (PHONONMOD)
  # Source code
  set(SOURCES_PHONONMOD
    ${PROJECT_SOURCE_DIR}/c/dynmat.c
    ${PROJECT_SOURCE_DIR}/c/lapack_wrapper.c
    ${PROJECT_SOURCE_DIR}/c/phonon.c
    ${PROJECT_SOURCE_DIR}/c/phononmod.c)

  # Shared library
  add_library(phmod SHARED ${SOURCES_PHONONMOD})

  if(NOT MSVC)
    target_link_libraries(phmod m ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} ${openmp})
  endif()
  set_property(TARGET phmod PROPERTY VERSION ${serial})
  set_property(TARGET phmod PROPERTY SOVERSION ${soserial})
  install(TARGETS phmod LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

  # Static link library
  add_library(phmod_static STATIC ${SOURCES_PHONONMOD})
  set_property(TARGET phmod_static PROPERTY VERSION ${serial})
  set_property(TARGET phmod_static PROPERTY SOVERSION ${soserial})
  set_property(TARGET phmod_static PROPERTY OUTPUT_NAME phmod)
  install(TARGETS phmod_static ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

  # Header file
  install(FILES ${PROJECT_SOURCE_DIR}/c/phononmod.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()

########
# grid #
########
if (GRIDSYS)
  # Source code
  set(SOURCES_GRIDSYSMOD
    ${PROJECT_SOURCE_DIR}/c/bzgrid.c
    ${PROJECT_SOURCE_DIR}/c/grgrid.c
    ${PROJECT_SOURCE_DIR}/c/gridsys.c
    ${PROJECT_SOURCE_DIR}/c/lagrid.c
    ${PROJECT_SOURCE_DIR}/c/phonoc_utils.c
    ${PROJECT_SOURCE_DIR}/c/snf3x3.c
    ${PROJECT_SOURCE_DIR}/c/tetrahedron_method.c
    ${PROJECT_SOURCE_DIR}/c/triplet.c
    ${PROJECT_SOURCE_DIR}/c/triplet_grid.c
    ${PROJECT_SOURCE_DIR}/c/triplet_iw.c)

  # Shared library
  add_library(gridsysmod SHARED ${SOURCES_GRIDSYSMOD})

  # if(NOT MSVC)
  #   target_link_libraries(gridsysmod m ${openmp})
  # endif()
  set_property(TARGET gridsysmod PROPERTY VERSION ${serial})
  set_property(TARGET gridsysmod PROPERTY SOVERSION ${soserial})
  install(TARGETS gridsysmod LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

  # Static link library
  add_library(gridsysmod_static STATIC ${SOURCES_GRIDSYSMOD})
  set_property(TARGET gridsysmod_static PROPERTY VERSION ${serial})
  set_property(TARGET gridsysmod_static PROPERTY SOVERSION ${soserial})
  set_property(TARGET gridsysmod_static PROPERTY OUTPUT_NAME gridsysmod)
  install(TARGETS gridsysmod_static ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

  # Header file
  install(FILES ${PROJECT_SOURCE_DIR}/c/gridsys.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()
