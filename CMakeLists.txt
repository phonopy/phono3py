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
message(STATUS "CMAKE_SYSTEM_PREFIX_PATH: ${CMAKE_SYSTEM_PREFIX_PATH}")

include(GNUInstallDirs)
# set(CMAKE_POSITION_INDEPENDENT_CODE ON)


# Version numbers
file(READ ${PROJECT_SOURCE_DIR}/phono3py/version.py version_file)
string(REGEX MATCH "__version__.*([0-9]+)[.]([0-9]+)[.]([0-9]+)" phono3py_version ${version_file})
set(phono3py_major_version ${CMAKE_MATCH_1})
set(phono3py_minor_version ${CMAKE_MATCH_2})
set(phono3py_micro_version ${CMAKE_MATCH_3})
set(serial "${phono3py_major_version}.${phono3py_minor_version}.${phono3py_micro_version}")
set(soserial "1")

if (USE_CONDA_PATH)
  message(STATUS "$ENV{CONDA_PREFIX}")
  set(CMAKE_MODULE_PATH $ENV{CONDA_PREFIX})
  set(MY_INCLUDES $ENV{CONDA_PREFIX}/include ${PROJECT_SOURCE_DIR}/c)
  link_directories($ENV{CONDA_PREFIX}/lib)
else()
  set(MY_INCLUDES ${PROJECT_SOURCE_DIR}/c)
endif()

find_package(OpenMP)
if (OpenMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    message(STATUS "OpenMP libs: ${OpenMP_C_LIBRARIES}")
    message(STATUS "OpenMP flags: ${OpenMP_C_FLAGS}")
endif()

if (PHONO3PY OR PHONONMOD)
  find_package(BLAS REQUIRED)    # set BLAS_LIBRARIES
  if (BLAS_FOUND)
    message(STATUS "BLAS libs: ${BLAS_LIBRARIES}")
    message(STATUS "BLAS flags: ${BLAS_LINKER_FLAGS}")
  endif()
  find_package(LAPACK REQUIRED)  # set LAPACK_LIBRARIES
  if (LAPACK_FOUND)
    message(STATUS "LAPACK libs: ${LAPACK_LIBRARIES}")
    message(STATUS "LAPACK flags: ${LAPACK_LINKER_FLAGS}")
  endif()
endif()

if (PHONO3PY OR PHONONMOD)
  if (BLAS_LIBRARIES MATCHES "libmkl")
    message(STATUS "MKL detected: Set C-macros MKL_LAPACKE MULTITHREADED_BLAS")
    add_compile_definitions(MKL_LAPACKE)
    add_compile_definitions(MULTITHREADED_BLAS)  # phono3py macro
    add_compile_definitions(NPY_NO_DEPRECATED_API=0)  # phono3py macro
    add_compile_definitions(THM_EPSILON=1e-10)  # phono3py macro
  endif()
  if (BLAS_LIBRARIES MATCHES "libopenblas")
    message(STATUS "OpenBLAS detected: Set C-macro MULTITHREADED_BLAS")
    add_compile_definitions(MULTITHREADED_BLAS)  # phono3py macro
    add_compile_definitions(NPY_NO_DEPRECATED_API=0)  # phono3py macro
    add_compile_definitions(THM_EPSILON=1e-10)  # phono3py macro
  endif()
endif()

############
# phono3py #
############
if (PHONO3PY)
  # Source code
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

  if (BUILD_SHARED_LIBRARIES)
    # Shared library
    add_library(ph3py SHARED ${SOURCES_PHONO3PY})
    target_link_libraries(ph3py m ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} ${OpenMP_C_LIBRARIES})
    target_include_directories(ph3py PRIVATE ${MY_INCLUDES})
    set_property(TARGET ph3py PROPERTY VERSION ${serial})
    set_property(TARGET ph3py PROPERTY SOVERSION ${soserial})
    install(TARGETS ph3py LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
  endif()

  # Static link library
  add_library(ph3py_static STATIC ${SOURCES_PHONO3PY})
  target_link_libraries(ph3py_static m ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} ${OpenMP_C_LIBRARIES})
  target_include_directories(ph3py_static PRIVATE ${MY_INCLUDES})
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

  if (BUILD_SHARED_LIBRARIES)
    # Shared library
    add_library(phmod SHARED ${SOURCES_PHONONMOD})
    target_link_libraries(phmod m ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} ${OpenMP_C_LIBRARIES})
    target_include_directories(phmod PRIVATE ${MY_INCLUDES})
    set_property(TARGET phmod PROPERTY VERSION ${serial})
    set_property(TARGET phmod PROPERTY SOVERSION ${soserial})
    install(TARGETS phmod LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
  endif()

  # Static link library
  add_library(phmod_static STATIC ${SOURCES_PHONONMOD})
  target_link_libraries(phmod_static m ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} ${OpenMP_C_LIBRARIES})
  target_include_directories(phmod_static PRIVATE ${MY_INCLUDES})
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


  if (BUILD_SHARED_LIBRARIES)
    # Shared library
    add_library(gridsysmod SHARED ${SOURCES_GRIDSYSMOD})
    target_link_libraries(gridsysmod m ${OpenMP_C_LIBRARIES})
    target_include_directories(gridsysmod PRIVATE ${MY_INCLUDES})
    set_property(TARGET gridsysmod PROPERTY VERSION ${serial})
    set_property(TARGET gridsysmod PROPERTY SOVERSION ${soserial})
    install(TARGETS gridsysmod LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
  endif()

  # Static link library
  add_library(gridsysmod_static STATIC ${SOURCES_GRIDSYSMOD})
  target_link_libraries(gridsysmod_static m ${OpenMP_C_LIBRARIES})
  target_include_directories(gridsysmod_static PRIVATE ${MY_INCLUDES})
  set_property(TARGET gridsysmod_static PROPERTY VERSION ${serial})
  set_property(TARGET gridsysmod_static PROPERTY SOVERSION ${soserial})
  set_property(TARGET gridsysmod_static PROPERTY OUTPUT_NAME gridsysmod)
  install(TARGETS gridsysmod_static ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

  # Header file
  install(FILES ${PROJECT_SOURCE_DIR}/c/gridsys.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()
