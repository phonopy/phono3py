cmake_minimum_required(VERSION 3.15)
project(phono3py C)

option(PHONO3PY "Option to build phono3py module" OFF)
option(PHONONCALC "Option to build phonon calculation module" OFF)
option(GRIDSYS "Option to build gridsys module" OFF)
option(WITH_Fortran "enable fortran interface" OFF)
option(USE_OpenMP "Option to search OpenMP library" ON)
option(WITH_TESTS "build unit tests" OFF)
option(BUILD_SHARED_LIBRARIES "Option to build shared library" OFF)
if(WITH_TESTS)
  set(BUILD_SHARED_LIBRARIES ON)
endif()

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
string(REGEX MATCH "__version__.*([0-9]+)[.]([0-9]+)[.]([0-9]+)"
             phono3py_version ${version_file})
set(MAJOR_VERSION ${CMAKE_MATCH_1})
set(MINOR_VERSION ${CMAKE_MATCH_2})
set(MICRO_VERSION ${CMAKE_MATCH_3})
set(SERIAL "${MAJOR_VERSION}.${MINOR_VERSION}.${MICRO_VERSION}")
set(SOSERIAL "1")

if(USE_CONDA_PATH)
  message(STATUS "$ENV{CONDA_PREFIX}")
  set(CMAKE_MODULE_PATH $ENV{CONDA_PREFIX})
  set(MY_INCLUDES $ENV{CONDA_PREFIX}/include ${PROJECT_SOURCE_DIR}/c)
  link_directories($ENV{CONDA_PREFIX}/lib)
else()
  set(MY_INCLUDES ${PROJECT_SOURCE_DIR}/c)
endif()

if(USE_OpenMP)
  message(STATUS "Find OpenMP library")
  find_package(OpenMP)

  if(OpenMP_FOUND)
    message(STATUS "OpenMP libs: ${OpenMP_C_LIBRARIES}")
    message(STATUS "OpenMP flags: ${OpenMP_C_FLAGS}")
  endif()
endif()

if(PHONO3PY OR PHONONCALC)
  find_package(BLAS REQUIRED) # set BLAS_LIBRARIES

  if(BLAS_FOUND)
    message(STATUS "BLAS libs: ${BLAS_LIBRARIES}")
    message(STATUS "BLAS flags: ${BLAS_LINKER_FLAGS}")
  endif()

  find_package(LAPACK REQUIRED) # set LAPACK_LIBRARIES

  if(LAPACK_FOUND)
    message(STATUS "LAPACK libs: ${LAPACK_LIBRARIES}")
    message(STATUS "LAPACK flags: ${LAPACK_LINKER_FLAGS}")
  endif()
endif()

if(PHONO3PY OR PHONONCALC)
  if(BLAS_LIBRARIES MATCHES "libmkl")
    message(STATUS "MKL detected: Set C-macros MKL_LAPACKE MULTITHREADED_BLAS")
  endif()

  if(BLAS_LIBRARIES MATCHES "libopenblas")
    message(STATUS "OpenBLAS detected: Set C-macro MULTITHREADED_BLAS")
  endif()
endif()

# ##############################################################################
# phono3py #
# ##############################################################################
if(PHONO3PY)
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

  if(BUILD_SHARED_LIBRARIES)
    # Shared library
    add_library(ph3py SHARED ${SOURCES_PHONO3PY})
    target_link_libraries(ph3py m ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES}
                          ${OpenMP_C_LIBRARIES})
    target_include_directories(ph3py PRIVATE ${MY_INCLUDES})
    target_compile_options(ph3py PRIVATE ${OpenMP_C_FLAGS})

    if(BLAS_LIBRARIES MATCHES "libmkl")
      target_compile_definitions(ph3py PRIVATE MKL_LAPACKE THM_EPSILON=1e-10)
    endif()

    if(BLAS_LIBRARIES MATCHES "libopenblas")
      target_compile_definitions(ph3py PRIVATE THM_EPSILON=1e-10)
    endif()

    set_property(TARGET ph3py PROPERTY VERSION ${SERIAL})
    set_property(TARGET ph3py PROPERTY SOVERSION ${SOSERIAL})
    install(TARGETS ph3py LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
  endif()

  # Static link library
  add_library(ph3py_static STATIC ${SOURCES_PHONO3PY})
  target_link_libraries(ph3py_static m ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES}
                        ${OpenMP_C_LIBRARIES})
  target_include_directories(ph3py_static PRIVATE ${MY_INCLUDES})
  target_compile_options(ph3py_static PRIVATE ${OpenMP_C_FLAGS})

  if(BLAS_LIBRARIES MATCHES "libmkl")
    target_compile_definitions(ph3py_static PRIVATE MKL_LAPACKE
                                                    THM_EPSILON=1e-10)
  endif()

  if(BLAS_LIBRARIES MATCHES "libopenblas")
    target_compile_definitions(ph3py_static PRIVATE THM_EPSILON=1e-10)
  endif()

  set_property(TARGET ph3py_static PROPERTY VERSION ${SERIAL})
  set_property(TARGET ph3py_static PROPERTY SOVERSION ${SOSERIAL})
  set_property(TARGET ph3py_static PROPERTY OUTPUT_NAME ph3py)
  install(TARGETS ph3py_static ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

  # Header file
  install(FILES ${PROJECT_SOURCE_DIR}/c/phono3py.h
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()

# ##############################################################################
# phononcalc #
# ##############################################################################
if(PHONONCALC)
  # Source code
  set(SOURCES_PHONONCALC
      ${PROJECT_SOURCE_DIR}/c/dynmat.c ${PROJECT_SOURCE_DIR}/c/lapack_wrapper.c
      ${PROJECT_SOURCE_DIR}/c/phonon.c ${PROJECT_SOURCE_DIR}/c/phononcalc.c)

  if(BUILD_SHARED_LIBRARIES)
    # Shared library
    add_library(phcalc SHARED ${SOURCES_PHONONCALC})
    target_link_libraries(phcalc m ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES}
                          ${OpenMP_C_LIBRARIES})
    target_include_directories(phcalc PRIVATE ${MY_INCLUDES})
    target_compile_options(phcalc PRIVATE ${OpenMP_C_FLAGS})

    if(BLAS_LIBRARIES MATCHES "libmkl")
      target_compile_definitions(phcalc PRIVATE MKL_LAPACKE MULTITHREADED_BLAS)
    endif()

    if(BLAS_LIBRARIES MATCHES "libopenblas")
      target_compile_definitions(phcalc PRIVATE MULTITHREADED_BLAS)
    endif()

    set_property(TARGET phcalc PROPERTY VERSION ${SERIAL})
    set_property(TARGET phcalc PROPERTY SOVERSION ${SOSERIAL})
    install(TARGETS phcalc LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
  endif()

  # Static link library
  add_library(phcalc_static STATIC ${SOURCES_PHONONCALC})
  target_link_libraries(phcalc_static m ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES}
                        ${OpenMP_C_LIBRARIES})
  target_include_directories(phcalc_static PRIVATE ${MY_INCLUDES})
  target_compile_options(phcalc_static PRIVATE ${OpenMP_C_FLAGS})

  if(BLAS_LIBRARIES MATCHES "libmkl")
    target_compile_definitions(phcalc_static PRIVATE MKL_LAPACKE
                                                     MULTITHREADED_BLAS)
  endif()

  if(BLAS_LIBRARIES MATCHES "libopenblas")
    target_compile_definitions(phcalc_static PRIVATE MULTITHREADED_BLAS)
  endif()

  set_property(TARGET phcalc_static PROPERTY VERSION ${SERIAL})
  set_property(TARGET phcalc_static PROPERTY SOVERSION ${SOSERIAL})
  set_property(TARGET phcalc_static PROPERTY OUTPUT_NAME phcalc)
  install(TARGETS phcalc_static ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

  # Header file
  install(FILES ${PROJECT_SOURCE_DIR}/c/phononcalc.h
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()

# ##############################################################################
# grid #
# ##############################################################################
if(GRIDSYS)
  # Source code
  set(SOURCES_GRIDSYS
      ${PROJECT_SOURCE_DIR}/c/bzgrid.c
      ${PROJECT_SOURCE_DIR}/c/grgrid.c
      ${PROJECT_SOURCE_DIR}/c/gridsys.c
      ${PROJECT_SOURCE_DIR}/c/lagrid.c
      ${PROJECT_SOURCE_DIR}/c/niggli.c
      ${PROJECT_SOURCE_DIR}/c/phonoc_utils.c
      ${PROJECT_SOURCE_DIR}/c/snf3x3.c
      ${PROJECT_SOURCE_DIR}/c/tetrahedron_method.c
      ${PROJECT_SOURCE_DIR}/c/triplet.c
      ${PROJECT_SOURCE_DIR}/c/triplet_grid.c
      ${PROJECT_SOURCE_DIR}/c/triplet_iw.c)

  if(BUILD_SHARED_LIBRARIES)
    # Shared library
    add_library(gridsys SHARED ${SOURCES_GRIDSYS})
    target_link_libraries(gridsys m ${OpenMP_C_LIBRARIES})
    target_include_directories(gridsys PRIVATE ${MY_INCLUDES})
    target_compile_options(gridsys PRIVATE ${OpenMP_C_FLAGS})
    target_compile_definitions(gridsys PRIVATE THM_EPSILON=1e-10)
    set_property(TARGET gridsys PROPERTY VERSION ${SERIAL})
    set_property(TARGET gridsys PROPERTY SOVERSION ${SOSERIAL})
    install(TARGETS gridsys LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
  endif()

  # Static link library
  add_library(gridsys_static STATIC ${SOURCES_GRIDSYS})
  target_link_libraries(gridsys_static m ${OpenMP_C_LIBRARIES})
  target_include_directories(gridsys_static PRIVATE ${MY_INCLUDES})
  target_compile_options(gridsys_static PRIVATE ${OpenMP_C_FLAGS})
  target_compile_definitions(gridsys_static PRIVATE THM_EPSILON=1e-10)
  set_property(TARGET gridsys_static PROPERTY VERSION ${SERIAL})
  set_property(TARGET gridsys_static PROPERTY SOVERSION ${SOSERIAL})
  set_property(TARGET gridsys_static PROPERTY OUTPUT_NAME gridsys)
  install(TARGETS gridsys_static ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

  # Header file
  install(FILES ${PROJECT_SOURCE_DIR}/c/gridsys.h
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()

if(WITH_TESTS)
  add_subdirectory(ctest)
endif()

if(WITH_Fortran)
  enable_language(Fortran)
  add_subdirectory(fortran)
endif()
