cmake_minimum_required(VERSION 3.20)

if(DEFINED ENV{PHONO3PY_NO_C_EXT} AND NOT "$ENV{PHONO3PY_NO_C_EXT}" STREQUAL "0")
    # Declare project with no language so CMake skips C/C++ compiler detection.
    # scikit-build-core's build step still calls CMake even with no targets, so the
    # project() call has to happen before we bail out.
    project(${SKBUILD_PROJECT_NAME} NONE)
    message(STATUS "PHONO3PY_NO_C_EXT is set; skipping _phono3py and _phononcalc")
    return()
endif()

option(PHONO3PY_USE_OMP "Option to search OpenMP library" ON)
option(PHONO3PY_USE_MTBLAS "Use multithread BLAS if it exists" ON)
option(BUILD_WITHOUT_LAPACKE "Option to build without LAPACKE" ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
project(${SKBUILD_PROJECT_NAME})
set(DEV_MODULE Development.Module)
find_package(
    Python 3.8 REQUIRED
    COMPONENTS Interpreter ${DEV_MODULE}
    OPTIONAL_COMPONENTS Development.SABIModule)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE
        Release
        CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
                                                 "MinSizeRel" "RelWithDebInfo")
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_SYSTEM_PREFIX_PATH: ${CMAKE_SYSTEM_PREFIX_PATH}")

if((NOT BUILD_WITHOUT_LAPACKE)
   AND USE_CONDA_PATH
   AND DEFINED ENV{CONDA_PREFIX})
    message(STATUS "$ENV{CONDA_PREFIX}")
    set(CMAKE_MODULE_PATH $ENV{CONDA_PREFIX})
    set(MY_INCLUDES $ENV{CONDA_PREFIX}/include ${PROJECT_SOURCE_DIR}/c)
    link_directories($ENV{CONDA_PREFIX}/lib)
else()
    set(MY_INCLUDES ${PROJECT_SOURCE_DIR}/c)
endif()

if(PHONO3PY_USE_OMP)
    message(STATUS "Find OpenMP library")
    find_package(OpenMP)

    if(OpenMP_FOUND)
        message(STATUS "OpenMP libs: ${OpenMP_C_LIBRARIES}")
        message(STATUS "OpenMP flags: ${OpenMP_C_FLAGS}")
    endif()
else()
    message(STATUS "OpenMP is not used.")
    set(OpenMP_FOUND OFF) # cmake-lint: disable=C0103
endif()

if(NOT BUILD_WITHOUT_LAPACKE)
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

    if(BLAS_LIBRARIES MATCHES "libmkl")
        message(STATUS "MKL detected.")

        if(PHONO3PY_USE_MTBLAS)
            message(
                STATUS "Set C-macro MULTITHREADED_BLAS to avoid nested OpenMP calls."
            )
        endif()
    endif()

    if(BLAS_LIBRARIES MATCHES "libopenblas")
        message(STATUS "OpenBLAS detected.")

        if(PHONO3PY_USE_MTBLAS)
            message(
                STATUS "Set C-macro MULTITHREADED_BLAS to avoid nested OpenMP calls."
            )
        endif()
    endif()
endif()

# ###################################################################################
# Reciprocal space grid library (internal, linked into _phono3py) #
# ###################################################################################
set(SOURCES_RECGRID
    ${PROJECT_SOURCE_DIR}/c/bzgrid.c ${PROJECT_SOURCE_DIR}/c/grgrid.c
    ${PROJECT_SOURCE_DIR}/c/lagrid.c ${PROJECT_SOURCE_DIR}/c/snf3x3.c
    ${PROJECT_SOURCE_DIR}/c/recgrid.c)
add_library(recgrid_lib STATIC ${SOURCES_RECGRID})

# ###################################################################################
# Ph-ph calculation library (internal, linked into _phono3py) #
# ###################################################################################
set(SOURCES_PHPHCALC
    ${PROJECT_SOURCE_DIR}/c/collision_matrix.c
    ${PROJECT_SOURCE_DIR}/c/fc3.c
    ${PROJECT_SOURCE_DIR}/c/imag_self_energy_with_g.c
    ${PROJECT_SOURCE_DIR}/c/interaction.c
    ${PROJECT_SOURCE_DIR}/c/isotope.c
    ${PROJECT_SOURCE_DIR}/c/lapack_wrapper.c
    ${PROJECT_SOURCE_DIR}/c/phono3py.c
    ${PROJECT_SOURCE_DIR}/c/funcs.c
    ${PROJECT_SOURCE_DIR}/c/pp_collision.c
    ${PROJECT_SOURCE_DIR}/c/real_self_energy.c
    ${PROJECT_SOURCE_DIR}/c/real_to_reciprocal.c
    ${PROJECT_SOURCE_DIR}/c/reciprocal_to_normal.c
    ${PROJECT_SOURCE_DIR}/c/tetrahedron_method.c
    ${PROJECT_SOURCE_DIR}/c/triplet.c
    ${PROJECT_SOURCE_DIR}/c/triplet_grid.c
    ${PROJECT_SOURCE_DIR}/c/triplet_iw.c)
add_library(phphcalc_lib STATIC ${SOURCES_PHPHCALC})

if(BUILD_WITHOUT_LAPACKE)
    if(OpenMP_FOUND)
        target_link_libraries(phphcalc_lib recgrid_lib OpenMP::OpenMP_C)
    else()
        target_link_libraries(phphcalc_lib recgrid_lib)
    endif()

    target_compile_definitions(phphcalc_lib PRIVATE NO_INCLUDE_LAPACKE
                                                    THM_EPSILON=1e-10)
else()
    if(OpenMP_FOUND)
        target_link_libraries(phphcalc_lib recgrid_lib BLAS::BLAS LAPACK::LAPACK
                              OpenMP::OpenMP_C)
    else()
        target_link_libraries(phphcalc_lib recgrid_lib BLAS::BLAS LAPACK::LAPACK)
    endif()

    if(BLAS_LIBRARIES MATCHES "libmkl")
        if(PHONO3PY_USE_MTBLAS)
            target_compile_definitions(
                phphcalc_lib PRIVATE MKL_BLAS MULTITHREADED_BLAS THM_EPSILON=1e-10)
        else()
            target_compile_definitions(phphcalc_lib PRIVATE MKL_BLAS
                                                            THM_EPSILON=1e-10)
        endif()
    endif()

    if(BLAS_LIBRARIES MATCHES "libopenblas")
        if(PHONO3PY_USE_MTBLAS)
            target_compile_definitions(phphcalc_lib PRIVATE MULTITHREADED_BLAS
                                                            THM_EPSILON=1e-10)
        else()
            target_compile_definitions(phphcalc_lib PRIVATE THM_EPSILON=1e-10)
        endif()
    endif()
endif()

target_include_directories(phphcalc_lib PRIVATE ${MY_INCLUDES})

# ###################################################################################
# Phonon calculation library (internal, linked into _phononcalc) #
# ###################################################################################
set(SOURCES_PHONONCALC
    ${PROJECT_SOURCE_DIR}/c/dynmat.c ${PROJECT_SOURCE_DIR}/c/lapack_wrapper.c
    ${PROJECT_SOURCE_DIR}/c/phonon.c ${PROJECT_SOURCE_DIR}/c/phononcalc.c)
add_library(phononcalc_lib STATIC ${SOURCES_PHONONCALC})

if(BUILD_WITHOUT_LAPACKE)
    if(OpenMP_FOUND)
        target_link_libraries(phononcalc_lib PRIVATE OpenMP::OpenMP_C)
    endif()

    target_compile_definitions(phononcalc_lib PRIVATE NO_INCLUDE_LAPACKE)
else()
    if(OpenMP_FOUND)
        target_link_libraries(phononcalc_lib PRIVATE BLAS::BLAS LAPACK::LAPACK
                                                     OpenMP::OpenMP_C)
    else()
        target_link_libraries(phononcalc_lib PRIVATE BLAS::BLAS LAPACK::LAPACK)
    endif()

    if(BLAS_LIBRARIES MATCHES "libmkl")
        target_compile_definitions(phononcalc_lib PRIVATE MKL_BLAS
                                                          MULTITHREADED_BLAS)
    endif()

    if(BLAS_LIBRARIES MATCHES "libopenblas")
        target_compile_definitions(phononcalc_lib PRIVATE MULTITHREADED_BLAS)
    endif()
endif()

target_include_directories(phononcalc_lib PRIVATE ${MY_INCLUDES})

# ###################################################################################
# nanobind modules #
# ###################################################################################
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_phono3py STABLE_ABI ${PROJECT_SOURCE_DIR}/c/phono3py.h
                    ${PROJECT_SOURCE_DIR}/c/_phono3py.cpp)
nanobind_add_module(_phononcalc STABLE_ABI ${PROJECT_SOURCE_DIR}/c/phononcalc.h
                    ${PROJECT_SOURCE_DIR}/c/_phononcalc.cpp)

target_link_libraries(_phono3py PRIVATE phphcalc_lib)
target_link_libraries(_phononcalc PRIVATE phononcalc_lib)

if(BUILD_WITHOUT_LAPACKE)
    target_compile_definitions(_phono3py PRIVATE NO_INCLUDE_LAPACKE
                                                 THM_EPSILON=1e-10)
else()
    if(BLAS_LIBRARIES MATCHES "libmkl")
        target_compile_definitions(_phono3py PRIVATE MKL_BLAS THM_EPSILON=1e-10)
    else()
        target_compile_definitions(_phono3py PRIVATE THM_EPSILON=1e-10)
    endif()
endif()

install(TARGETS _phono3py LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})
install(TARGETS _phononcalc LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})
