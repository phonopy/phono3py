cmake_minimum_required(VERSION 3.25...3.31)

#[=============================================================================[
#                           Basic project definition                           #
]=============================================================================]

list(APPEND CMAKE_MESSAGE_CONTEXT phonopy)
project(phono3py
        VERSION ${SKBUILD_PROJECT_VERSION}
        DESCRIPTION "A simulation package of phonon-phonon interaction related properties"
        HOMEPAGE_URL https://github.com/phonopy/phono3py
        LANGUAGES C CXX
)

# Specify C/C++ standard used in the project.
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


#[=============================================================================[
#                                   Options                                   #
]=============================================================================]

include(CMakeDependentOption)
include(FeatureSummary)

option(PHONO3PY_SHARED_LIBS "Phono3py: Build as a shared library" ${PROJECT_IS_TOP_LEVEL})
option(PHONO3PY_USE_OMP "Phono3py: Build with OpenMP support" ON)
option(PHONO3PY_USE_MTBLAS "Phono3py: Use multithread BLAS if it exists" ON)

if (DEFINED BUILD_PHPHCALC_LIB
        OR DEFINED BUILD_PHONONCALC_LIB
        OR DEFINED BUILD_RECGRID_LIB
        OR DEFINED BUILD_GRIDSYS_LIB
        OR DEFINED PHONO3PY_WITH_Fortran
        OR DEFINED PHONO3PY_WITH_TESTS
)
    if (PHONO3PY_IGNORE_OPTIONS)
        set(message_type WARNING)
    else ()
        set(message_type SEND_ERROR)
    endif ()
    message(${message_type}
            "You have set an option that has been removed\n"
            "Please remove it or set PHONO3PY_IGNORE_OPTIONS to ignore this message\n"
            "BUILD_PHPHCALC_LIB = ${BUILD_PHPHCALC_LIB}\n"
            "BUILD_PHONONCALC_LIB = ${BUILD_PHONONCALC_LIB}\n"
            "BUILD_RECGRID_LIB = ${BUILD_RECGRID_LIB}\n"
            "BUILD_GRIDSYS_LIB = ${BUILD_GRIDSYS_LIB}\n"
            "PHONO3PY_WITH_Fortran = ${PHONO3PY_WITH_Fortran}\n"
            "PHONO3PY_WITH_TESTS = ${PHONO3PY_WITH_TESTS}"
    )
    if (BUILD_GRIDSYS_LIB OR PHONO3PY_WITH_Fortran)
        message(${message_type}
                "The non-python libraries have been removed.\n"
                "Please contact us and tell us where and how you are using it:\n"
                "https://github.com/phonopy/phono3py"
        )
    endif ()
endif ()

#[=============================================================================[
#                            Project configuration                            #
]=============================================================================]

include(GNUInstallDirs)

set(BUILD_SHARED_LIBS ${PHONO3PY_SHARED_LIBS})

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

#[=============================================================================[
#                              External packages                              #
]=============================================================================]

if (PHONOPY_USE_OMP)
    find_package(OpenMP REQUIRED COMPONENTS C)
endif ()
find_package(Python 3.9 REQUIRED
        COMPONENTS Development.Module
)
find_package(nanobind REQUIRED CONFIG)

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

#[=============================================================================[
#                               Main definition                               #
]=============================================================================]

add_library(phono3py_lib)
set_target_properties(phono3py_lib PROPERTIES
        VERSION ${PROJECT_VERSION}
        OUTPUT_NAME phono3py
)
nanobind_add_module(_phono3py)
nanobind_add_module(_phononcalc)
nanobind_add_module(_recgrid)
add_subdirectory(c)
