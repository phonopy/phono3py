cmake_minimum_required(VERSION 3.20)

option(BUILD_PHPHCALC_LIB "Option to build phph calculation module" OFF)
option(BUILD_PHONONCALC_LIB "Option to build phonon calculation module" OFF)
option(BUILD_GRIDSYS_LIB "Option to build gridsys module" OFF)
option(PHONO3PY_WITH_Fortran "enable fortran interface" OFF)
option(PHONO3PY_USE_OMP "Option to search OpenMP library" ON)
option(PHONO3PY_USE_MTBLAS "Use multithread BLAS if it exists" ON)
option(PHONO3PY_WITH_TESTS "build unit tests" OFF)
option(BUILD_SHARED_LIBS "Option to build shared library" OFF)

if(PHONO3PY_WITH_Fortran)
    enable_language(Fortran)
    set(BUILD_GRIDSYS_LIB ON)
endif()

if(PHONO3PY_WITH_TESTS)
    set(BUILD_SHARED_LIBS ON)
    set(BUILD_GRIDSYS_LIB ON)
endif()

if((NOT BUILD_PHPHCALC_LIB)
   AND (NOT BUILD_PHONONCALC_LIB)
   AND (NOT BUILD_GRIDSYS_LIB))
    set(BUILD_NANOBIND_MODULE ON)
    message(STATUS "Build nanobind module of ${SKBUILD_PROJECT_NAME}")
else()
    set(BUILD_NANOBIND_MODULE OFF)
endif()

if(BUILD_NANOBIND_MODULE)
    project(${SKBUILD_PROJECT_NAME})
    set(DEV_MODULE Development.Module)
    find_package(
        Python 3.8 REQUIRED
        COMPONENTS Interpreter ${DEV_MODULE}
        OPTIONAL_COMPONENTS Development.SABIModule)
else()
    project(phono3py C)
    set(CMAKE_MACOSX_RPATH 1)
    set(CMAKE_C_FLAGS_RELEASE "-Wall -O2")
    set(CMAKE_C_FLAGS_DEBUG "-g -DLAGWARNING -DTHMWARNING")

    # Version numbers
    file(READ ${PROJECT_SOURCE_DIR}/phono3py/version.py version_file)
    string(REGEX MATCH "__version__.*([0-9]+)[.]([0-9]+)[.]([0-9]+)"
                 phono3py_version ${version_file})
    set(MAJOR_VERSION ${CMAKE_MATCH_1})
    set(MINOR_VERSION ${CMAKE_MATCH_2})
    set(MICRO_VERSION ${CMAKE_MATCH_3})
    set(SERIAL "${MAJOR_VERSION}.${MINOR_VERSION}.${MICRO_VERSION}")
    set(SOSERIAL "1")
    include(GNUInstallDirs)
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE
        Release
        CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
                                                 "MinSizeRel" "RelWithDebInfo")
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_SYSTEM_PREFIX_PATH: ${CMAKE_SYSTEM_PREFIX_PATH}")

if(USE_CONDA_PATH AND DEFINED ENV{CONDA_PREFIX})
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

if(BUILD_PHPHCALC_LIB
   OR BUILD_PHONONCALC_LIB
   OR BUILD_NANOBIND_MODULE)
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
        message(STATUS "MKL detected: Set C-macro MKL_LAPACKE.")

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
# Build phphcalc module #
# ###################################################################################
if(BUILD_PHPHCALC_LIB OR BUILD_NANOBIND_MODULE)
    # Source code
    set(SOURCES_PHPHCALC
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

    if(BUILD_SHARED_LIBS)
        # Shared library
        add_library(phphcalc_lib SHARED ${SOURCES_PHPHCALC})

        if(OpenMP_FOUND)
            target_link_libraries(phphcalc_lib PRIVATE BLAS::BLAS LAPACK::LAPACK
                                                       OpenMP::OpenMP_C)
        else()
            target_link_libraries(phphcalc_lib PRIVATE BLAS::BLAS LAPACK::LAPACK)
        endif()

        target_include_directories(phphcalc_lib PRIVATE ${MY_INCLUDES})

        if(BLAS_LIBRARIES MATCHES "libmkl")
            if(PHONO3PY_USE_MTBLAS)
                target_compile_definitions(
                    phphcalc_lib PRIVATE MKL_LAPACKE MULTITHREADED_BLAS
                                         THM_EPSILON=1e-10)
            else()
                target_compile_definitions(phphcalc_lib PRIVATE MKL_LAPACKE
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
    else()
        # Static link library
        add_library(phphcalc_lib STATIC ${SOURCES_PHPHCALC})

        if(OpenMP_FOUND)
            target_link_libraries(phphcalc_lib BLAS::BLAS LAPACK::LAPACK
                                  OpenMP::OpenMP_C)
        else()
            target_link_libraries(phphcalc_lib BLAS::BLAS LAPACK::LAPACK)
        endif()

        target_include_directories(phphcalc_lib PRIVATE ${MY_INCLUDES})

        if(BLAS_LIBRARIES MATCHES "libmkl")
            if(PHONO3PY_USE_MTBLAS)
                target_compile_definitions(
                    phphcalc_lib PRIVATE MKL_LAPACKE MULTITHREADED_BLAS
                                         THM_EPSILON=1e-10)
            else()
                target_compile_definitions(phphcalc_lib PRIVATE MKL_LAPACKE
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

    if(NOT BUILD_NANOBIND_MODULE)
        if(BUILD_SHARED_LIBS)
            set_property(TARGET phphcalc_lib PROPERTY VERSION ${SERIAL})
            set_property(TARGET phphcalc_lib PROPERTY SOVERSION ${SOSERIAL})
            install(TARGETS phphcalc_lib LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
        else()
            set_property(TARGET phphcalc_lib PROPERTY VERSION ${SERIAL})
            set_property(TARGET phphcalc_lib PROPERTY SOVERSION ${SOSERIAL})
            set_property(TARGET phphcalc_lib PROPERTY OUTPUT_NAME phphcalc_lib)
            install(TARGETS phphcalc_lib ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
        endif()

        # Header file
        install(FILES ${PROJECT_SOURCE_DIR}/c/phono3py.h
                DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    endif()
endif()

# ###################################################################################
# phononcalc #
# ###################################################################################
if(BUILD_PHONONCALC_LIB OR BUILD_NANOBIND_MODULE)
    # Source code
    set(SOURCES_PHONONCALC
        ${PROJECT_SOURCE_DIR}/c/dynmat.c ${PROJECT_SOURCE_DIR}/c/lapack_wrapper.c
        ${PROJECT_SOURCE_DIR}/c/phonon.c ${PROJECT_SOURCE_DIR}/c/phononcalc.c)

    if(BUILD_SHARED_LIBS)
        # Shared library
        add_library(phononcalc_lib SHARED ${SOURCES_PHONONCALC})

        if(OpenMP_FOUND)
            target_link_libraries(phononcalc_lib BLAS::BLAS LAPACK::LAPACK
                                  OpenMP::OpenMP_C)
        else()
            target_link_libraries(phononcalc_lib BLAS::BLAS LAPACK::LAPACK)
        endif()

        target_include_directories(phononcalc_lib PRIVATE ${MY_INCLUDES})

        if(BLAS_LIBRARIES MATCHES "libmkl")
            target_compile_definitions(phononcalc_lib PRIVATE MKL_LAPACKE
                                                              MULTITHREADED_BLAS)
        endif()

        if(BLAS_LIBRARIES MATCHES "libopenblas")
            target_compile_definitions(phononcalc_lib PRIVATE MULTITHREADED_BLAS)
        endif()

    else()
        # Static link library
        add_library(phononcalc_lib STATIC ${SOURCES_PHONONCALC})

        if(OpenMP_FOUND)
            target_link_libraries(phononcalc_lib PRIVATE BLAS::BLAS LAPACK::LAPACK
                                                         OpenMP::OpenMP_C)
        else()
            target_link_libraries(phononcalc_lib PRIVATE BLAS::BLAS LAPACK::LAPACK)
        endif()

        target_include_directories(phononcalc_lib PRIVATE ${MY_INCLUDES})

        if(BLAS_LIBRARIES MATCHES "libmkl")
            target_compile_definitions(phononcalc_lib PRIVATE MKL_LAPACKE
                                                              MULTITHREADED_BLAS)
        endif()

        if(BLAS_LIBRARIES MATCHES "libopenblas")
            target_compile_definitions(phononcalc_lib PRIVATE MULTITHREADED_BLAS)
        endif()
    endif()

    if(NOT BUILD_NANOBIND_MODULE)
        if(BUILD_SHARED_LIBS)
            set_property(TARGET phononcalc_lib PROPERTY VERSION ${SERIAL})
            set_property(TARGET phononcalc_lib PROPERTY SOVERSION ${SOSERIAL})
            install(TARGETS phononcalc_lib
                    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

        else()
            set_property(TARGET phononcalc_lib PROPERTY VERSION ${SERIAL})
            set_property(TARGET phononcalc_lib PROPERTY SOVERSION ${SOSERIAL})
            set_property(TARGET phononcalc_lib PROPERTY OUTPUT_NAME phononcalc_lib)
            install(TARGETS phononcalc_lib
                    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
        endif()

        # Header file
        install(FILES ${PROJECT_SOURCE_DIR}/c/phononcalc.h
                DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    endif()
endif()

# ###################################################################################
# grid #
# ###################################################################################
if(BUILD_GRIDSYS_LIB)
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

    if(BUILD_SHARED_LIBS)
        # Shared library
        add_library(gridsys SHARED ${SOURCES_GRIDSYS})

        if(OpenMP_FOUND)
            target_link_libraries(gridsys PRIVATE OpenMP::OpenMP_C)
        endif()

        target_include_directories(gridsys PRIVATE ${MY_INCLUDES})
        target_compile_definitions(gridsys PRIVATE THM_EPSILON=1e-10)
        set_property(TARGET gridsys PROPERTY VERSION ${SERIAL})
        set_property(TARGET gridsys PROPERTY SOVERSION ${SOSERIAL})
        install(TARGETS gridsys LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
    else()
        # Static link library
        add_library(gridsys_static STATIC ${SOURCES_GRIDSYS})

        if(OpenMP_FOUND)
            target_link_libraries(gridsys_static PRIVATE OpenMP::OpenMP_C)
        endif()

        target_include_directories(gridsys_static PRIVATE ${MY_INCLUDES})
        target_compile_definitions(gridsys_static PRIVATE THM_EPSILON=1e-10)
        set_property(TARGET gridsys_static PROPERTY VERSION ${SERIAL})
        set_property(TARGET gridsys_static PROPERTY SOVERSION ${SOSERIAL})
        set_property(TARGET gridsys_static PROPERTY OUTPUT_NAME gridsys)
        install(TARGETS gridsys_static ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
    endif()

    # Header file
    install(FILES ${PROJECT_SOURCE_DIR}/c/gridsys.h
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()

if(PHONO3PY_WITH_Fortran)
    add_subdirectory(fortran)
endif()

if(PHONO3PY_WITH_TESTS)
    if(PHONO3PY_WITH_Fortran)
        set(LIB_MOD_DIR ${CMAKE_CURRENT_BINARY_DIR}/fortran)
    endif()

    add_subdirectory(ctest)
endif()

if(BUILD_NANOBIND_MODULE)
    set_target_properties(phphcalc_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)
    set_target_properties(phononcalc_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)

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

    target_compile_definitions(_phono3py PRIVATE THM_EPSILON=1e-10)
    install(TARGETS _phono3py LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})
    install(TARGETS _phononcalc LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})
endif()
