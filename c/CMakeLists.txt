target_sources(phono3py_lib PRIVATE
        # Reciprocal space grid library
        bzgrid.c
        grgrid.c
        lagrid.c
        snf3x3.c
        recgrid.c
        # Ph-ph calculation library
        collision_matrix.c
        fc3.c
        funcs.c
        imag_self_energy_with_g.c
        interaction.c
        isotope.c
        lapack_wrapper.c
        phono3py.c
        pp_collision.c
        real_self_energy.c
        real_to_reciprocal.c
        reciprocal_to_normal.c
        tetrahedron_method.c
        triplet.c
        triplet_grid.c
        triplet_iw.c
        # Phonon calculation library
        dynmat.c
        phonon.c
        phononcalc.c
        # gridsys
        gridsys.c
        niggli.c
)
target_sources(_phono3py PRIVATE _phono3py.cpp)
target_sources(_phononcalc PRIVATE _phononcalc.cpp)
target_sources(_recgrid PRIVATE _recgrid.cpp)
target_include_directories(phono3py_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(phono3py_lib PRIVATE
        BLAS::BLAS
        LAPACK::LAPACK
        $<$<BOOL:${PHONOPY_USE_OMP}>:OpenMP::OpenMP_C>
)

target_compile_definitions(phono3py_lib PRIVATE
        THM_EPSILON=1e-10
        $<$<BOOL:${PHONO3PY_USE_MTBLAS}>:MULTITHREADED_BLAS>
)

target_link_libraries(_phono3py PRIVATE
        phono3py_lib
)
install(TARGETS phono3py_lib)
# Fix RPATH for unix systems
set_target_properties(_phono3py _phononcalc _recgrid PROPERTIES
        INSTALL_RPATH "$<IF:$<BOOL:${APPLE}>,@loader_path,$ORIGIN>/${CMAKE_INSTALL_LIBDIR}"
)
# For windows systems we need to also install a copy of the dll files
install(TARGETS phono3py_lib RUNTIME DESTINATION .)
install(TARGETS _phono3py _phononcalc _recgrid DESTINATION .)
