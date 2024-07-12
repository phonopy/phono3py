if(USE_OpenMP)
    find_package(
        OpenMP
        COMPONENTS Fortran
        REQUIRED)
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} ${OpenMP_Fortran_FLAGS}")
endif()

add_library(gridsysf_static STATIC gridsysf.f90)
set_property(TARGET gridsysf_static PROPERTY VERSION ${PROJECT_VERSION})
set_property(TARGET gridsysf_static PROPERTY OUTPUT_NAME gridsysf)
target_link_libraries(gridsysf_static PRIVATE gridsys_static)
install(TARGETS gridsysf_static ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(BUILD_SHARED_LIBRARIES)
    add_library(gridsysf SHARED gridsysf.f90)
    set_property(TARGET gridsysf PROPERTY VERSION ${PROJECT_VERSION})
    set_property(TARGET gridsysf PROPERTY SOVERSION ${soserial})
    target_link_libraries(gridsysf PUBLIC gridsys)
    install(TARGETS gridsysf LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/gridsysf.mod
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
