set(NAME gridsysf)

add_executable(test_${NAME} test_${NAME}.f90)
target_link_libraries(test_${NAME} PRIVATE gridsysf)
# cmake-lint: disable=C0307
set_target_properties(test_${NAME} PROPERTIES Fortran_MODULE_DIRECTORY
                                              ${LIB_MOD_DIR})

add_test(
    NAME ${NAME}
    COMMAND $<TARGET_FILE:test_${NAME}> ${CMAKE_CURRENT_BINARY_DIR}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
