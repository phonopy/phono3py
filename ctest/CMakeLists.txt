enable_language(CXX)
enable_testing()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(GTest)
if(NOT GTest_FOUND)
  # pthread (required for GoogleTest)
  # https://stackoverflow.com/questions/1620918/cmake-and-libpthread
  set(THREADS_PREFER_PTHREAD_FLAG ON)
  find_package(Threads REQUIRED)

  # Fetch GoogleTest
  include(FetchContent)

  # cmake-lint inline directives must be specified at body scope. See
  # https://cmake-format.readthedocs.io/en/latest/lint-inline.html

  # cmake-lint: disable=C0301
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.tar.gz
    URL_HASH MD5=e82199374acdfda3f425331028eb4e2a)
  FetchContent_MakeAvailable(googletest)
endif()

foreach(testcase IN ITEMS test_gridsys test_niggli)
  add_executable(${testcase} ${testcase}.cpp)
  target_link_libraries(
    ${testcase}
    PUBLIC gridsys
    PRIVATE GTest::gtest GTest::gtest_main)
  target_include_directories(${testcase} PUBLIC ${PROJECT_SOURCE_DIR}/c)
  gtest_discover_tests(${testcase})
endforeach()
