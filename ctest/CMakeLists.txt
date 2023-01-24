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
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.tar.gz
    URL_HASH MD5=e82199374acdfda3f425331028eb4e2a)
  FetchContent_MakeAvailable(googletest)
endif()

foreach(TESTCASE IN ITEMS test_gridsys)
  add_executable(${TESTCASE} ${TESTCASE}.cpp)
  target_link_libraries(
    ${TESTCASE}
    PUBLIC gridsys
    PRIVATE GTest::gtest GTest::gtest_main)
  target_include_directories(${TESTCASE} PUBLIC ${PROJECT_SOURCE_DIR}/c)
  gtest_discover_tests(${TESTCASE})
endforeach()