cmake_minimum_required(VERSION 3.5.1)
project(KMeans)
set(CMAKE_CXX_STANDARD 11)

# Main executable
include_directories(include)
add_executable(KMeans src/main.cpp)
target_link_libraries(KMeans boost_program_options)

# Google Test unit tests
enable_testing()
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIR})
add_executable(unit_tests src/unit_tests.cpp)
target_link_libraries(unit_tests ${GTEST_LIBRARIES} pthread)

add_test(
    NAME unit_tests
    COMMAND unit_tests
)
