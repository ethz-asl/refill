cmake_minimum_required(VERSION 2.8.3)
project(refill_catkin)

add_definitions(-std=c++1y)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

find_package(Eigen3 REQUIRED)
include_directories(${Eigen_INCLUDE_DIRS})

get_filename_component(REFILL_MAIN_DIR ${CMAKE_SOURCE_DIR} DIRECTORY)

include_directories(SYSTEM
  ../include
)

# LIBRARY
file(GLOB_RECURSE SRC_LIST "../src/*.cc")
file(GLOB_RECURSE TEST_SOURCES "../src/*_test.cc")
list(REMOVE_ITEM SRC_LIST ${TEST_SOURCES})

cs_add_library(refill ${SRC_LIST})

# Tests
catkin_add_gtest(refill_test ${TEST_SOURCES})
target_link_libraries(refill_test refill ${GTEST_BOTH_LIBRARIES} ${catkin_LIBRARIES} -lpthread -lm)

cs_install()
cs_export(
  INCLUDE_DIRS ../include
)
