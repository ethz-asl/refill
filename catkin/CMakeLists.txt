cmake_minimum_required(VERSION 2.8.3)
project(refill_catkin)

add_definitions(-std=c++1y)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

get_filename_component(REFILL_MAIN_DIR ${CMAKE_SOURCE_DIR} DIRECTORY)

include_directories(SYSTEM
  ../include
)

# LIBRARY
file(GLOB_RECURSE SRC_LIST "../src/*.cc")
file(GLOB_RECURSE TEST_SOURCES "../src/*_test.cc")
file(GLOB_RECURSE EXAMPLE_SOURCES "../src/*_example.cc")
list(REMOVE_ITEM SRC_LIST ${TEST_SOURCES})
list(REMOVE_ITEM SRC_LIST ${EXAMPLE_SOURCES})

cs_add_library(refill ${SRC_LIST})

# TESTS
catkin_add_gtest(refill_test ${TEST_SOURCES})
target_link_libraries(refill_test refill ${GTEST_BOTH_LIBRARIES} ${catkin_LIBRARIES} -lpthread -lm)

# EXAMPLES
foreach (EXAMPLE_SOURCE ${EXAMPLE_SOURCES})
  string(REGEX REPLACE "(.+)/|.cc" "" EXAMPLE_NAME ${EXAMPLE_SOURCE})
  cs_add_executable(${EXAMPLE_NAME} ${EXAMPLE_SOURCE})
  target_link_libraries(${EXAMPLE_NAME} refill)
endforeach (EXAMPLE_SOURCE)

cs_install()
cs_export(
  INCLUDE_DIRS ../include
)
