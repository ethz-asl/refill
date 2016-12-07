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
file(GLOB_RECURSE REMOVE_SOURCES "../src/*_test.cc")
list(REMOVE_ITEM SRC_LIST ${REMOVE_SOURCES})

cs_add_library(refill ${SRC_LIST})

cs_install()
cs_export(
  INCLUDE_DIRS ../include
)
