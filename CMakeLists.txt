project(refill)
cmake_minimum_required(VERSION 2.8)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)

# Use C++14
set(CMAKE_CXX_STANDARD 14)

# SETUP EIGEN Change module path in order to resolve:
# https://github.com/Homebrew/legacy-homebrew/issues/44937
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} /usr/local/share/cmake/Modules)
find_package(Eigen3 REQUIRED)

# SETUP GTEST
enable_testing()
find_package(GTest REQUIRED)

# Setup glog
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(Glog)
if(NOT GLOG_FOUND)
  message(FATAL_ERROR "Can't find Glog")
endif()

include_directories(
  SYSTEM
  include
  ${GTEST_INCLUDE_DIRS}
  ${GLOG_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

# LIBRARY
file(GLOB_RECURSE SRC_LIST ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc)
file(GLOB_RECURSE TESTS src/*_test.cc)
file(GLOB_RECURSE EXAMPLES src/*_example.cc)
list(REMOVE_ITEM SRC_LIST ${TESTS})
list(REMOVE_ITEM SRC_LIST ${EXAMPLES})

add_library(${PROJECT_NAME} ${SRC_LIST})
target_link_libraries(${PROJECT_NAME} ${GLOG_LIBRARIES})

# Enable warnings and set warnings as errors
target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -pedantic -Werror)

# TESTS
add_executable(filtertests ${TESTS})
target_link_libraries(
  filtertests
  ${GTEST_BOTH_LIBRARIES}
  ${GLOG_LIBRARIES}
  ${PROJECT_NAME}
  -lpthread
  -lm)
add_test(AllTests filtertests)

# EXAMPLES
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin/examples)
foreach(ExampleSource ${EXAMPLES})
  string(REGEX REPLACE "(.+)/|.cc$" "" ExampleName ${ExampleSource})
  add_executable(${ExampleName} ${ExampleSource})
  target_link_libraries(${ExampleName} ${PROJECT_NAME})
endforeach()

# Qt Creator Compatibility ensuring include dir is shown See
# http://stackoverflow.com/questions/28384935/qtcreator-cmake-project-how-to-show-all-project-files
file(GLOB_RECURSE LibFiles "include/*")
add_custom_target(headers SOURCES ${LibFiles})
