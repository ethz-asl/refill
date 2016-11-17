cmake_minimum_required(VERSION 2.8.3)
project(refill_catkin)

set(CMAKE_CXX_STANDARD 14)

find_package(catkin REQUIRED)

catkin_package(
  CATKIN_DEPENDS catkin
)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../ ${CMAKE_CURRENT_BINARY_DIR}/refill)
