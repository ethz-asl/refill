project(Refill)
cmake_minimum_required(VERSION 2.8)

# SETUP EIGEN
# Change module path in order to resolve https://github.com/Homebrew/legacy-homebrew/issues/44937
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} /usr/local/share/cmake/Modules)
find_package(Eigen3 REQUIRED)


# SETUP GTEST
enable_testing()
find_package(GTest REQUIRED)

include_directories(SYSTEM
    include
    ${GTEST_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

# LIBRARY
aux_source_directory(. SRC_LIST)
aux_source_directory(src SRC_LIST)
add_library(${PROJECT_NAME} ${SRC_LIST})

# TESTS
FILE(GLOB MyTestSources src/*_test.cc)
add_executable(filtertests ${MyTestSources})
target_link_libraries(filtertests ${GTEST_BOTH_LIBRARIES} Refill)
add_test(AllTests filtertests)

# Qt Creator Compatibility ensuring include dir is shown
# See http://stackoverflow.com/questions/28384935/qtcreator-cmake-project-how-to-show-all-project-files
FILE(GLOB_RECURSE LibFiles "include/*")
add_custom_target(headers SOURCES ${LibFiles})
