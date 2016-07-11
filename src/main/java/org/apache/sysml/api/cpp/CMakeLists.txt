cmake_minimum_required ( VERSION 3.1 )
include_directories( "$ENV{JAVA_HOME}/include")
include_directories( "$ENV{JAVA_HOME}/include/linux")
include_directories( "$ENV{JAVA_HOME}/include/darwin")
include_directories( "$ENV{EIGEN3_INCLUDE_DIR}" )


enable_language(CXX)

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

project (SystemML)
add_library(SystemML SHARED SystemML.cpp)
