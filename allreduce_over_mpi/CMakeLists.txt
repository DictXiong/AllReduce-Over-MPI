cmake_minimum_required(VERSION 3.0.2)
project(aompiproj)
set(CMAKE_C_COMPILER "gcc")
set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS} -O2 -Wall -Wextra -Wreorder -fPIC -fopenmp -mavx2")

find_package(glog REQUIRED)
find_package(MPI REQUIRED)

include_directories(SYSTEM ${MPI_INCLUDE_PATH})
message("${MPI_INCLUDE_PATH}")

add_executable(allreduce_over_mpi benchmark.cpp)
target_link_libraries(allreduce_over_mpi ${MPI_CXX_LIBRARIES} glog pthread)