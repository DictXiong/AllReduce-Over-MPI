cmake_minimum_required(VERSION 3.0.2)
project(aompiproj)
set(CMAKE_C_COMPILER "gcc")
set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS} -O2 -Wall -Wextra -Wreorder -fPIC -fopenmp -mavx2")

find_package(glog REQUIRED)

add_executable(aompi allreduce_over_mpi.cpp)
target_link_libraries(aompi glog pthread)
