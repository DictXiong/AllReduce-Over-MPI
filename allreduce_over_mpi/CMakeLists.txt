cmake_minimum_required(VERSION 3.0.2)
project(aompiproj)
SET(CMAKE_C_COMPILER "gcc")
SET(CMAKE_CXX_COMPILER "g++")

find_package(glog REQUIRED)

add_executable(aompi allreduce_over_mpi.cpp)
target_link_libraries(aompi glog pthread)
