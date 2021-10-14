cmake_minimum_required(VERSION 3.0.2)
project(aompiproj)
set(CMAKE_C_COMPILER "gcc")
set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS} -O2 -Wall -Wextra -Wreorder -fPIC -fopenmp -mavx2")

find_package(glog REQUIRED)
find_package(MPI REQUIRED)

find_package(Git)
# 生成版本描述字符串类似 TAG-X-gHASH
execute_process(COMMAND ${GIT_EXECUTABLE} describe --abbrev=6 --dirty --always --tags
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE  GIT_REPO_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
# 获取最新 commit 日期，YYYY-MM-DD
execute_process(COMMAND ${GIT_EXECUTABLE} log -1 --format=%cd --date=short
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE  GIT_REPO_DATE
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
# 获取最新 commit Hash
execute_process(COMMAND ${GIT_EXECUTABLE} log -1 --format=%H
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE  GIT_REPO_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
add_definitions( -DGIT_REPO_VERSION=\"${GIT_REPO_VERSION}\")
add_definitions( -DGIT_REPO_DATE=\"${GIT_REPO_DATE}\")
add_definitions( -DGIT_REPO_HASH=\"${GIT_REPO_HASH}\")

include_directories(SYSTEM ${MPI_INCLUDE_PATH})
message("${MPI_INCLUDE_PATH}")

add_executable(allreduce_over_mpi benchmark.cpp)
target_link_libraries(allreduce_over_mpi ${MPI_CXX_LIBRARIES} glog pthread)